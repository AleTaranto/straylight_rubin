"""Build the enrichment caches shared by all 13 per-artifact analyses.

Three products, each cached so the driver can be re-run freely:

1. `star_pairs.parquet`  -- every (exposure, bright star) pair within 30 deg of
   the boresight, with the star's direction expressed in the camera, dome and sky
   frames. This is what lets us ask "is there always a star at a specific
   off-axis distance", which the CSV's own two star columns cannot answer.
2. `consdb.parquet` / `efd_20s.parquet` -- extended backwards to cover every
   inspected night. The existing caches start at 20250711 (the arc epoch), but
   tape/banana/sail/ghoul/pillow/shoe/nibble labels begin as early as 20250417,
   so without the backfill the telescope-state questions are unanswerable exactly
   where those artifacts live.

Run:  python build_star_geometry.py [--no-consdb] [--no-efd]
"""
from __future__ import annotations

# matplotlib.font_manager must be imported before the LSST stack on this cluster:
# the lsst-scipipe conda env has a broken pyexpat that only surfaces when
# something else pulls in xml.parsers.expat first.
import matplotlib  # noqa: F401
import matplotlib.font_manager  # noqa: F401

import argparse
import asyncio
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, search_around_sky
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))

import artifact_stats as A  # noqa: E402

CSV = "../data/raw/stray_light_20260506.csv"
YBSC = "../L3_baffle/rubin_one_year_output/ybsc_v5.fits"
CACHE = pathlib.Path("../outputs/artifact_stats/cache")
ENV_FILE = "../.env"

# Match radius. 30 deg covers every annulus we scan (0.5-25 deg) with margin.
MATCH_RADIUS_DEG = 30.0
# Magnitude floor for the pair table. V<4.5 keeps ~900 stars, which is deep
# enough for a dose-response curve while keeping the table ~6 M rows.
VMAG_MAX = 4.5

CONSDB_URL = "https://usdf-rsp.slac.stanford.edu/consdb"
EFD_NAME = "usdf_efd"
EFD_BUCKET = "20s"

# name -> (InfluxQL aggregate, topic)   [same topics as _build_arc_nb.py §3.4]
EFD_TOPICS = {
    "dome_az":        ('mean("positionActual")',    "lsst.sal.MTDome.azimuth"),
    "dome_az_cmd":    ('mean("positionCommanded")', "lsst.sal.MTDome.azimuth"),
    "dome_shutter_a": ('mean("positionActual0")',   "lsst.sal.MTDome.apertureShutter"),
    "dome_shutter_b": ('mean("positionActual1")',   "lsst.sal.MTDome.apertureShutter"),
    "mirror_cover_0": ('mean("actualPosition0")',   "lsst.sal.MTMount.mirrorCover"),
    "mirror_cover_1": ('mean("actualPosition1")',   "lsst.sal.MTMount.mirrorCover"),
    "mirror_cover_2": ('mean("actualPosition2")',   "lsst.sal.MTMount.mirrorCover"),
    "mirror_cover_3": ('mean("actualPosition3")',   "lsst.sal.MTMount.mirrorCover"),
    "louver_max":     ('max("positionActual0")',    "lsst.sal.MTDome.louvers"),
}


def consdb_token():
    """CONSDB_TOKEN from the environment, falling back to parsing .env."""
    tok = os.environ.get("CONSDB_TOKEN")
    if tok:
        return tok
    path = pathlib.Path(ENV_FILE)
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "CONSDB_TOKEN":
                tok = val.strip().strip("'\"")
                os.environ["CONSDB_TOKEN"] = tok
                return tok
    raise RuntimeError(
        f"CONSDB_TOKEN not in the environment and not found in {path}. "
        "Regenerate it with `python common/create_dot_env.py` (see README).")


# ======================================================================
# the analysis frame: every inspected night, so all 13 artifacts fit in it
# ======================================================================
def build_frame():
    df, n_cal = A.load_labels(CSV)
    A.verify_bit_mapping(df)
    insp = A.inspected_nights(df)
    frame = df[df["day_obs"].isin(insp)].copy()
    # Sorting by (night, seq_num) is load-bearing: the circular-shift null rolls
    # each night's labels along this axis, so it must be observation order.
    frame = frame.sort_values(["day_obs", "seq_num"]).reset_index(drop=True)
    print(f"dropped {n_cal} dark/flat/bias; {len(frame)} exposures on "
          f"{frame['day_obs'].nunique()} inspected nights "
          f"({frame['day_obs'].min()}-{frame['day_obs'].max()})")
    return frame


# ======================================================================
# bright-star pair table
# ======================================================================
def load_ybsc():
    """Yale BSC with V, B-V and the per-band magnitudes the L3 model uses.

    `horseshoe_detailer.py` evaluates its magnitude limit in the *observing band*,
    not in V, so the band magnitudes are carried here to let the analysis match
    that definition instead of approximating it.
    """
    t = Table.read(YBSC)
    cat = pd.DataFrame({
        "star_name": [str(x).strip() for x in t["Name"]],
        "vmag": np.asarray(t["Vmag"], dtype=float),
        "bv": np.asarray(t["B-V"], dtype=float),
        "ra": np.asarray(t["coord_ra"], dtype=float),
        "dec": np.asarray(t["coord_dec"], dtype=float),
    }).dropna(subset=["ra", "dec", "vmag"]).reset_index(drop=True)

    # Colour transforms copied from horseshoe_detailer.py so the band magnitudes
    # here are the same numbers that model uses.
    bv = cat["bv"].fillna(0.0).values
    V = cat["vmag"].values
    cat["g"] = V + 0.552 * bv - 0.099
    cat["r"] = V - 0.543 * bv + 0.128
    cat["i"] = V - 1.040 * bv + 0.312
    cat["z"] = V - 1.302 * bv + 0.417
    cat["y"] = V - 1.416 * bv + 0.504
    cat["u"] = V + 2.0 * bv - 0.7
    return cat


def build_star_pairs(frame, cat, out_path):
    """Cross-match and express each star's direction in three frames.

    The three frames are the whole point: a feature fixed in the camera implicates
    the optics, one fixed in the dome implicates the structure, one fixed on the
    sky implicates the atmosphere -- or, far more often, means we simply kept
    revisiting the same field.
    """
    sel = cat[cat["vmag"] < VMAG_MAX].reset_index(drop=True)
    print(f"cross-matching {len(frame)} exposures against {len(sel)} stars "
          f"(V < {VMAG_MAX}) within {MATCH_RADIUS_DEG} deg")

    bore = SkyCoord(frame["ra"].values * u.deg, frame["dec"].values * u.deg)
    stars = SkyCoord(sel["ra"].values * u.deg, sel["dec"].values * u.deg)
    i_exp, i_star, sep, _ = search_around_sky(bore, stars,
                                              MATCH_RADIUS_DEG * u.deg)
    print(f"  {len(i_exp)} pairs ({len(i_exp)/len(frame):.1f} per exposure)")

    loc = EarthLocation.of_site("rubin")
    # Chunked so peak memory stays modest; astropy's AltAz transform is the
    # accurate one (a hand-rolled airless version is off by ~0.4 deg, which is
    # more than the width of the annuli we care about).
    n = len(i_exp)
    pa_sky = np.empty(n, dtype=np.float32)
    pa_azalt = np.empty(n, dtype=np.float32)
    star_az = np.empty(n, dtype=np.float32)
    star_alt = np.empty(n, dtype=np.float32)
    CH = 400_000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        ie, is_ = i_exp[s:e], i_star[s:e]
        sub = frame.iloc[ie]
        t = Time(sub["exp_midpt_mjd"].values, format="mjd", scale="utc")
        aa = AltAz(location=loc, obstime=t)
        b = SkyCoord(sub["ra"].values * u.deg, sub["dec"].values * u.deg)
        st = SkyCoord(sel["ra"].values[is_] * u.deg, sel["dec"].values[is_] * u.deg)
        pa_sky[s:e] = b.position_angle(st).deg
        ba, sa = b.transform_to(aa), st.transform_to(aa)
        pa_azalt[s:e] = ba.position_angle(sa).deg
        star_az[s:e] = sa.az.deg
        star_alt[s:e] = sa.alt.deg
        print(f"  frames {e}/{n}", end="\r")
    print()

    rot = frame["rotator_angle"].values[i_exp]
    bore_az = frame["azimuth"].values[i_exp]
    pairs = pd.DataFrame({
        "exp_idx": i_exp.astype(np.int32),
        "star_idx": i_star.astype(np.int32),
        "sep": sep.deg.astype(np.float32),
        # sky frame: position angle on the sky, East of North
        "pa_sky": pa_sky,
        # alt-az / horizon frame
        "pa_azalt": pa_azalt,
        # camera frame: what the optics actually sees
        "pa_cam": A.wrap360(pa_azalt - rot).astype(np.float32),
        # dome/structure frame: azimuth offset from where the telescope points
        "daz": A.wrap180(star_az - bore_az).astype(np.float32),
        "star_alt": star_alt,
        "vmag": sel["vmag"].values[i_star].astype(np.float32),
    })
    # Magnitude in the band actually being observed, for the L3-model definition.
    band = frame["band"].values[i_exp]
    bandmag = np.full(n, np.nan, dtype=np.float32)
    for b in ("u", "g", "r", "i", "z", "y"):
        k = band == b
        if k.any():
            bandmag[k] = sel[b].values[i_star[k]]
    pairs["bandmag"] = bandmag

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    sel.to_parquet(out_path.parent / "star_catalog.parquet", index=False)
    print(f"wrote {out_path} ({len(pairs)} rows, "
          f"{out_path.stat().st_size/1e6:.0f} MB)")
    return pairs, sel


# ======================================================================
# ConsDB
# ======================================================================
def fetch_consdb(day_min, day_max):
    """Same two queries as _build_arc_nb.py §3.3, over a wider date range."""
    from lsst.summit.utils import ConsDbClient
    client = ConsDbClient(CONSDB_URL, token=consdb_token())

    q_visit = f"""
        SELECT v.visit_id, v.can_see_sky, v.scheduler_note, v.pressure,
               vq.physical_rotator_angle, vq.sky_bg_median, vq.sky_noise_median,
               vq.psf_sigma_median, vq.zero_point_median, vq.eff_time_median,
               vq.seeing_zenith_500nm_median, vq.astrom_offset_mean_median,
               vq.aos_fwhm, vq.donut_blur_fwhm, vq.n_psf_star_median
        FROM cdb_lsstcam.visit1 AS v
        LEFT JOIN cdb_lsstcam.visit1_quicklook AS vq ON v.visit_id = vq.visit_id
        WHERE v.day_obs BETWEEN {day_min} AND {day_max}
    """
    q_exp = f"""
        SELECT exposure_id, mount_jitter_rms, mount_jitter_rms_az,
               mount_jitter_rms_el, mount_jitter_rms_rot,
               mount_motion_image_degradation, mount_jitter_rms_cam_hexapod,
               mount_jitter_rms_m2_hexapod
        FROM cdb_lsstcam.exposure_quicklook
        WHERE day_obs BETWEEN {day_min} AND {day_max}
    """
    vis = client.query(q_visit).to_pandas()
    exp = client.query(q_exp).to_pandas()
    vis["visit_id"] = vis["visit_id"].astype("int64")
    exp["exposure_id"] = exp["exposure_id"].astype("int64")
    num = [c for c in vis.columns if c not in ("visit_id", "can_see_sky",
                                               "scheduler_note")]
    vis[num] = vis[num].apply(pd.to_numeric, errors="coerce")
    num = [c for c in exp.columns if c != "exposure_id"]
    exp[num] = exp[num].apply(pd.to_numeric, errors="coerce")
    out = vis.merge(exp, left_on="visit_id", right_on="exposure_id", how="outer")
    out["visit_id"] = out["visit_id"].fillna(out["exposure_id"]).astype("int64")
    return out.drop(columns=["exposure_id"]).rename(
        columns={"visit_id": "exposure_id"})


# ======================================================================
# EFD
# ======================================================================
async def fetch_efd(nights):
    """Per-night, server-side-downsampled EFD pull, one row per 20 s bucket."""
    from lsst_efd_client import EfdClient
    client = EfdClient(EFD_NAME)

    by_topic = {}
    for name, (agg, topic) in EFD_TOPICS.items():
        by_topic.setdefault(topic, []).append((name, agg))

    chunks = []
    for i, night in enumerate(nights):
        d0 = pd.to_datetime(str(night), format="%Y%m%d")
        t0 = (d0 + pd.Timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
        t1 = (d0 + pd.Timedelta(hours=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
        series = {}
        for topic, cols in by_topic.items():
            selq = ", ".join(f"{agg} AS {name}" for name, agg in cols)
            q = (f'SELECT {selq} FROM "efd"."autogen"."{topic}" '
                 f"WHERE time >= '{t0}' AND time <= '{t1}' "
                 f"GROUP BY time({EFD_BUCKET}) fill(none)")
            try:
                res = await client.influxql_query(q)
            except Exception as exc:          # a topic can be absent for a night
                print(f"  [{night}] {topic}: {type(exc).__name__} {str(exc)[:60]}")
                continue
            if res is None or not len(res):
                continue
            for name, _ in cols:
                if name in res:
                    series[name] = pd.to_numeric(res[name],
                                                 errors="coerce").dropna()
        if series:
            for name, s in series.items():
                chunks.append(pd.DataFrame({"night": night, "topic": name,
                                            "t_ns": s.index.values.astype("int64"),
                                            "value": s.values.astype(float)}))
        if i % 25 == 0:
            print(f"  EFD {i+1}/{len(nights)} nights ({night})")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def interp_efd_to_exposures(frame, raw):
    """Linear-interpolate each EFD topic onto every exposure midpoint."""
    want = Time(frame["exp_midpt_mjd"].values, format="mjd", scale="utc")
    want_ns = pd.DatetimeIndex(want.datetime, tz="UTC").values.astype("int64")
    out = pd.DataFrame(index=frame.index)
    for topic, g in raw.groupby("topic"):
        col = np.full(len(frame), np.nan)
        for night, gg in g.groupby("night"):
            rows = np.where(frame["day_obs"].values == night)[0]
            if len(rows) == 0 or len(gg) < 2:
                continue
            gg = gg.sort_values("t_ns")
            col[rows] = np.interp(want_ns[rows], gg["t_ns"].values,
                                  gg["value"].values, left=np.nan, right=np.nan)
        out[topic] = col
    return out


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-consdb", action="store_true")
    ap.add_argument("--no-efd", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a cache exists")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    frame.to_parquet(CACHE / "frame.parquet", index=False)

    # ---- star pairs -------------------------------------------------
    pair_path = CACHE / "star_pairs.parquet"
    if pair_path.exists() and not args.force:
        print(f"star pairs cached: {pair_path}")
    else:
        build_star_pairs(frame, load_ybsc(), pair_path)

    # ---- ConsDB -----------------------------------------------------
    cdb_path = CACHE / "consdb.parquet"
    if args.no_consdb:
        print("ConsDB skipped (--no-consdb)")
    elif cdb_path.exists() and not args.force:
        print(f"ConsDB cached: {cdb_path}")
    else:
        d0, d1 = int(frame["day_obs"].min()), int(frame["day_obs"].max())
        print(f"fetching ConsDB {d0}-{d1} (extends the arc-epoch cache backwards)")
        cdb = fetch_consdb(d0, d1)
        cdb.to_parquet(cdb_path, index=False)
        print(f"wrote {cdb_path} ({len(cdb)} rows)")

    # ---- EFD --------------------------------------------------------
    efd_path = CACHE / f"efd_{EFD_BUCKET}_raw.parquet"
    if args.no_efd:
        print("EFD skipped (--no-efd)")
    elif efd_path.exists() and not args.force:
        print(f"EFD cached: {efd_path}")
    else:
        nights = sorted(frame["day_obs"].unique())
        print(f"fetching EFD for {len(nights)} nights")
        raw = asyncio.run(fetch_efd(nights))
        raw.to_parquet(efd_path, index=False)
        print(f"wrote {efd_path} ({len(raw)} rows)")

    print("\nenrichment caches ready in", CACHE)


if __name__ == "__main__":
    main()
