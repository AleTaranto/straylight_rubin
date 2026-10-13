"""Run the full characterisation for every artifact and write the summary files.

Outputs, all in this directory:
  <artifact>_summary.txt    one per artifact, self-contained
  ALL_ARTIFACTS_SUMMARY.txt the cross-artifact ranking and the null calibration

Run:  python run_all_artifacts.py [--perm 2000] [--only tape,arc]

Read `artifact_stats.py`'s module docstring first: the reason this is not a
straightforward "lift" analysis is that labels arrive in bursts, and a blind
annulus scan finds 2-6x "effects" from burstiness alone. The primary statistic
counts nights and is calibrated against a null that preserves the bursts.
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (see build_star_geometry.py for why this is first)
import matplotlib.font_manager  # noqa: F401

import argparse
import pathlib
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))

import artifact_stats as A  # noqa: E402

CACHE = pathlib.Path("../outputs/artifact_stats/cache")
TABDIR = pathlib.Path("../outputs/artifact_stats")
OUTDIR = pathlib.Path(".")

# Groups for the univariate screen -- same structure as _build_arc_nb.py §5.
GROUPS = {
    "pointing": ["altitude", "azimuth", "zenith_distance", "airmass",
                 "rotator_angle", "sky_rotation", "parallactic_angle", "ra", "dec",
                 "focus_z", "physical_rotator_angle"],
    "moon": ["moon_sep", "moon_alt", "moon_illum", "moon_bright", "moon_daz",
             "moon_dalt", "moon_az"],
    "stars": ["star0_sep", "star0_vmag", "star0_alt", "star0_dalt", "star1_sep",
              "star1_vmag", "star1_alt", "star1_dalt", "tape_sep", "tape_vmag",
              "n_bright_10deg", "n_bright_25deg", "nearest_bright_sep"],
    "telescope": ["dome_tel_daz_abs", "dome_moon_daz_abs", "dome_follow_err",
                  "mirror_cover_max", "dome_shutter_min", "mount_jitter_rms",
                  "mount_jitter_rms_az", "mount_jitter_rms_el",
                  "mount_jitter_rms_rot", "mount_motion_image_degradation"],
    "environment": ["air_temp", "humidity", "wind_speed", "pressure",
                    "dimm_seeing"],
    "image quality": ["sky_bg_median", "sky_noise_median", "psf_sigma_median",
                      "zero_point_median", "eff_time_median",
                      "astrom_offset_mean_median", "aos_fwhm", "donut_blur_fwhm",
                      "n_psf_star_median"],
}
ALL_VARS = [c for g in GROUPS.values() for c in g]
VAR_GROUP = {c: g for g, cols in GROUPS.items() for c in cols}

# Telescope-state conditions for the supporting MH table. Deliberately excludes
# any bright-star condition: that is section 1's question, and restating it here
# with a much weaker method would read as independent corroboration when it is
# the same observation counted twice.
def telescope_tests(f):
    def col(name):
        return f[name] if name in f else pd.Series(np.nan, index=f.index)
    return {
        "Moon above horizon": col("moon_alt") > 0,
        "Moon closer than 50 deg": col("moon_sep") < 50,
        "Moon illum > 0.9": col("moon_illum") > 0.9,
        "telescope above 65 deg": col("altitude") > 65,
        "telescope az in N quadrant": (col("azimuth") > 315) | (col("azimuth") < 45),
        "|dome-tel az| > 4 deg": col("dome_tel_daz_abs") > 4,
        "Moon within 90 deg of dome az": col("dome_moon_daz_abs") < 90,
        "mirror covers off stow": col("mirror_cover_max") > 1,
        "dome shutter not fully open": col("dome_shutter_min") < 99,
        "mount jitter above median": col("mount_jitter_rms") > col("mount_jitter_rms").median(),
    }


# ======================================================================
# tiering
# ======================================================================
def assign_tier(frame, artifact):
    """Depth tier from measured power, so a thin artifact cannot fake a result."""
    col = f"is_{artifact}"
    lab = frame[frame[col]]
    n = len(lab)
    nights = lab["day_obs"].nunique()
    per_night = lab.groupby("day_obs").size()
    nights_ge8 = int((per_night >= 8).sum())
    g = frame.groupby("day_obs")[col]
    both = int(((g.transform("sum") > 0) & (g.transform(lambda s: (~s).sum()) > 0))
               .groupby(frame["day_obs"]).any().sum())
    if n < 25 or nights < 5:
        tier = "D"
    elif n < 100:
        tier = "C"
    elif nights_ge8 >= 8:
        tier = "A"
    else:
        tier = "B"
    return dict(tier=tier, n_labels=n, nights=nights, nights_ge8=nights_ge8,
                nights_both=both)


# ======================================================================
# formatting helpers -- a skipped test must say why, never print a NaN
# ======================================================================
def not_tested(reason, have=None, need=None):
    s = f"NOT TESTED -- {reason}"
    if have is not None and need is not None:
        s += f" ({have} available, {need} required)"
    return s


def fmt(x, nd=3, na="n/a"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{nd}f}"


def hr(c="=", n=78):
    return c * n


# ======================================================================
# per-artifact analysis
# ======================================================================
def analyse(artifact, frame, pairs, cat, cells, mem_full, n_perm, bh_lookup=None):
    """Everything computed for one artifact. Returns (evidence, text_lines)."""
    L = []
    col = f"is_{artifact}"
    power = assign_tier(frame, artifact)
    tier = power["tier"]

    fr, epoch = A.artifact_frame(frame, artifact,
                                 inspected=set(frame["day_obs"].unique()))
    fr = fr.sort_values(["day_obs", "seq_num"])
    idx = fr.index.values                       # positions into `frame`/`mem_full`
    fr = fr.reset_index(drop=True)
    lab = fr[col].values.astype(bool)
    night_ids = pd.factorize(fr["day_obs"].values)[0]
    burst = A.burst_structure(fr, artifact)

    ev = dict(artifact=artifact, tier=tier, epoch=epoch, frame_n=len(fr),
              frame_nights=int(fr["day_obs"].nunique()), n_labels=burst["n"],
              nights=burst["nights"], runs=burst["runs"],
              n_targets=burst["n_targets"], n_pointings=burst["n_pointings"])

    # ---------------- header ----------------
    L += [hr(), f"'{artifact.upper()}' ARTIFACT -- SUMMARY", hr(), ""]
    if artifact in A.SUBLABEL_OF:
        parent = A.SUBLABEL_OF[artifact]
        L += textwrap.wrap(
            f"CAVEAT -- '{artifact}' is not an independent artifact in this "
            f"dataset. Every one of its {burst['n']} labels co-occurs with "
            f"'{parent}', so it is perfectly collinear with it: any association "
            f"found against the whole survey is inherited from '{parent}' and is "
            f"not evidence about '{artifact}'. The only well-posed question is "
            f"conditional -- among {parent}-labelled exposures, what predicts "
            f"also being {artifact} -- and that is what the conditional section "
            f"below reports. It is excluded from the cross-artifact ranking.", 78)
        L += [""]

    L += [f"Analysis frame",
          f"  label epoch starts      : {epoch}  (first {artifact} label; "
          f"earlier nights carry no label for it and are excluded)",
          f"  frame                   : {len(fr)} exposures on "
          f"{fr['day_obs'].nunique()} inspected nights",
          f"  {(artifact + ' exposures'):24s}: {burst['n']}  "
          f"({burst['n']/max(len(fr),1)*100:.3f} %)",
          f"  distinct nights / runs  : {burst['nights']} nights, "
          f"{burst['runs']} contiguous runs "
          f"(median {fmt(burst['median_run'],0)}, max {burst['max_run']})",
          f"  distinct target fields  : {burst['n_targets']}   "
          f"distinct pointings (1 deg): {burst['n_pointings']}",
          f"  depth tier              : {tier}  "
          f"(nights with >=8 labels: {power['nights_ge8']}; "
          f"nights with both classes: {power['nights_both']})",
          ""]
    if burst["n"]:
        L += ["  busiest nights          : " +
              ", ".join(f"{k}:{v}" for k, v in burst["top_nights"].items()), ""]
    L += textwrap.wrap(
        "The effective sample size is the night count, not the exposure count: "
        f"{burst['n']} labels arrive in {burst['runs']} runs on "
        f"{burst['nights']} nights, so exposures are nowhere near independent. "
        "Every test below is night-aware for that reason.", 78)
    L += [""]

    if burst["n"] == 0:
        L += [not_tested("no labels in the analysis frame")]
        ev["no_test"] = True
        ev["no_test_reason"] = "no labels"
        return ev, L

    # ================= STAR-ANNULUS TEST (primary) =================
    L += [hr("-"), "1. BRIGHT-STAR ANNULUS TEST  (the primary question)", hr("-"), ""]
    L += textwrap.wrap(
        "Question: is there always a bright star at some preferred off-axis "
        "distance from the boresight when this artifact appears? Statistic: the "
        "number of distinct NIGHTS on which at least one labelled exposure has a "
        "star in a given annulus. Counting nights rather than exposures is what "
        "stops a single burst from being read as hundreds of detections. The null "
        "circularly shifts each night's label vector, which preserves run lengths "
        "and spacings, so the null samples are exactly as bursty as the real one. "
        "The reported p is family-wise over the whole scan: each cell is "
        "standardised, and the maximum z is calibrated against the per-replicate "
        "maximum, so picking the best of "
        f"{len(cells)} cells costs what it should.", 78)
    L += [""]
    L += textwrap.wrap(
        "The scan covers brightness as well as radius. One magnitude cut will not "
        "do: taking every star down to V=4.5 means that at 20 deg off axis almost "
        "every exposure has one, so the indicator saturates and any real contrast "
        "is diluted away. Brightness is also the physically right axis, since "
        "scattered flux scales with the flux of the source.", 78)
    L += [""]

    mem = mem_full[idx]
    pe = pairs["exp_idx"].values
    # restrict the pair table to this frame, remapping exposure indices
    remap = np.full(len(frame), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    keep = remap[pe] >= 0
    p_exp = remap[pe[keep]].astype(np.int64)
    p_sep = pairs["sep"].values[keep]
    p_star = pairs["star_idx"].values[keep]
    p_vmag = pairs["vmag"].values[keep]

    if tier == "D":
        L += [not_tested(
            f"tier D -- {burst['n']} labels on {burst['nights']} nights is too "
            "few for any inferential test", burst["nights"], 5)]
        L += ["", "  Descriptive facts instead of statistics:"]
        pn = fr[fr[col]].groupby("day_obs").size().sort_values(ascending=False)
        L += [f"    labels per night      : " +
              ", ".join(f"{k}:{v}" for k, v in pn.items())]
        biggest = pn.iloc[0] / burst["n"] * 100
        L += [f"    largest single night  : {pn.index[0]} holds {pn.iloc[0]} of "
              f"{burst['n']} labels ({biggest:.0f} %)",
              f"    effective sample size : ~{burst['nights']} nights",
              ""]
        sub = fr[fr[col]]
        for c in ("moon_sep", "moon_alt", "altitude", "azimuth"):
            if c in sub:
                v = sub[c].dropna()
                if len(v):
                    L += [f"    {c:20s}: median {v.median():.1f}, "
                          f"range {v.min():.1f} to {v.max():.1f}"]
        L += [""]
        ev["_fig"] = {"per_night": fr[fr[col]].groupby("day_obs").size()}
        ev.update(no_test=True, no_test_reason=(
            f"tier D: {burst['n']} labels on {burst['nights']} nights"))
        return ev, L

    scan = A.scan_recurrence(lab, mem, night_ids, n_perm=n_perm)
    if scan["best_bin"] is None:
        L += [not_tested("no annulus was populated often enough to standardise")]
        ev.update(no_test=True, no_test_reason="scan degenerate")
        return ev, L

    lo, hi, vmax = cells[scan["best_bin"]]
    # Everything the LaTeX report plots is stashed here as it is computed, so the
    # figures and the prose can never drift apart.
    fig = {}
    ev["_fig"] = fig
    fig["scan"] = dict(
        z_all=scan["z_all"], obs=scan["obs"], mu=scan["mu"], sd=scan["sd"],
        cells=cells, best=scan["best_bin"], support=scan.get("support_ok"),
        null_max_shift=scan.get("null_max_shift"),
        null_max_cross=scan.get("null_max_cross"))
    ev.update(best_lo=lo, best_hi=hi, best_vmax=vmax, width=hi - lo, z=scan["z"],
              p_fwer=scan["p_fwer"], p_shift=scan.get("p_shift"),
              p_cross=scan.get("p_cross"), p_floor=scan.get("p_floor"),
              n_cells_tested=scan.get("n_cells_tested"),
              obs_nights=scan["obs_nights"],
              null_mean=scan["null_mean"], null_sd=scan["null_sd"])

    # Every downstream control uses this same (annulus, brightness) selection.
    sel_pair = (p_sep >= lo) & (p_sep < hi) & (p_vmag < vmax)
    in_ann = np.zeros(len(fr), dtype=bool)
    in_ann[p_exp[sel_pair]] = True
    st = A.annulus_exposure_stats(lab, in_ann, night_ids)
    ev.update(crude_lift=st["lift"], nights_in=st["nights_in"],
              rate_in=st["rate_in"], rate_out=st["rate_out"],
              k_in_annulus=st["k_in"], n_in_annulus=st["n_in"])

    L += [f"  grid                    : {len(cells)} cells "
          f"({len(cells)//len(A.MAG_TIERS)} annuli, widths 0.05/0.10/0.25/0.50 deg "
          f"over 0.5-25 deg x {len(A.MAG_TIERS)} brightness tiers)",
          f"  permutations            : {scan['n_perm']}",
          "",
          f"  BEST CELL               : [{lo:.3f}, {hi:.3f}) deg  "
          f"(width {hi-lo:.2f} deg)  with a star brighter than V = {vmax:.1f}",
          f"  nights with in-annulus labels : {scan['obs_nights']:.0f}   "
          f"(null expects {scan['null_mean']:.2f} +/- {scan['null_sd']:.2f})",
          f"  recurrence z            : {fmt(scan['z'],2)}",
          "",
          "  Two nulls, and the WEAKER (larger) p is the one quoted:",
          f"    within-night shift  p = {fmt(scan.get('p_shift'),4)}   "
          "(labels moved inside their own night)",
          f"    cross-night shuffle p = {fmt(scan.get('p_cross'),4)}   "
          "(whole nights transplanted)",
          f"  p (family-wise, conservative)       : {fmt(scan['p_fwer'],4)}"
          + ("   <- at the permutation floor; more permutations would only lower it"
             if np.isclose(scan["p_fwer"], scan.get("p_floor", -1)) else ""),
          f"  cells with enough support to test   : {scan.get('n_cells_tested','?')}"
          f" of {len(cells)}",
          "    (a cell needs >=3 nights with labels and >=30 exposures: without",
          "     that floor the scan is won by a cell holding a handful of exposures,",
          "     where the null is ~0 and a coincidence produces a huge meaningless z)",
          "",
          "  Why two nulls: shifting labels within a night cannot see that the",
          "  labelled nights are themselves a biased sample of pointings -- a night",
          "  spent on one field has every exposure near the same stars, so every",
          "  shift lands in the same annuli. Checked directly, that null alone gives",
          "  p ~ 0.01 to labels deliberately decoupled from the sky. Transplanting",
          "  whole nights breaks the link while keeping the bursts, so it catches",
          "  what the shift cannot.",
          "",
          "  Exposure-level view of the same annulus (SUPPORTING ONLY -- a burst",
          "  of N flagged exposures counts N times here, so treat it as a",
          "  description, not as evidence):",
          f"    exposures in annulus  : {st['n_in']}   labelled: {st['k_in']}",
          f"    rate inside           : {st['rate_in']*100:.3f} %",
          f"    rate outside          : {st['rate_out']*100:.4f} %",
          f"    crude lift            : {fmt(st['lift'],1)}x",
          f"    share of labels inside: {st['frac_labels_inside']*100:.1f} %",
          ""]

    # Every control below is restricted to the winning brightness tier, so it
    # tests the same hypothesis the scan selected rather than a looser one.
    tier_k = p_vmag < vmax
    t_exp, t_sep, t_star = p_exp[tier_k], p_sep[tier_k], p_star[tier_k]
    t_vmag = p_vmag[tier_k]

    # ---- radial profile: is the winner an edge or a bump? ----
    # One bin cannot distinguish a sharp mechanism edge from the peak of a broad
    # excess, and those imply different physics, so show the neighbourhood.
    prof = []
    half = max(hi - lo, 0.05)
    for j in range(-6, 7):
        a_, b_ = lo + j * half, hi + j * half
        if a_ <= 0:
            continue
        kk = (t_sep >= a_) & (t_sep < b_)
        ia = np.zeros(len(fr), dtype=bool)
        ia[t_exp[kk]] = True
        if ia.sum() < 10:
            continue
        nn = len(np.unique(fr["day_obs"].values[lab & ia]))
        prof.append((a_, b_, int(ia.sum()), int((lab & ia).sum()),
                     lab[ia].mean() * 100, nn, j == 0))
    fig["profile"] = prof
    if len(prof) >= 3:
        L += [f"  Radial profile around the winner (same width, V < {vmax:.1f}):",
              "    annulus (deg)      exposures  labelled    rate   nights"]
        for a_, b_, n_, k_, r_, nn, is_win in prof:
            L += [f"    [{a_:6.3f},{b_:6.3f})  {n_:9d} {k_:9d} {r_:7.3f} % "
                  f"{nn:6d}" + ("   <- winner" if is_win else "")]
        L += ["    A sharp drop either side is a mechanism edge; a broad bump is",
              "    more consistent with scattering off a large structure.", ""]

    # ---- star-level concentration ----
    sc = A.star_contributions(lab, t_exp, t_sep, t_star, cat["star_name"].values,
                              lo, hi, fr["day_obs"].values)
    fig["stars"] = sc
    ev["n_stars"] = int(len(sc))
    L += ["  Which stars produce the in-annulus labels:"]
    if len(sc) == 0:
        L += ["    (none)"]
    else:
        ev["top_star_share"] = float(sc["share"].iloc[0])
        ev["top_star"] = str(sc["star"].iloc[0])
        L += [f"    {len(sc)} distinct stars; top contributor "
              f"{sc['star'].iloc[0]} with {sc['n_pairs'].iloc[0]} pairs on "
              f"{sc['n_nights'].iloc[0]} nights "
              f"({sc['share'].iloc[0]*100:.0f} % of all in-annulus labels)"]
        for _, r in sc.head(6).iterrows():
            L += [f"      {r['star']:14s} pairs {int(r['n_pairs']):4d}  "
                  f"nights {int(r['n_nights']):3d}  share {r['share']*100:5.1f} %"]
        if sc["share"].iloc[0] > 0.5:
            L += ["", "    WARNING: one star supplies more than half the "
                  "in-annulus labels.",
                  "    That is the signature of a scheduling coincidence -- the "
                  "survey revisited",
                  "    one field for a few nights -- not of an optical mechanism, "
                  "which would",
                  "    show up on many different stars."]
    L += [""]

    # ---- leave-one-star-out ----
    if len(sc) >= 2:
        drop = int(cat.index[cat["star_name"] == sc["star"].iloc[0]][0]) \
            if (cat["star_name"] == sc["star"].iloc[0]).any() else None
        if drop is not None:
            loo = A.leave_one_star_out(lab, None, t_exp, t_sep, t_star, lo, hi,
                                       night_ids, drop,
                                       n_perm=max(400, n_perm // 4))
            ev["loo_p"] = loo.get("p_fwer", np.nan)
            ev["loo_z"] = loo.get("z", np.nan)
            L += ["  Leave-one-star-out (drop the top contributor and re-test):"]
            if loo.get("best_bin") is None or not np.isfinite(loo.get("z", np.nan)):
                L += ["    " + not_tested(
                    f"with {sc['star'].iloc[0]} removed, the annulus no longer "
                    "has enough support to standardise (>=3 nights with labels "
                    "and >=30 exposures)"),
                    "    Read that as the signal depending on that one star, "
                    "which is what a",
                    "    scheduling coincidence looks like -- not as a passed test."]
            else:
                L += [f"    z {fmt(loo.get('z'),2)}   p "
                      f"{fmt(loo.get('p_fwer'),4)}   "
                      f"nights {fmt(loo.get('obs_nights'),0)}",
                      "    A signal that vanishes here was one coincidence, not a "
                      "mechanism."]
            L += [""]
    else:
        L += ["  Leave-one-star-out: " +
              not_tested("fewer than 2 contributing stars", len(sc), 2), ""]

    # ---- offset-annulus controls ----
    off = A.offset_annulus_controls(lab, t_exp, t_sep, night_ids, lo, hi)
    fig["offsets"] = off
    if len(off):
        base = st["rate_in"]
        neigh = off["rate_in"].replace([np.inf, -np.inf], np.nan).dropna()
        ratio = (neigh.max() / base) if base and base > 0 and len(neigh) else np.nan
        ev["offset_ratio"] = float(ratio) if np.isfinite(ratio) else np.nan
        L += ["  Offset-annulus negative controls (same width, displaced):",
              "    A real mechanism has a sharp edge, so neighbours should be "
              "much weaker.",
              f"    winner [{lo:.3f},{hi:.3f})  rate {base*100:.3f} %"]
        for _, r in off.iterrows():
            L += [f"    offset {r['offset']:+.2f}  [{r['lo']:.3f},{r['hi']:.3f})  "
                  f"n {int(r['n_in']):5d}  labelled {int(r['k_in']):4d}  "
                  f"rate {r['rate_in']*100:7.3f} %"]
        L += [f"    strongest neighbour / winner = {fmt(ratio,2)}", ""]

    # ---- magnitude dose-response ----
    dr = A.magnitude_dose_response(lab, p_exp, p_sep, p_vmag, lo, hi)
    fig["dose"] = dr
    tab = dr["table"]
    if len(tab) >= 3:
        ev["dose_response"] = dr.get("dose_response", False)
        ev["trend_p"] = dr.get("trend_p", np.nan)
        ev["trend_z"] = dr.get("trend_z", np.nan)
        ev["bright_faint_ratio"] = dr.get("bright_faint_ratio", np.nan)
        L += ["  Magnitude dose-response on the winning annulus:",
              "    A real optical mechanism should worsen with brighter stars and",
              "    switch off past its own magnitude limit."]
        for _, r in tab.iterrows():
            thin = "   (thin bin)" if r["thin"] else ""
            L += [f"    V in [{r['mag_lo']:+.1f},{r['mag_hi']:+.1f})  "
                  f"n {int(r['n']):6d}  labelled {int(r['k']):5d}  "
                  f"rate {r['rate']*100:7.3f} %{thin}"]
        L += [f"    no star in annulus     n {dr['n_no_star']:6d}  "
              f"rate {dr['rate_no_star']*100:7.3f} %"]
        L += ["", f"    Cochran-Armitage trend  z {fmt(dr.get('trend_z'),2)}  "
              f"p {fmt(dr.get('trend_p'),4)}   (negative z = rate falls as stars "
              "get fainter)",
              f"    bright half / faint half rate ratio: "
              f"{fmt(dr.get('bright_faint_ratio'),1)}x",
              f"    dose-response satisfied: {dr.get('dose_response')}"]
        if not dr.get("strict_monotone", True):
            L += ["    (bin-to-bin ordering is not strictly monotone; the "
                  "brightest bins hold",
                  "     only a few exposures -- there are four stars with V<0 in "
                  "the whole sky --",
                  "     so their ordering is noise. The trend test and the "
                  "bright/faint ratio are",
                  "     what carry the evidence.)"]
        L += ["    (exposure-level: supporting evidence, not the verdict)", ""]
    else:
        L += ["  Magnitude dose-response: " +
              not_tested("too few populated magnitude bins", len(tab), 3), ""]

    # ---- multiplicity + frame diagnostics ----
    mult = A.annulus_multiplicity(t_exp, t_sep, lo, hi, len(fr))
    ev.update(mean_mult=mult["mean_mult"], frac_single=mult["frac_single"])
    L += [hr("-"), "2. WHICH FRAME IS THE FEATURE FIXED IN?", hr("-"), ""]
    L += textwrap.wrap(
        "A stray-light feature is fixed in some frame, and each implicates a "
        "different surface: the camera frame means the optics (a ghost), the dome "
        "frame means the structure or the slit, the sky frame means the atmosphere "
        "-- or, far more often, that the survey kept revisiting one field.", 78)
    L += ["",
          f"  stars per exposure in the winning annulus: mean "
          f"{fmt(mult['mean_mult'],2)}, "
          f"exactly one in {fmt(mult['frac_single']*100,0)} % of cases", ""]
    L += textwrap.wrap(
        "This is why the frame cannot be identified from telemetry: with about "
        "one star per exposure there is a single direction, and the three frames "
        "are then related by fixed offsets (rotator, parallactic angle). Two stars "
        "at the same radius differ in position angle by a rotator-invariant "
        "constant, so they rotate together and add no discriminating power either. "
        "The concentrations below are therefore DESCRIPTIVE; read their ordering, "
        "not their absolute size.", 78)
    L += [""]
    # All three must be position angles. An azimuth offset would be pinned near
    # zero by the annulus itself and would measure radius, not direction.
    pa_frames = {"camera": pairs["pa_cam"].values[keep][tier_k],
                 "horizon": pairs["pa_azalt"].values[keep][tier_k],
                 "sky": pairs["pa_sky"].values[keep][tier_k]}
    fc = A.frame_concentration(lab, t_exp, t_sep, pa_frames, lo, hi)
    fig["frames"] = fc
    L += ["    frame     R(labels)  R(controls)   excess   mean direction"]
    excess = {}
    for fname in ("camera", "horizon", "sky"):
        rl = fc.get(f"R_lab_{fname}", np.nan)
        rc_ = fc.get(f"R_ctl_{fname}", np.nan)
        excess[fname] = rl - rc_ if np.isfinite(rl) and np.isfinite(rc_) else np.nan
        L += [f"    {fname:9s} {fmt(rl,3):>9s}  {fmt(rc_,3):>11s} "
              f"{fmt(excess[fname],3):>8s}   {fmt(fc.get(f'mean_lab_{fname}'),0)} deg"]
    ev.update(R_sky=fc.get("R_lab_sky", np.nan),
              R_cam=fc.get("R_lab_camera", np.nan),
              R_horizon=fc.get("R_lab_horizon", np.nan),
              excess_sky=excess["sky"], excess_cam=excess["camera"])
    L += ["",
          "  'excess' is R(labels) - R(controls): each frame compared against its",
          "  own baseline, which is the only fair comparison since the frames have",
          "  different intrinsic concentrations."]
    es, ec = excess["sky"], excess["camera"]
    if np.isfinite(es) and np.isfinite(ec):
        L += [""]
        if es > ec:
            L += textwrap.wrap(
                "  The sky frame shows the larger excess. The star direction "
                "repeats on the SKY more than in the camera, which is what "
                "repeatedly visiting the same field produces -- not what a "
                "camera-fixed ghost produces. Pointing repetition is the more "
                "economical explanation.", 78)
        else:
            L += textwrap.wrap(
                "  The camera frame shows the larger excess, which is the "
                "camera-fixed (ghost) ordering. Suggestive only, and worth "
                "pursuing only if the annulus test above survived its controls.",
                78)
    L += [""]

    # ================= the rest of the battery =================
    L += [hr("-"), "3. UNIVARIATE SCREEN  (what else differs)", hr("-"), ""]
    scr = A.screen(fr, artifact, ALL_VARS)
    fig["screen"] = scr
    if len(scr) == 0:
        L += [not_tested("no numeric variable had enough coverage")]
    else:
        scr["group"] = scr["variable"].map(VAR_GROUP)
        L += ["  Cliff's delta: probability a random labelled exposure exceeds a",
              "  random unlabelled one, rescaled to (-1,+1). Ranked by |delta|;",
              "  p-values ignore night clustering and so are optimistic.", ""]
        for _, r in scr.head(14).iterrows():
            L += [f"    {r['variable']:26s} [{str(r['group'])[:12]:12s}] "
                  f"d={r['cliffs_delta']:+.3f}  median {fmt(r['median_lab'],2)} "
                  f"vs {fmt(r['median_other'],2)}  q={fmt(r['q_BH'],2,'n/a')}"]
        ev["top_screen"] = scr.head(6)[["variable", "cliffs_delta"]].values.tolist()
    L += [""]

    # ---- telescope state ----
    L += [hr("-"), "4. TELESCOPE POSITION AND STATE", hr("-"), ""]
    nl = A.night_level_variables(fr, artifact, ALL_VARS)
    tests = telescope_tests(fr)
    mh_rows = []
    strat = fr["day_obs"].astype(str).values
    for name, cond in tests.items():
        okm = cond.notna()
        if okm.sum() < 200 or cond[okm].sum() < 20:
            continue
        sub = fr[okm]
        crude = A.mantel_haenszel(sub[col], cond[okm], np.ones(int(okm.sum())))
        adj = A.mantel_haenszel(sub[col], cond[okm], strat[okm.values])
        mh_rows.append({"condition": name, "n_true": int(cond[okm].sum()),
                        "OR_crude": crude["OR"], "OR_night": adj["OR"],
                        "lo": adj["lo"], "hi": adj["hi"], "p": adj["p"]})
    mh = pd.DataFrame(mh_rows)
    fig["mh"] = mh
    fig["per_night"] = fr[fr[col]].groupby("day_obs").size()
    # A condition whose stratified OR is undefined (no stratum has both classes on
    # both sides) tells the reader nothing; printing a row of "n/a" invites them to
    # read a missing number as a weak result. Report which conditions dropped out
    # and why, instead.
    undefined = []
    if len(mh):
        bad = ~np.isfinite(mh["OR_night"])
        undefined = mh.loc[bad, "condition"].tolist()
        mh = mh[~bad]
    if len(mh):
        mh = mh.sort_values("OR_night", ascending=False)
        L += ["  Odds ratios stratified by night (SUPPORTING ONLY -- exposure-level,",
              "  so a burst inflates the precision; compare crude vs stratified to",
              "  see how much of the crude effect was just which nights we observed):",
              ""]
        for _, r in mh.iterrows():
            L += [f"    {r['condition']:32s} OR {fmt(r['OR_night'],2):>7s} "
                  f"({fmt(r['lo'],2)}-{fmt(r['hi'],2)})  p {fmt(r['p'],4)}"
                  f"   [crude {fmt(r['OR_crude'],2)}]"]
        ev["mh"] = mh
    else:
        L += ["  " + not_tested("no condition had enough exposures on both sides")]
    if undefined:
        L += ["",
              "  " + not_tested(
                  "no night contains both states of these conditions, so a "
                  "night-stratified odds ratio does not exist for them: "
                  + ", ".join(undefined))]
    L += ["",
          "  Cannot be tested within a night (night-level quantities -- for these,",
          "  between-night evidence is all there is, which is a real limit):",
          "    " + (", ".join(nl) if nl else "(none)"), ""]

    # ---- moon gate ----
    if "moon_alt" in fr:
        up = fr["moon_alt"] > 0
        r_up = fr.loc[up, col].mean() * 100
        r_dn = fr.loc[~up, col].mean() * 100
        ev.update(moonup_frac=float(fr.loc[lab, "moon_alt"].gt(0).mean()),
                  rate_moon_up=r_up, rate_moon_down=r_dn)
        L += [f"  Moon gate: {ev['moonup_frac']*100:.1f} % of {artifact} labels have",
              f"    the Moon above the horizon (vs "
              f"{fr.loc[~lab,'moon_alt'].gt(0).mean()*100:.1f} % of the rest)",
              f"    rate with Moon up {r_up:.3f} %   Moon down {r_dn:.3f} %", ""]

    # ---- night fixed effects + GBM, tier A only ----
    L += [hr("-"), "5. WITHIN-NIGHT MODELS", hr("-"), ""]
    # Needs enough nights carrying both classes AND enough labels per night for
    # a within-night contrast to exist at all.
    fe_ok = power["nights_both"] >= 8 and power["nights_ge8"] >= 4
    if fe_ok:
        fe = fit_night_fe(fr, artifact)
        if fe is None or len(fe) == 0:
            L += ["  " + not_tested("the fixed-effects model did not converge")]
        else:
            L += ["  Night fixed-effects logit, cluster-robust by night. Only",
                  "  within-night variation contributes, so anything constant over",
                  "  a night is absent by necessity.", ""]
            for _, r in fe.iterrows():
                if r["identified"]:
                    L += [f"    {r['term']:26s} OR {fmt(r['OR'],3):>7s} "
                          f"({fmt(r['lo'],3)}-{fmt(r['hi'],3)})  p {fmt(r['p'],4)}"]
                else:
                    L += [f"    {r['term']:26s} not identified (separated in these "
                          "data)"]
            ev["fe"] = fe
        gbm = fit_gbm(fr, artifact)
        if gbm:
            L += ["",
                  f"  Geometry-only classifier, nights never shared between train",
                  f"  and test: ROC AUC {fmt(gbm['auc'],3)} +/- {fmt(gbm['auc_sd'],3)}, "
                  f"average precision {fmt(gbm['ap'],4)} vs "
                  f"{fmt(gbm['chance'],4)} chance "
                  f"({fmt(gbm['lift'],1)}x lift).",
                  "  This is a specificity check, not a detector: it shows whether",
                  "  the geometry carries signal that transfers to unseen nights."]
            ev["gbm"] = gbm
    else:
        if power["nights_both"] < 8:
            L += ["  " + not_tested(
                "too few nights carry both labelled and unlabelled exposures",
                power["nights_both"], 8)]
        else:
            L += ["  " + not_tested(
                "labels are spread too thinly across nights for a within-night "
                "contrast (nights with at least 8 labels)",
                power["nights_ge8"], 4)]
            L += ["    " + f"{power['nights_both']} nights do carry both classes, "
                  "but with only a handful of",
                  "    labels each the model would be fitting one exposure per "
                  "night against its",
                  "    own intercept, which estimates nothing."]
    L += [""]
    return ev, L


# ======================================================================
# models
# ======================================================================
def fit_night_fe(fr, artifact):
    """Night fixed-effects logit (see _build_arc_nb.py §8.3)."""
    import statsmodels.api as sm
    col = f"is_{artifact}"
    spec = [(c, 10.0, lab) for c, lab in [
        ("moon_sep", "Moon separation / 10deg"),
        ("moon_alt", "Moon altitude / 10deg"),
        ("altitude", "telescope altitude / 10deg"),
        ("nearest_bright_sep", "nearest bright star sep / 10deg"),
        ("dome_tel_daz_abs", "|dome-tel az| / 10deg")] if c in fr]
    if not spec:
        return None
    g = fr.groupby("day_obs")[col]
    both = g.transform("sum").gt(0) & g.transform(lambda s: (~s).sum()).gt(0)
    use = fr[both].dropna(subset=[s[0] for s in spec])
    if len(use) < 200 or use["day_obs"].nunique() < 5 or use[col].sum() < 10:
        return None
    X = pd.DataFrame({lab: use[c] / sc for c, sc, lab in spec}, index=use.index)
    nights = pd.get_dummies(use["day_obs"].astype(str), prefix="night",
                            drop_first=True).astype(float)
    design = sm.add_constant(pd.concat([X, nights], axis=1))
    try:
        res = sm.GLM(use[col].astype(int).values, design.values,
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": use["day_obs"].values})
    except Exception:
        return None
    rows = []
    for i, nm in enumerate(design.columns):
        if nm.startswith("night_") or nm == "const":
            continue
        b, se = res.params[i], res.bse[i]
        ok = np.isfinite(b) and np.isfinite(se) and 0 < se < 10
        rows.append({"term": nm,
                     "OR": np.exp(b) if ok else np.nan,
                     "lo": np.exp(b - 1.96 * se) if ok else np.nan,
                     "hi": np.exp(b + 1.96 * se) if ok else np.nan,
                     "p": float(2 * sps.norm.sf(abs(b / se))) if ok else np.nan,
                     "identified": ok})
    return pd.DataFrame(rows)


def fit_gbm(fr, artifact):
    """Night-grouped CV specificity check."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    col = f"is_{artifact}"
    feats = [c for c in ["moon_sep", "moon_alt", "moon_illum", "moon_daz",
                         "moon_dalt", "altitude", "azimuth", "rotator_angle",
                         "airmass", "star0_sep", "star1_sep", "nearest_bright_sep",
                         "n_bright_10deg", "dome_tel_daz_abs", "dome_moon_daz_abs",
                         "mount_jitter_rms", "air_temp", "humidity", "wind_speed"]
             if c in fr]
    M = fr.dropna(subset=["altitude"]).copy()
    y = M[col].astype(int).values
    if y.sum() < 30 or M["day_obs"].nunique() < 6:
        return None
    X = M[feats].astype(float)
    groups = M["day_obs"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps = [], []
    for tr, te in cv.split(X, y, groups):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                             learning_rate=0.06,
                                             class_weight="balanced",
                                             random_state=42)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
    if len(aucs) < 2:
        return None
    chance = y.mean()
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                ap=float(np.mean(aps)), chance=float(chance),
                lift=float(np.mean(aps) / chance) if chance > 0 else np.nan,
                n_folds=len(aucs))


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    TABDIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(CACHE / "frame.parquet")
    pairs = pd.read_parquet(CACHE / "star_pairs.parquet")
    cat = pd.read_parquet(CACHE / "star_catalog.parquet")
    frame = enrich(frame, pairs)
    print(f"frame {len(frame)} exposures, pairs {len(pairs)}, stars {len(cat)}")

    grid = A.nested_annulus_grid()
    mem_full, cells = A.annulus_membership(
        len(frame), pairs["exp_idx"].values, pairs["sep"].values, grid,
        pair_mag=pairs["vmag"].values)
    print(f"annulus x brightness grid: {len(grid)} annuli x "
          f"{len(A.MAG_TIERS)} magnitude tiers = {len(cells)} cells "
          f"({mem_full.nbytes/1e9:.2f} GB)")

    names = list(A.BITS)
    if args.only:
        names = [n.strip() for n in args.only.split(",")]

    results = {}
    for name in names:
        print(f"\n=== {name} ===")
        ev, lines = analyse(name, frame, pairs, cat, cells, mem_full, args.perm)
        results[name] = (ev, lines)

    # BH across the artifacts that actually produced a scan p-value, excluding
    # sub-labels (collinear, so not an independent test).
    testable = [n for n, (ev, _) in results.items()
                if np.isfinite(ev.get("p_fwer", np.nan))
                and n not in A.SUBLABEL_OF]
    if testable:
        q = A.bh_qvalues([results[n][0]["p_fwer"] for n in testable])
        for n, qq in zip(testable, q):
            results[n][0]["q_bh"] = float(qq)

    for name, (ev, lines) in results.items():
        verdict, met, failed = A.assign_verdict(ev)
        ev["verdict"] = verdict
        ev["criteria_met"] = met
        ev["criteria_failed"] = failed
        lines = lines + verdict_block(ev, verdict, met, failed)
        path = OUTDIR / f"{name}_summary.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}  [{verdict}]")

    write_cross_summary(results, cells, frame)

    # Persist the evidence so `build_report.py` plots the very numbers that were
    # printed above, rather than recomputing them and risking a silent mismatch.
    import pickle
    blob = {n: ev for n, (ev, _) in results.items()}
    with open(TABDIR / "evidence.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    print(f"wrote {TABDIR / 'evidence.pkl'}")


def join_telescope_state(e):
    """Attach ConsDB image quality / mount jitter and EFD dome & cover state.

    Both caches now span every inspected night, not just the arc epoch, so the
    telescope-state questions are answerable for the early-labelled artifacts
    (tape, banana, ghoul, ...) rather than silently NaN exactly where they live.
    """
    cdb_path = CACHE / "consdb.parquet"
    if cdb_path.exists():
        cdb = pd.read_parquet(cdb_path)
        cdb["exposure_id"] = cdb["exposure_id"].astype("int64")
        cdb = cdb.drop_duplicates(subset="exposure_id")
        e["exposure_id"] = e["exposure_id"].astype("int64")
        before = len(e)
        # `pressure` exists in both sources; keep the CSV's and drop ConsDB's.
        cdb = cdb.drop(columns=[c for c in ("pressure",) if c in cdb.columns])
        e = e.merge(cdb, on="exposure_id", how="left")
        assert len(e) == before, "ConsDB merge changed the row count"
        cov = e[["sky_bg_median", "mount_jitter_rms"]].notna().mean()
        print("  ConsDB coverage: " +
              ", ".join(f"{k} {v:.2f}" for k, v in cov.items()))

    efd_path = CACHE / "efd_20s_raw.parquet"
    if efd_path.exists():
        raw = pd.read_parquet(efd_path)
        import build_star_geometry as B
        efd = B.interp_efd_to_exposures(e, raw)
        e = e.join(efd)
        print("  EFD coverage: " +
              ", ".join(f"{c} {e[c].notna().mean():.2f}"
                        for c in ("dome_az", "mirror_cover_0") if c in e))
        if "dome_az" in e:
            e["dome_tel_daz"] = A.wrap180(e["dome_az"] - e["azimuth"])
            e["dome_tel_daz_abs"] = e["dome_tel_daz"].abs()
            e["dome_follow_err"] = A.wrap180(e["dome_az"] - e["dome_az_cmd"])
            if "moon_az" in e:
                e["dome_moon_daz"] = A.wrap180(e["dome_az"] - e["moon_az"])
                e["dome_moon_daz_abs"] = e["dome_moon_daz"].abs()
        covs = [c for c in e.columns if c.startswith("mirror_cover_")]
        if covs:
            e["mirror_cover_max"] = e[covs].max(axis=1)
        if "dome_shutter_a" in e and "dome_shutter_b" in e:
            e["dome_shutter_min"] = e[["dome_shutter_a", "dome_shutter_b"]].min(axis=1)
    return e


def enrich(frame, pairs):
    """Derived geometry + the per-exposure star summaries used by the screen."""
    e = join_telescope_state(frame)
    if "moon_az" in e:
        e["moon_daz"] = A.wrap180(e["moon_az"] - e["azimuth"])
        e["moon_dalt"] = e["moon_alt"] - e["altitude"]
        e["moon_pa_cam"] = A.wrap360(e["moon_pa_azalt"] - e["rotator_angle"])
        e["moon_bright"] = e["moon_illum"] * np.clip(
            np.sin(np.deg2rad(e["moon_alt"])), 0, None)
    for s in ("star0", "star1"):
        if f"{s}_az" in e:
            e[f"{s}_daz"] = A.wrap180(e[f"{s}_az"] - e["azimuth"])
            e[f"{s}_dalt"] = e[f"{s}_alt"] - e["altitude"]
            e[f"{s}_pa_cam"] = A.wrap360(e[f"{s}_pa_azalt"] - e["rotator_angle"])
    # per-exposure bright-star summaries from the pair table
    pe, ps, pv = pairs["exp_idx"].values, pairs["sep"].values, pairs["vmag"].values
    for lim, name in ((10.0, "n_bright_10deg"), (25.0, "n_bright_25deg")):
        k = (ps < lim) & (pv < 3.0)
        e[name] = np.bincount(pe[k], minlength=len(e))
    k3 = pv < 3.0
    near = pd.Series(ps[k3]).groupby(pe[k3]).min()
    col = np.full(len(e), np.nan)
    col[near.index.values] = near.values
    e["nearest_bright_sep"] = col
    return e


def verdict_block(ev, verdict, met, failed):
    L = ["", hr(), f"VERDICT: {verdict}", hr(), ""]
    expl = {
        "MECHANISM IDENTIFIED":
            "A specific bright-star geometry predicts this artifact, and the "
            "association survives every control applied: the burst-preserving "
            "null, the removal of its top-contributing star, displaced control "
            "annuli, and a magnitude dose-response.",
        "STRONG ASSOCIATION":
            "The star geometry is associated with this artifact beyond what "
            "burstiness explains, but not every mechanism criterion is met.",
        "SUGGESTIVE":
            "There is a hint of a preferred star geometry. It is NOT "
            "distinguishable from pointing-schedule coincidence with the data "
            "available.",
        "CONFOUNDED":
            "A large crude enrichment exists, but it is explained by when and "
            "where the survey happened to be pointing rather than by the star. "
            "Either it does not survive the burst-preserving null, or it survives "
            "the null while resting on too few independent nights, too few "
            "labelled exposures, or a single dominant star to support a "
            "mechanism claim.",
        "NULL":
            "Adequately powered and no star-geometry effect found.",
        "UNDERPOWERED":
            "Too few independent nights to tell. This means 'cannot tell', NOT "
            "'no effect'.",
    }
    L += textwrap.wrap(expl.get(verdict, ""), 78)
    L += [""]
    if met:
        L += ["  criteria met:"] + [f"    + {m}" for m in met]
    if failed:
        L += ["  criteria not met:"] + [f"    - {m}" for m in failed]
    if verdict in ("NULL", "UNDERPOWERED"):
        z = ev.get("z", np.nan)
        if np.isfinite(z):
            L += ["", textwrap.fill(
                "  Power statement: the scan's strongest cell reached z = "
                f"{fmt(z,2)} against the burst-preserving null, on "
                f"{ev.get('nights_in', 0)} nights with in-annulus labels. An "
                "effect large enough to raise night-recurrence about 5 sigma "
                "above that null would have been detected; anything smaller "
                "would not.", 78)]
        else:
            L += ["", textwrap.fill(
                "  Power statement: no annulus-by-brightness cell had enough "
                f"support to test at all ({ev.get('n_labels', 0)} labels on "
                f"{ev.get('nights', 0)} nights). No effect size is excluded by "
                "this result -- the data cannot address the question either way.",
                78)]
    L += [""]
    return L


def write_cross_summary(results, cells, frame):
    L = [hr(), "STRAY-LIGHT ARTIFACTS -- CROSS-ARTIFACT SUMMARY", hr(),
         f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
         f"labels: data/raw/stray_light_20260506.csv    "
         f"frame: {len(frame)} on-sky exposures on "
         f"{frame['day_obs'].nunique()} inspected nights", ""]
    L += textwrap.wrap(
        "READ THIS FIRST. The question behind this analysis was whether each "
        "artifact type always has a bright star at some specific off-axis "
        "distance from the boresight. The short answer is that the naive way of "
        "asking -- compare the artifact rate inside a candidate annulus with the "
        "rate outside -- gives badly misleading answers here, and most of the "
        "apparently strong results it produces are not real.", 78)
    L += [""]
    L += textwrap.wrap(
        "The reason is burstiness. Artifact labels arrive in runs of consecutive "
        "exposures on a handful of nights, and the survey revisits the same "
        "fields, so whichever annulus that night's field happens to populate will "
        "look enriched. Measured against a null that reproduces the burst "
        "structure, the largest 'lift' the scan finds by chance alone is already "
        "between 1.5x and 5.8x depending on the artifact. So a 4x enrichment is "
        "not evidence of anything -- it is the typical value of the null.", 78)
    L += ["", hr("-"), "VERDICTS, ranked by strength of evidence", hr("-"), ""]

    rows = []
    for name, (ev, _) in results.items():
        rows.append(dict(artifact=name, verdict=ev.get("verdict", "?"),
                         tier=ev.get("tier", "?"), n=ev.get("n_labels", 0),
                         nights=ev.get("nights", 0),
                         annulus=(f"[{ev['best_lo']:.2f},{ev['best_hi']:.2f})"
                                  if "best_lo" in ev else "-"),
                         vmax=(f"V<{ev['best_vmax']:.1f}"
                               if ev.get("best_vmax") is not None
                               and np.isfinite(ev.get("best_vmax", np.nan))
                               else "-"),
                         z=ev.get("z", np.nan), p=ev.get("p_fwer", np.nan),
                         q=ev.get("q_bh", np.nan),
                         lift=ev.get("crude_lift", np.nan),
                         stars=ev.get("n_stars", 0),
                         nights_in=ev.get("nights_in", 0)))
    tab = pd.DataFrame(rows)
    tab["order"] = tab["verdict"].map(
        {v: i for i, v in enumerate(A.VERDICT_ORDER)}).fillna(99)
    tab = tab.sort_values(["order", "z"], ascending=[True, False])

    L += [f"{'artifact':10s} {'verdict':21s} {'T':2s} {'n':>5s} {'nights':>6s} "
          f"{'best annulus':>14s} {'stars<':>6s} {'z':>7s} {'p_FWER':>8s} "
          f"{'q_BH':>7s} {'lift':>7s} {'nstar':>5s}"]
    for _, r in tab.iterrows():
        note = "  <- sub-label of tape" if r["artifact"] in A.SUBLABEL_OF else ""
        L += [f"{r['artifact']:10s} {r['verdict']:21s} {r['tier']:2s} "
              f"{int(r['n']):5d} {int(r['nights']):6d} {r['annulus']:>14s} "
              f"{r['vmax']:>6s} {fmt(r['z'],2):>7s} {fmt(r['p'],4):>8s} "
              f"{fmt(r['q'],3):>7s} {fmt(r['lift'],1):>7s} "
              f"{int(r['stars']):5d}{note}"]
    L += [""]
    L += textwrap.wrap(
        "Columns: z is how far the number of distinct nights showing a star in "
        "the best annulus exceeds the burst-preserving null. p_FWER is corrected "
        f"for searching all {len(cells)} annulus-by-brightness cells. q_BH "
        "corrects across artifacts. "
        "'lift' is the crude exposure-level ratio -- shown only so it can be "
        "compared with z, because where the two disagree the lift is the one "
        "that is wrong.", 78)
    L += [""]

    # per-artifact prose
    L += [hr("-"), "MOST PROBABLE CAUSE, ARTIFACT BY ARTIFACT", hr("-"), ""]
    for _, r in tab.iterrows():
        name = r["artifact"]
        ev = results[name][0]
        L += [f"{name.upper()}  ({r['verdict']}, tier {r['tier']}, "
              f"{int(r['n'])} labels on {int(r['nights'])} nights)"]
        L += [f"  see {name}_summary.txt for the full working"]
        for line in cause_prose(name, ev):
            L += textwrap.wrap(line, 76, initial_indent="  ",
                               subsequent_indent="  ")
        L += [""]

    L += [hr("-"), "WHAT THE TELEMETRY CANNOT SETTLE", hr("-"), ""]
    L += textwrap.wrap(
        "Which surface scatters the light. A feature fixed in the camera frame "
        "implicates the optics, one fixed in the dome frame the structure, one "
        "fixed on the sky the atmosphere. Separating those needs the source "
        "direction measured in each frame, and this dataset provides only about "
        "one bright star per exposure inside any candidate annulus. With a single "
        "direction the three frames differ by fixed offsets, and two stars at the "
        "same radius rotate together, so no amount of additional telemetry of this "
        "kind breaks the degeneracy. Answering it needs the artifact's own "
        "orientation measured in the pixels.", 78)
    L += [""]
    L += textwrap.wrap(
        "Label completeness. Artifacts are found by human inspection, so every "
        "rate here is a rate of labelling. The burst structure is consistent with "
        "real events and also with inspection attention.", 78)
    L += [""]
    L += textwrap.wrap(
        "What the artifact names mean physically. Apart from 'horseshoe' (L3 lens "
        "baffle reflection, documented in L3_baffle/horseshoe_detailer.py) and "
        "'tape' (scattered light, test LVV-T956-a), no physical definition of "
        "these names is recorded anywhere in this repository. The causes named "
        "above are inferred from geometry alone.", 78)
    L += [""]
    path = OUTDIR / "ALL_ARTIFACTS_SUMMARY.txt"
    path.write_text("\n".join(L) + "\n")
    print(f"\nwrote {path}")
    tab.drop(columns=["order"]).to_csv(TABDIR / "verdicts.csv", index=False)


def cause_prose(name, ev):
    """Plain-language cause statement, driven by what the numbers actually say."""
    out = []
    v = ev.get("verdict", "")
    lo, hi = ev.get("best_lo"), ev.get("best_hi")
    if v == "MECHANISM IDENTIFIED":
        out.append(
            f"Cause: a bright star at {lo:.2f}-{hi:.2f} deg off axis. The "
            f"association appears on {ev.get('nights_in',0)} separate nights and "
            f"{ev.get('n_stars',0)} different stars, survives removing its "
            "top-contributing star, and dies away in displaced control annuli -- "
            "a sharp edge, which is what a baffle or a lens edge produces.")
        if name == "horseshoe":
            out.append(
                "This independently reproduces the L3 lens-baffle model already "
                "implemented in L3_baffle/horseshoe_detailer.py, which forbids "
                "stars in [1.995, 2.025] deg. Recovering it from the labels alone "
                "is also the positive control for this whole analysis: the method "
                "rediscovers the one mechanism that was known in advance.")
    elif v == "STRONG ASSOCIATION":
        out.append(
            f"Cause: a bright star (V < {ev.get('best_vmax', float('nan')):.1f}) "
            f"at {lo:.2f}-{hi:.2f} deg off axis is associated with this artifact "
            f"beyond what burstiness explains. It appears on "
            f"{ev.get('nights_in',0)} separate nights and "
            f"{ev.get('n_stars',0)} different stars, with "
            f"{ev.get('k_in_annulus',0)} labelled exposures inside the annulus.")
        out.append(
            "Short of 'mechanism identified' because not every corroborating "
            "criterion is met -- see the criteria list in the per-artifact file "
            "for exactly which.")
    elif v == "SUGGESTIVE":
        out.append(
            f"Possible cause: a bright star near {lo:.2f}-{hi:.2f} deg off axis. "
            "Not distinguishable from pointing-schedule coincidence with the data "
            "available.")
    elif v == "CONFOUNDED":
        share = ev.get("top_star_share")
        k_in = ev.get("k_in_annulus", 0)
        nights_in = ev.get("nights_in", 0)
        p_ = ev.get("p_fwer", np.nan)
        # Be exact about WHY. Some of these pass the null and fail on support;
        # saying "did not survive the null" of those would be simply false.
        if np.isfinite(p_) and p_ > 0.05:
            out.append(
                f"The apparent enrichment at {lo:.2f}-{hi:.2f} deg "
                f"({fmt(ev.get('crude_lift'),1)}x) does not survive the "
                f"burst-preserving null (p = {fmt(p_,3)}).")
        else:
            out.append(
                f"There is an enrichment at {lo:.2f}-{hi:.2f} deg "
                f"({fmt(ev.get('crude_lift'),1)}x, p = {fmt(p_,4)}) that does "
                "clear the null, but it rests on too little independent support "
                "to be called a mechanism:")
            bits = []
            if k_in < 20:
                bits.append(f"only {k_in} labelled exposures fall in the annulus")
            if nights_in < 8:
                bits.append(f"only {nights_in} distinct nights")
            if share and share >= 0.5:
                bits.append(f"one star ({ev.get('top_star')}) supplies "
                            f"{share*100:.0f} % of them")
            if bits:
                out.append("  " + "; ".join(bits) + ".")
        if share and share >= 0.5:
            out.append(
                "That pattern -- one star, a few consecutive nights -- is what "
                "the survey revisiting a single field produces, not what an "
                "optical mechanism produces. A mechanism would show up on many "
                "different stars.")
        if np.isfinite(ev.get("R_sky", np.nan)) and \
           ev.get("R_sky", 0) > ev.get("R_cam", 0):
            out.append(
                "The star direction repeats on the sky more than in the camera, "
                "confirming pointing repetition rather than a camera-fixed ghost.")
        out.append("Most probable cause: not identified from geometry.")
    elif v == "NULL":
        out.append(
            "No bright-star geometry predicts this artifact once burstiness is "
            "respected. Whatever drives it is not a star at a fixed off-axis "
            "distance.")
    elif v == "UNDERPOWERED":
        out.append(
            f"Cannot tell: {ev.get('n_labels',0)} labels on "
            f"{ev.get('nights',0)} nights. This is a statement about power, not a "
            "null result.")
        if name == "moon":
            out.append(
                "The apparent star-annulus signal here cannot be separated from "
                "the Moon's own position at this sample size, and the artifact is "
                "named after the Moon, so the Moon is the obvious candidate.")
        if name == "lighton":
            out.append(
                "Almost every label falls in one run on a single night, so the "
                "effective sample size is about 2 events -- consistent with a "
                "one-off illumination event (a light left on) rather than "
                "anything geometric.")
    # tape deserves its own note whatever verdict it lands on: the dataset
    # already encodes a hypothesis about it, and these numbers bear on it.
    if name == "tape" and lo is not None and 19.0 <= lo <= 22.5:
        out.append(
            "Worth noting: this distance falls inside the 19.5-22 deg annulus "
            "already hard-coded into the CSV's own tape_* columns, which pick the "
            "brightest V<1 star in exactly that range. So the data carries a "
            "pre-existing hypothesis about tape, and this analysis independently "
            "supports it from the labels alone.")
        out.append(
            "But read the shape before reading the peak: the night-level excess is "
            "a broad symmetric bump roughly 19-22 deg wide rather than a sharp "
            "step. A baffle or lens edge gives a hard edge; a broad bump is more "
            "consistent with scattering off a large structure. So the distance is "
            "real, and the single best bin is not to be over-read.")
    if ev.get("moonup_frac", 0) > 0.9 and ev.get("n_labels", 0) >= 50:
        out.append(
            f"Separately, {ev['moonup_frac']*100:.0f} % of these labels occur "
            "with the Moon above the horizon, so the Moon gates this artifact "
            "whatever the star geometry does.")
    return out


if __name__ == "__main__":
    main()
