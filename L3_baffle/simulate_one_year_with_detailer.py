#!/usr/bin/env python3
"""Run a full Rubin survey simulation with the Horseshoe Avoidance Detailer.

This script runs a Rubin scheduler simulation (by default 1 full year = 365.25 days)
with `HorseshoeAvoidanceDetailer` attached to all scheduler surveys. Whenever a
bright star (V <= 3.0 from YBSC v5) falls within the critical L3 horseshoe baffle
annulus [1.995, 2.025] deg, the detailer shifts the visit pointing by +0.02 deg
in both RA and Dec to avoid stray light artifacts.

Outputs generated:
1. SQLite database: Full simulation records produced by rubin_scheduler.
2. Parquet and CSV catalogs: Processed visits with RA/Dec in degrees & radians,
   UTC timestamps, bands, airmass, notes, and a boolean flag `horseshoe_avoided`.
3. Shifted visits report (CSV): Detailed log of all pointings modified by the
   detailer, with initial and shifted coordinates and the matched stars.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.time import Time

# Ensure rubin_sim_data directory is set to the user's valid directory
rubin_data_dir = Path.home() / "rubin-user/rubin_sim_data"
os.environ["RUBIN_SIM_DATA_DIR"] = str(rubin_data_dir)

try:
    from rubin_scheduler.scheduler.example import (
        generate_baseline_coresched,
        run_sched,
    )
    from rubin_scheduler.scheduler.example.lsst_surveys import (
        BLOB_SURVEY_PARAMS_DEFAULTS,
    )
    # from rubin_scheduler.utils import DEFAULT_NSIDE
except ImportError as err:
    print(
        f"Error importing rubin_scheduler: {err}\n"
        "Please run this script using the LSST conda environment, e.g.:\n"
        "/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.1.0/bin/python "
        f"{sys.argv[0]}",
        file=sys.stderr,
    )
    sys.exit(1)

# Import local detailer
from horseshoe_detailer import (
    HorseshoeAvoidanceDetailer,
    add_horseshoe_detailer_to_scheduler,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simulate Rubin survey observations with Horseshoe Avoidance Detailer."
    )
    parser.add_argument(
        "--survey-length",
        type=float,
        default=365.25,
        help="Survey simulation length in days (default: 365.25 for 1 full year).",
    )
    parser.add_argument(
        "--survey-start-mjd",
        type=float,
        default=61284.0,
        help="Survey start MJD (default: 61284.0).",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="HEALPix nside resolution (default: 32).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rubin_one_year_output",
        help="Directory to save simulation outputs (default: ./rubin_one_year_output).",
    )
    parser.add_argument(
        "--db-filename",
        type=str,
        default="rubin_one_year_horseshoe_avoided.db",
        help="Filename for SQLite database output.",
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default="rubin_visits_horseshoe_avoided.csv",
        help="Filename for processed visits CSV catalog.",
    )
    parser.add_argument(
        "--parquet-filename",
        type=str,
        default="rubin_visits_horseshoe_avoided.parquet",
        help="Filename for processed visits Parquet catalog.",
    )
    parser.add_argument(
        "--shifted-report-filename",
        type=str,
        default="horseshoe_shifted_visits.csv",
        help="Filename for shifted visits summary CSV.",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Path to star catalog FITS file (defaults to L3_baffle/ybsc_v5.fits).",
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=3.0,
        help="Faintest magnitude in observation filter band to trigger avoidance (default: 3.0).",
    )
    parser.add_argument(
        "--vmag-limit",
        type=float,
        default=None,
        help="Deprecated alias for --mag-limit.",
    )
    parser.add_argument(
        "--use-vmag-only",
        action="store_true",
        help="Use standard V-band magnitude instead of reference filter band.",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=1.995,
        help="Inner radius of forbidden annulus in degrees (default: 1.995).",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=2.025,
        help="Outer radius of forbidden annulus in degrees (default: 2.025).",
    )
    parser.add_argument(
        "--offset-ra",
        type=float,
        default=0.05,
        help="Offset in RA in degrees to apply when compromised (default: 0.05).",
    )
    parser.add_argument(
        "--offset-dec",
        type=float,
        default=0.05,
        help="Offset in Dec in degrees to apply when compromised (default: 0.05).",
    )
    parser.add_argument(
        "--offset-mode",
        type=str,
        default="gnomonic",
        choices=["gnomonic", "direct"],
        help="Offset projection mode: 'gnomonic' (tangent plane) or 'direct' (default: gnomonic).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during simulation.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = output_dir / args.db_filename
    csv_path = output_dir / args.csv_filename
    parquet_path = output_dir / args.parquet_filename
    shifted_report_path = output_dir / args.shifted_report_filename

    catalog_path = (
        Path(args.catalog_path).resolve()
        if args.catalog_path
        else Path(__file__).resolve().parent / "ybsc_v5.fits"
    )

    print("=" * 70)
    print("RUBIN SURVEY SIMULATION WITH HORSESHOE AVOIDANCE DETAILER")
    print("=" * 70)
    print(f"Simulation length:     {args.survey_length:.2f} days")
    print(f"Survey start MJD:      {args.survey_start_mjd:.1f}")
    print(f"HEALPix NSIDE:         {args.nside}")
    print(f"Output directory:      {output_dir}")
    print(f"SQLite DB:             {sqlite_path.name}")
    print(f"CSV catalog:           {csv_path.name}")
    print(f"Parquet catalog:       {parquet_path.name}")
    mag_limit = args.vmag_limit if args.vmag_limit is not None else args.mag_limit
    use_obs_band = not args.use_vmag_only
    threshold_desc = f"V <= {mag_limit:.2f}" if args.use_vmag_only else f"obs-band mag <= {mag_limit:.2f}"

    print(f"Star catalog:          {catalog_path.name}")
    print(f"Mag threshold:         {threshold_desc}")
    print(f"Forbidden annulus:     [{args.inner_radius:.3f}, {args.outer_radius:.3f}] deg")
    print(f"Applied offsets:       ΔRA = +{args.offset_ra:.3f}°, ΔDec = +{args.offset_dec:.3f}°")
    print("=" * 70, flush=True)

    # Work around a rubin_scheduler bug: BLOB_SURVEY_PARAMS_DEFAULTS hard-codes
    # "nside" to DEFAULT_NSIDE (32), and generate_blobs/generate_short_blobs/
    # gen_long_gaps_survey build BlobSurvey instances from this dict without
    # overriding nside with the value passed in. This causes a mismatch between
    # the basis functions' HEALPix maps (built with args.nside) and the
    # BlobSurvey's own ra/dec/fields arrays (built with the stale default),
    # leading to IndexError in generate_observations_rough whenever
    # args.nside != DEFAULT_NSIDE.
    BLOB_SURVEY_PARAMS_DEFAULTS["nside"] = args.nside

    # 1. Create Baseline Scheduler
    print("\n[1/4] Generating baseline core scheduler...", flush=True)
    t_start_sched = time.time()
    scheduler = generate_baseline_coresched(
        nside=args.nside,
        survey_start_mjd=args.survey_start_mjd,
    )
    t_sched = time.time() - t_start_sched
    print(f"      Scheduler generated in {t_sched:.1f} s.", flush=True)

    # 2. Instantiate and attach the Horseshoe Avoidance Detailer
    print("\n[2/4] Initializing and attaching HorseshoeAvoidanceDetailer...", flush=True)
    detailer = HorseshoeAvoidanceDetailer(
        catalog_path=catalog_path,
        mag_limit=mag_limit,
        use_observation_band=use_obs_band,
        inner_radius_deg=args.inner_radius,
        outer_radius_deg=args.outer_radius,
        offset_ra_deg=args.offset_ra,
        offset_dec_deg=args.offset_dec,
        offset_mode=args.offset_mode,
        annotate_note=True,
        note_tag="hs_avoid",
        nside=args.nside,
    )

    _, attached_count = add_horseshoe_detailer_to_scheduler(
        scheduler=scheduler,
        detailer=detailer,
        position="prepend_tracking",
    )
    print(f"      Detailer successfully attached to {attached_count} surveys.", flush=True)

    # 3. Run Survey Simulation
    print(f"\n[3/4] Running survey simulation for {args.survey_length:.2f} days...", flush=True)
    if args.survey_length >= 300:
        print("      (Note: Full 1-year simulation typically takes ~1.5 to 2 hours)", flush=True)

    t_start_sim = time.time()
    observatory, scheduler, observations = run_sched(
        scheduler,
        survey_length=args.survey_length,
        survey_start_mjd=args.survey_start_mjd,
        nside=args.nside,
        filename=str(sqlite_path),
        verbose=args.verbose,
    )
    t_sim = time.time() - t_start_sim
    print(f"      Simulation completed in {t_sim / 60.0:.2f} minutes ({t_sim:.1f} s).", flush=True)
    print(f"      Total visits completed: {len(observations)}", flush=True)

    # Print detailer statistics
    print("\n" + "-" * 50)
    detailer.summary()
    print("-" * 50)

    # 4. Post-processing and export
    print("\n[4/4] Post-processing catalog and exporting files...", flush=True)
    df = pd.DataFrame.from_records(observations)

    # UTC timestamps
    if "mjd" in df.columns:
        times = Time(df["mjd"].values, format="mjd", scale="tai")
        utc_times = times.utc
        df["datetime_utc"] = utc_times.to_datetime()
        df["date_utc"] = [dt.strftime("%Y-%m-%d") for dt in df["datetime_utc"]]
        df["year"] = [dt.year for dt in df["datetime_utc"]]
        df["month"] = [dt.month for dt in df["datetime_utc"]]
        df["day"] = [dt.day for dt in df["datetime_utc"]]
        df["hour_utc"] = [
            dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            for dt in df["datetime_utc"]
        ]

    # Coordinate columns in both radians and degrees
    if "RA" in df.columns and "dec" in df.columns:
        df["ra_rad"] = df["RA"]
        df["dec_rad"] = df["dec"]
        df["ra_deg"] = np.degrees(df["ra_rad"])
        df["dec_deg"] = np.degrees(df["dec_rad"])

    # Identification flag for avoided visits
    note_col = df["scheduler_note"] if "scheduler_note" in df.columns else ""
    aux_note_col = df["note"] if "note" in df.columns else ""
    df["horseshoe_avoided"] = (
        note_col.astype(str).str.contains("hs_avoid", na=False)
        | aux_note_col.astype(str).str.contains("hs_avoid", na=False)
    )

    # Column ordering
    preferred_order = [
        "ID",
        "mjd",
        "datetime_utc",
        "date_utc",
        "year",
        "month",
        "day",
        "hour_utc",
        "ra_rad",
        "dec_rad",
        "ra_deg",
        "dec_deg",
        "band",
        "filter",
        "alt",
        "az",
        "airmass",
        "rotSkyPos",
        "rotTelPos",
        "exptime",
        "horseshoe_avoided",
        "scheduler_note",
        "fivesigmadepth",
        "skybrightness",
        "night",
        "slewtime",
        "visittime",
        "slewdist",
    ]
    present_preferred = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in present_preferred]
    export_df = df[present_preferred + remaining_cols]

    # Save CSV
    export_df.to_csv(csv_path, index=False)
    print(f"      Exported CSV:     {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # Save Parquet
    try:
        export_df.to_parquet(parquet_path, index=False)
        print(f"      Exported Parquet: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"      Parquet export skipped: {e}")

    # Save shifted visits report
    df_shifted = detailer.get_shifted_dataframe()
    if not df_shifted.empty:
        df_shifted.to_csv(shifted_report_path, index=False)
        print(f"      Exported Shifted Report: {shifted_report_path} ({len(df_shifted)} visits)")
    else:
        print("      No visits required shifting.")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"Total simulated visits:  {len(df)}")
    print(f"Visits with hs_avoid:    {df['horseshoe_avoided'].sum()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
