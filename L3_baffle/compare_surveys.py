#!/usr/bin/env python3
"""Compare 1-Year Nominal Rubin Survey vs Horseshoe Avoidance Survey.

This script performs a comprehensive scientific comparison between:
1. Nominal 1-year baseline survey (`rubin_visits_ra_dec.parquet` / `rubin_one_year.db`)
2. Horseshoe-avoided 1-year survey (`rubin_visits_horseshoe_avoided.parquet` / `rubin_one_year_horseshoe_avoided.db`)

Metrics analyzed:
- Overall throughput and band distribution (u, g, r, i, z, y)
- Sky coverage area (deg^2)
- Number of passes (visit count) per sky pixel & differential visit maps (ΔN)
- Expected coadded 5-sigma depth (m5_coadded) & differential depth maps (Δm5)
- Spatial impact and verification of the 254 shifted visits
"""

import os
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import healpy as hp

# Paths
BASE_DIR = Path("/sdf/home/a/ataranto/straylight/L3_baffle")
OUT_DIR = BASE_DIR / "rubin_one_year_output"
PLOT_DIR = OUT_DIR / "comparison_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

NOM_PARQUET = OUT_DIR / "rubin_visits_ra_dec.parquet"
NOM_DB = OUT_DIR / "rubin_one_year.db"
AVOID_PARQUET = OUT_DIR / "rubin_visits_horseshoe_avoided.parquet"
AVOID_DB = OUT_DIR / "rubin_one_year_horseshoe_avoided.db"
SHIFTED_CSV = OUT_DIR / "horseshoe_shifted_visits.csv"

# Global analysis parameters
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
PIX_AREA = hp.nside2pixarea(NSIDE, degrees=True)
FOV_RADIUS_RAD = np.radians(1.75)  # LSST 3.5 deg field of view
BANDS = ["u", "g", "r", "i", "z", "y"]
BAND_COLORS = {
    "u": "#1f77b4",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#ff7f0e",
    "z": "#9467bd",
    "y": "#8c564b",
}


def load_datasets():
    print("[1/6] Loading datasets...")
    # 1. Load Nominal
    if NOM_DB.exists():
        con_nom = sqlite3.connect(NOM_DB)
        query = (
            "SELECT observationId as ID, fieldRA as ra_deg, fieldDec as dec_deg, "
            "band, fiveSigmaDepth as m5, airmass, visitExposureTime as exptime, "
            "seeingFwhmEff as seeing, slewTime, night, observationStartMJD as mjd "
            "FROM observations"
        )
        df_nom = pd.read_sql_query(query, con_nom)
        con_nom.close()
    else:
        df_nom = pd.read_parquet(NOM_PARQUET)
        if "fivesigmadepth" in df_nom.columns:
            df_nom["m5"] = df_nom["fivesigmadepth"]

    # 2. Load Avoided
    if AVOID_PARQUET.exists():
        df_avoid = pd.read_parquet(AVOID_PARQUET)
        if "fivesigmadepth" in df_avoid.columns:
            df_avoid["m5"] = df_avoid["fivesigmadepth"]
    else:
        con_avoid = sqlite3.connect(AVOID_DB)
        query = (
            "SELECT observationId as ID, fieldRA as ra_deg, fieldDec as dec_deg, "
            "band, fiveSigmaDepth as m5, airmass, visitExposureTime as exptime, "
            "seeingFwhmEff as seeing, slewTime, night, observationStartMJD as mjd "
            "FROM observations"
        )
        df_avoid = pd.read_sql_query(query, con_avoid)
        con_avoid.close()

    # 3. Load Shifted visits report
    df_shifted = pd.read_csv(SHIFTED_CSV) if SHIFTED_CSV.exists() else pd.DataFrame()

    print(f"      Nominal survey: {len(df_nom):,} visits")
    print(f"      Avoided survey: {len(df_avoid):,} visits (diff: {len(df_avoid)-len(df_nom):+,} visits)")
    print(f"      Shifted visits: {len(df_shifted):,} visits")
    return df_nom, df_avoid, df_shifted


def compute_skymaps(df, band=None):
    """Compute visit count and coadded 5-sigma depth on HEALPix grid."""
    if band is not None:
        sub = df[df["band"] == band]
    else:
        sub = df

    ra_rad = np.radians(sub["ra_deg"].values)
    dec_rad = np.radians(sub["dec_deg"].values)
    m5 = sub["m5"].values

    theta = np.pi / 2.0 - dec_rad
    phi = ra_rad
    vecs = hp.ang2vec(theta, phi)

    n_visits = np.zeros(NPIX, dtype=int)
    flux_sum = np.zeros(NPIX, dtype=float)

    for i in range(len(sub)):
        pix = hp.query_disc(NSIDE, vecs[i], FOV_RADIUS_RAD)
        n_visits[pix] += 1
        if np.isfinite(m5[i]):
            flux_sum[pix] += 10.0 ** (0.8 * m5[i])

    coadded_m5 = np.full(NPIX, np.nan)
    valid = n_visits > 0
    coadded_m5[valid] = 1.25 * np.log10(flux_sum[valid])

    return n_visits, coadded_m5


def plot_mollweide_trio(map_nom, map_avoid, title_prefix, unit, filename, is_diff_mag=False, vmin=None, vmax=None):
    """Plot a 3-panel figure: Nominal, Avoided, and Differential Map."""
    diff_map = map_avoid.copy()
    valid_both = (map_nom > 0) & (map_avoid > 0) if not is_diff_mag else (np.isfinite(map_nom) & np.isfinite(map_avoid))
    
    diff_data = np.full(NPIX, np.nan)
    diff_data[valid_both] = map_avoid[valid_both] - map_nom[valid_both]

    fig = plt.figure(figsize=(18, 5))
    
    # 1. Nominal
    plt.subplot(131)
    nom_disp = map_nom.astype(float).copy()
    nom_disp[~valid_both] = np.nan
    v_min = vmin if vmin is not None else np.nanpercentile(nom_disp, 1)
    v_max = vmax if vmax is not None else np.nanpercentile(nom_disp, 99)
    hp.mollview(
        nom_disp,
        title=f"Nominal: {title_prefix}",
        unit=unit,
        min=v_min,
        max=v_max,
        cmap="viridis",
        hold=True,
        sub=(1, 3, 1),
    )
    hp.graticule(verbose=False)

    # 2. Avoided
    plt.subplot(132)
    av_disp = map_avoid.astype(float).copy()
    av_disp[~valid_both] = np.nan
    hp.mollview(
        av_disp,
        title=f"Horseshoe Avoided: {title_prefix}",
        unit=unit,
        min=v_min,
        max=v_max,
        cmap="viridis",
        hold=True,
        sub=(1, 3, 2),
    )
    hp.graticule(verbose=False)

    # 3. Differential
    plt.subplot(133)
    diff_vals = diff_data[valid_both]
    bound = np.nanpercentile(np.abs(diff_vals), 98) if len(diff_vals) > 0 else 1.0
    if bound == 0:
        bound = 1.0
    
    hp.mollview(
        diff_data,
        title=f"Differential (Avoided - Nominal): Δ{title_prefix}",
        unit=f"Δ{unit}",
        min=-bound,
        max=bound,
        cmap="coolwarm",
        hold=True,
        sub=(1, 3, 3),
    )
    hp.graticule(verbose=False)
    out_path = PLOT_DIR / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {out_path.name}")


def generate_band_differential_grid(df_nom, df_avoid, metric="visits"):
    """Generate 6-panel Mollweide grid of differential maps per band."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Per-Band Differential {'Visit Count (ΔN)' if metric == 'visits' else 'Coadded 5-σ Depth (Δm5)'} (Avoided - Nominal)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, band in enumerate(BANDS, 1):
        n_nom, m5_nom = compute_skymaps(df_nom, band=band)
        n_av, m5_av = compute_skymaps(df_avoid, band=band)

        if metric == "visits":
            valid = (n_nom > 0) & (n_av > 0)
            diff = np.full(NPIX, np.nan)
            diff[valid] = n_av[valid] - n_nom[valid]
            bound = max(5, np.nanpercentile(np.abs(diff[valid]), 98))
            unit = "ΔVisits"
            cmap = "coolwarm"
        else:
            valid = np.isfinite(m5_nom) & np.isfinite(m5_av)
            diff = np.full(NPIX, np.nan)
            diff[valid] = m5_av[valid] - m5_nom[valid]
            bound = max(0.05, np.nanpercentile(np.abs(diff[valid]), 98))
            unit = "Δmag"
            cmap = "coolwarm"

        plt.subplot(2, 3, idx)
        hp.mollview(
            diff,
            title=f"Band {band} (bound: ±{bound:.2f} {unit})",
            unit=unit,
            min=-bound,
            max=bound,
            cmap=cmap,
            hold=True,
            sub=(2, 3, idx),
        )
        hp.graticule(verbose=False)

    filename = f"03_band_differential_{metric}.png"
    out_path = PLOT_DIR / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {out_path.name}")


def generate_quantitative_figures(df_nom, df_avoid, df_shifted, stats_table):
    """Generate multi-panel histograms, cumulative distributions, and bar charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar Chart of Visits per Band
    ax1 = axes[0, 0]
    x = np.arange(len(BANDS))
    width = 0.35
    nom_counts = [len(df_nom[df_nom["band"] == b]) for b in BANDS]
    avoid_counts = [len(df_avoid[df_avoid["band"] == b]) for b in BANDS]

    rects1 = ax1.bar(x - width / 2, nom_counts, width, label="Nominal", color="#4a90e2", edgecolor="black")
    rects2 = ax1.bar(x + width / 2, avoid_counts, width, label="Horseshoe Avoided", color="#50e3c2", edgecolor="black")

    ax1.set_xlabel("Filter Band", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Number of Visits", fontsize=12, fontweight="bold")
    ax1.set_title("Visit Count per Band (Nominal vs Avoided)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(BANDS, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate percentage differences
    for i, b in enumerate(BANDS):
        diff_pct = (avoid_counts[i] - nom_counts[i]) / nom_counts[i] * 100.0
        diff_val = avoid_counts[i] - nom_counts[i]
        ax1.annotate(
            f"{diff_val:+d}\n({diff_pct:+.1f}%)",
            xy=(x[i] + width / 2, avoid_counts[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 2. Histogram of Coadded Depth Differences per Band
    ax2 = axes[0, 1]
    for b in BANDS:
        n_nom, m5_nom = compute_skymaps(df_nom, band=b)
        n_av, m5_av = compute_skymaps(df_avoid, band=b)
        valid = np.isfinite(m5_nom) & np.isfinite(m5_av)
        diff = m5_av[valid] - m5_nom[valid]
        ax2.hist(
            diff,
            bins=np.linspace(-0.25, 0.25, 60),
            histtype="step",
            linewidth=2,
            label=f"{b} (mean: {np.mean(diff):+.3f})",
            color=BAND_COLORS[b],
        )

    ax2.axvline(0, color="gray", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("ΔCoadded Depth [mag] (Avoided - Nominal)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Sky Pixels", fontsize=12, fontweight="bold")
    ax2.set_title("Distribution of Coadded 5-σ Depth Difference (Δm5)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim(-0.25, 0.25)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Airmass Cumulative Distributions
    ax3 = axes[1, 0]
    nom_airmass = np.sort(df_nom["airmass"].dropna().values)
    avoid_airmass = np.sort(df_avoid["airmass"].dropna().values)
    p_nom = np.linspace(0, 1, len(nom_airmass))
    p_avoid = np.linspace(0, 1, len(avoid_airmass))

    ax3.plot(nom_airmass, p_nom, label=f"Nominal (median: {np.median(nom_airmass):.2f})", color="#2c3e50", linewidth=2)
    ax3.plot(avoid_airmass, p_avoid, label=f"Avoided (median: {np.median(avoid_airmass):.2f})", color="#e74c3c", linestyle="--", linewidth=2)
    ax3.set_xlabel("Airmass", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Cumulative Fraction", fontsize=12, fontweight="bold")
    ax3.set_title("Survey Airmass Cumulative Distribution", fontsize=13, fontweight="bold")
    ax3.set_xlim(1.0, 2.0)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Summary Table Graphic
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    headers = ["Band", "Nom. Visits", "Avoid. Visits", "ΔVisits", "ΔVisits (%)", "ΔDepth (mag)"]
    for row in stats_table:
        table_data.append([
            row["band"],
            f"{row['n_nom']:,}",
            f"{row['n_avoid']:,}",
            f"{row['delta_n']:+,}",
            f"{row['delta_n_pct']:+.2f}%",
            f"{row['delta_m5']:+.4f}",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "#ffffff")

    ax4.set_title("Survey Key Comparison Metrics", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    out_path = PLOT_DIR / "05_histograms_and_distributions.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {out_path.name}")


def generate_shifted_visits_impact_figure(df_shifted, df_nom):
    """Plot detailed map, band breakdown, and separation diagnostics for shifted visits."""
    if df_shifted.empty:
        return

    n_shifted = len(df_shifted)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Sky distribution of shifted visits
    ax1 = axes[0]
    ax1.scatter(
        df_nom["ra_deg"].values[::20],
        df_nom["dec_deg"].values[::20],
        s=1,
        color="#bdc3c7",
        alpha=0.3,
        label="Rubin Survey Footprint (sample)",
    )
    scatter = ax1.scatter(
        df_shifted["orig_ra_deg"],
        df_shifted["orig_dec_deg"],
        c=df_shifted["night"],
        cmap="plasma",
        s=35,
        edgecolor="black",
        linewidth=0.5,
        label=f"Shifted Visits (N={n_shifted})",
    )
    cb = plt.colorbar(scatter, ax=ax1, orientation="horizontal", pad=0.15)
    cb.set_label("Survey Night", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Right Ascension [deg]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Declination [deg]", fontsize=12, fontweight="bold")
    ax1.set_title(f"Spatial Distribution of {n_shifted} Shifted Pointings", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Shifted visits by band
    ax2 = axes[1]
    b_counts = [len(df_shifted[df_shifted["band"] == b]) for b in BANDS]
    b_colors = [BAND_COLORS.get(b, "#3498db") for b in BANDS]
    bars = ax2.bar(BANDS, b_counts, color=b_colors, edgecolor="black", width=0.6)
    for bar, count in zip(bars, b_counts):
        pct = (count / n_shifted) * 100.0 if n_shifted > 0 else 0
        ax2.annotate(
            f"{count}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_xlabel("Observation Filter Band", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Shifted Visits", fontsize=12, fontweight="bold")
    ax2.set_title("Shifted Visits by Filter Band (Band-Specific Trigger)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(b_counts) * 1.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Separation before vs after
    ax3 = axes[2]
    import ast
    seps_before = []
    for stars_str in df_shifted["stars"]:
        try:
            stars_list = ast.literal_eval(stars_str) if isinstance(stars_str, str) else stars_str
            for s in stars_list:
                seps_before.append(s["sep_deg"])
        except Exception:
            pass

    ax3.axvspan(1.995, 2.025, color="red", alpha=0.25, label="Horseshoe Zone [1.995°, 2.025°]")
    if seps_before:
        ax3.hist(
            seps_before,
            bins=np.linspace(1.98, 2.04, 30),
            color="#e74c3c",
            edgecolor="black",
            alpha=0.8,
            label=f"Nominal Separation (N={len(seps_before)})",
        )

    ax3.set_xlabel("Angular Separation to Bright Star [deg]", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Number of Occurrences", fontsize=12, fontweight="bold")
    ax3.set_title("Initial Star Separations in Forbidden Annulus", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = PLOT_DIR / "06_shifted_visits_impact.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {out_path.name}")


def main():
    print("=" * 75)
    print("RUBIN SURVEY COMPARISON: NOMINAL vs HORSESHOE AVOIDED")
    print("=" * 75)

    df_nom, df_avoid, df_shifted = load_datasets()

    print("\n[2/6] Computing all-sky visit count and coadded depth maps (nside=64)...")
    n_nom_total, m5_nom_total = compute_skymaps(df_nom)
    n_av_total, m5_av_total = compute_skymaps(df_avoid)

    cov_nom = (n_nom_total > 0).sum() * PIX_AREA
    cov_av = (n_av_total > 0).sum() * PIX_AREA
    print(f"      Total Sky Coverage: Nominal = {cov_nom:.1f} deg^2 | Avoided = {cov_av:.1f} deg^2 (Δ: {cov_av - cov_nom:+.1f} deg^2)")

    print("\n[3/6] Generating All-Sky Mollweide Trio figures...")
    plot_mollweide_trio(
        map_nom=n_nom_total,
        map_avoid=n_av_total,
        title_prefix="Total Visits (All Bands)",
        unit="Visits",
        filename="01_total_visits_maps.png",
        is_diff_mag=False,
    )
    plot_mollweide_trio(
        map_nom=m5_nom_total,
        map_avoid=m5_av_total,
        title_prefix="Coadded 5-σ Depth (All Bands)",
        unit="mag",
        filename="02_total_coadded_depth_maps.png",
        is_diff_mag=True,
    )

    print("\n[4/6] Generating per-band differential grids...")
    generate_band_differential_grid(df_nom, df_avoid, metric="visits")
    generate_band_differential_grid(df_nom, df_avoid, metric="depth")

    print("\n[5/6] Calculating per-band summary table and statistics...")
    stats_table = []
    print(f"\n{'Band':<5} | {'Nominal':<10} | {'Avoided':<10} | {'ΔVisits':<10} | {'Δ%':<8} | {'Cov. Nom':<10} | {'Cov. Avoid':<11} | {'ΔDepth (mag)':<12}")
    print("-" * 95)
    for b in BANDS:
        n_nom_b, m5_nom_b = compute_skymaps(df_nom, band=b)
        n_av_b, m5_av_b = compute_skymaps(df_avoid, band=b)
        v_n = len(df_nom[df_nom["band"] == b])
        v_a = len(df_avoid[df_avoid["band"] == b])
        c_n = (n_nom_b > 0).sum() * PIX_AREA
        c_a = (n_av_b > 0).sum() * PIX_AREA
        valid_both = np.isfinite(m5_nom_b) & np.isfinite(m5_av_b)
        delta_m5 = np.nanmean(m5_av_b[valid_both] - m5_nom_b[valid_both])
        diff_n = v_a - v_n
        diff_pct = diff_n / v_n * 100.0

        stats_table.append({
            "band": b,
            "n_nom": v_n,
            "n_avoid": v_a,
            "delta_n": diff_n,
            "delta_n_pct": diff_pct,
            "cov_nom": c_n,
            "cov_avoid": c_a,
            "delta_m5": delta_m5,
        })
        print(f"{b:<5} | {v_n:<10,d} | {v_a:<10,d} | {diff_n:<+10,d} | {diff_pct:<+7.2f}% | {c_n:<10.1f} | {c_a:<11.1f} | {delta_m5:+12.4f}")

    print("\n[6/6] Generating quantitative distribution and shifted visit figures...")
    generate_quantitative_figures(df_nom, df_avoid, df_shifted, stats_table)
    generate_shifted_visits_impact_figure(df_shifted, df_nom)

    # Save summary table to CSV
    summary_df = pd.DataFrame(stats_table)
    summary_csv = OUT_DIR / "survey_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV: {summary_csv}")

    print("\n" + "=" * 75)
    print("ALL COMPARISON MAPS AND FIGURES GENERATED SUCCESSFULLY!")
    print(f"Figures saved in: {PLOT_DIR.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
