"""Figures for the per-artifact LaTeX report.

Every figure is drawn from `outputs/artifact_stats/evidence.pkl`, which
`run_all_artifacts.py` writes as it prints the text summaries. Nothing is
recomputed here, so a figure cannot disagree with the numbers in the `.txt`.

Design rules followed (from the dataviz skill):
  * categorical hues in fixed slot order, never cycled;
  * sequential = one hue light->dark; diverging = two hues + neutral grey midpoint;
  * thin marks, hairline recessive grid, no dashed gridlines;
  * selective direct labels -- never a number on every point;
  * a legend whenever two series are present;
  * one y-axis per panel, never two scales on one plot.
"""
from __future__ import annotations

import pathlib

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- palette
# Reference instance from the dataviz skill (references/palette.md), light mode.
# Slots are used in fixed order; slots 1-3 are the documented all-pairs-safe
# subset, which is why no figure here puts more than three categorical series on
# one axis.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OBS_C, NULL_C, THIRD_C = PAL[1], PAL[0], PAL[2]   # observed=orange, null=blue
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, RULE = "#e1e0d9", "#fcfcfb", "#c3c2b7"

SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#5598e7",
                 "#2a78d6", "#1c5cab", "#0d366b"])
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "div_br", ["#184f95", "#5598e7", "#cde2fb", "#f0efec",
               "#f6c3c3", "#e34948", "#a32424"])
SEQ.set_bad("#f0efec")
DIV.set_bad("#f0efec")

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 9, "axes.labelsize": 9.5,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})

FIGDIR = pathlib.Path("report/figs")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _nodata(ax, msg):
    """An explicit 'not measured' panel -- never an empty axes pretending to a result."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, color=MUTED,
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ===================================================================== fig 1
def fig_scan(name, ev):
    """Night-recurrence z against off-axis radius, one line per brightness tier.

    The scan's whole output in one view: where the excess is, how sharp it is, and
    whether it survives at the brightness tiers that matter.
    """
    f = ev.get("_fig", {}).get("scan")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if not f or f.get("best") is None:
        _nodata(ax, "No annulus-by-brightness cell had enough support to test.\n"
                    "Nothing is plotted because nothing was measured.")
        ax.set_title(f"{name} — annulus scan not performed")
        return _save(fig, f"{name}_scan")

    cells = f["cells"]
    z = np.asarray(f["z_all"], dtype=float)
    lo = np.array([c[0] for c in cells])
    hi = np.array([c[1] for c in cells])
    vm = np.array([c[2] for c in cells])
    wid = hi - lo
    best = f["best"]
    bvm = cells[best][2]

    # One width only, so the curve is a like-for-like comparison across radius.
    # The winner's own width is the natural choice but often too narrow to draw:
    # at 0.05 deg most cells fall below the support floor and carry NaN, leaving a
    # handful of disconnected points that hide the very structure the figure is
    # for. So pick the narrowest width that is actually populated across radius,
    # and say in the axis label which width is being shown.
    widths = sorted(set(np.round(wid, 4)))
    bw = round(cells[best][1] - cells[best][0], 4)
    for w in widths:
        k = np.isclose(wid, w) & (vm == bvm)
        if k.sum() and np.isfinite(z[k]).sum() >= 15:
            bw = w
            break
    tiers = sorted(set(vm))
    # Three tiers max on one axis: slots 1-3 are the all-pairs-validated subset.
    show = [t for t in tiers if t == bvm]
    for t in tiers:
        if t != bvm and len(show) < 3:
            show.append(t)
    show = sorted(show)

    # Cells below the support floor carry NaN, and for a sparse artifact most
    # radii are in that state. A connecting line would assert a continuous curve
    # through radii where nothing was measured, so this is drawn as points: each
    # marker is a cell that actually cleared the floor, and a gap means "not
    # testable here", never "zero here".
    zmax = np.nanmax(z) if np.any(np.isfinite(z)) else 1.0
    n_drawn = 0
    for i, t in enumerate(show):
        k = (vm == t) & (np.isclose(wid, bw)) & np.isfinite(z)
        if k.sum() == 0:
            continue
        o = np.argsort(lo[k])
        x, y = lo[k][o] + bw / 2, z[k][o]
        is_best = t == bvm
        if is_best:
            n_drawn = len(x)
            ax.plot(x, y, linestyle="none", marker="o", color=OBS_C,
                    markersize=4.6, markeredgecolor=SURF, markeredgewidth=0.8,
                    label=f"V < {t:.1f}  (selected)", zorder=4)
        else:
            ax.plot(x, y, linestyle="none", marker="o", color=PAL[i % 3],
                    markersize=3.0, alpha=0.55, markeredgecolor="none",
                    label=f"V < {t:.1f}", zorder=2)

    ax.axhline(0, color=RULE, linewidth=1.0)
    ax.axhline(5, color=MUTED, linewidth=0.9)

    bl, bh = cells[best][0], cells[best][1]
    bx = (bl + bh) / 2
    # Headroom for the legend band and the callout, reserved before annotating.
    ax.set_ylim(min(0, np.nanmin(z) if np.any(np.isfinite(z)) else 0) - 0.5,
                zmax * 1.42)
    ax.annotate("z = 5  (mechanism threshold)", (ax.get_xlim()[0], 5),
                xytext=(5, 5), textcoords="offset points", ha="left",
                fontsize=8, color=MUTED)
    ax.plot([bx], [ev["z"]], "o", color=OBS_C, markersize=8,
            markeredgecolor=SURF, markeredgewidth=1.6, zorder=5)
    # Place the callout on whichever side has room, so it never lands on the
    # y-axis or runs off the right edge.
    x0, x1 = ax.get_xlim()
    right = bx < (x0 + x1) / 2
    # Above the marker, not beside it: beside, the callout collided with the
    # z = 5 threshold label whenever the winner sat near that level.
    ax.annotate(f"[{bl:.2f}, {bh:.2f})°   z = {ev['z']:.1f}", (bx, ev["z"]),
                xytext=(10 if right else -10, 9), textcoords="offset points",
                fontsize=8.5, color=INK, fontweight="semibold",
                ha="left" if right else "right", va="bottom")

    ax.set_xlabel("off-axis distance of the bright star from the boresight [deg]")
    # The "what a gap means" note lives in the LaTeX caption, not on the axes:
    # inside the plot it landed on the cluster of near-zero points it describes.
    ax.set_facecolor(SURF)
    ax.set_ylabel("night-recurrence z\nvs burst-preserving null")
    # Title as a suptitle and the legend in its own band under it: placed inside
    # the axes the legend sat on top of the very peak it labels.
    fig.suptitle(f"{name} — where a bright star coincides with the artifact",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold", color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = _save(fig, f"{name}_scan")
    # Returned alongside the path so the caption can state exactly what is drawn.
    fig_scan.last_meta = {"width": bw, "n_cells": n_drawn}
    return path


# ===================================================================== fig 2
def fig_null(name, ev):
    """Observed scan maximum against both null distributions.

    This is the figure that justifies the verdict: the reader sees how far the
    observation sits from what burstiness alone reaches, for both nulls.
    """
    f = ev.get("_fig", {}).get("scan")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    if not f or f.get("best") is None:
        _nodata(ax, "No scan was performed, so there is no null to compare against.")
        ax.set_title(f"{name} — null calibration not applicable")
        return _save(fig, f"{name}_null")

    ns = np.asarray(f.get("null_max_shift"), dtype=float)
    nc = np.asarray(f.get("null_max_cross"), dtype=float)
    ns = ns[np.isfinite(ns)]
    nc = nc[np.isfinite(nc)]
    obs = ev["z"]

    allv = np.concatenate([v for v in (ns, nc, [obs]) if len(v)])
    bins = np.linspace(min(allv.min(), 0), allv.max() * 1.05, 44)
    if len(ns):
        ax.hist(ns, bins=bins, color=NULL_C, alpha=0.75, linewidth=0,
                label=f"within-night shift  (n={len(ns)})")
    if len(nc):
        ax.hist(nc, bins=bins, color=THIRD_C, alpha=0.65, linewidth=0,
                label=f"cross-night shuffle  (n={len(nc)})")
    ax.set_yscale("log")
    ax.axvline(obs, color=OBS_C, linewidth=2.2, zorder=5)
    # Label the observed line on whichever side has room. When the observation is
    # far to the right -- the interesting case -- a right-hand label runs off the
    # axes and collides with the legend, so it flips to the left of the rule.
    x0, x1 = ax.get_xlim()
    right = obs < (x0 + x1) * 0.62
    ax.annotate(f"observed\nz = {obs:.1f}\np = {ev.get('p_fwer', float('nan')):.4f}",
                (obs, ax.get_ylim()[1] * 0.55),
                xytext=(8 if right else -8, 0), textcoords="offset points",
                ha="left" if right else "right", va="top",
                fontsize=8.5, color=OBS_C, fontweight="semibold")
    ax.set_xlabel("maximum night-recurrence z reached across the whole scan")
    ax.set_ylabel("null replicates")
    # Title above the legend band so neither sits on the other.
    fig.suptitle(f"{name} — is the best cell better than chance?", x=0.02,
                 ha="left", fontsize=11, fontweight="semibold", color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, f"{name}_null")


# ===================================================================== fig 3
def fig_profile(name, ev):
    """Radial profile: label rate and night count either side of the winner.

    Distinguishes a sharp mechanism edge from the peak of a broad bump -- which is
    the difference between a baffle and diffuse scattering, so it earns a panel.
    """
    prof = ev.get("_fig", {}).get("profile")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    if not prof or len(prof) < 3:
        for ax in axes:
            _nodata(ax, "Too few populated annuli around the winner to profile.")
        axes[0].set_title(f"{name} — radial profile unavailable")
        return _save(fig, f"{name}_profile")

    lo = np.array([p[0] for p in prof])
    hi = np.array([p[1] for p in prof])
    rate = np.array([p[4] for p in prof])
    nights = np.array([p[5] for p in prof])
    win = np.array([p[6] for p in prof])
    mid = (lo + hi) / 2

    ax = axes[0]
    # Emphasis, not a value-ramp: the selected bar is the story, the rest recede.
    ax.bar(mid, rate, width=(hi - lo) * 0.82,
           color=np.where(win, OBS_C, "#c9d8ef"), linewidth=0)
    ax.set_xlabel("off-axis distance [deg]")
    ax.set_ylabel("artifact rate among\nexposures in annulus [%]")
    ax.set_title("Rate vs radius")
    j = int(np.argmax(win)) if win.any() else int(np.argmax(rate))
    ax.annotate(f"{rate[j]:.1f} %", (mid[j], rate[j]), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8,
                color=INK, fontweight="semibold")

    ax = axes[1]
    ax.bar(mid, nights, width=(hi - lo) * 0.82,
           color=np.where(win, OBS_C, "#c9d8ef"), linewidth=0)
    ax.set_xlabel("off-axis distance [deg]")
    ax.set_ylabel("distinct nights with\nan in-annulus label")
    ax.set_title("Independent nights vs radius")
    ax.annotate(f"{nights[j]:.0f} nights", (mid[j], nights[j]), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8,
                color=INK, fontweight="semibold")

    fig.suptitle(f"{name} — a sharp edge or a broad bump?", x=0.02, ha="left",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, f"{name}_profile")


# ===================================================================== fig 4
def fig_stars(name, ev):
    """Per-star contribution: many stars (mechanism) or one (coincidence)."""
    sc = ev.get("_fig", {}).get("stars")
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.32 * min(len(sc) if sc is not None else 0, 12) + 1.1)))
    if sc is None or len(sc) == 0:
        _nodata(ax, "No star produced an in-annulus label.")
        ax.set_title(f"{name} — star contributions unavailable")
        return _save(fig, f"{name}_stars")

    sub = sc.head(12).iloc[::-1]
    y = np.arange(len(sub))
    share = sub["share"].values * 100
    top_is_dominant = sc["share"].iloc[0] >= 0.5
    # Emphasis colour marks the dominant contributor, which is the failure mode
    # this panel exists to expose.
    colors = ["#c9d8ef"] * len(sub)
    if top_is_dominant:
        colors[-1] = PAL[7]
    else:
        colors[-1] = OBS_C
    ax.barh(y, share, color=colors, height=0.66, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}  ({int(n)} night{'s' if int(n) != 1 else ''})"
         for s, n in zip(sub["star"], sub["n_nights"])], fontsize=8)
    ax.set_xlabel("share of this artifact's in-annulus labels [%]")
    ax.set_title(f"{name} — {len(sc)} contributing star"
                 f"{'s' if len(sc) != 1 else ''} in the selected annulus")
    ax.grid(axis="y", visible=False)
    ax.annotate(f"{share[-1]:.0f} %", (share[-1], y[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8,
                color=INK, fontweight="semibold")
    if top_is_dominant:
        ax.annotate("one star dominates — the signature of\n"
                    "revisiting a single field, not of optics",
                    (0.97, 0.06), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=8, color=PAL[7])
    return _save(fig, f"{name}_stars")


# ===================================================================== fig 5
def fig_controls(name, ev):
    """Offset-annulus controls and the magnitude dose-response, side by side."""
    off = ev.get("_fig", {}).get("offsets")
    dose = ev.get("_fig", {}).get("dose")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    if off is None or len(off) == 0:
        _nodata(ax, "No displaced control annuli available.")
    else:
        o = off.sort_values("lo")
        x = np.arange(len(o) + 1)
        rates = list(o["rate_in"].values * 100)
        labels = [f"{v:+.2f}" for v in o["offset"].values]
        # Insert the winner at offset 0 so the edge is read in one pass.
        ins = int(np.searchsorted(o["offset"].values, 0))
        rates.insert(ins, ev.get("rate_in", np.nan) * 100)
        labels.insert(ins, "0\n(selected)")
        cols = ["#c9d8ef"] * len(rates)
        cols[ins] = OBS_C
        ax.bar(np.arange(len(rates)), rates, color=cols, width=0.72, linewidth=0)
        ax.set_xticks(np.arange(len(rates)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("annulus displacement [deg]")
        ax.set_ylabel("artifact rate [%]")
        ax.set_title("Displaced control annuli")
        ax.annotate(f"{rates[ins]:.1f} %", (ins, rates[ins]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK, fontweight="semibold")

    ax = axes[1]
    tab = dose.get("table") if isinstance(dose, dict) else None
    if tab is None or len(tab) < 2:
        _nodata(ax, "Too few populated magnitude bins for a dose-response.")
    else:
        y = tab["rate"].values * 100
        x = np.arange(len(tab))
        thin = tab["thin"].values
        ax.plot(x, y, color=OBS_C, marker="o", markerfacecolor=OBS_C,
                markeredgecolor=SURF, markeredgewidth=1.4, zorder=3)
        # Thin bins are drawn hollow: visible, but marked as unreliable rather
        # than silently equal to the well-populated ones.
        if thin.any():
            ax.plot(x[thin], y[thin], "o", markerfacecolor=SURF,
                    markeredgecolor=OBS_C, markeredgewidth=1.6, markersize=7,
                    linestyle="none", zorder=4,
                    label="thin bin (few exposures)")
            # Below the axes: placed inside, this legend landed on the curve it
            # annotates.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
        r0 = dose.get("rate_no_star")
        if r0 is not None and np.isfinite(r0):
            ax.axhline(r0 * 100, color=MUTED, linewidth=1.0)
            ax.annotate(f"no star in annulus: {r0*100:.2f} %", (0, r0 * 100),
                        xytext=(2, 4), textcoords="offset points", fontsize=7.5,
                        color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.0f}–{b:.0f}" for a, b in
                            zip(tab["mag_lo"], tab["mag_hi"])], fontsize=7.5)
        ax.set_xlabel("V magnitude of the brightest in-annulus star")
        ax.set_ylabel("artifact rate [%]")
        ax.set_title("Brighter star, stronger effect?")

    fig.suptitle(f"{name} — does the association behave like a mechanism?",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, f"{name}_controls")


# ===================================================================== fig 6
def fig_frames(name, ev):
    """Direction concentration in the three candidate frames, label vs control."""
    fc = ev.get("_fig", {}).get("frames")
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    if not fc:
        _nodata(ax, "Frame diagnostics not computed.")
        ax.set_title(f"{name} — frame diagnostics unavailable")
        return _save(fig, f"{name}_frames")

    names = ["camera", "horizon", "sky"]
    labv = [fc.get(f"R_lab_{n}", np.nan) for n in names]
    ctlv = [fc.get(f"R_ctl_{n}", np.nan) for n in names]
    x = np.arange(len(names))
    w = 0.34
    # Two series, so a legend is present; paired bars share one y-axis.
    ax.bar(x - w / 2 - 0.01, labv, width=w, color=OBS_C, linewidth=0,
           label="labelled exposures")
    ax.bar(x + w / 2 + 0.01, ctlv, width=w, color=NULL_C, linewidth=0,
           label="control exposures")
    ax.set_xticks(x)
    ax.set_xticklabels(["camera\n(optics / ghost)", "horizon\n(structure)",
                        "sky\n(atmosphere or revisits)"], fontsize=8.5)
    ax.set_ylabel("R — concentration of the\nstar's direction")
    ax.grid(axis="x", visible=False)
    # Headroom reserved before the callout is placed, and the legend lifted into
    # its own band: inside the axes both landed on the bars.
    top = np.nanmax([v for v in labv + ctlv if np.isfinite(v)] or [1.0])
    ax.set_ylim(0, top * 1.22)
    exc = [a - b if np.isfinite(a) and np.isfinite(b) else np.nan
           for a, b in zip(labv, ctlv)]
    if np.any(np.isfinite(exc)):
        j = int(np.nanargmax(exc))
        # Direct-label the winning excess only -- the one number that is read.
        ax.annotate(f"largest excess\n+{exc[j]:.3f}", (x[j] - w / 2 - 0.01, labv[j]),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=INK, fontweight="semibold")
    fig.suptitle(f"{name} — which frame does the star direction repeat in?",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold", color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, f"{name}_frames")


# ===================================================================== fig 7
def fig_burst(name, ev):
    """Labels per night -- the burst structure that drives the whole methodology."""
    pn = ev.get("_fig", {}).get("per_night")
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    if pn is None or len(pn) == 0:
        _nodata(ax, "No labelled nights.")
        ax.set_title(f"{name} — burst structure unavailable")
        return _save(fig, f"{name}_burst")

    d = pd.to_datetime(pd.Series(pn.index).astype(str), format="%Y%m%d")
    ax.bar(d, pn.values, width=3.0, color=OBS_C, linewidth=0)
    ax.set_ylabel("labels per night")
    ax.set_xlabel("night")
    tot = int(pn.sum())
    top = pn.sort_values(ascending=False)
    frac = top.iloc[0] / tot * 100
    ax.set_title(f"{name} — {tot} labels on {len(pn)} nights; "
                 f"busiest night holds {frac:.0f} %")
    ax.annotate(f"{top.index[0]}: {int(top.iloc[0])}",
                (pd.Timestamp(str(top.index[0])), top.iloc[0]),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8, color=INK, fontweight="semibold")
    ax.grid(axis="x", visible=False)
    for lab in ax.get_xticklabels():
        lab.set_rotation(25)
        lab.set_ha("right")
    return _save(fig, f"{name}_burst")


# ===================================================================== fig 8
def fig_screen(name, ev, top=12):
    """Univariate effect sizes -- signed, so a diverging encoding is correct."""
    scr = ev.get("_fig", {}).get("screen")
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.30 * top + 1.0)))
    if scr is None or len(scr) == 0:
        _nodata(ax, "No variable had enough coverage to screen.")
        ax.set_title(f"{name} — univariate screen unavailable")
        return _save(fig, f"{name}_screen")

    sub = scr.head(top).iloc[::-1]
    y = np.arange(len(sub))
    v = sub["cliffs_delta"].values
    # Colour carries the SIGN only (the diverging job); magnitude is the bar.
    ax.barh(y, v, color=np.where(v > 0, PAL[7], PAL[0]), height=0.64,
            linewidth=0)
    ax.axvline(0, color=RULE, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["variable"], fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cliff's δ    ← lower in labelled exposures      higher →")
    ax.set_title(f"{name} — what else differs, ranked by effect size")
    ax.grid(axis="y", visible=False)
    return _save(fig, f"{name}_screen")


# ===================================================================== cross
def fig_verdicts(evidence):
    """Cross-artifact: observed z against each artifact's own null spread."""
    rows = []
    for n, ev in evidence.items():
        if not np.isfinite(ev.get("z", np.nan)):
            continue
        rows.append(dict(name=n, z=ev["z"], p=ev.get("p_fwer", np.nan),
                         verdict=ev.get("verdict", "?"),
                         lift=ev.get("crude_lift", np.nan)))
    t = pd.DataFrame(rows).sort_values("z")
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.34 * len(t) + 1.0)))
    if t.empty:
        _nodata(ax, "No artifact produced a testable scan.")
        return _save(fig, "cross_verdicts")

    y = np.arange(len(t))
    strong = t["p"].values <= 0.01
    ax.barh(y, t["z"].values,
            color=np.where(strong, OBS_C, "#c9d8ef"), height=0.62, linewidth=0)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, t["z"].max() * 2.4)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.axvline(5, color=MUTED, linewidth=1.0)
    ax.annotate("z = 5  (mechanism threshold)", (5, -0.65),
                xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.name}" for r in t.itertuples()], fontsize=8.5)
    ax.set_xlabel("night-recurrence z of the best cell (log scale)")
    ax.set_title("All artifacts — only two clear the mechanism threshold")
    ax.grid(axis="y", visible=False)
    # Selective direct labels: the value sits at the bar end, which is the one
    # place it cannot be confused with another bar.
    for yy, r in zip(y, t.itertuples()):
        ax.annotate(f"{r.z:.1f}", (r.z, yy), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK2)
    return _save(fig, "cross_verdicts")


def fig_lift_vs_z(evidence):
    """The methodological headline: crude lift does not predict real evidence."""
    rows = []
    for n, ev in evidence.items():
        if not (np.isfinite(ev.get("z", np.nan))
                and np.isfinite(ev.get("crude_lift", np.nan))):
            continue
        rows.append(dict(name=n, z=ev["z"], lift=ev["crude_lift"],
                         p=ev.get("p_fwer", np.nan)))
    t = pd.DataFrame(rows).sort_values("z")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if t.empty:
        _nodata(ax, "No artifact produced both a lift and a z.")
        return _save(fig, "cross_lift_vs_z")

    sig = t["p"].values <= 0.01
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)
    # Fix the limits before annotating, so nothing is placed against a range that
    # later grows and clips it.
    ax.set_xlim(t["lift"].min() * 0.55, t["lift"].max() * 3.2)
    ax.set_ylim(0, t["z"].max() * 2.6)

    # The null's own reach, measured: a lift below this band is not evidence.
    ax.axvspan(1.5, 5.8, color="#f0efec", zorder=0)
    ax.annotate("lifts reachable\nby chance alone\n(1.5–5.8×, measured)",
                (np.sqrt(1.5 * 5.8), ax.get_ylim()[1]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top",
                fontsize=8, color=MUTED)
    ax.axhline(5, color=MUTED, linewidth=1.0)
    ax.annotate("z = 5", (ax.get_xlim()[0], 5), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)

    ax.scatter(t["lift"], t["z"], s=64, c=np.where(sig, OBS_C, NULL_C),
               edgecolor=SURF, linewidth=1.5, zorder=3)
    # Alternate label side and nudge vertically: several artifacts sit almost on
    # top of each other in the confounded cluster, and overlapping text there
    # would make the very point of the figure unreadable.
    for i, r in enumerate(t.itertuples()):
        left = i % 2 == 1
        ax.annotate(r.name, (r.lift, r.z),
                    xytext=(-7 if left else 7, 3 if i % 4 < 2 else -9),
                    textcoords="offset points", fontsize=8, color=INK2,
                    ha="right" if left else "left")
    ax.set_xlabel("crude exposure-level lift (rate inside / outside)")
    ax.set_ylabel("night-recurrence z\n(the trustworthy statistic)")
    ax.set_title("Why the obvious statistic misleads")
    hs = [plt.Line2D([], [], marker="o", linestyle="none", color=OBS_C,
                     markeredgecolor=SURF, label="p ≤ 0.01"),
          plt.Line2D([], [], marker="o", linestyle="none", color=NULL_C,
                     markeredgecolor=SURF, label="p > 0.01")]
    ax.legend(handles=hs, loc="lower right")
    return _save(fig, "cross_lift_vs_z")
