"""Statistical machinery for the 13-artifact stray-light characterisation.

The single-artifact ("arc") analysis lives in `_build_arc_nb.py`; the helpers it
proved out are lifted here unchanged so the two agree by construction, and the
new star-ensemble machinery is added alongside.

The central methodological point, and the reason this module exists rather than a
loop over `CFG.ARTIFACT`: artifact labels arrive in **bursts**. A night with 100
consecutive flagged exposures carries roughly one night's worth of independent
information, not 100. Every exposure-level statistic (lift, odds ratio, logit
coefficient) therefore overstates its own significance by a large factor, and a
blind scan over annuli finds "effects" of 2-6x purely from that burstiness --
measured, not assumed; see `scan_recurrence` and the null it calibrates against.

So the primary statistic here counts **nights**, not exposures, and its null
preserves the burst structure by circularly shifting each night's label vector.
Exposure-level models are retained but demoted to supporting evidence and flagged
as such wherever they are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Artifact bit map. Verified against the data in `verify_bit_mapping` rather
# than trusted -- `_build_arc_nb.py` §2 does the same, for the same reason.
# --------------------------------------------------------------------------
BITS = {
    "tape": 1, "shoe": 2, "pillow": 4, "sail": 8, "nibble": 16, "banana": 32,
    "ghoul": 64, "moon": 128, "airplane": 256, "arc": 512, "lighton": 1024,
    "smudge": 2048, "horseshoe": 4096,
}

# `sail` never occurs alone: all 704 labels are "tape;sail", so it is perfectly
# collinear with tape and cannot be ranked against it. Handled as a sub-label.
SUBLABEL_OF = {"sail": "tape"}

CAL_IMG_TYPES = r"dark|flat|bias"


# ======================================================================
# angles
# ======================================================================
def wrap180(x):
    """Wrap an angle difference into (-180, 180]."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def wrap360(x):
    return np.asarray(x, dtype=float) % 360.0


def circ_R(deg):
    """Resultant length in [0, 1]: 0 = uniform, 1 = all one direction."""
    r = np.deg2rad(np.asarray(deg, dtype=float))
    r = r[np.isfinite(r)]
    return float(np.hypot(np.cos(r).mean(), np.sin(r).mean())) if r.size > 1 else np.nan


def circ_mean(deg):
    r = np.deg2rad(np.asarray(deg, dtype=float))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def circ_stats(deg):
    from astropy.stats import rayleightest
    r = np.deg2rad(np.asarray(deg, dtype=float))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return dict(R=np.nan, mean=np.nan, n=0, rayleigh_p=np.nan)
    C, S = np.cos(r).mean(), np.sin(r).mean()
    return dict(R=float(np.hypot(C, S)),
                mean=float(np.rad2deg(np.arctan2(S, C)) % 360),
                n=int(r.size),
                rayleigh_p=float(rayleightest(r)) if r.size > 1 else np.nan)


# ======================================================================
# labels and the analysis frame
# ======================================================================
def load_labels(csv_path):
    """On-sky exposures with the artifact bitmask expanded into columns."""
    raw = pd.read_csv(csv_path, low_memory=False)
    is_cal = raw["img_type"].astype(str).str.contains(CAL_IMG_TYPES, case=False,
                                                     na=False, regex=True)
    df = raw[~is_cal].reset_index(drop=True)
    flag = df["flag"].astype(int)
    df["any_flag"] = flag > 0
    for name, bit in BITS.items():
        df[f"is_{name}"] = (flag & bit) > 0
    return df, int(is_cal.sum())


def verify_bit_mapping(df):
    """Derive bit<->name from exclusively-single-label rows and check BITS.

    Trusting a hard-coded constant here would silently mis-analyse an entire
    artifact, so the mapping is re-derived from the data on every run.
    """
    single = df[df["artifact"].notna()]
    found = {}
    for name, grp in single.groupby(single["artifact"].astype(str)):
        if ";" in name:
            continue
        vals = sorted(grp["flag"].astype(int).unique())
        if len(vals) == 1:
            found[name] = int(vals[0])
    bad = {k: (v, BITS.get(k)) for k, v in found.items() if BITS.get(k) != v}
    if bad:
        raise AssertionError(f"bit mapping disagrees with the data: {bad}")
    return found


def inspected_nights(df):
    """Nights carrying at least one flag of any kind.

    A night with zero flags was probably never inspected, so its exposures are
    false controls -- unflagged because nobody looked, not because nothing was
    there.
    """
    per_night = df.groupby("day_obs")["any_flag"].mean()
    return set(per_night.index[per_night > 0])


def artifact_frame(df, artifact, inspected=None):
    """Analysis frame for one artifact: its own label epoch, inspected nights.

    Each artifact started being labelled at a different date, so each gets its
    own epoch. Using a shared epoch would import other artifacts' blind spots.

    The original index is deliberately PRESERVED: callers use it to slice the
    shared annulus-membership matrix, which is indexed by position in `df`.
    """
    col = f"is_{artifact}"
    labelled = df[df[col]]
    if labelled.empty:
        return df.iloc[:0].copy(), None
    epoch = int(labelled["day_obs"].min())
    sel = df["day_obs"] >= epoch
    if inspected is not None:
        sel &= df["day_obs"].isin(inspected)
    return df[sel].copy(), epoch


def burst_structure(frame, artifact):
    """Contiguous runs of flagged exposures -- the real unit of independence."""
    col = f"is_{artifact}"
    lab = frame[frame[col]].sort_values("exposure_id")
    if lab.empty:
        return dict(n=0, nights=0, runs=0, median_run=np.nan, max_run=np.nan,
                    top_nights={}, n_targets=0, n_pointings=0)
    gap = (lab["exposure_id"].astype("int64").diff().ne(1)
           | lab["day_obs"].diff().ne(0))
    runs = lab.assign(run_id=gap.cumsum()).groupby("run_id").size()
    pts = set(zip(lab["ra"].round(0), lab["dec"].round(0)))
    return dict(
        n=int(len(lab)), nights=int(lab["day_obs"].nunique()), runs=int(len(runs)),
        median_run=float(runs.median()), max_run=int(runs.max()),
        top_nights=lab["day_obs"].value_counts().head(5).to_dict(),
        n_targets=int(lab["target_name"].nunique()), n_pointings=len(pts),
    )


# ======================================================================
# univariate screen  (lifted from _build_arc_nb.py §5)
# ======================================================================
def cliffs_delta(a, b):
    """Cliff's delta and the Mann-Whitney p. delta>0 means `a` tends to exceed `b`."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return np.nan, np.nan, a.size, b.size
    uu, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2.0 * uu / (a.size * b.size) - 1.0, p, a.size, b.size


def bh_qvalues(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranks = np.arange(1, pv.size + 1)
    adj = np.empty(pv.size)
    adj[order] = np.minimum.accumulate((pv[order] * pv.size / ranks)[::-1])[::-1]
    q[ok] = np.clip(adj, 0, 1)
    return q


def screen(frame, artifact, cols):
    """Cliff's delta screen with BH-FDR, ranked by |delta|.

    p-values ignore night clustering and so are optimistic; they rank candidates
    and nothing more.
    """
    col = f"is_{artifact}"
    m = frame[col].values
    rows = []
    for c in cols:
        if c not in frame:
            continue
        d, p, na, nb = cliffs_delta(frame.loc[m, c], frame.loc[~m, c])
        rows.append({"variable": c, "cliffs_delta": d, "p": p,
                     "median_lab": frame.loc[m, c].median(),
                     "median_other": frame.loc[~m, c].median(),
                     "n_lab": na, "n_other": nb})
    out = pd.DataFrame(rows).dropna(subset=["cliffs_delta"])
    if out.empty:
        return out
    out["abs_delta"] = out["cliffs_delta"].abs()
    out["q_BH"] = bh_qvalues(out["p"].values)
    return out.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def night_level_variables(frame, artifact, cols, thresh=0.05):
    """Variables with almost no within-night spread -- untestable within a night.

    Reported explicitly rather than silently omitted: for these, between-night
    evidence is all there is, and that is a real limit on what can be concluded.
    """
    out = []
    for c in cols:
        if c not in frame:
            continue
        v = frame[c]
        if not np.issubdtype(v.dtype, np.number) or v.notna().sum() < 50:
            continue
        overall = v.std()
        if not np.isfinite(overall) or overall == 0:
            out.append(c)
            continue
        within = frame.groupby("day_obs")[c].std().median()
        if np.isfinite(within) and within / overall < thresh:
            out.append(c)
    return out


# ======================================================================
# rates
# ======================================================================
def wilson(k, n, z=1.959963984540054):
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_profile(frame, artifact, col, bins, min_n=30):
    lab = f"is_{artifact}"
    g = frame.assign(_b=pd.cut(frame[col], bins)).groupby("_b", observed=True)[lab]
    out = g.agg(n="size", k="sum")
    out = out[out["n"] >= min_n]
    if out.empty:
        return out
    p, lo, hi = wilson(out["k"], out["n"])
    out["rate"], out["lo"], out["hi"] = p * 100, lo * 100, hi * 100
    out["centre"] = [iv.mid for iv in out.index]
    return out


# ======================================================================
# Mantel-Haenszel  (lifted from _build_arc_nb.py §8) -- SUPPORTING ONLY
# ======================================================================
def mantel_haenszel(y, x, strata):
    """MH odds ratio with Robins-Breslow-Greenland variance.

    NOTE this is an *exposure-level* statistic: a 100-exposure burst contributes
    100 counts of evidence, so its CI is far too narrow when labels are bursty.
    Reported as supporting evidence only, always with that caveat attached.
    """
    y = np.asarray(y).astype(int)
    x = np.asarray(x).astype(int)
    s = pd.Series(np.asarray(strata)).astype(str).values
    num = den = 0.0
    P = Q = R = S = PSQR = 0.0
    used = 0
    for lev in pd.unique(s):
        k = s == lev
        a = int(np.sum(k & (x == 1) & (y == 1))); b = int(np.sum(k & (x == 1) & (y == 0)))
        c = int(np.sum(k & (x == 0) & (y == 1))); d = int(np.sum(k & (x == 0) & (y == 0)))
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        used += 1
        num += a * d / n; den += b * c / n
        p_, q_ = (a + d) / n, (b + c) / n
        r_, s_ = a * d / n, b * c / n
        P += p_ * r_; Q += q_ * s_; R += r_; S += s_; PSQR += p_ * s_ + q_ * r_
    if den <= 0 or R <= 0 or S <= 0:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, strata_used=used)
    orr = num / den
    se = np.sqrt(P / (2 * R**2) + PSQR / (2 * R * S) + Q / (2 * S**2))
    return dict(OR=orr, lo=orr * np.exp(-1.96 * se), hi=orr * np.exp(1.96 * se),
                p=2 * stats.norm.sf(abs(np.log(orr) / se)), strata_used=used)


# ======================================================================
# THE PRIMARY TEST: night-recurrence over an annulus grid
# ======================================================================
def nested_annulus_grid(r_min=0.5, r_max=25.0, widths=(0.05, 0.10, 0.25, 0.50)):
    """Annuli at several widths, each stepped by half its width.

    A single width cannot serve: the horseshoe mechanism is 0.03 deg wide and a
    0.25 deg grid dilutes its signal by ~3x, while a 0.05 deg grid is too narrow
    to hold enough exposures for a broad feature like tape's ~2 deg bump.
    """
    grid = []
    for w in widths:
        lo = r_min
        while lo + w <= r_max:
            grid.append((round(lo, 4), round(lo + w, 4)))
            lo += w / 2
    return grid


#: Brightness tiers scanned alongside radius. A single magnitude cut cannot
#: serve: with V<4.5 there are ~900 stars, so at 20 deg off axis almost every
#: exposure has one and the annulus indicator saturates -- 13 % of exposures, and
#: any real contrast is diluted away. Restricting to the few very bright stars
#: restores it (measured for tape: 1.1x at V<4.5 vs 2.9x at V<2). Brightness is
#: also physically the right axis, since scattered flux scales with source flux.
MAG_TIERS = (0.0, 1.0, 2.0, 3.0, 4.5)


def annulus_membership(n_exp, pair_exp, pair_sep, grid, pair_mag=None,
                       mag_tiers=MAG_TIERS):
    """Boolean (n_exp x n_cells): is there a star in this annulus *and* tier?

    Returns (membership, cells) where each cell is (lo, hi, vmax). Without
    `pair_mag` this degrades to the radius-only grid.
    """
    if pair_mag is None:
        mem = np.zeros((n_exp, len(grid)), dtype=bool)
        for b, (lo, hi) in enumerate(grid):
            k = (pair_sep >= lo) & (pair_sep < hi)
            mem[pair_exp[k], b] = True
        return mem, [(lo, hi, np.inf) for lo, hi in grid]

    cells = [(lo, hi, v) for v in mag_tiers for lo, hi in grid]
    mem = np.zeros((n_exp, len(cells)), dtype=bool)
    c = 0
    for v in mag_tiers:
        bright = pair_mag < v
        pe_v, ps_v = pair_exp[bright], pair_sep[bright]
        for lo, hi in grid:
            k = (ps_v >= lo) & (ps_v < hi)
            mem[pe_v[k], c] = True
            c += 1
    return mem, cells


def _night_bounds(night_ids):
    """Start/stop offsets of each night, assuming rows are sorted by night."""
    counts = np.bincount(night_ids)
    return np.r_[0, np.cumsum(counts)]


def _recurrence(lab_bool, mem, night_ids, n_nights):
    """Per bin: how many distinct nights have at least one in-annulus label.

    This is the statistic that makes the analysis honest. One burst of 100
    flagged exposures on a single night contributes exactly 1, so a night-long
    coincidence cannot masquerade as a hundred independent detections.

    Rows are required to be in night order (the caller sorts by day_obs, which the
    circular-shift null also depends on), so the per-night OR is a `reduceat` over
    contiguous segments -- about 12x faster than scattering into a night x bin
    accumulator, which matters because this runs 2000 times per artifact.
    """
    out = np.zeros(mem.shape[1])
    idx = np.where(lab_bool)[0]
    if idx.size == 0:
        return out
    g = night_ids[idx]
    starts = np.r_[0, np.where(np.diff(g) != 0)[0] + 1]
    return np.logical_or.reduceat(mem[idx], starts, axis=0).sum(0).astype(float)


def cross_night_shuffle(lab, night_bounds, rng):
    """Move each night's whole label pattern onto a different night.

    A stricter null than the within-night shift. The within-night null asks "were
    the labels placed specially *within* this night?", which leaves untouched the
    fact that the labelled nights are themselves a biased sample of pointings. A
    night spent on one field has all its exposures near the same stars, so every
    within-night shift lands in the same annuli and the null cannot see that
    confound at all.

    Transplanting whole nights breaks it: the burst structure survives, but the
    labels now sit on a different night's pointings. Measured on this dataset the
    two nulls disagree materially -- see `cross_night_calibration`.
    """
    out = np.zeros_like(lab)
    counts = np.diff(night_bounds)
    order = rng.permutation(len(counts))
    for dst, src in enumerate(order):
        s0, e0 = night_bounds[dst], night_bounds[dst + 1]
        s1, e1 = night_bounds[src], night_bounds[src + 1]
        block = lab[s1:e1]
        n = e0 - s0
        if len(block) >= n:
            out[s0:e0] = block[:n]
        else:
            out[s0:s0 + len(block)] = block
    return out


def cross_night_calibration(lab, mem, night_ids, mu, sd, valid, n_shuffle=200,
                            seed=4242):
    """Max standardised recurrence reachable by transplanting nights wholesale.

    Reuses the main scan's per-cell mu/sd, so this costs one recurrence pass per
    shuffle rather than a nested permutation -- cheap enough to run for every
    artifact. Returns the distribution of the scan maximum and the resulting
    p-value, which is the conservative one to quote.
    """
    bounds = _night_bounds(night_ids)
    n_nights = int(night_ids.max()) + 1 if night_ids.size else 0
    rng = np.random.default_rng(seed)
    maxz = np.full(n_shuffle, np.nan)
    for i in range(n_shuffle):
        fake = cross_night_shuffle(lab, bounds, rng)
        r = _recurrence(fake, mem, night_ids, n_nights)
        z = np.full(mem.shape[1], np.nan)
        z[valid] = (r[valid] - mu[valid]) / sd[valid]
        if np.any(np.isfinite(z)):
            maxz[i] = np.nanmax(z)
    return maxz


def circular_shift_null(lab, night_bounds, rng):
    """Roll each night's label vector by a random offset.

    Preserves run lengths *and* the spacing between runs, so the null sample is
    as bursty as the real one. A plain within-night permutation breaks runs
    apart, which makes any bursty artifact look significant in every annulus at
    once -- the failure mode this null exists to avoid.
    """
    out = lab.copy()
    for g in range(len(night_bounds) - 1):
        s, e = night_bounds[g], night_bounds[g + 1]
        if e - s > 1:
            out[s:e] = np.roll(lab[s:e], rng.integers(0, e - s))
    return out


def scan_recurrence(lab, mem, night_ids, n_perm=2000, seed=42, return_null=False,
                    min_nights=3, min_exposures=30, n_cross=200):
    """Scan all annuli for excess night-recurrence; correct for the scan itself.

    Per cell the observed recurrence is standardised against the circular-shift
    null, then the maximum z over cells is calibrated against the *per-replicate
    maximum*. That is what makes "the best of thousands of cells" cost what it
    should: the p-value is family-wise over the whole scan, not per cell.

    Cells with very little support are excluded before the maximum is taken.
    Without that floor the scan is won by a cell holding a handful of exposures,
    where the null recurrence is ~0 and its standard deviation is correspondingly
    tiny, so a couple of coincident nights produce an enormous z that means
    nothing. Requiring a few nights and a few tens of exposures keeps the
    standardisation meaningful; the excluded cells are still reported in `z_all`.

    TWO nulls are used and the conservative p is returned, because the obvious one
    is not sufficient -- verified empirically, not assumed. See
    `cross_night_shuffle`; `_negative_control.py` is the check that motivated it.
    """
    lab = np.asarray(lab, dtype=bool)
    n_nights = int(night_ids.max()) + 1 if night_ids.size else 0
    bounds = _night_bounds(night_ids)
    obs = _recurrence(lab, mem, night_ids, n_nights)
    support_exp = mem.sum(0)
    support_ok = (obs >= min_nights) & (support_exp >= min_exposures)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, mem.shape[1]))
    for p in range(n_perm):
        null[p] = _recurrence(circular_shift_null(lab, bounds, rng),
                              mem, night_ids, n_nights)

    mu = null.mean(0)
    sd = null.std(0, ddof=1)
    # A cell the null never populates has sd 0; it cannot be standardised, and
    # calling that infinite z would invent a detection out of one stray night.
    valid = (sd > 1e-9) & support_ok
    z = np.full(mem.shape[1], np.nan)
    z[valid] = (obs[valid] - mu[valid]) / sd[valid]
    # The scan maximum and its null must be taken over the SAME set of cells,
    # otherwise the observed maximum is compared against a differently-sized
    # search and the correction is wrong.
    zn = np.full_like(null, np.nan)
    zn[:, valid] = (null[:, valid] - mu[valid]) / sd[valid]

    if not np.any(np.isfinite(z)):
        res = dict(best_bin=None, z=np.nan, p_fwer=np.nan, p_shift=np.nan,
                   p_cross=np.nan, obs=obs, mu=mu, sd=sd, z_all=z,
                   n_perm=n_perm, n_cross=0, n_cells_tested=0,
                   p_floor=1.0 / (n_perm + 1))
        return (res, null) if return_null else res

    T = np.nanmax(z)
    Tn = np.nanmax(zn, axis=1)
    p_shift = (np.sum(Tn >= T) + 1) / (n_perm + 1)
    best = int(np.nanargmax(z))

    # The within-night shift null is not enough on its own. It cannot see that
    # the labelled nights are a biased sample of pointings, so on this dataset it
    # assigns p ~ 0.01 to labels that have been deliberately decoupled from the
    # sky (measured: arc and ghoul both reach 0.007-0.01 under a cross-night
    # shuffle). Calibrating additionally against whole-night transplants fixes
    # that, and the conservative maximum of the two is what gets quoted.
    p_cross = np.nan
    maxz = np.array([])
    if n_cross > 0:
        maxz = cross_night_calibration(lab, mem, night_ids, mu, sd, valid,
                                       n_shuffle=n_cross, seed=seed + 9901)
        ok = np.isfinite(maxz)
        if ok.sum() >= 20:
            p_cross = (np.sum(maxz[ok] >= T) + 1) / (ok.sum() + 1)

    p_fwer = np.nanmax([p_shift, p_cross]) if np.isfinite(p_cross) else p_shift
    res = dict(best_bin=best, z=float(T), p_fwer=float(p_fwer),
               p_shift=float(p_shift),
               p_cross=float(p_cross) if np.isfinite(p_cross) else np.nan,
               obs_nights=float(obs[best]), null_mean=float(mu[best]),
               null_sd=float(sd[best]), obs=obs, mu=mu, sd=sd, z_all=z,
               n_perm=n_perm, n_cross=n_cross, n_cells_tested=int(valid.sum()),
               p_floor=1.0 / (n_perm + 1),
               # Null distributions of the scan maximum, kept so the report can
               # show the observed z against the null it was calibrated on rather
               # than just quoting a p-value.
               null_max_shift=Tn, null_max_cross=maxz, support_ok=valid)
    return (res, null) if return_null else res


# ======================================================================
# supporting controls for a surviving annulus
# ======================================================================
def annulus_exposure_stats(lab, in_ann, night_ids):
    """Crude exposure-level lift -- reported only next to its null calibration."""
    lab = np.asarray(lab, bool); in_ann = np.asarray(in_ann, bool)
    n_in, n_out = int(in_ann.sum()), int((~in_ann).sum())
    k_in, k_out = int((lab & in_ann).sum()), int((lab & ~in_ann).sum())
    r_in = k_in / n_in if n_in else np.nan
    r_out = k_out / n_out if n_out else np.nan
    return dict(n_in=n_in, n_out=n_out, k_in=k_in, k_out=k_out,
                rate_in=r_in, rate_out=r_out,
                lift=r_in / r_out if r_out and r_out > 0 else np.nan,
                frac_labels_inside=k_in / max(k_in + k_out, 1),
                nights_in=int(len(np.unique(night_ids[lab & in_ann]))))


def star_contributions(lab, pair_exp, pair_sep, pair_star, star_names, lo, hi,
                       night_of_exp):
    """Which stars and nights actually produce the in-annulus labels.

    A mechanism should show up on many stars; a scheduling coincidence shows up
    on one star over a few consecutive nights. This is the cheapest and most
    decisive way to tell those apart.
    """
    k = (pair_sep >= lo) & (pair_sep < hi) & lab[pair_exp]
    if not np.any(k):
        return pd.DataFrame(columns=["star", "n_pairs", "n_nights", "share"])
    names = np.asarray(star_names)[pair_star[k]]
    nights = night_of_exp[pair_exp[k]]
    tab = (pd.DataFrame({"star": names, "night": nights})
           .groupby("star").agg(n_pairs=("night", "size"),
                                n_nights=("night", "nunique"))
           .sort_values("n_pairs", ascending=False))
    tab["share"] = tab["n_pairs"] / tab["n_pairs"].sum()
    return tab.reset_index()


def leave_one_star_out(lab, mem_bin, pair_exp, pair_sep, pair_star, lo, hi,
                       night_ids, drop_star, n_perm=400, seed=42):
    """Re-test the winning annulus with its top-contributing star removed.

    If the signal evaporates, it was one coincidence wearing a mechanism's
    clothes.
    """
    keep = ~((pair_sep >= lo) & (pair_sep < hi) & (pair_star == drop_star))
    mem = np.zeros((len(lab), 1), dtype=bool)
    k = keep & (pair_sep >= lo) & (pair_sep < hi)
    mem[pair_exp[k], 0] = True
    return scan_recurrence(lab, mem, night_ids, n_perm=n_perm, seed=seed)


def offset_annulus_controls(lab, pair_exp, pair_sep, night_ids, lo, hi,
                            offsets=(-0.15, -0.10, -0.05, 0.05, 0.10, 0.15)):
    """Same-width annuli displaced slightly -- a mechanism has a sharp edge.

    If the neighbours light up as brightly as the winner, the "annulus" is just a
    broad excess and no edge has been located.
    """
    w = hi - lo
    rows = []
    for off in offsets:
        a, b = lo + off, hi + off
        if a <= 0:
            continue
        in_ann = np.zeros(len(lab), dtype=bool)
        k = (pair_sep >= a) & (pair_sep < b)
        in_ann[pair_exp[k]] = True
        st = annulus_exposure_stats(lab, in_ann, night_ids)
        rows.append({"lo": round(a, 4), "hi": round(b, 4), "offset": off,
                     "width": round(w, 4), **{kk: st[kk] for kk in
                     ("n_in", "k_in", "rate_in", "lift", "nights_in")}})
    return pd.DataFrame(rows)


def magnitude_dose_response(lab, pair_exp, pair_sep, pair_mag, lo, hi,
                            edges=(-2.0, 0.0, 1.0, 2.0, 3.0, 4.5), min_n=20):
    """Label rate vs the brightness of the in-annulus star, plus a trend test.

    A real optical mechanism should get worse with brighter stars and switch off
    past its own magnitude limit. Cochran-Armitage trend on the ordered bins;
    exposure-level, so it supports rather than decides.

    Strict bin-to-bin monotonicity is deliberately NOT the criterion: the
    brightest bins hold only a handful of exposures (there are four stars with
    V<0 in the whole sky), so one label more or less flips the ordering by tens
    of percent. What a dose-response actually asserts is that the trend is
    *significantly* downward and that bright stars carry a much higher rate than
    faint ones -- both robust to that noise, unlike the raw ordering.
    """
    lab = np.asarray(lab, bool)
    k = (pair_sep >= lo) & (pair_sep < hi)
    pe, pm = pair_exp[k], pair_mag[k]
    rows = []
    # brightest in-annulus star per exposure decides its bin
    brightest = pd.Series(pm).groupby(pe).min()
    assign = np.full(len(lab), np.nan)
    assign[brightest.index.values] = brightest.values
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        sel = (assign >= a) & (assign < b)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({"mag_lo": a, "mag_hi": b, "n": n, "k": int(lab[sel].sum()),
                     "rate": float(lab[sel].mean()), "score": i,
                     "thin": n < min_n})
    none_sel = ~np.isfinite(assign)
    tab = pd.DataFrame(rows)
    out = {"table": tab,
           "rate_no_star": float(lab[none_sel].mean()) if none_sel.any() else np.nan,
           "n_no_star": int(none_sel.sum())}
    if len(tab) >= 3:
        n_i = tab["n"].values.astype(float); k_i = tab["k"].values.astype(float)
        s_i = tab["score"].values.astype(float)
        N, K = n_i.sum(), k_i.sum()
        pbar = K / N
        num = np.sum(s_i * (k_i - n_i * pbar))
        den = pbar * (1 - pbar) * (np.sum(n_i * s_i**2) - np.sum(n_i * s_i)**2 / N)
        if den > 0:
            zct = num / np.sqrt(den)
            out["trend_z"] = float(zct)
            out["trend_p"] = float(2 * stats.norm.sf(abs(zct)))
        out["strict_monotone"] = bool(np.all(np.diff(tab["rate"].values) <= 1e-12))
        # Robust form: pool bins with enough exposures, compare bright vs faint
        # halves, and require the trend to point downward.
        soli = tab[~tab["thin"]]
        if len(soli) >= 2:
            mid = len(soli) // 2
            bright = soli.iloc[:mid] if mid else soli.iloc[:1]
            faint = soli.iloc[mid:] if mid else soli.iloc[1:]
            rb = bright["k"].sum() / max(bright["n"].sum(), 1)
            rf = faint["k"].sum() / max(faint["n"].sum(), 1)
            out["rate_bright"] = float(rb)
            out["rate_faint"] = float(rf)
            out["bright_faint_ratio"] = float(rb / rf) if rf > 0 else np.inf
            out["dose_response"] = bool(
                out.get("trend_z", 0) < 0
                and np.isfinite(out.get("trend_p", np.nan))
                and out["trend_p"] <= 0.05
                and rb > 2 * rf)
    return out


# ======================================================================
# frame diagnostics
# ======================================================================
def frame_concentration(lab, pair_exp, pair_sep, pa_by_frame, lo, hi,
                        max_control=40000, seed=42):
    """R of the star direction for labels vs controls, in each frame.

    All three frames must be *position angles* (clocking angles measured East of
    North). An azimuth *offset* will not do: a star 2 deg off axis necessarily has
    an azimuth within a few degrees of the boresight, so its azimuth offset is
    pinned near zero by the annulus itself and its concentration measures the
    radius rather than any direction.

    Compare each R against its own control, not against the other frames: the
    excess over control is the interpretable quantity. Descriptive only -- with
    ~1 star per exposure the three frames differ by fixed offsets, so this cannot
    identify a frame on its own; see the module docstring and §6.3 of the arc
    analysis.
    """
    lab = np.asarray(lab, bool)
    ann = (pair_sep >= lo) & (pair_sep < hi)
    kl = ann & lab[pair_exp]
    kc = ann & ~lab[pair_exp]
    ic = np.where(kc)[0]
    if len(ic) > max_control:
        ic = np.random.default_rng(seed).choice(ic, max_control, replace=False)
    out = {}
    for fname, pa in pa_by_frame.items():
        out[f"R_lab_{fname}"] = circ_R(pa[kl])
        out[f"R_ctl_{fname}"] = circ_R(pa[ic])
        out[f"mean_lab_{fname}"] = circ_mean(pa[kl])
    out["n_lab_pairs"] = int(kl.sum())
    out["n_ctl_pairs"] = int(len(ic))
    return out


def annulus_multiplicity(pair_exp, pair_sep, lo, hi, n_exp):
    """How many in-annulus stars an exposure typically has.

    Decides whether the star ensemble can break the frame degeneracy: with one
    star per exposure it cannot, and co-radial stars rotate together so they add
    nothing either.
    """
    k = (pair_sep >= lo) & (pair_sep < hi)
    if not np.any(k):
        return dict(mean_mult=np.nan, frac_single=np.nan, n_exp_with=0)
    c = pd.Series(pair_exp[k]).value_counts()
    return dict(mean_mult=float(c.mean()), frac_single=float((c == 1).mean()),
                n_exp_with=int(len(c)))


# ======================================================================
# verdicts
# ======================================================================
VERDICT_ORDER = ["MECHANISM IDENTIFIED", "STRONG ASSOCIATION", "SUGGESTIVE",
                 "CONFOUNDED", "NULL", "UNDERPOWERED", "NOT ANALYSED"]


def assign_verdict(ev):
    """Map the assembled evidence to one ordered verdict, first match wins.

    Returns (verdict, reasons_met, reasons_failed) so the text file can show why
    a verdict was reached rather than asserting it.
    """
    def chk(cond, label):
        return (label, bool(cond))

    z = ev.get("z", np.nan)
    p = ev.get("p_fwer", np.nan)
    q = ev.get("q_bh", np.nan)
    nights = ev.get("nights_in", 0)
    stars = ev.get("n_stars", 0)
    loo_p = ev.get("loo_p", np.nan)
    width = ev.get("width", np.inf)

    if ev.get("tier") == "D" or ev.get("no_test"):
        return "UNDERPOWERED", [], [ev.get("no_test_reason", "insufficient data")]

    mech = [
        chk(np.isfinite(p) and p <= 0.01, "p_FWER <= 0.01"),
        chk(np.isfinite(q) and q <= 0.05, "BH q <= 0.05 across artifacts"),
        chk(np.isfinite(z) and z >= 5, "recurrence z >= 5"),
        chk(nights >= 8, ">= 8 nights with in-annulus labels"),
        chk(stars >= 5, ">= 5 distinct stars"),
        chk(ev.get("k_in_annulus", 0) >= 20,
            ">= 20 labelled exposures in the annulus"),
        chk(ev.get("top_star_share", 0.0) < 0.5,
            "no single star supplies >50 % of the in-annulus labels"),
        chk(np.isfinite(loo_p) and loo_p <= 0.05, "leave-one-star-out p <= 0.05"),
        chk(ev.get("dose_response", False),
            "magnitude dose-response (downward trend p <= 0.05 and bright stars "
            ">2x the faint rate)"),
        chk(ev.get("offset_ratio", np.inf) < 0.20,
            "offset controls < 20 % of the excess"),
        chk(width <= 0.5, "annulus width <= 0.5 deg"),
    ]
    if all(v for _, v in mech):
        return "MECHANISM IDENTIFIED", [k for k, _ in mech], []

    strong = [
        chk(np.isfinite(p) and p <= 0.05, "p_FWER <= 0.05"),
        chk(np.isfinite(q) and q <= 0.10, "BH q <= 0.10"),
        chk(np.isfinite(z) and z >= 3, "recurrence z >= 3"),
        chk(nights >= 8, ">= 8 nights with in-annulus labels"),
        chk(ev.get("k_in_annulus", 0) >= 20,
            ">= 20 labelled exposures in the annulus"),
        chk(stars >= 3, ">= 3 stars"),
        chk(ev.get("top_star_share", 0.0) < 0.5,
            "no single star supplies >50 % of the in-annulus labels"),
        chk(np.isfinite(loo_p) and loo_p <= 0.10, "leave-one-star-out p <= 0.10"),
    ]
    if all(v for _, v in strong):
        return ("STRONG ASSOCIATION", [k for k, _ in strong],
                [k for k, v in mech if not v])

    # Thresholds here are set by the negative control, not by convention. With
    # labels deliberately decoupled from the sky, individual shuffles still reach
    # p = 0.01-0.07 (see `_negative_control.py`), so a p below 0.20 -- or even
    # below 0.05 on its own -- is not evidence. Hence p <= 0.05 *plus* real
    # support: the scan's own floor admits a cell at 3 nights, so requiring only
    # 3 nights here would rubber-stamp every cell that cleared the floor.
    k_in = ev.get("k_in_annulus", 0)
    share = ev.get("top_star_share", 0.0)
    sugg = [
        chk(np.isfinite(p) and p <= 0.05, "p_FWER <= 0.05"),
        chk(np.isfinite(z) and z >= 3, "recurrence z >= 3"),
        chk(nights >= 5, ">= 5 nights with in-annulus labels"),
        chk(k_in >= 10, ">= 10 labelled exposures in the annulus"),
        chk(stars >= 2, ">= 2 stars"),
        chk(share < 0.7, "no single star supplies >70 % of the in-annulus labels"),
    ]
    if all(v for _, v in sugg):
        return ("SUGGESTIVE", [k for k, _ in sugg],
                [k for k, v in strong if not v])

    # A large crude lift that fails the above because one star or one week
    # supplies it is not merely "not significant" -- it is positively explained by
    # the confound, and saying so is more informative than "null".
    if (ev.get("crude_lift", 0) >= 3 and
            (share >= 0.5 or nights < 5 or k_in < 10)):
        return ("CONFOUNDED", [], [k for k, v in sugg if not v])

    # Adequately powered and nothing survived -> distinguish "confounded"
    # (a big crude lift that the null erases) from a plain null.
    powered = ev.get("n_labels", 0) >= 100 and ev.get("frame_nights", 0) >= 8
    if powered:
        if ev.get("crude_lift", 0) >= 3 and (not np.isfinite(p) or p > 0.20):
            return ("CONFOUNDED", [], [k for k, v in sugg if not v])
        return "NULL", [], [k for k, v in sugg if not v]
    return "UNDERPOWERED", [], [k for k, v in sugg if not v]
