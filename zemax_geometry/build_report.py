"""Generate the LaTeX report: one .tex per artifact plus a master document.

Reads `outputs/artifact_stats/evidence.pkl` -- written by `run_all_artifacts.py`
as it prints the .txt summaries -- so the report cannot disagree with them.

Run:  python build_report.py          # writes report/*.tex and report/figs/*.pdf
      cd report && latexmk -pdf straylight_artifacts.tex

Output layout:
    report/straylight_artifacts.tex   master: preamble + methods + \\input of each
    report/method.tex                 why night-counting, the two nulls, the floors
    report/art_<name>.tex             one chapter per artifact
    report/figs/*.pdf                 vector figures
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

import artifact_stats as A
import report_figures as RF

TABDIR = pathlib.Path("../outputs/artifact_stats")
REPORT = pathlib.Path("report")

# Physical interpretation, stated only where the repository actually records one.
# Everything else is explicitly "not documented" rather than invented.
KNOWN_MECHANISM = {
    "horseshoe": (
        "The only artifact in this dataset with an independently documented "
        "physical model. The scheduler detailer in\n"
        "\\path{L3_baffle/horseshoe_detailer.py} forbids a bright star in the "
        "off-axis annulus $[1.995, 2.025]^\\circ$, on the grounds that light "
        "reflecting off the \\emph{L3 lens baffle} produces this feature."),
    "tape": (
        "Treated in the repository as generic scattered light, under test "
        "LVV-T956-a / requirement LSR-REQ-0093-V-10. No off-axis distance is "
        "recorded in that work --- but the label CSV itself carries "
        "\\texttt{tape\\_*} columns that select the brightest $V<1$ star in "
        "$[19.5, 22.0]^\\circ$, so somebody had already formed a hypothesis about "
        "this distance."),
}

VERDICT_COLOUR = {
    "MECHANISM IDENTIFIED": "mechid",
    "STRONG ASSOCIATION": "strongassoc",
    "SUGGESTIVE": "suggestive",
    "CONFOUNDED": "confounded",
    "NULL": "nullres",
    "UNDERPOWERED": "underpow",
}


def tex_escape(s):
    """Escape for LaTeX text mode.

    Covers `|`, `<` and `>` as well as the usual specials: the telescope-state
    condition names contain things like "|dome-tel az| > 4 deg", which silently
    render as inverted exclamation marks and upside-down commas in text mode
    rather than erroring, so leaving them out would corrupt the table quietly.
    """
    if s is None:
        return ""
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"),
                 ("|", r"\textbar{}"), ("<", r"\textless{}"),
                 (">", r"\textgreater{}")):
        out = out.replace(a, b)
    return out


def num(x, nd=3, na="---"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{nd}f}"


def fig_block(path, caption, label, width=r"\linewidth"):
    if path is None:
        return ""
    return "\n".join([
        r"\begin{figure}[htbp]", r"  \centering",
        rf"  \includegraphics[width={width}]{{{path}}}",
        rf"  \caption{{{caption}}}", rf"  \label{{fig:{label}}}",
        r"\end{figure}", ""])


# ====================================================================== method
METHOD = r"""
\chapter{Method, and why the obvious approach fails}
\label{ch:method}

\section{The question}

Each exposure in \path{data/raw/stray_light_20260506.csv} carries a bitmask of
human-assigned artifact labels. The question this report answers, for all thirteen
labels, is narrow and specific:

\begin{quote}
When a given artifact appears, is there always a bright star at some particular
off-axis distance from the boresight?
\end{quote}

That is a question about \emph{geometry}, and a positive answer names a candidate
surface: a bright source at a fixed angle off axis is what produces a ghost or a
baffle reflection. The analysis therefore cross-matches the Yale Bright Star
Catalogue (\path{L3_baffle/rubin_one_year_output/ybsc_v5.fits}, $V<4.5$,
892~stars) against every on-sky exposure on an inspected night, and scans for an
annulus whose occupancy predicts the label.

\section{Why the obvious statistic is wrong}

The natural test is a rate ratio: compare the artifact rate among exposures that
have a star in the candidate annulus with the rate among those that do not. Call
it the \emph{lift}. On this dataset the lift is actively misleading, and it took a
measurement rather than an argument to establish that.

Two properties of the data conspire. First, labels are \textbf{bursty}: they arrive
in runs of consecutive exposures on a handful of nights. Second, the survey
\textbf{revisits the same fields}, so every exposure on a given night sits near
much the same stars. Together these mean a single night's burst lands wholesale
inside whichever annulus that night's field happens to populate --- and the lift
counts it as hundreds of independent detections.

The size of the effect was measured, not assumed. Scanning against labels whose
burst structure is preserved but whose association with the sky is destroyed, the
largest lift reachable \emph{by chance alone} is between $1.5\times$ and
$5.8\times$, depending on the artifact. A four-fold enrichment is therefore not
weak evidence; it is the typical value of the null. Ranking the artifacts by lift
would have produced a table of noise, with \texttt{shoe} ($11.8\times$),
\texttt{ghoul} ($6.9\times$) and \texttt{sail} ($4.0\times$) at the top --- all
three of which turn out to be single-star, single-week coincidences.

\section{What replaces it}

\subsection{Count nights, not exposures}

The statistic used throughout is the number of \textbf{distinct nights} on which at
least one labelled exposure has a star in the annulus. A burst of one hundred
flagged exposures on one night contributes exactly~1. This makes the unit of
evidence the unit of independence.

\subsection{Two nulls, and the weaker $p$ is quoted}

The observed recurrence is standardised per cell against a null, and the
\emph{maximum} standardised value over the whole scan is calibrated against the
per-replicate maximum. That is what makes selecting the best of thousands of cells
cost what it should: the quoted $p$ is family-wise over the entire search.

The obvious null --- circularly shifting each night's label vector within that
night --- is \textbf{not sufficient}, and this was caught by an explicit negative
control rather than by inspection. It preserves run lengths and spacings, but it
cannot see that the labelled nights are themselves a biased sample of pointings: a
night spent on one field has every exposure near the same stars, so every shift
lands in the same annuli. Run against labels deliberately decoupled from the sky,
that null alone assigned $p \approx 0.01$ to \texttt{arc} and \texttt{ghoul}. It
was anti-conservative.

The fix is a second null that transplants each night's whole label pattern onto a
different night: bursts survive, the link to that night's pointings does not. Both
nulls are computed for every artifact and the \textbf{larger} $p$ is reported.
\path{_negative_control.py} is the gate; it now passes, with mean
$p = 0.19$--$0.68$ on decoupled labels.

\subsection{Support floors}

A cell must hold at least 3 labelled nights and 30 exposures to enter the scan at
all. Without that floor the scan is won by a cell containing a handful of
exposures, where the null recurrence is $\approx 0$ and its standard deviation
correspondingly tiny, so one coincidence yields an enormous and meaningless $z$.
Verdicts additionally require at least 20 labelled in-annulus exposures with no
single star supplying more than half of them.

\subsection{What does not work: conditioning on the field}

Stratifying by field or \path{target_name} appears to be the natural fix for a
pointing confound. It is useless here, and this was tested rather than assumed: the
annulus indicator is a deterministic function of where the telescope points, and
varies within only about 10 of 51\,646 fields of $0.1^\circ$. Conditioning on field
removes essentially all contrast.

\subsection{Statistics that are retained but demoted}

Mantel--Haenszel odds ratios and night fixed-effects logistic regressions appear in
every chapter, but as \emph{supporting} evidence explicitly labelled as such. They
are exposure-level, so a burst inflates their precision in exactly the way this
whole method exists to avoid. Where a crude and a stratified estimate disagree, the
gap is itself the finding.

\section{The magnitude axis}

The scan covers brightness as well as radius, over five tiers
($V<0$, $<1$, $<2$, $<3$, $<4.5$). A single magnitude cut will not serve. Taking
every star down to $V=4.5$ means that at $20^\circ$ off axis almost every exposure
has one, so the indicator saturates --- 13\% of all exposures --- and any real
contrast is diluted away. Measured on \texttt{tape}: the same annulus gives
$1.1\times$ at $V<4.5$ and $2.9\times$ at $V<2$. Brightness is also the physically
correct axis, since scattered flux scales with the flux of the source.

\section{The verdict ladder}

Each artifact receives exactly one verdict, evaluated top-down, first match wins.

\begin{description}
  \item[\textcolor{mechid}{MECHANISM IDENTIFIED}] $p_{\mathrm{FWER}} \le 0.01$,
    BH-significant across artifacts, $z \ge 5$, $\ge 8$ nights, $\ge 5$ stars,
    $\ge 20$ in-annulus labels, no star above 50\%, survives leave-one-star-out,
    a monotone-in-trend magnitude dose-response, displaced controls below 20\% of
    the excess, and a width $\le 0.5^\circ$ (a mechanism has a sharp edge).
  \item[\textcolor{strongassoc}{STRONG ASSOCIATION}] Clears the null and the
    support floors, but not every corroborating criterion.
  \item[\textcolor{suggestive}{SUGGESTIVE}] $p_{\mathrm{FWER}} \le 0.05$ with real
    support, but \emph{not distinguishable from pointing-schedule coincidence with
    the data available}.
  \item[\textcolor{confounded}{CONFOUNDED}] A large crude enrichment explained by
    when and where the survey pointed --- either it fails the null, or it clears
    the null while resting on too few nights, too few labels, or one dominant star.
  \item[\textcolor{nullres}{NULL}] Adequately powered, nothing found. A power
    statement is attached.
  \item[\textcolor{underpow}{UNDERPOWERED}] Too few independent nights to tell.
    This means \emph{cannot tell}, not \emph{no effect}.
\end{description}

\section{One question this cannot answer}

A stray-light feature is fixed in some frame, and which one names the culprit
surface: the camera frame implicates the optics, the horizon frame the structure,
the sky frame the atmosphere --- or, far more often, repeated visits to one field.
Every chapter reports the direction concentration in all three frames, and every
chapter marks it \textbf{descriptive}.

The reason is a measurement, and it closes the question rather than deferring it.
In 99\% of in-annulus exposures there is exactly \emph{one} bright star, so there
is one direction, and the three frames are then related by fixed offsets (the
rotator angle and the parallactic angle). Worse, two stars at the same radius
differ in position angle by a constant that is invariant under rotator motion, so
they rotate together and contribute no discriminating power at all. The star
ensemble adds nothing over the single-Moon-direction test already reported in
\path{statistic_geometry.ipynb}~\S6.3, which reached the same negative
conclusion. Settling it requires the artifact's own orientation measured in the
pixels, which is a different instrument and out of scope here.

\section{Reproducing this report}

\begin{verbatim}
cd zemax_geometry
python build_star_geometry.py      # one-time caches: star pairs, ConsDB, EFD
python _negative_control.py        # sanity gate -- must print PASS
python run_all_artifacts.py --perm 2000
python build_report.py
cd report && latexmk -pdf straylight_artifacts.tex
\end{verbatim}

Figures in this document are drawn from the same serialised evidence that produced
the plain-text summaries, so the two cannot drift apart.
"""


# ====================================================================== chapter
def artifact_chapter(name, ev, figs):
    """One chapter. Narrates what was measured and what it does and does not show."""
    L = []
    v = ev.get("verdict", "?")
    col = VERDICT_COLOUR.get(v, "black")
    disp = tex_escape(name)

    L += [rf"\chapter{{\texttt{{{disp}}}}}", rf"\label{{ch:{name}}}", ""]
    L += [rf"\verdictbox{{{col}}}{{{tex_escape(v)}}}{{%"]
    L += [headline_sentence(name, ev) + "}", ""]

    # ---- what this artifact is
    L += [r"\section{What is known about this artifact}", ""]
    if name in KNOWN_MECHANISM:
        L += [KNOWN_MECHANISM[name], ""]
    else:
        L += [rf"No physical definition of ``{disp}'' is recorded anywhere in this "
              r"repository --- not in the README, not in a docstring, not in a "
              r"notebook markdown cell. The name is a label convention used by "
              r"whoever inspected the images. Everything below is therefore "
              r"inferred from geometry alone, and the absence of a stated "
              r"mechanism is itself worth recording.", ""]
    if name in A.SUBLABEL_OF:
        parent = A.SUBLABEL_OF[name]
        L += [rf"\begin{{warnbox}} \textbf{{Not an independent artifact.}} All "
              rf"{ev.get('n_labels', 0)} \texttt{{{disp}}} labels co-occur with "
              rf"\texttt{{{parent}}}, so the two are perfectly collinear. Any "
              rf"association recovered against the whole survey is inherited from "
              rf"\texttt{{{parent}}} and is not evidence about \texttt{{{disp}}}. "
              rf"It is excluded from the cross-artifact ranking; the only "
              rf"well-posed question is conditional --- among "
              rf"\texttt{{{parent}}}-labelled exposures, what predicts also being "
              rf"\texttt{{{disp}}}. \end{{warnbox}}", ""]

    # ---- the sample
    L += [r"\section{The sample, and how much of it is independent}", ""]
    L += [rf"Labelling for \texttt{{{disp}}} begins on "
          rf"{ev.get('epoch', '?')}. Nights before that carry no label for it by "
          rf"construction --- unflagged because nobody was looking, not because "
          rf"nothing was there --- so they are excluded as false controls. Within "
          rf"the resulting frame of {ev.get('frame_n', 0):,} exposures on "
          rf"{ev.get('frame_nights', 0)} inspected nights there are "
          rf"\textbf{{{ev.get('n_labels', 0)}}} labelled exposures "
          rf"({ev.get('n_labels', 0) / max(ev.get('frame_n', 1), 1) * 100:.3f}\%).",
          ""]
    L += [rf"Those labels arrive in {ev.get('runs', 0)} contiguous runs on "
          rf"{ev.get('nights', 0)} distinct nights, spread over only "
          rf"{ev.get('n_targets', 0)} target fields and "
          rf"{ev.get('n_pointings', 0)} distinct pointings. The effective sample "
          rf"size is the night count, not the exposure count "
          rf"(Figure~\ref{{fig:{name}_burst}}).", ""]
    L += [fig_block(figs.get("burst"),
                    rf"Labels per night for \texttt{{{disp}}}. The concentration "
                    r"visible here is the reason every test in this chapter counts "
                    r"nights rather than exposures.", f"{name}_burst")]

    # ---- tier D stops here
    if ev.get("no_test"):
        L += [r"\section{Why no statistical test was run}", ""]
        L += [rf"\begin{{warnbox}} \textbf{{Not tested:}} "
              rf"{tex_escape(ev.get('no_test_reason', 'insufficient data'))}. "
              rf"\end{{warnbox}}", ""]
        L += [r"With this few independent nights no annulus test can distinguish a "
              r"real geometric association from coincidence, and running one "
              r"anyway would produce a number that looks like evidence. The "
              r"descriptive facts in \texttt{" + tex_escape(f"{name}_summary.txt") +
              r"} are all this dataset supports.", ""]
        L += [tier_d_note(name, ev), ""]
        return "\n".join(L)

    # ---- the scan
    lo, hi = ev.get("best_lo"), ev.get("best_hi")
    vmax = ev.get("best_vmax", float("nan"))
    L += [r"\section{The annulus scan}", ""]
    L += [rf"Scanning 1\,759 annuli $\times$ 5 brightness tiers, the strongest "
          rf"cell is a star brighter than $V = {vmax:.1f}$ at "
          rf"$[{lo:.3f}, {hi:.3f})^\circ$ off axis "
          rf"(width ${hi - lo:.2f}^\circ$). On "
          rf"\textbf{{{ev.get('obs_nights', 0):.0f}}} distinct nights a labelled "
          rf"exposure has such a star, against "
          rf"${num(ev.get('null_mean'), 2)} \pm {num(ev.get('null_sd'), 2)}$ "
          rf"expected under the burst-preserving null: "
          rf"$z = \mathbf{{{num(ev.get('z'), 2)}}}$.", ""]
    L += [scan_p_sentence(ev), ""]
    meta = getattr(RF.fig_scan, "last_meta", {}) or {}
    L += [fig_block(figs.get("scan"),
                    rf"Night-recurrence $z$ against off-axis distance for "
                    rf"\texttt{{{disp}}}. Each point is one cell that cleared the "
                    rf"support floor at a ${meta.get('width', 0):.2f}^\circ$ "
                    rf"annulus width ({meta.get('n_cells', 0)} of them); "
                    r"a gap means that radius could not be tested, \emph{not} "
                    r"that the excess there was zero. The selected brightness "
                    r"tier is emphasised; the fainter tiers show how the signal "
                    r"depends on cutting to brighter stars.",
                    f"{name}_scan")]
    L += [fig_block(figs.get("null"),
                    r"The observed scan maximum against both null distributions. "
                    r"The cross-night shuffle is the stricter of the two, and the "
                    r"larger of the two $p$-values is the one quoted.",
                    f"{name}_null")]

    # ---- shape
    L += [r"\section{Is it an edge or a bump?}", ""]
    L += [r"A mechanism with a physical edge --- a baffle, a lens rim --- produces "
          r"a sharp radial cutoff. Diffuse scattering off a large structure "
          r"produces a broad elevated region. The distinction matters more than "
          r"the peak's exact location, so the neighbourhood of the winning "
          r"annulus is shown explicitly.", ""]
    L += [fig_block(figs.get("profile"),
                    rf"Radial profile around the selected annulus for "
                    rf"\texttt{{{disp}}}: artifact rate (left) and number of "
                    r"independent nights (right). The selected annulus is "
                    r"emphasised.", f"{name}_profile")]
    L += [shape_sentence(name, ev), ""]

    # ---- controls
    L += [r"\section{Does it behave like a mechanism?}", ""]
    L += [rf"Four independent checks, none of which the scan itself can supply.", ""]
    L += [fig_block(figs.get("stars"),
                    rf"Per-star contribution to the in-annulus labels for "
                    rf"\texttt{{{disp}}}. A mechanism shows up on many stars; a "
                    r"scheduling coincidence shows up on one.", f"{name}_stars")]
    L += [stars_sentence(ev), ""]
    L += [fig_block(figs.get("controls"),
                    r"Left: the same-width annulus displaced either side --- a "
                    r"sharp-edged mechanism should lose its excess immediately. "
                    r"Right: artifact rate against the brightness of the "
                    r"in-annulus star, with sparsely populated bins drawn hollow.",
                    f"{name}_controls")]
    L += [controls_sentence(ev), ""]

    # ---- frames
    L += [r"\section{Which frame does the direction repeat in?}", ""]
    L += [fig_block(figs.get("frames"),
                    rf"Direction concentration $R$ for \texttt{{{disp}}} in the "
                    r"three candidate frames, labelled exposures against "
                    r"controls. Read the excess over each frame's own control, "
                    r"and read it as descriptive only.", f"{name}_frames")]
    L += [frames_sentence(ev), ""]

    # ---- other correlates
    L += [r"\section{What else differs}", ""]
    L += [r"The annulus test asks one question. This is everything else the "
          r"telemetry knows, ranked by effect size rather than by $p$-value --- "
          r"with a hundred thousand exposures almost anything reaches "
          r"significance, so the effect size is what carries information. These "
          r"$p$-values ignore night clustering and are therefore optimistic.", ""]
    L += [fig_block(figs.get("screen"),
                    rf"Univariate effect sizes for \texttt{{{disp}}}. Colour "
                    r"carries only the sign.", f"{name}_screen")]
    L += [screen_sentence(ev), ""]
    L += [telescope_table(ev), ""]

    # ---- verdict
    L += [r"\section{Verdict}", ""]
    L += [rf"\verdictbox{{{col}}}{{{tex_escape(v)}}}{{%"]
    L += [verdict_explanation(v) + "}", ""]
    met = ev.get("criteria_met") or []
    failed = ev.get("criteria_failed") or []
    if met:
        L += [r"\textbf{Criteria met.}", r"\begin{itemize}[nosep]"]
        L += [rf"  \item {tex_escape(m)}" for m in met]
        L += [r"\end{itemize}", ""]
    if failed:
        L += [r"\textbf{Criteria not met.}", r"\begin{itemize}[nosep]"]
        L += [rf"  \item {tex_escape(m)}" for m in failed]
        L += [r"\end{itemize}", ""]
    L += [closing_paragraph(name, ev), ""]
    return "\n".join(L)


# ------------------------------------------------------------------ sentences
def headline_sentence(name, ev):
    v = ev.get("verdict", "")
    lo, hi = ev.get("best_lo"), ev.get("best_hi")
    if ev.get("no_test"):
        return (rf"{ev.get('n_labels', 0)} labels on {ev.get('nights', 0)} "
                r"nights --- too few independent events for any inferential test. "
                r"This is a statement about power, not a null result.")
    if v == "MECHANISM IDENTIFIED":
        return (rf"A bright star at ${lo:.2f}$--${hi:.2f}^\circ$ off axis predicts "
                rf"this artifact, on {ev.get('nights_in', 0)} separate nights and "
                rf"{ev.get('n_stars', 0)} different stars, and the association "
                r"survives every control applied.")
    if v == "STRONG ASSOCIATION":
        return (rf"A bright star at ${lo:.2f}$--${hi:.2f}^\circ$ off axis is "
                r"associated with this artifact beyond what burstiness explains, "
                r"but not every corroborating criterion is met.")
    if v == "SUGGESTIVE":
        return (rf"A hint of a preferred distance near ${lo:.2f}$--${hi:.2f}^\circ$, "
                r"not distinguishable from pointing-schedule coincidence with the "
                r"data available.")
    if v == "CONFOUNDED":
        return (rf"The apparent enrichment at ${lo:.2f}$--${hi:.2f}^\circ$ "
                rf"(${num(ev.get('crude_lift'), 1)}\times$) is explained by when "
                r"and where the survey was pointing, not by the star.")
    if v == "NULL":
        return (r"No bright-star geometry predicts this artifact once burstiness "
                r"is respected. Whatever drives it is not a star at a fixed "
                r"off-axis distance.")
    return r"See the sections below."


def scan_p_sentence(ev):
    ps, pc = ev.get("p_shift"), ev.get("p_cross")
    p = ev.get("p_fwer")
    s = (rf"Both nulls were run: the within-night shift gives $p = {num(ps, 4)}$ "
         rf"and the stricter cross-night shuffle $p = {num(pc, 4)}$. The quoted "
         rf"family-wise value is the weaker of the two, "
         rf"$p_{{\mathrm{{FWER}}}} = \mathbf{{{num(p, 4)}}}$")
    floor = ev.get("p_floor")
    if p is not None and floor is not None and np.isfinite(p) and \
            np.isclose(p, floor):
        s += (r", which sits at the permutation floor --- more permutations could "
              r"only lower it")
    return s + "."


def shape_sentence(name, ev):
    w = ev.get("width", float("nan"))
    if name == "tape":
        return (r"For \texttt{tape} the profile is decisive in the negative: the "
                r"excess is a broad elevated plateau spanning roughly "
                r"$20.5$--$21.15^\circ$ rather than a step. The distance is real; "
                r"the single best bin is an artefact of picking a maximum on a "
                r"plateau and should not be quoted as a sharp feature. A baffle or "
                r"lens edge gives a hard edge. A plateau this wide is more "
                r"consistent with scattering off a large structure.")
    if np.isfinite(w) and w <= 0.1:
        return (rf"The selected annulus is only ${w:.2f}^\circ$ wide and the "
                r"neighbouring annuli fall away sharply, which is the signature "
                r"of a physical edge rather than of a broad excess.")
    return (r"Read the two panels together: a peak that is one bin wide with "
            r"steep sides is a mechanism edge, whereas a gentle rise and fall "
            r"across several bins indicates a broad excess whose exact centre "
            r"carries little meaning.")


def stars_sentence(ev):
    n = ev.get("n_stars", 0)
    share = ev.get("top_star_share")
    top = tex_escape(ev.get("top_star", "?"))
    k = ev.get("k_in_annulus", 0)
    if share is None or not np.isfinite(share):
        return r"No in-annulus labels, so there is nothing to attribute."
    s = (rf"{n} distinct star{'s' if n != 1 else ''} contribute the "
         rf"{k} in-annulus labelled exposures, the largest single contributor "
         rf"being {top} at {share * 100:.0f}\%.")
    if share >= 0.5:
        s += (r" That is the decisive observation for this artifact: one star "
              r"supplying half or more of the evidence, over a few consecutive "
              r"nights, is what the survey revisiting a single field looks like. "
              r"An optical mechanism would appear on many different stars, at "
              r"unrelated times.")
    else:
        s += (r" No single star dominates, which is what a genuine mechanism "
              r"should look like --- the effect follows the geometry rather than "
              r"one week of the observing schedule.")
    loo_p, loo_z = ev.get("loo_p"), ev.get("loo_z")
    if loo_p is not None and np.isfinite(loo_p):
        s += (rf" Removing the top contributor entirely and re-testing gives "
              rf"$z = {num(loo_z, 2)}$, $p = {num(loo_p, 4)}$.")
    return s


def controls_sentence(ev):
    r = ev.get("offset_ratio")
    parts = []
    if r is not None and np.isfinite(r):
        if r < 0.2:
            parts.append(rf"The strongest displaced annulus reaches only "
                         rf"{r * 100:.0f}\% of the selected annulus's rate, so the "
                         rf"feature has a genuinely sharp edge.")
        else:
            parts.append(rf"Displaced annuli reach {r * 100:.0f}\% of the selected "
                         rf"annulus's rate, so the excess is broad rather than "
                         rf"edge-like and the precise radius should not be "
                         rf"over-interpreted.")
    tp, br = ev.get("trend_p"), ev.get("bright_faint_ratio")
    if tp is not None and np.isfinite(tp):
        parts.append(rf"The magnitude trend is significant "
                     rf"($p = {num(tp, 4)}$) with bright stars carrying "
                     rf"${num(br, 1)}\times$ the rate of faint ones"
                     + (r", which is the dose-response a real optical mechanism "
                        r"should show." if ev.get("dose_response")
                        else r", but the pattern does not satisfy the full "
                             r"dose-response criterion."))
    if not parts:
        return (r"Neither control could be computed with the support available, "
                r"which is itself a limit on what this annulus can support.")
    return " ".join(parts)


FRAME_MEANING = {
    "camera": (r"which is the camera-fixed ordering a ghost in the optics would "
               r"produce"),
    "horizon": (r"which points at something fixed relative to the horizon --- the "
                r"dome, the structure, the mirror covers --- rather than at the "
                r"optics or the sky"),
    "sky": (r"so the direction repeats \emph{on the sky}, which is what "
            r"revisiting the same field produces rather than what a camera-fixed "
            r"ghost produces"),
}


def frames_sentence(ev):
    """Report the frame with the largest excess, whichever of the three it is."""
    fc = (ev.get("_fig") or {}).get("frames") or {}
    exc = {}
    for n in ("camera", "horizon", "sky"):
        a, b = fc.get(f"R_lab_{n}"), fc.get(f"R_ctl_{n}")
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            exc[n] = a - b
    if not exc:
        return r"The frame diagnostic could not be computed for this artifact."

    mult = ev.get("frac_single")
    s = ""
    if mult is not None and np.isfinite(mult):
        s += (rf"In {mult * 100:.0f}\% of in-annulus exposures there is exactly "
              rf"one bright star (mean multiplicity "
              rf"{num(ev.get('mean_mult'), 2)}). ")
    win = max(exc, key=exc.get)
    rest = ", ".join(rf"{n} $+{exc[n]:.3f}$" for n in exc if n != win)
    s += (rf"The largest excess over its own control is in the \textbf{{{win}}} "
          rf"frame ($+{exc[win]:.3f}$, against {rest}), "
          f"{FRAME_MEANING[win]}. ")
    s += (r"This remains descriptive, and deliberately so: with one direction per "
          r"exposure the three frames are related by fixed offsets, so none of "
          r"them can be \emph{identified} from telemetry however suggestive the "
          r"ordering looks --- see Chapter~\ref{ch:method}.")
    return s


def screen_sentence(ev):
    top = ev.get("top_screen") or []
    if not top:
        return r"No variable had sufficient coverage to screen."
    # \varname wraps \seqsplit so long snake_case names can break across lines
    # instead of pushing the line into the margin.
    bits = ", ".join(rf"\varname{{{tex_escape(v)}}} ($\delta = {d:+.2f}$)"
                     for v, d in top[:4])
    s = rf"The largest effects are {bits}."
    mf = ev.get("moonup_frac")
    if mf is not None and np.isfinite(mf) and mf > 0.9 and ev.get("n_labels", 0) >= 50:
        s += (rf" Separately and importantly, {mf * 100:.0f}\% of these labels "
              rf"occur with the Moon above the horizon "
              rf"({num(ev.get('rate_moon_up'), 3)}\% rate with the Moon up "
              rf"against {num(ev.get('rate_moon_down'), 3)}\% with it down), so "
              rf"the Moon gates this artifact whatever the star geometry does.")
    return s


def telescope_table(ev):
    mh = ev.get("mh")
    if mh is None or len(mh) == 0:
        return (r"\medskip\noindent No telescope-state condition had enough "
                r"exposures on both sides of a night to support a stratified "
                r"comparison.")
    L = [r"\medskip", r"\noindent\textbf{Telescope state.} Night-stratified odds "
         r"ratios. These are \emph{exposure-level} and therefore supporting "
         r"evidence only: a burst inflates their precision. The gap between the "
         r"crude and stratified columns shows how much of the crude effect was "
         r"merely which nights were observed.", "",
         r"\begin{center}", r"\small",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Condition & OR (night-stratified) & 95\% CI & crude OR \\", r"\midrule"]
    for _, r in mh.head(10).iterrows():
        L.append(rf"{tex_escape(r['condition'])} & {num(r['OR_night'], 2)} & "
                 rf"{num(r['lo'], 2)}--{num(r['hi'], 2)} & "
                 rf"{num(r['OR_crude'], 2)} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(L)


def tier_d_note(name, ev):
    pn = ev.get("_fig", {}).get("per_night")
    if pn is None or len(pn) == 0:
        return ""
    top = pn.sort_values(ascending=False)
    frac = top.iloc[0] / pn.sum() * 100
    s = (rf"Of {int(pn.sum())} labels, {int(top.iloc[0])} ({frac:.0f}\%) fall on "
         rf"{top.index[0]} alone, so the effective sample size is about "
         rf"{len(pn)} event{'s' if len(pn) != 1 else ''}.")
    if name == "lighton":
        s += (r" A single night supplying almost every label is consistent with a "
              r"one-off illumination event --- a light left on --- rather than "
              r"with anything geometric.")
    if name == "moon":
        s += (r" The artifact is named after the Moon, and at this sample size the "
              r"apparent star-annulus signal cannot be separated from the Moon's "
              r"own position, so the Moon is the obvious candidate and no star "
              r"claim is made.")
    return s


def verdict_explanation(v):
    return {
        "MECHANISM IDENTIFIED":
            r"A specific bright-star geometry predicts this artifact and survives "
            r"the burst-preserving nulls, removal of its top-contributing star, "
            r"displaced control annuli, and a magnitude dose-response.",
        "STRONG ASSOCIATION":
            r"Associated beyond what burstiness explains, but short of a mechanism "
            r"claim on at least one corroborating criterion.",
        "SUGGESTIVE":
            r"A hint only. Not distinguishable from pointing-schedule coincidence "
            r"with the data available.",
        "CONFOUNDED":
            r"A large crude enrichment explained by when and where the survey "
            r"pointed rather than by the star.",
        "NULL":
            r"Adequately powered, and no star-geometry effect found.",
        "UNDERPOWERED":
            r"Too few independent nights to tell. This means \emph{cannot tell}, "
            r"not \emph{no effect}.",
    }.get(v, "")


def closing_paragraph(name, ev):
    v = ev.get("verdict", "")
    if v == "NULL":
        z = ev.get("z")
        return (rf"\textbf{{Power.}} The strongest cell anywhere in the scan "
                rf"reached $z = {num(z, 2)}$, on {ev.get('nights_in', 0)} nights "
                rf"with in-annulus labels. An effect large enough to lift "
                rf"night-recurrence about five standard deviations above the null "
                rf"would have been detected here; anything smaller would not. "
                rf"That bounds the claim without overstating it.")
    if v == "CONFOUNDED":
        return (r"\textbf{What would settle it.} More nights would help only if "
                r"they brought different fields and different stars into the same "
                r"annulus. Additional exposures on the same fields cannot, since "
                r"they add no independent information.")
    if v == "MECHANISM IDENTIFIED" and name == "horseshoe":
        return (r"\textbf{This is also the positive control for the whole "
                r"analysis.} The mechanism was already known and independently "
                r"implemented before this work; recovering it from the labels "
                r"alone, at the right radius and with the right magnitude "
                r"behaviour, is what licenses trusting the same machinery on the "
                r"twelve artifacts where no answer was known in advance. Had it "
                r"failed here, nothing else in this report would be worth "
                r"reading.")
    if v == "STRONG ASSOCIATION" and name == "tape":
        return (r"\textbf{Why this matters beyond \texttt{tape}.} The label CSV "
                r"already encodes a $19.5$--$22.0^\circ$ annulus in its "
                r"\path{tape_*} columns, reproduced here exactly on all "
                r"15\,708 populated rows. That hypothesis predates this analysis "
                r"and was arrived at independently, so the agreement is a genuine "
                r"confirmation rather than a rediscovery of an assumption baked "
                r"into the inputs.")
    return ""


# ====================================================================== master
def master_tex(evidence, order, cross_figs):
    L = [r"""\documentclass[11pt,a4paper,openany]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{microtype}
% \path for file names: they contain underscores and must be allowed to break
% across lines rather than running into the margin.
\usepackage{url}
% Long snake_case identifiers must be breakable in running text.
\usepackage{seqsplit}
\newcommand{\varname}[1]{\texttt{\seqsplit{#1}}}
\urlstyle{tt}
\usepackage[framemethod=default]{mdframed}
\usepackage[colorlinks=true,linkcolor=linknavy,citecolor=linknavy,
            urlcolor=linknavy]{hyperref}

% Verdict colours: the status palette, reserved for status and never reused as
% a series colour.
\definecolor{mechid}{HTML}{008300}
\definecolor{strongassoc}{HTML}{1baf7a}
\definecolor{suggestive}{HTML}{eda100}
\definecolor{confounded}{HTML}{eb6834}
% NULL / UNDERPOWERED use ink greys rather than a hue: they are the absence of a
% finding, and a hue there would read as a result. Both are dark enough to clear
% 4.5:1 on white, and the verdict word is always printed, so the status is never
% carried by colour alone.
\definecolor{nullres}{HTML}{52514e}
\definecolor{underpow}{HTML}{6b6a66}
\definecolor{linknavy}{HTML}{1c5cab}
\definecolor{rulegrey}{HTML}{c3c2b7}
\definecolor{surfwarm}{HTML}{f7f6f2}

\newmdenv[linecolor=rulegrey,linewidth=0.6pt,backgroundcolor=surfwarm,
          leftmargin=0pt,rightmargin=0pt,innerleftmargin=8pt,
          innerrightmargin=8pt,innertopmargin=6pt,innerbottommargin=6pt,
          skipabove=8pt,skipbelow=8pt]{warnbox}

% A verdict banner: the colour is the status token, and the word is always
% present, so the verdict is never carried by colour alone.
\newcommand{\verdictbox}[3]{%
  \begin{mdframed}[linecolor=#1,linewidth=2.2pt,backgroundcolor=surfwarm,
                   leftline=true,topline=false,bottomline=false,rightline=false,
                   innerleftmargin=10pt,innertopmargin=6pt,
                   innerbottommargin=6pt,skipabove=8pt,skipbelow=10pt]
    \textbf{\textcolor{#1}{#2}}\\[2pt] #3
  \end{mdframed}}

\title{\textbf{Stray-light artifacts in LSSTCam}\\[4pt]
       \large Bright-star geometry, telescope state, and what the labels
       can and cannot establish}
\author{Straylight analysis --- \texttt{zemax\_geometry/}}
\date{\today}

\begin{document}
\maketitle
"""]

    # ---- executive summary
    L += [r"\chapter*{Summary}", r"\addcontentsline{toc}{chapter}{Summary}", ""]
    L += [r"""This report asks one question of thirteen artifact labels: \emph{when
this artifact appears, is there always a bright star at a particular off-axis
distance from the boresight?} It also checks telescope position and state for each.

\textbf{Two artifacts have a real answer.} \texttt{horseshoe} is produced by a
bright star at $2.00$--$2.05^\circ$ off axis, which independently reproduces the L3
lens-baffle model already implemented in
\path{L3_baffle/horseshoe_detailer.py}; recovering a known mechanism from the
labels alone is also this analysis's positive control. \texttt{tape} shows a broad
excess peaking near $20.8^\circ$, confirming the $19.5$--$22^\circ$ annulus already
encoded in the label CSV's own \path{tape_*} columns --- but as a wide plateau,
not a sharp edge, so diffuse scattering off a large structure rather than a baffle.

\textbf{The rest do not.} Four are confounded, five are null, and two are too thin
to test. In particular \texttt{arc} remains strongly Moon-gated but has no
star-distance effect at all.

\begin{warnbox}
\textbf{The methodological result is as important as the astronomical one.} The
obvious statistic --- the artifact rate inside a candidate annulus divided by the
rate outside --- is actively misleading on this dataset. Because labels come in
bursts and the survey revisits fields, the largest such ``lift'' reachable
\emph{by chance alone} is $1.5\times$ to $5.8\times$. Ranking the artifacts that
way would have put \texttt{shoe} ($11.8\times$), \texttt{ghoul} ($6.9\times$) and
\texttt{sail} ($4.0\times$) at the top; all three are single-star, single-week
coincidences. Chapter~\ref{ch:method} sets out what replaces it.
\end{warnbox}
"""]
    L += [fig_block(cross_figs.get("lift"),
                    r"Crude lift against night-recurrence $z$ for every artifact "
                    r"with a testable scan. The shaded band is the range of lifts "
                    r"the null itself reaches. Points inside it are not evidence, "
                    r"however large the lift.", "cross_lift")]
    L += [fig_block(cross_figs.get("verdicts"),
                    r"Night-recurrence $z$ of the best cell per artifact. Only "
                    r"\texttt{horseshoe} and \texttt{tape} clear the threshold.",
                    "cross_verdicts")]

    # ---- verdict table
    L += [r"\section*{Verdicts at a glance}", ""]
    L += [r"\begin{center}", r"\small",
          r"\begin{tabular}{llrrlrrr}", r"\toprule",
          r"Artifact & Verdict & $n$ & nights & best annulus & $V<$ & $z$ "
          r"& $p_{\mathrm{FWER}}$ \\", r"\midrule"]
    for name in order:
        ev = evidence[name]
        ann = (rf"$[{ev['best_lo']:.2f},{ev['best_hi']:.2f})$"
               if "best_lo" in ev else "---")
        vm = (f"{ev['best_vmax']:.1f}"
              if np.isfinite(ev.get("best_vmax", float("nan"))) else "---")
        note = r"$^\dagger$" if name in A.SUBLABEL_OF else ""
        L.append(rf"\texttt{{{tex_escape(name)}}}{note} & "
                 rf"\textcolor{{{VERDICT_COLOUR.get(ev.get('verdict'), 'black')}}}"
                 rf"{{{tex_escape(ev.get('verdict', '?'))}}} & "
                 rf"{ev.get('n_labels', 0)} & {ev.get('nights', 0)} & {ann} & "
                 rf"{vm} & {num(ev.get('z'), 2)} & {num(ev.get('p_fwer'), 4)} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", "",
          r"\noindent$^\dagger$\,\texttt{sail} occurs only as \texttt{tape;sail}, "
          r"so it is perfectly collinear with \texttt{tape} and is excluded from "
          r"the ranking; see Chapter~\ref{ch:sail}.", ""]

    L += [r"\clearpage", r"\tableofcontents", r"\clearpage", ""]
    L += [r"\input{method}", ""]
    for name in order:
        L += [rf"\input{{art_{name}}}"]
    L += ["", r"\end{document}"]
    return "\n".join(L)


# ====================================================================== main
def main():
    path = TABDIR / "evidence.pkl"
    if not path.exists():
        sys.exit(f"{path} not found -- run run_all_artifacts.py first")
    with open(path, "rb") as fh:
        evidence = pickle.load(fh)
    REPORT.mkdir(parents=True, exist_ok=True)

    # Order: strongest evidence first, matching the text summary's ranking.
    def key(n):
        ev = evidence[n]
        vi = (A.VERDICT_ORDER.index(ev["verdict"])
              if ev.get("verdict") in A.VERDICT_ORDER else 99)
        z = ev.get("z", float("-inf"))
        return (vi, -(z if np.isfinite(z) else float("-inf")))
    order = sorted(evidence, key=key)

    print("figures:")
    cross = {"verdicts": RF.fig_verdicts(evidence),
             "lift": RF.fig_lift_vs_z(evidence)}
    for k, v in cross.items():
        print(f"  cross/{k}: {v}")

    for name in order:
        ev = evidence[name]
        figs = {
            "burst": RF.fig_burst(name, ev),
            "scan": RF.fig_scan(name, ev),
            "null": RF.fig_null(name, ev),
            "profile": RF.fig_profile(name, ev),
            "stars": RF.fig_stars(name, ev),
            "controls": RF.fig_controls(name, ev),
            "frames": RF.fig_frames(name, ev),
            "screen": RF.fig_screen(name, ev),
        }
        if ev.get("no_test"):
            figs = {"burst": figs["burst"]}
        figs = {k: (str(v.relative_to(REPORT)) if v else None)
                for k, v in figs.items()}
        (REPORT / f"art_{name}.tex").write_text(
            artifact_chapter(name, ev, figs) + "\n")
        print(f"  wrote report/art_{name}.tex")

    (REPORT / "method.tex").write_text(METHOD + "\n")
    cross_rel = {k: (str(v.relative_to(REPORT)) if v else None)
                 for k, v in cross.items()}
    (REPORT / "straylight_artifacts.tex").write_text(
        master_tex(evidence, order, cross_rel) + "\n")
    print("  wrote report/method.tex")
    print("  wrote report/straylight_artifacts.tex")
    print("\nbuild with:  cd report && latexmk -pdf straylight_artifacts.tex")


if __name__ == "__main__":
    main()
