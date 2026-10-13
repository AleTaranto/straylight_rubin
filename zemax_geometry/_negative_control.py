"""Negative control: does the scan correction actually neutralise the search?

The scan picks the best of ~8800 (annulus x brightness) cells. If the
max-statistic calibration works, then feeding it labels that have been shuffled
in a way that destroys any real star association -- while preserving the burst
structure -- must yield p-values scattered over (0, 1), not small ones. If they
came out small, every "detection" in the real analysis would be an artefact of
searching.

Run:  python _negative_control.py
"""
import numpy as np
import pandas as pd

import artifact_stats as A

CACHE = "../outputs/artifact_stats/cache"


def main():
    frame = pd.read_parquet(f"{CACHE}/frame.parquet")
    pairs = pd.read_parquet(f"{CACHE}/star_pairs.parquet")
    mem, cells = A.annulus_membership(
        len(frame), pairs["exp_idx"].values, pairs["sep"].values,
        A.nested_annulus_grid(), pair_mag=pairs["vmag"].values)

    print("NEGATIVE CONTROL -- scan run against shuffled labels")
    print("The shuffle moves each night's whole label pattern to a different "
          "night, so")
    print("burstiness is preserved but any association with the sky is broken.")
    print("p_FWER should be scattered over (0,1). Small p here would mean the")
    print("correction is not doing its job.\n")

    rows = []
    for art in ("tape", "banana", "arc", "ghoul"):
        fr, _ = A.artifact_frame(frame, art, set(frame["day_obs"].unique()))
        fr = fr.sort_values(["day_obs", "seq_num"])
        idx = fr.index.values
        lab = fr[f"is_{art}"].values.astype(bool)
        nid = pd.factorize(fr["day_obs"].values)[0]
        m = mem[idx]
        bounds = A._night_bounds(nid)
        counts = np.diff(bounds)

        ps = []
        for s in range(5):
            rng = np.random.default_rng(1000 + s)
            # Move each night's label block to a *different* night of the same
            # length where possible: this keeps run structure but decouples the
            # labels from that night's pointings, which is what we want broken.
            fake = np.zeros_like(lab)
            order = rng.permutation(len(counts))
            for g_dst, g_src in enumerate(order):
                s0, e0 = bounds[g_dst], bounds[g_dst + 1]
                s1, e1 = bounds[g_src], bounds[g_src + 1]
                src = lab[s1:e1]
                n = e0 - s0
                if len(src) >= n:
                    fake[s0:e0] = src[:n]
                else:
                    fake[s0:s0 + len(src)] = src
            r = A.scan_recurrence(fake, m, nid, n_perm=400, seed=7 + s)
            ps.append(r["p_fwer"])
            print(f"  {art:8s} shuffle {s}: labels {int(fake.sum()):5d}  "
                  f"z {r['z']:6.2f}  p_FWER {r['p_fwer']:.4f}")
        print(f"  {art:8s} -> mean p {np.mean(ps):.3f}, min p {np.min(ps):.3f}\n")
        rows.append(dict(artifact=art, mean_p=np.mean(ps), min_p=np.min(ps)))

    t = pd.DataFrame(rows)
    ok = (t["mean_p"] > 0.15).all()
    print("=" * 70)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} -- mean p per artifact: " +
          ", ".join(f"{r.artifact} {r.mean_p:.2f}" for r in t.itertuples()))
    print("A pass means the scan correction is not manufacturing detections.")
    print()
    print("History worth keeping: with the within-night shift null alone this test")
    print("FAILED -- arc and ghoul reached p = 0.007-0.010 on labels deliberately")
    print("decoupled from the sky, i.e. the correction was anti-conservative. That")
    print("is why scan_recurrence also calibrates against whole-night transplants")
    print("and quotes the larger p. Individual shuffles can still land near 0.01")
    print("(see arc above), so p just under 0.05 should not be read as a detection;")
    print("the verdict thresholds require 0.01 for a mechanism claim for this reason.")


if __name__ == "__main__":
    main()
