"""Paired per-fold comparison from a finished frozen-mode run.

Reads folds.json (the per-fold record written by src/proneuro/deep/train.py)
and compares freesurfer_plus_vit_ridge against freesurfer_ridge_same_folds
fold by fold — the paired differences, not the summary SDs, are what decide
whether the combined model genuinely differs from the FreeSurfer-only ridge.

Usage (on XAI01, in the ProNeuro-AD folder, .venv active):
    python scripts\\paired_folds.py
    python scripts\\paired_folds.py --folds outputs\\vit_mtl_pca128\\folds.json

Exit code 0 always; this is a report, not a gate.
"""

import argparse
import json
import math
from pathlib import Path

METRICS = ("amyloid_auc", "conversion_cindex")
ARMS = ("freesurfer_plus_vit_ridge", "ridge", "vit_features_ridge", "vit")
LABELS = {
    "freesurfer_plus_vit_ridge": "FreeSurfer + ViT ridge",
    "ridge": "FreeSurfer ridge",
    "vit_features_ridge": "ViT-features ridge",
    "vit": "Tri-planar ViT (neural)",
}


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(diffs):
    """Two-sided paired t-test p-value; uses scipy when available, else
    falls back to reporting t and df without a p-value."""
    n = len(diffs)
    if n < 2 or sd(diffs) == 0:
        return None, None
    t = mean(diffs) / (sd(diffs) / math.sqrt(n))
    try:
        from scipy import stats

        return t, float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return t, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="outputs/vit_mtl/folds.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.folds).read_text())
    rows.sort(key=lambda r: r["fold"])
    print(f"{args.folds}: {len(rows)} folds\n")

    for metric in METRICS:
        print(f"== {metric} ==")
        for arm in ARMS:
            vals = [r[arm][metric] for r in rows if arm in r and metric in r[arm]]
            if vals:
                print(f"  {LABELS[arm]:26s} mean {mean(vals):.3f}  sd {sd(vals):.3f}  "
                      f"folds {[round(v, 3) for v in vals]}")
        diffs = [
            r["freesurfer_plus_vit_ridge"][metric] - r["ridge"][metric]
            for r in rows
            if metric in r.get("freesurfer_plus_vit_ridge", {}) and metric in r.get("ridge", {})
        ]
        if diffs:
            wins = sum(1 for d in diffs if d > 0)
            t, p = paired_t(diffs)
            line = (f"  PAIRED combined - ridge: mean diff {mean(diffs):+.3f}  "
                    f"sd of diff {sd(diffs):.3f}  combined wins {wins}/{len(diffs)} folds")
            if t is not None:
                line += f"  t({len(diffs) - 1}) = {t:.2f}"
            if p is not None:
                line += f"  p = {p:.3f}"
            print(line)
            print(f"  per-fold diffs {[round(d, 3) for d in diffs]}")
        print()


if __name__ == "__main__":
    main()
