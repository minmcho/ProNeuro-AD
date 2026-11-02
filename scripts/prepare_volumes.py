"""Convert FastSurfer outputs into normalized brain volumes for deep learning.

    python scripts/prepare_volumes.py --fastsurfer-dir outputs/fastsurfer

Writes data/processed/volumes/<RID>.npy (float16, 128^3 by default, ~4 MB each).
"""

import argparse
from pathlib import Path

import numpy as np

from proneuro.deep.volumes import load_subject, preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fastsurfer-dir", default="outputs/fastsurfer")
    p.add_argument("--out", default="data/processed/volumes")
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = [d for d in Path(args.fastsurfer_dir).iterdir() if d.is_dir() and d.name.isdigit()]
    ok = skipped = failed = 0
    for d in sorted(subjects, key=lambda x: int(x.name)):
        dest = out / f"{d.name}.npy"
        if dest.exists():
            skipped += 1
            continue
        try:
            np.save(dest, preprocess(*load_subject(d), size=args.size))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"RID {d.name}: {e}")
    print(f"{ok} volumes written, {skipped} already present, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
