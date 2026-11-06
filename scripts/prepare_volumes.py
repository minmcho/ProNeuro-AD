"""Convert FastSurfer outputs into normalized inputs for deep learning.

    python scripts/prepare_volumes.py                 # whole brain -> data/processed/volumes
    python scripts/prepare_volumes.py --type mtl      # hippocampus crops -> data/processed/mtl

brain: <RID>.npy float16 (128,128,128), skull-stripped whole brain
mtl:   <RID>.npy float16 (2,64,64,64), 64 mm cube around each hippocampus
"""

import argparse
from pathlib import Path

import numpy as np

from proneuro.deep.volumes import load_seg, load_subject, mtl_crops, preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fastsurfer-dir", default="outputs/fastsurfer")
    p.add_argument("--type", choices=["brain", "mtl"], default="brain")
    p.add_argument("--out", default=None, help="default: data/processed/volumes or .../mtl")
    p.add_argument("--size", type=int, default=None, help="default: 128 (brain) or 64 (mtl)")
    args = p.parse_args()
    args.out = args.out or ("data/processed/mtl" if args.type == "mtl" else "data/processed/volumes")
    args.size = args.size or (64 if args.type == "mtl" else 128)

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
            if args.type == "mtl":
                np.save(dest, mtl_crops(*load_subject(d), load_seg(d), size=args.size))
            else:
                np.save(dest, preprocess(*load_subject(d), size=args.size))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"RID {d.name}: {e}")
    print(f"{ok} volumes written, {skipped} already present, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
