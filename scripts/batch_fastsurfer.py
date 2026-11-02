"""Run FastSurfer on every downloaded baseline T1 scan (skips ones already done).

    python scripts/batch_fastsurfer.py --download-dir <LONI download folder>

LONI downloads look like  ADNI/<PTID>/<sequence>/<date>/I<image id>/*.dcm (or *.nii).
Scans are matched to data/processed/mri_selection/baseline_t1_selection.csv by image ID.
DICOM folders are converted to NIfTI first. Output: outputs/fastsurfer/<RID>/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from proneuro.preprocessing.fastsurfer import DEFAULT_IMAGE, fastsurfer_command


def find_image_dirs(root: Path) -> dict[int, Path]:
    found = {}
    for p in root.rglob("I*"):
        if p.is_dir() and p.name[1:].isdigit():
            found[int(p.name[1:])] = p
    return found


def to_nifti(image_dir: Path, nifti_dir: Path) -> Path:
    """Return a NIfTI for this image folder, converting DICOM if needed."""
    for pattern in ("*.nii.gz", "*.nii"):
        hits = sorted(image_dir.glob(pattern))
        if hits:
            return hits[0]
    out = nifti_dir / f"{image_dir.name}.nii.gz"
    if not out.exists():
        import dicom2nifti

        nifti_dir.mkdir(parents=True, exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(str(image_dir), str(out), reorient_nifti=True)
    return out


def docker_running() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download-dir", required=True)
    p.add_argument("--selection", default="data/processed/mri_selection/baseline_t1_selection.csv")
    p.add_argument("--out", default="outputs/fastsurfer")
    p.add_argument("--nifti-dir", default="data/processed/nifti")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--limit", type=int, default=0, help="process at most N scans (0 = all)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not docker_running():
        sys.exit("Docker is not running. Open Docker Desktop, wait for 'Engine running', "
                 "then run this again.")
    sel = pd.read_csv(args.selection)
    dirs = find_image_dirs(Path(args.download_dir))
    todo = sel[sel["image_id"].isin(dirs)]
    print(f"{len(dirs)} image folders found; {len(todo)} of {len(sel)} selected scans present")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for _, row in todo.iterrows():
        if args.limit and done + failed >= args.limit:
            break
        rid = str(int(row["RID"]))
        if (out / rid / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
            continue
        t0 = time.time()
        try:
            nii = to_nifti(dirs[int(row["image_id"])], Path(args.nifti_dir))
            cmd = fastsurfer_command(nii, rid, out, image=args.image, use_gpu=not args.cpu)
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode != 0:
                lines = (r.stdout + "\n" + r.stderr).strip().splitlines()
                tail = "\n    ".join(lines[-15:]) or "(no output)"
                raise RuntimeError(f"FastSurfer exit code {r.returncode}. Last output:\n    {tail}"
                                   f"\n  Command: {' '.join(cmd)}")
            done += 1
            print(f"[{done + failed}/{len(todo)}] RID {rid} ok ({time.time() - t0:.0f}s)")
        except Exception as e:  # keep going; log the failure
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            print(f"[{done + failed}/{len(todo)}] RID {rid} FAILED: {msg}", file=sys.stderr)
            with open(out / "failures.txt", "a", encoding="utf-8") as f:
                f.write(f"{rid}\t{row['image_id']}\t{msg[:2000]}\n")
    print(f"Done: {done} processed, {failed} failed")


if __name__ == "__main__":
    main()
