"""Run FastSurfer on every downloaded baseline T1 scan (skips ones already done).

    python scripts/batch_fastsurfer.py --download-dir <LONI download folder>

LONI downloads look like  ADNI/<PTID>/<sequence>/<date>/I<image id>/*.dcm (or *.nii).
Scans are matched to data/processed/mri_selection/baseline_t1_selection.csv by image ID.
DICOM folders are converted to NIfTI first. Output: outputs/fastsurfer/<RID>/
"""

import argparse
import shutil
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


def to_3d(nii: Path, nifti_dir: Path) -> Path:
    """FastSurfer needs one 3D image. Some ADNI T1 series hold several frames
    (multi-echo MPRAGE or repeats); combine them by root-mean-square, the standard
    way to merge multi-echo MPRAGE (for identical repeats it is close to the mean)."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(nii))
    if img.ndim == 3:
        return nii
    out = nifti_dir / (nii.name.replace(".nii.gz", "").replace(".nii", "") + "_3d.nii.gz")
    if not out.exists():
        data = np.asarray(img.dataobj, dtype=np.float32)
        data = data.reshape(data.shape[:3] + (-1,))
        combined = np.sqrt((data ** 2).mean(axis=3))
        nifti_dir.mkdir(parents=True, exist_ok=True)
        hdr = img.header.copy()
        hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(combined, img.affine, hdr), str(out))
    return out


def to_nifti(image_dir: Path, nifti_dir: Path) -> Path:
    """Return a single-frame 3D NIfTI for this image folder, converting DICOM if needed."""
    for pattern in ("*.nii.gz", "*.nii"):
        hits = sorted(image_dir.glob(pattern))
        if hits:
            return to_3d(hits[0], nifti_dir)
    out = nifti_dir / f"{image_dir.name}.nii.gz"
    if not out.exists():
        import dicom2nifti

        nifti_dir.mkdir(parents=True, exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(str(image_dir), str(out), reorient_nifti=True)
    return to_3d(out, nifti_dir)


def docker_running() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_one(rid: str, image_dir: Path, image_id, args, out: Path) -> tuple[bool, float, str]:
    """Convert + segment one scan. Returns (ok, seconds, error message)."""
    t0 = time.time()
    try:
        # A scan interrupted mid-run leaves a partial folder that FastSurfer refuses
        # to reuse; it holds nothing useful, so start that subject fresh.
        subj = out / rid
        if subj.exists() and not (subj / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
            shutil.rmtree(subj, ignore_errors=True)
        nii = to_nifti(image_dir, Path(args.nifti_dir))
        cmd = fastsurfer_command(nii, rid, out, image=args.image, threads=args.threads,
                                 use_gpu=not args.cpu)
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            lines = (r.stdout + "\n" + r.stderr).strip().splitlines()
            tail = "\n    ".join(lines[-15:]) or "(no output)"
            raise RuntimeError(f"FastSurfer exit code {r.returncode}. Last output:\n    {tail}"
                               f"\n  Command: {' '.join(cmd)}")
        return True, time.time() - t0, ""
    except Exception as e:  # keep going; report the failure
        return False, time.time() - t0, f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download-dir", required=True)
    p.add_argument("--selection", default="data/processed/mri_selection/baseline_t1_selection.csv")
    p.add_argument("--out", default="outputs/fastsurfer")
    p.add_argument("--nifti-dir", default="data/processed/nifti")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--limit", type=int, default=0, help="process at most N scans (0 = all)")
    p.add_argument("--workers", type=int, default=1,
                   help="scans processed in parallel (2-3 on a 12 GB GPU; watch nvidia-smi)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per FastSurfer run")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not docker_running():
        sys.exit("Docker is not running. Open Docker Desktop, wait for 'Engine running', "
                 "then run this again.")
    sel = pd.read_csv(args.selection)
    dirs = find_image_dirs(Path(args.download_dir))
    present = sel[sel["image_id"].isin(dirs)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(str(int(r["RID"])), dirs[int(r["image_id"])], r["image_id"])
            for _, r in present.iterrows()
            if not (out / str(int(r["RID"])) / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(dirs)} image folders found; {len(present)} of {len(sel)} selected scans present; "
          f"{len(present) - len(todo) if not args.limit else 'some'} already done; "
          f"{len(todo)} to process with {args.workers} worker(s) x {args.threads} threads",
          flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = failed = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, rid, d, iid, args, out): (rid, iid) for rid, d, iid in todo}
        for fut in as_completed(futures):
            rid, iid = futures[fut]
            ok, secs, msg = fut.result()
            n = done + failed + 1
            if ok:
                done += 1
            else:
                failed += 1
                with open(out / "failures.txt", "a", encoding="utf-8") as f:
                    f.write(f"{rid}\t{iid}\t{msg[:2000]}\n")
            rate = (time.time() - t_start) / n
            eta_h = rate * (len(todo) - n) / 3600
            status = f"ok ({secs:.0f}s)" if ok else f"FAILED: {msg}"
            print(f"[{n}/{len(todo)}] RID {rid} {status}   ~{eta_h:.1f} h left", flush=True,
                  file=sys.stdout if ok else sys.stderr)
    print(f"Done: {done} processed, {failed} failed")


if __name__ == "__main__":
    main()
