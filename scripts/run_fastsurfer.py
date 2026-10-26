"""Run FastSurfer segmentation on one T1 MRI and print key brain volumes.

Example (PowerShell):
    python scripts\\run_fastsurfer.py --t1 data\\raw\\sample\\sub-01_T1w.nii.gz --sid sub-01

Docker Desktop must be running. The first run downloads the FastSurfer image (several GB).
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from proneuro.preprocessing.fastsurfer import DEFAULT_IMAGE, fastsurfer_command

KEY_REGIONS = [
    "Left-Hippocampus", "Right-Hippocampus", "Left-Lateral-Ventricle",
    "Right-Lateral-Ventricle", "Left-Amygdala", "Right-Amygdala",
]


def read_volumes(stats_file: Path) -> dict[str, float]:
    """Parse region volumes (mm^3) from a FreeSurfer-style .stats file."""
    vols = {}
    for line in stats_file.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 5:
            vols[parts[4]] = float(parts[3])
    return vols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t1", required=True, help="T1-weighted MRI (.nii, .nii.gz or .mgz)")
    p.add_argument("--sid", default="sub-01", help="subject ID (output folder name)")
    p.add_argument("--out", default="outputs/fastsurfer", help="output folder")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="FastSurfer Docker image")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="run without GPU")
    args = p.parse_args()

    t1 = Path(args.t1)
    if not t1.exists():
        sys.exit(f"T1 file not found: {t1}")
    if shutil.which("docker") is None:
        sys.exit("Docker not found. Install Docker Desktop and make sure it is running.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cmd = fastsurfer_command(t1, args.sid, out, image=args.image,
                             threads=args.threads, use_gpu=not args.cpu)
    print("Running:\n  " + " ".join(cmd) + "\n")
    start = time.time()
    result = subprocess.run(cmd)
    minutes = (time.time() - start) / 60
    if result.returncode != 0:
        sys.exit(f"FastSurfer failed (exit code {result.returncode}) after {minutes:.1f} min. "
                 "Copy the error above.")

    print(f"\nFinished in {minutes:.1f} min. Output: {out / args.sid}")
    stats = out / args.sid / "stats" / "aseg+DKT.stats"
    if stats.exists():
        vols = read_volumes(stats)
        print("\nKey volumes (mm^3):")
        for region in KEY_REGIONS:
            if region in vols:
                print(f"  {region:26s} {vols[region]:10.0f}")
    else:
        print(f"(Stats file not found at {stats}; check the output folder.)")


if __name__ == "__main__":
    main()
