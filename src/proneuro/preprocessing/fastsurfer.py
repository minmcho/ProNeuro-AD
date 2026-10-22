"""Stage 2 (planned): run FastSurfer segmentation on raw T1 MRI.

FastSurfer runs best via its official Docker image with GPU access.
This builds the command; check FastSurfer's docs for current flags
and the FreeSurfer license file it requires.
"""

from pathlib import Path


def fastsurfer_command(t1: Path, subject_id: str, out_dir: Path, license_file: Path) -> list[str]:
    return [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{t1.parent.resolve()}:/data",
        "-v", f"{out_dir.resolve()}:/output",
        "-v", f"{license_file.parent.resolve()}:/fs_license",
        "deepmi/fastsurfer:latest",
        "--t1", f"/data/{t1.name}",
        "--sid", subject_id,
        "--sd", "/output",
        "--fs_license", f"/fs_license/{license_file.name}",
        "--seg_only",
    ]
