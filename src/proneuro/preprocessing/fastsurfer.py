"""Stage 2: FastSurfer brain segmentation on a T1-weighted MRI, run in Docker.

Segmentation only (--seg_only) does not need a FreeSurfer license.
Surface reconstruction (later) does; pass `license_file` then.
"""

import os
from pathlib import Path

DEFAULT_IMAGE = "deepmi/fastsurfer:latest"


def fastsurfer_command(
    t1: Path,
    subject_id: str,
    out_dir: Path,
    image: str = DEFAULT_IMAGE,
    threads: int = 4,
    use_gpu: bool = True,
    license_file: Path | None = None,
    seg_only: bool = True,
    user: str | None = None,
) -> list[str]:
    """Build the `docker run` command. Input folder -> /data, output folder -> /output.

    FastSurfer must not run as its default container user. On Linux/macOS we pass the
    host user (uid:gid); on Windows (no uid) Docker Desktop maps file permissions
    itself, so a regular non-root id works.
    """
    t1, out_dir = Path(t1).resolve(), Path(out_dir).resolve()
    if user is None:
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
    cmd = ["docker", "run", "--rm", "--user", user]
    if use_gpu:
        cmd += ["--gpus", "all"]
    cmd += ["-v", f"{t1.parent}:/data:ro", "-v", f"{out_dir}:/output"]
    if license_file is not None:
        license_file = Path(license_file).resolve()
        cmd += ["-v", f"{license_file.parent}:/fs_license:ro"]
    cmd += [image, "--t1", f"/data/{t1.name}", "--sid", subject_id, "--sd", "/output",
            "--threads", str(threads)]
    if license_file is not None:
        cmd += ["--fs_license", f"/fs_license/{license_file.name}"]
    if seg_only:
        cmd += ["--seg_only"]
    return cmd
