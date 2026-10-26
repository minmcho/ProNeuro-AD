"""Tests for the FastSurfer command builder (no Docker needed)."""

from pathlib import Path

from proneuro.preprocessing.fastsurfer import fastsurfer_command


def test_gpu_seg_only_command(tmp_path: Path):
    t1 = tmp_path / "in" / "sub-01_T1w.nii.gz"
    cmd = fastsurfer_command(t1, "sub-01", tmp_path / "out")
    assert cmd[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "/data/sub-01_T1w.nii.gz" in cmd
    assert "--seg_only" in cmd and "--fs_license" not in cmd


def test_cpu_with_license(tmp_path: Path):
    cmd = fastsurfer_command(tmp_path / "t1.mgz", "s1", tmp_path / "out", use_gpu=False,
                             license_file=tmp_path / "lic" / "license.txt", seg_only=False)
    assert "--gpus" not in cmd
    assert "/fs_license/license.txt" in cmd and "--seg_only" not in cmd
