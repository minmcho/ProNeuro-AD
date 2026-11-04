"""Tests for the FastSurfer command builder (no Docker needed)."""

from pathlib import Path

from proneuro.preprocessing.fastsurfer import fastsurfer_command


def test_gpu_seg_only_command(tmp_path: Path):
    t1 = tmp_path / "in" / "sub-01_T1w.nii.gz"
    cmd = fastsurfer_command(t1, "sub-01", tmp_path / "out")
    assert cmd[:3] == ["docker", "run", "--rm"] and "--user" in cmd
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    assert "/data/sub-01_T1w.nii.gz" in cmd
    assert "--seg_only" in cmd and "--fs_license" not in cmd


def test_cpu_with_license(tmp_path: Path):
    cmd = fastsurfer_command(tmp_path / "t1.mgz", "s1", tmp_path / "out", use_gpu=False,
                             license_file=tmp_path / "lic" / "license.txt", seg_only=False)
    assert "--gpus" not in cmd
    assert "/fs_license/license.txt" in cmd and "--seg_only" not in cmd


def test_multiframe_nifti_combined_to_3d(tmp_path: Path):
    import importlib.util
    import sys

    import nibabel as nib
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "batch_fastsurfer", Path(__file__).parents[1] / "scripts" / "batch_fastsurfer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["batch_fastsurfer"] = mod
    spec.loader.exec_module(mod)

    four_d = tmp_path / "I1.nii.gz"
    nib.save(nib.Nifti1Image(np.full((8, 8, 8, 3), 2.0, np.float32), np.eye(4)), str(four_d))
    out = mod.to_3d(four_d, tmp_path / "nifti")
    img = nib.load(str(out))
    assert img.ndim == 3 and np.allclose(img.get_fdata(), 2.0)

    three_d = tmp_path / "I2.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4)), str(three_d))
    assert mod.to_3d(three_d, tmp_path / "nifti") == three_d
