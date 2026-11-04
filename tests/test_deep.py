"""Stage 3 tests on tiny synthetic brains (CPU, no pretrained download)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from proneuro.deep.model import (TriPlanarViT, discrete_time_loss, extract_slices,  # noqa: E402
                                 risk_score, time_bin)
from proneuro.deep.volumes import preprocess  # noqa: E402

TINY = dict(backbone="vit_tiny_patch16_224", pretrained=False, img_size=32, patch_size=8,
            slices_per_plane=4, fusion_layers=1, time_bins=5)


def test_preprocess_shapes_and_normalization():
    img = np.random.default_rng(0).normal(100, 20, (60, 60, 60)).astype(np.float32)
    mask = np.zeros_like(img, dtype=bool)
    mask[10:50, 15:45, 12:48] = True
    v = preprocess(img, mask, size=32)
    assert v.shape == (32, 32, 32) and v.dtype == np.float16
    assert abs(float(v[v != 0].mean())) < 0.5


def test_slices_and_forward():
    vol = torch.randn(2, 32, 32, 32)
    assert extract_slices(vol, 0, 4).shape == (2, 4, 3, 32, 32)
    model = TriPlanarViT(**TINY)
    out = model(vol)
    assert out["amyloid_logit"].shape == (2,) and out["hazard_logits"].shape == (2, 5)
    assert len(model.backbones) == 3  # three parallel branches


def test_discrete_time_loss():
    assert time_bin(torch.tensor([1.0, 12.0, 13.0, 60.0, 90.0]), 60, 5).tolist() == [0, 0, 1, 4, 4]
    logits = torch.zeros(3, 5, requires_grad=True)
    loss = discrete_time_loss(logits, torch.tensor([10.0, 30.0, 60.0]),
                              torch.tensor([1.0, 0.0, 1.0]), 60)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    assert risk_score(torch.tensor([[2.0] * 5, [-2.0] * 5]))[0] > risk_score(
        torch.tensor([[2.0] * 5, [-2.0] * 5]))[1]


def _synthetic(tmp_path: Path):
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 40
    rids = list(range(1, n + 1))
    sev = rng.normal(size=n)
    vol_dir = tmp_path / "vols"
    vol_dir.mkdir()
    for rid, s in zip(rids, sev):
        v = rng.normal(0, 1, (32, 32, 32)).astype(np.float32)
        v[8:24, 8:24, 8:24] += s  # "atrophy" signal
        np.save(vol_dir / f"{rid}.npy", v.astype(np.float16))
    idx = pd.Index(rids)
    d = {
        "X": pd.DataFrame({"age": rng.normal(70, 5, n)}, index=idx),
        "regions": pd.DataFrame({"fs_a": sev + rng.normal(0, .3, n), "fs_b": rng.normal(size=n)},
                                index=idx),
        "baseline_dx": pd.Series(np.where(np.arange(n) % 2, 2, 1), index=idx),
        "amyloid": pd.Series((sev > 0).astype(float), index=idx),
        "conversion": pd.DataFrame({"duration": rng.uniform(6, 60, n // 2),
                                    "event": (sev[1::2] > 0).astype(int)}, index=idx[1::2]),
    }
    cfg = {"seed": 0, "n_splits": 2, "horizon_months": 60,
           "data": {"feature_dir": str(tmp_path / "feats")},
           "frozen": {"epochs": 5, "patience": 3, "batch_size": 8, "lr": 1e-3,
                      "weight_decay": 0.0},
           "model": {**{k: TINY[k] for k in ("backbone", "pretrained", "img_size", "slices_per_plane",
                                            "fusion_layers", "time_bins")},
                     "shared_backbone": False, "dropout": 0.0,
                     "backbone_kwargs": {"patch_size": 8}},
           "train": {"epochs": 2, "patience": 5, "batch_size": 4, "accumulate": 2,
                     "lr_backbone": 1e-4, "lr_head": 1e-3, "weight_decay": 0.0,
                     "amyloid_weight": 1.0, "val_fraction": 0.2, "amp": False, "num_workers": 0}}
    return cfg, d, vol_dir


def test_cross_validate_finetune(tmp_path: Path):
    from proneuro.deep.train import cross_validate

    cfg, d, vol_dir = _synthetic(tmp_path)
    cfg["model"]["freeze_backbone"] = False
    summary = cross_validate(cfg, d, vol_dir, tmp_path / "out", torch.device("cpu"), folds=[1])
    assert summary["mode"] == "finetune" and summary["folds_run"] == 1
    assert "amyloid_auc" in summary["tri_planar_vit"]
    assert "amyloid_auc" in summary["freesurfer_ridge_same_folds"]
    assert (tmp_path / "out" / "summary.json").exists()


def test_cross_validate_frozen(tmp_path: Path):
    from proneuro.deep.train import cross_validate

    cfg, d, vol_dir = _synthetic(tmp_path)
    cfg["model"]["freeze_backbone"] = True
    summary = cross_validate(cfg, d, vol_dir, tmp_path / "out", torch.device("cpu"), folds=[1])
    assert summary["mode"] == "frozen" and summary["folds_run"] == 1
    assert "amyloid_auc" in summary["tri_planar_vit"]
    feats = list((tmp_path / "feats").rglob("*.npy"))
    assert len(feats) == 40 and np.load(feats[0]).shape == (2, 3, 4, 192)
