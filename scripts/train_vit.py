"""Fine-tune the tri-planar parallel ViT with cross-validation (Stage 3).

    python scripts/train_vit.py --config configs/vit.yaml
    python scripts/train_vit.py --config configs/vit.yaml --folds 1      # quick: one fold only

Prints and saves (outputs/vit/) per-fold and summary results, next to a FreeSurfer
ridge baseline trained on exactly the same folds.
"""

import argparse
from pathlib import Path

import torch

from proneuro.config import load_config
from proneuro.data.adsp import build_dataset
from proneuro.deep.train import cross_validate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vit.yaml")
    p.add_argument("--folds", default="", help="comma-separated fold numbers to run, e.g. 1 or 1,2")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                 "  -- no GPU found, training will be very slow"))
    m = cfg["matching"]
    d = build_dataset(cfg["data"]["adsp_dir"], m["dx_window_years"], m["pet_window_years"],
                      cfg["horizon_months"])
    folds = [int(f) for f in args.folds.split(",") if f]
    cross_validate(cfg, d, Path(cfg["data"]["volume_dir"]), Path(cfg["output_dir"]), device, folds)


if __name__ == "__main__":
    main()
