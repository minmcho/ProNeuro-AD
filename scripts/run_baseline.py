"""Run the Stage 1 baseline: amyloid classifier + MCI conversion survival model.

    python scripts/run_baseline.py --config configs/baseline_adsp.yaml   # ADSP tables (recommended)
    python scripts/run_baseline.py --config configs/baseline.yaml        # legacy ADNIMERGE.csv
    add --data-dir <folder> to point at the ADSP CSV folder without editing the config
"""

import argparse
import json
from pathlib import Path

from proneuro.config import load_config
from proneuro.models.baseline import amyloid_cv, conversion_cv


def run_adsp(cfg: dict, data_dir: str | None) -> dict:
    from proneuro.data.adsp import build_dataset

    m = cfg["matching"]
    d = build_dataset(data_dir or cfg["data"]["adsp_dir"], m["dx_window_years"],
                      m["pet_window_years"], cfg["conversion"]["horizon_months"])
    X, k, seed = d["X"], cfg["eval"]["n_splits"], cfg["eval"]["seed"]
    counts = d["baseline_dx"].map({1: "CN", 2: "MCI", 3: "AD", 4: "other_dementia",
                                   5: "other"}).value_counts().to_dict()
    return {
        "participants_with_baseline_mri": int(len(X)),
        "baseline_diagnosis": counts,
        "features": list(X.columns),
        "amyloid": amyloid_cv(X, d["amyloid"], k, seed),
        "conversion": conversion_cv(X.loc[X.index.isin(d["conversion"].index)],
                                    d["conversion"], k, seed),
    }


def run_adnimerge(cfg: dict) -> dict:
    from proneuro.data.adnimerge import (amyloid_labels, baseline_features,
                                         conversion_labels, load_adnimerge)

    df = load_adnimerge(cfg["data"]["adnimerge_csv"])
    X = baseline_features(df, cfg["features"]["volumes"], cfg["features"]["covariates"])
    k, seed = cfg["eval"]["n_splits"], cfg["eval"]["seed"]
    return {
        "amyloid": amyloid_cv(X, amyloid_labels(df, cfg["amyloid"]["av45_cutoff"]), k, seed),
        "conversion": conversion_cv(
            X, conversion_labels(df, cfg["conversion"]["baseline_dx"],
                                 cfg["conversion"]["horizon_months"]), k, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_adsp.yaml")
    parser.add_argument("--data-dir", default=None, help="folder with the ADSP CSVs")
    args = parser.parse_args()
    cfg = load_config(args.config)

    results = run_adsp(cfg, args.data_dir) if cfg.get("source") == "adsp" else run_adnimerge(cfg)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
