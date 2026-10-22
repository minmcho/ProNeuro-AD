"""Run the Stage 1 baseline: amyloid classifier + MCI conversion survival model."""

import argparse
import json
from pathlib import Path

from brainprog.config import load_config
from brainprog.data.adnimerge import (
    amyloid_labels,
    baseline_features,
    conversion_labels,
    load_adnimerge,
)
from brainprog.models.baseline import amyloid_cv, conversion_cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(parser.parse_args().config)

    df = load_adnimerge(cfg["data"]["adnimerge_csv"])
    X = baseline_features(df, cfg["features"]["volumes"], cfg["features"]["covariates"])
    seed, k = cfg["eval"]["seed"], cfg["eval"]["n_splits"]

    results = {
        "amyloid": amyloid_cv(X, amyloid_labels(df, cfg["amyloid"]["av45_cutoff"]), k, seed),
        "conversion": conversion_cv(
            X,
            conversion_labels(df, cfg["conversion"]["baseline_dx"],
                              cfg["conversion"]["horizon_months"]),
            k, seed,
        ),
    }

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
