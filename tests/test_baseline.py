"""Tests on synthetic ADNIMERGE-like data (no real patient data)."""

import numpy as np
import pandas as pd

from brainprog.data.adnimerge import amyloid_labels, baseline_features, conversion_labels
from brainprog.models.baseline import amyloid_cv, conversion_cv

VOLS = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
COVS = ["AGE", "PTEDUCAT", "PTGENDER", "APOE4"]


def synthetic_adnimerge(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(n):
        atrophy = rng.normal()
        dx = rng.choice(["CN", "MCI"], p=[0.4, 0.6])
        base = {
            "RID": rid, "AGE": rng.normal(73, 7), "PTEDUCAT": rng.integers(8, 20),
            "PTGENDER": rng.choice(["Male", "Female"]), "APOE4": rng.integers(0, 3),
            "ICV": rng.normal(1.5e6, 1e5),
            "AV45": 1.1 + 0.15 * atrophy + rng.normal(0, 0.05),
        }
        for v in VOLS:
            base[v] = rng.normal(5000, 500) * (1 - 0.05 * atrophy)
        convert_m = 12 * (3 - atrophy) if dx == "MCI" and atrophy > 0 else None
        for m in [0, 6, 12, 24, 36, 48]:
            vis_dx = "Dementia" if convert_m is not None and m >= convert_m else dx
            rows.append({**base, "VISCODE": "bl" if m == 0 else f"m{m:02d}", "M": m, "DX": vis_dx})
    return pd.DataFrame(rows)


def test_labels_and_features():
    df = synthetic_adnimerge()
    X = baseline_features(df, VOLS, COVS)
    assert len(X) == 200 and "Hippocampus_icv" in X and "female" in X
    y = amyloid_labels(df, 1.11)
    assert set(y.dropna().unique()) <= {0.0, 1.0}
    surv = conversion_labels(df, "MCI", 60)
    assert (surv["duration"] > 0).all() and surv["event"].isin([0, 1]).all()


def test_models_run_and_beat_chance():
    df = synthetic_adnimerge()
    X = baseline_features(df, VOLS, COVS)
    amy = amyloid_cv(X, amyloid_labels(df, 1.11), n_splits=3)
    conv = conversion_cv(X, conversion_labels(df), n_splits=3)
    assert amy["auc_mean"] > 0.6
    assert conv["cindex_mean"] > 0.6
