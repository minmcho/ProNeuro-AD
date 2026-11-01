"""ADSP loader tests on synthetic tables (no real patient data)."""

from pathlib import Path

import numpy as np
import pandas as pd

from proneuro.data.adsp import ICV_COLUMN, VOLUME_COLUMNS, build_dataset
from proneuro.models.baseline import amyloid_cv, conversion_cv


def write_synthetic(folder: Path, n: int = 240, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    dx, mri, amy, apoe = [], [], [], []
    for rid in range(1, n + 1):
        atrophy = rng.normal()
        age0 = rng.normal(73, 6)
        MCI = 2
        base = MCI if rng.random() < 0.6 else 1
        row = {"RID": rid, "PHC_Age_T1": age0, ICV_COLUMN: rng.normal(1.5e6, 1e5)}
        for cols in VOLUME_COLUMNS.values():
            for c in cols:
                row[c] = rng.normal(4000, 300) * (1 - 0.06 * atrophy)
        mri.append(row)
        conv_year = 3 - atrophy if (base == MCI and atrophy > 0) else None
        for yr in [0, 1, 2, 3, 4]:
            d = 3 if conv_year is not None and yr >= conv_year else base
            dx.append({"RID": rid, "PHC_Age_Diagnosis": age0 + yr + 0.01, "PHC_Diagnosis": d,
                       "PHC_Sex": rng.integers(1, 3), "PHC_Education": rng.integers(10, 20)})
        amy.append({"RID": rid, "PHC_Age_PET": age0 + 0.2,
                    "PHC_AMYLOID_STATUS": int(atrophy + rng.normal(0, 0.5) > 0)})
        apoe.append({"RID": rid, "GENOTYPE": rng.choice(["3/3", "3/4", "4/4"])})
    pd.DataFrame(dx).to_csv(folder / "ADSP_PHC_DEMODX.csv", index=False)
    pd.DataFrame(mri).to_csv(folder / "ADSP_PHC_T1_FS.csv", index=False)
    pd.DataFrame(amy).to_csv(folder / "ADSP_PHC_PET_Amyloid_Simple.csv", index=False)
    pd.DataFrame(apoe).to_csv(folder / "APOERES.csv", index=False)


def test_build_dataset_and_models(tmp_path: Path):
    write_synthetic(tmp_path)
    d = build_dataset(tmp_path)
    X = d["X"]
    assert len(X) == 240 and {"hippocampus_icv", "apoe4", "female", "age"} <= set(X.columns)
    assert set(d["amyloid"].dropna().unique()) <= {0.0, 1.0}
    surv = d["conversion"]
    assert len(surv) > 50 and surv["event"].sum() > 5 and (surv["duration"] > 0).all()
    assert amyloid_cv(X, d["amyloid"], n_splits=3)["auc_mean"] > 0.6
    assert conversion_cv(X.loc[X.index.isin(surv.index)], surv, n_splits=3)["cindex_mean"] > 0.6


def test_run_baseline_comparison(tmp_path: Path):
    import importlib.util
    import sys

    write_synthetic(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "run_baseline", Path(__file__).parents[1] / "scripts" / "run_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_baseline"] = mod
    spec.loader.exec_module(mod)
    cfg = {"matching": {"dx_window_years": 0.5, "pet_window_years": 1.0},
           "conversion": {"horizon_months": 60}, "eval": {"n_splits": 3, "seed": 0}}
    res = mod.run_adsp(cfg, str(tmp_path))
    assert {"MRI only", "MRI all regions (ridge)", "MRI all regions (boosting)"} <= set(
        res["comparison"])
    assert res["n_region_features"] > 10
    assert res["comparison"]["MRI only"]["amyloid_auc"] > 0.6


def test_select_baseline_mri(tmp_path: Path):
    import importlib.util
    import sys

    write_synthetic(tmp_path)
    fs = pd.read_csv(tmp_path / "ADSP_PHC_T1_FS.csv")
    fs["PTID"] = "S" + fs["RID"].astype(str)
    fs["PHC_SCANDATE"] = "2020-01-01"
    fs.to_csv(tmp_path / "ADSP_PHC_T1_FS.csv", index=False)
    scans = pd.DataFrame({
        "subject_id": list(fs["PTID"]) * 2, "image_date": "2020-01-01",
        "image_id": list(range(1000, 1000 + len(fs))) + list(range(5000, 5000 + len(fs))),
        "series_type": "T1w", "acquisition_type": ["3D"] * len(fs) + ["2D"] * len(fs),
        "series_description": "MPRAGE", "acceleration": None,
        "magnetic_field_strength": 3.0, "scanner_manufacturer": "X", "image_visit": "bl"})
    spec = importlib.util.spec_from_file_location(
        "select_baseline_mri", Path(__file__).parents[1] / "scripts" / "select_baseline_mri.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_baseline_mri"] = mod
    spec.loader.exec_module(mod)
    sel = mod.select(scans, tmp_path)
    assert len(sel) == 240 and (sel["image_id"] < 5000).all()  # 3D chosen over 2D
    assert set(sel["baseline_dx"]) <= {"CN", "MCI", "AD"}
