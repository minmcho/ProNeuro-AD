"""Build Stage 1 features and labels from ADSP-PHC harmonized ADNI tables.

Tables (CSV, exported from the ADNIMERGE2 R package):
  ADSP_PHC_DEMODX.csv              diagnosis + demographics per cognitive visit
  ADSP_PHC_T1_FS.csv               FreeSurfer volumes per T1 MRI (ComBat-harmonized)
  ADSP_PHC_PET_Amyloid_Simple.csv  amyloid PET status per scan
  APOERES.csv                      APOE genotype

All visits are aligned on the participant's age (years), which every table carries.
Baseline = each participant's first usable T1 MRI.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# PHC_Diagnosis codes
CN, MCI, AD_DEMENTIA = 1, 2, 3

# ComBat-harmonized FreeSurfer columns -> feature name (left/right summed)
VOLUME_COLUMNS = {
    "hippocampus": ["Left_Hippocampus_combat", "Right_Hippocampus_combat"],
    "entorhinal": ["lh_entorhinal_volume_combat", "rh_entorhinal_volume_combat"],
    "parahippocampal": ["lh_parahippocampal_volume_combat", "rh_parahippocampal_volume_combat"],
    "fusiform": ["lh_fusiform_volume_combat", "rh_fusiform_volume_combat"],
    "middletemporal": ["lh_middletemporal_volume_combat", "rh_middletemporal_volume_combat"],
    "lat_ventricles": ["Left_Lateral_Ventricle_combat", "Right_Lateral_Ventricle_combat"],
    "brain": ["BrainSegVolNotVent_combat"],
}
ICV_COLUMN = "EstimatedTotalIntraCranialVol_combat"


def _read(folder: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / f"{name}.csv", low_memory=False, **kw)


def load_tables(folder: str | Path) -> dict[str, pd.DataFrame]:
    mri_cols = ["RID", "PHC_Age_T1", ICV_COLUMN] + sum(VOLUME_COLUMNS.values(), [])
    return {
        "dx": _read(folder, "ADSP_PHC_DEMODX"),
        "mri": _read(folder, "ADSP_PHC_T1_FS", usecols=mri_cols),
        "amy": _read(folder, "ADSP_PHC_PET_Amyloid_Simple"),
        "apoe": _read(folder, "APOERES"),
    }


def apoe4_count(apoe: pd.DataFrame) -> pd.Series:
    """Number of APOE e4 alleles per RID, from GENOTYPE like '3/4'."""
    g = apoe.dropna(subset=["GENOTYPE"]).drop_duplicates("RID").set_index("RID")["GENOTYPE"]
    return g.astype(str).str.count("4").astype(float)


def baseline_mri(mri: pd.DataFrame) -> pd.DataFrame:
    """First T1 MRI per participant with ICV-normalized regional volumes."""
    mri = mri.dropna(subset=["PHC_Age_T1", ICV_COLUMN]).sort_values(["RID", "PHC_Age_T1"])
    bl = mri.drop_duplicates("RID").set_index("RID")
    out = pd.DataFrame(index=bl.index)
    out["age_mri"] = bl["PHC_Age_T1"]
    for name, cols in VOLUME_COLUMNS.items():
        out[f"{name}_icv"] = bl[cols].sum(axis=1, min_count=len(cols)) / bl[ICV_COLUMN]
    return out


def _nearest(events: pd.DataFrame, age_col: str, ref_age: pd.Series, window: float) -> pd.DataFrame:
    """For each RID, the event row closest in age to ref_age, within +/- window years."""
    e = events.join(ref_age.rename("ref_age"), on="RID", how="inner")
    e["gap"] = (e[age_col] - e["ref_age"]).abs()
    e = e[e["gap"] <= window].sort_values(["RID", "gap"])
    return e.drop_duplicates("RID").set_index("RID")


def build_dataset(folder: str | Path, dx_window: float = 0.5, pet_window: float = 1.0,
                  horizon_months: float = 60) -> dict[str, pd.DataFrame | pd.Series]:
    """Return features X, baseline diagnosis, amyloid labels and MCI survival table."""
    t = load_tables(folder)
    bl = baseline_mri(t["mri"])

    dx = t["dx"].dropna(subset=["PHC_Age_Diagnosis", "PHC_Diagnosis"])
    bl_dx = _nearest(dx, "PHC_Age_Diagnosis", bl["age_mri"], dx_window)

    X = bl.copy()
    X["female"] = (bl_dx["PHC_Sex"] == 2).astype(float).reindex(X.index)
    X["education"] = bl_dx["PHC_Education"].reindex(X.index)
    X["apoe4"] = apoe4_count(t["apoe"]).reindex(X.index)
    X = X.rename(columns={"age_mri": "age"})
    X = X.loc[X.index.intersection(bl_dx.index)]
    baseline_dx = bl_dx["PHC_Diagnosis"].reindex(X.index)

    # Amyloid status from the PET scan nearest the baseline MRI (0 = neg, 1 = pos; 9 = excluded)
    amy = t["amy"][t["amy"]["PHC_AMYLOID_STATUS"].isin([0, 1])].dropna(subset=["PHC_Age_PET"])
    near_pet = _nearest(amy, "PHC_Age_PET", bl["age_mri"], pet_window)
    amyloid = near_pet["PHC_AMYLOID_STATUS"].astype(float).reindex(X.index)

    # MCI -> Alzheimer's dementia conversion, time from baseline MRI (months)
    rows = []
    mci_ids = baseline_dx.index[baseline_dx == MCI]
    later = dx[dx["RID"].isin(mci_ids)].join(bl["age_mri"], on="RID")
    later = later[later["PHC_Age_Diagnosis"] > later["age_mri"]]
    for rid, v in later.groupby("RID"):
        months = (v["PHC_Age_Diagnosis"] - v["age_mri"]) * 12
        conv = months[v["PHC_Diagnosis"] == AD_DEMENTIA]
        duration, event = (conv.min(), 1) if len(conv) else (months.max(), 0)
        if duration > horizon_months:
            duration, event = horizon_months, 0
        if duration > 0:
            rows.append({"RID": rid, "duration": float(duration), "event": int(event)})
    surv = pd.DataFrame(rows, columns=["RID", "duration", "event"]).set_index("RID")

    return {"X": X.replace([np.inf, -np.inf], np.nan), "baseline_dx": baseline_dx,
            "amyloid": amyloid, "conversion": surv}
