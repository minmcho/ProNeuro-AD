"""Load ADNIMERGE and build Stage 1 labels.

ADNIMERGE has one row per subject visit (RID, VISCODE, M = months from baseline,
DX = diagnosis: CN / MCI / Dementia, plus FreeSurfer volumes and PET summaries).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_adnimerge(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    return df


def baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: the baseline (VISCODE == 'bl') visit."""
    return df[df["VISCODE"] == "bl"].drop_duplicates("RID").set_index("RID")


def amyloid_labels(df: pd.DataFrame, av45_cutoff: float) -> pd.Series:
    """Amyloid positivity at baseline from florbetapir (AV45) SUVR. NaN if no PET."""
    bl = baseline_rows(df)
    suvr = pd.to_numeric(bl.get("AV45"), errors="coerce")
    return (suvr >= av45_cutoff).astype(float).where(suvr.notna())


def conversion_labels(
    df: pd.DataFrame, baseline_dx: str = "MCI", horizon_months: float = 60
) -> pd.DataFrame:
    """Time-to-event table for baseline-MCI subjects.

    Returns columns: duration (months), event (1 = converted to Dementia).
    Censored at last follow-up or at the horizon.
    """
    bl = baseline_rows(df)
    rids = bl.index[bl["DX"] == baseline_dx]
    rows = []
    for rid, visits in df[df["RID"].isin(rids)].groupby("RID"):
        visits = visits.dropna(subset=["M"]).sort_values("M")
        converted = visits[visits["DX"] == "Dementia"]
        if len(converted):
            duration, event = converted["M"].iloc[0], 1
        else:
            duration, event = visits["M"].max(), 0
        if duration > horizon_months:
            duration, event = horizon_months, 0
        if duration > 0:
            rows.append({"RID": rid, "duration": float(duration), "event": int(event)})
    return pd.DataFrame(rows).set_index("RID")


def baseline_features(df: pd.DataFrame, volumes: list[str], covariates: list[str]) -> pd.DataFrame:
    """Baseline volumes normalized by ICV, plus covariates (sex encoded 0/1)."""
    bl = baseline_rows(df)
    icv = pd.to_numeric(bl["ICV"], errors="coerce")
    feats = pd.DataFrame(index=bl.index)
    for col in volumes:
        feats[f"{col}_icv"] = pd.to_numeric(bl[col], errors="coerce") / icv
    for col in covariates:
        if col == "PTGENDER":
            feats["female"] = (bl[col] == "Female").astype(float)
        else:
            feats[col] = pd.to_numeric(bl[col], errors="coerce")
    return feats.replace([np.inf, -np.inf], np.nan)
