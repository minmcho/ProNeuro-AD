"""Pick one baseline T1 MRI per participant and list the image IDs to download from LONI.

Uses the same baseline scan as the Stage 1 tables (first scan in ADSP_PHC_T1_FS),
matched to the LONI scan list by subject ID and scan date.

    python scripts/select_baseline_mri.py --scan-list <ADNIMERGE_Key_MRI_*.csv or Tables_*.zip>

Writes to data/processed/mri_selection/:
  baseline_t1_selection.csv   one row per participant (image ID, date, sequence, diagnosis)
  image_ids_part*.txt         comma-separated image IDs, in batches, to paste into LONI search
"""

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from proneuro.data.adsp import build_dataset

DX_NAMES = {1: "CN", 2: "MCI", 3: "AD", 4: "other_dementia", 5: "other"}


def read_scan_list(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if "Key_MRI" in n and n.endswith(".csv"))
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
    return pd.read_csv(path, low_memory=False)


def rank_t1(scans: pd.DataFrame) -> pd.DataFrame:
    """Order candidate T1 scans: 3D first, non-repeat, non-accelerated, 3T."""
    desc = scans["series_description"].fillna("").str.lower()
    scans = scans.assign(
        _is3d=(scans["acquisition_type"] == "3D").astype(int),
        _repeat=desc.str.contains("repeat").astype(int),
        _accel=(desc.str.contains("accel|grappa|sense") | scans["acceleration"].notna()).astype(int),
        _field=scans["magnetic_field_strength"].fillna(0),
    )
    return scans.sort_values(["_is3d", "_repeat", "_accel", "_field", "image_id"],
                             ascending=[False, True, True, False, True])


def select(scan_list: pd.DataFrame, adsp_dir: str | Path) -> pd.DataFrame:
    fs = pd.read_csv(Path(adsp_dir) / "ADSP_PHC_T1_FS.csv",
                     usecols=["RID", "PTID", "PHC_SCANDATE", "PHC_Age_T1"])
    first = fs.dropna(subset=["PHC_Age_T1"]).sort_values(["RID", "PHC_Age_T1"]).drop_duplicates("RID")

    t1 = scan_list[scan_list["series_type"] == "T1w"].copy()
    t1["scan_date"] = pd.to_datetime(t1["image_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cand = first.merge(t1, left_on=["PTID", "PHC_SCANDATE"], right_on=["subject_id", "scan_date"])
    best = rank_t1(cand).drop_duplicates("RID")

    dx = build_dataset(adsp_dir)["baseline_dx"].map(DX_NAMES)
    out = best.set_index("RID")[["PTID", "scan_date", "image_id", "series_description",
                                 "acquisition_type", "magnetic_field_strength",
                                 "scanner_manufacturer", "image_visit"]]
    out["baseline_dx"] = dx.reindex(out.index)
    out["age"] = first.set_index("RID")["PHC_Age_T1"].reindex(out.index).round(1)
    return out.reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan-list", required=True, help="ADNIMERGE_Key_MRI CSV, or the Tables zip")
    p.add_argument("--adsp-dir", default="data/raw/ADNI_tables")
    p.add_argument("--out", default="data/processed/mri_selection")
    p.add_argument("--groups", default="CN,MCI,AD", help="baseline diagnoses to include")
    p.add_argument("--batch", type=int, default=500, help="image IDs per LONI search")
    args = p.parse_args()

    sel = select(read_scan_list(Path(args.scan_list)), args.adsp_dir)
    sel = sel[sel["baseline_dx"].isin(args.groups.split(","))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "baseline_t1_selection.csv", index=False)
    ids = sel["image_id"].astype(int).astype(str).tolist()
    for i in range(0, len(ids), args.batch):
        (out / f"image_ids_part{i // args.batch + 1}.txt").write_text(",".join(ids[i:i + args.batch]))

    print(f"Selected {len(sel)} baseline T1 scans -> {out}")
    print(sel["baseline_dx"].value_counts().to_string())
    print("\nField strength:", sel["magnetic_field_strength"].value_counts().to_dict())
    print("3D:", int((sel["acquisition_type"] == "3D").sum()), "of", len(sel))
    print(f"\nID batches: {(len(ids) - 1) // args.batch + 1} file(s) of up to {args.batch} IDs")


if __name__ == "__main__":
    main()
