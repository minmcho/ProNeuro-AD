"""Read-only intake check for a new ADNI download — run BEFORE FastSurfer.

    python scripts/intake_check.py --download-dir <LONI download folder>

Counts the I<image id> folders in the download, matches them against
data/processed/mri_selection/baseline_t1_selection.csv, and subtracts the
RIDs already completed in outputs/fastsurfer (same completion marker
batch_fastsurfer.py uses: mri/aparc.DKTatlas+aseg.deep.mgz).

Prints: folders found, selected scans present, already processed,
new scans FastSurfer would process, download folders not in the selection,
and selected scans still missing. Changes nothing on disk; needs no Docker.

Reading the numbers: point --download-dir at the NEW download alone and the
lines that matter are "NEW scans to process" (expect ~1,719 for the ADNI
remainder download) and "Folders not in selection" (expect 0); "Selected
scans missing" will then also count the scans from the earlier download.
Point it at a merged tree and "Selected scans missing" should be 0.
"""

import argparse
from pathlib import Path

import pandas as pd


def find_image_dirs(root: Path) -> dict[int, Path]:
    found = {}
    for p in root.rglob("I*"):
        if p.is_dir() and p.name[1:].isdigit():
            found[int(p.name[1:])] = p
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download-dir", required=True)
    p.add_argument("--selection", default="data/processed/mri_selection/baseline_t1_selection.csv")
    p.add_argument("--out", default="outputs/fastsurfer")
    args = p.parse_args()

    sel = pd.read_csv(args.selection)
    sel_ids = set(int(x) for x in sel["image_id"])
    dirs = find_image_dirs(Path(args.download_dir))
    found_ids = set(dirs)

    present = sel[sel["image_id"].isin(found_ids)]
    out = Path(args.out)
    done_rids = set()
    if out.exists():
        for rid_dir in out.iterdir():
            if rid_dir.is_dir() and (rid_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
                done_rids.add(rid_dir.name)
    present_rids = set(str(int(r)) for r in present["RID"])
    new_rids = present_rids - done_rids

    print(f"Image folders in download : {len(dirs)}")
    print(f"Selected scans (CSV rows) : {len(sel)}")
    print(f"Selected scans present    : {len(present)}")
    print(f"Already processed (RIDs)  : {len(present_rids & done_rids)}")
    print(f"NEW scans to process      : {len(new_rids)}")
    print(f"Folders not in selection  : {len(found_ids - sel_ids)}")
    print(f"Selected scans missing    : {len(sel_ids - found_ids)}")
    if found_ids - sel_ids:
        print("  unmatched image ids:", sorted(found_ids - sel_ids)[:20],
              "..." if len(found_ids - sel_ids) > 20 else "")


if __name__ == "__main__":
    main()
