#!/usr/bin/env python3
"""
Simplified utility: read PTID and IMAGEUID from ADNIMERGE.csv and locate matching NIfTI files
by filename using a simple glob pattern like "*{PTID}*{IMAGEUID}*.nii*".

No date or other preprocessing is performed.
"""
import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


def find_matches(nii_root: Path, ptid: str, imageuid: str) -> List[str]:
    """Return sorted list of file paths under nii_root matching *PTID*IMAGEUID*.nii*"""
    # The pattern below mirrors the suggested shell form: ls *PTID**IMAGEUID*
    # We also constrain to NIfTI extensions by adding .nii*
    pattern = f"*{ptid}*{imageuid}*.nii*"
    return sorted(str(p) for p in nii_root.rglob(pattern) if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Locate NIfTI images by PTID and IMAGEUID from ADNIMERGE.csv using simple filename globbing. "
            "No preprocessing of the CSV values is performed."
        )
    )
    ap.add_argument(
        "--adnimerge_csv",
        type=Path,
        default=Path("./data/tabular/ADNIMERGE.csv"),
        help="Path to ADNIMERGE.csv (must include PTID and IMAGEUID columns)",
    )
    ap.add_argument(
        "--nii",
        type=Path,
        default=Path("./data/nii"),
        help="Root directory to search for NIfTI files (searched recursively)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally limit to the first N rows of ADNIMERGE for a quick check (0 means no limit; default)",
    )
    ap.add_argument(
        "--unique",
        action="store_true",
        help="Only output rows that resolve to exactly one matching file",
    )
    ap.add_argument(
        "--sep",
        default=",",
        help="Field separator for stdout (default: ',')",
    )
    args = ap.parse_args()

    # Read only the columns we need, without any normalization
    try:
        df = pd.read_csv(args.adnimerge_csv, dtype=str, usecols=["PTID", "IMAGEUID"]).fillna("")
    except ValueError:
        # Fallback if pandas can't select usecols due to case differences
        df_all = pd.read_csv(args.adnimerge_csv, dtype=str).fillna("")
        cols = {c.lower(): c for c in df_all.columns}
        if "ptid" not in cols or "imageuid" not in cols:
            raise SystemExit(f"{args.adnimerge_csv} must contain PTID and IMAGEUID columns")
        df = df_all[[cols["ptid"], cols["imageuid"]]].rename(columns={cols["ptid"]: "PTID", cols["imageuid"]: "IMAGEUID"})

    if args.limit > 0:
        df = df.head(args.limit)

    # Output header
    sep = args.sep
    print(f"PTID{sep}IMAGEUID{sep}MATCH_COUNT{sep}PATHS")

    total = 0
    matched = 0
    for ptid, imageuid in df.itertuples(index=False):
        if not ptid or not imageuid:
            continue
        total += 1
        matches = find_matches(args.nii, ptid, imageuid)
        if args.unique and len(matches) != 1:
            continue
        if matches:
            matched += 1
        paths = " | ".join(matches)
        print(f"{ptid}{sep}{imageuid}{sep}{len(matches)}{sep}{paths}")

    print(f"\nSearched {total} row(s); resolved {matched} with >=1 match.", file=sys.stderr)


if __name__ == "__main__":
    main()
