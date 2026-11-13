"""
Pre-flight validator for the LISA Task 1a submission CSV.

Checks the file against the challenge spec BEFORE you spend one of your 2/day submission
slots: required filename, the 8 columns in the right order, integer grades in {0,1,2}, no
NaN/dupes, and -- if --val_dir is given -- that every validation image has exactly one row
and no stray rows. Exit code 0 = looks submittable, 1 = problems found.

    python lofi_radio/validate_submission.py LISA_LF_QC_predictions.csv --val_dir Task1Validation
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ARTIFACTS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
EXPECTED_NAME = "LISA_LF_QC_predictions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--val_dir", default=None, help="cross-check rows against the validation .nii.gz set")
    ap.add_argument("--id_col", default="filename", help="expected first-column header")
    args = ap.parse_args()
    csv = Path(args.csv)
    errs, warns = [], []

    if csv.name != EXPECTED_NAME:
        errs.append(f"file is named {csv.name!r}; the system expects exactly {EXPECTED_NAME!r}")
    if not csv.exists():
        print(f"FAIL: {csv} does not exist"); sys.exit(1)

    df = pd.read_csv(csv)

    # columns: first = id, then the 7 artifacts in order
    expected = [args.id_col] + ARTIFACTS
    if list(df.columns) != expected:
        errs.append(f"columns are {list(df.columns)}\n        expected {expected}")
    if df.shape[1] != 8:
        errs.append(f"{df.shape[1]} columns; expected 8")

    # grades: integer-valued, in {0,1,2}, no NaN
    for a in ARTIFACTS:
        if a not in df.columns:
            continue
        col = df[a]
        if col.isna().any():
            errs.append(f"{a}: has {int(col.isna().sum())} NaN/blank values")
            continue
        if not ((col == col.round()).all()):
            errs.append(f"{a}: has non-integer values")
        bad = sorted(set(col.dropna().unique()) - {0, 1, 2})
        if bad:
            errs.append(f"{a}: out-of-range grades {bad} (must be 0/1/2)")

    # ids: present, unique
    if args.id_col in df.columns:
        ids = df[args.id_col].astype(str)
        if ids.isna().any() or (ids.str.strip() == "").any():
            errs.append(f"{args.id_col}: blank id(s)")
        dups = ids[ids.duplicated()].unique().tolist()
        if dups:
            errs.append(f"{args.id_col}: {len(dups)} duplicate id(s), e.g. {dups[:3]}")

    # cross-check against the validation directory
    if args.val_dir and args.id_col in df.columns:
        val = sorted(p.name for p in Path(args.val_dir).glob("*.nii.gz"))
        if not val:
            warns.append(f"--val_dir {args.val_dir} has no .nii.gz files; skipped cross-check")
        else:
            sub_ids = set(df[args.id_col].astype(str))
            val_ids = set(val)
            missing = sorted(val_ids - sub_ids)        # in val set, no prediction
            extra = sorted(sub_ids - val_ids)          # predicted, not in val set
            if missing:
                errs.append(f"{len(missing)} validation image(s) have NO row, e.g. {missing[:3]}")
            if extra:
                errs.append(f"{len(extra)} row(s) are not in the validation set, e.g. {extra[:3]}")
            if not missing and not extra:
                print(f"  cross-check OK: {len(val)} validation images all present, no extras")

    print(f"\nrows: {len(df)}  columns: {list(df.columns)}")
    print("grade counts (0/1/2):")
    for a in ARTIFACTS:
        if a in df.columns:
            vc = df[a].value_counts().reindex([0, 1, 2], fill_value=0)
            print(f"  {a:12s} {int(vc[0]):4d} {int(vc[1]):4d} {int(vc[2]):4d}")

    for w in warns:
        print(f"WARN: {w}")
    if errs:
        print("\n".join(f"FAIL: {e}" for e in errs))
        print(f"\n{len(errs)} problem(s) -- DO NOT submit yet.")
        sys.exit(1)
    print("\nPASS: submission looks valid." + ("" if args.val_dir else
          "  (run with --val_dir to also confirm all validation images are covered.)"))


if __name__ == "__main__":
    main()
