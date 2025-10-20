import pandas as pd
import os
from typing import List

#
# Merge ADNI tabular + MRI QC + cognitive composites with strict visit alignment
#

# Input paths (as described in preprocessing/README.md)
PATH_ADNIMERGE = "./data/tabular/ADNIMERGE.csv"
PATH_MRI_QC = "./data/tabular/MRINCLUSIO_10Sep2025.csv"
PATH_UWNPSYCH = "./data/tabular/UWNPSYCHSUM_03Sep2025.csv"
MRI_MAPPING = "./data/tabular/ADSP-PHC__ADNI_T1.csv"

# Columns to drop from the final merged output (mirrors the initial version behavior)
DROP_COLUMNS = [
    "EcogPtMem",
    "EcogPtLang",
    "EcogPtVisspat",
    "EcogPtPlan",
    "EcogPtOrgan",
    "EcogPtDivatt",
    "EcogPtTotal",
    "EcogSPMem",
    "EcogSPLang",
    "EcogSPVisspat",
    "EcogSPPlan",
    "EcogSPOrgan",
    "EcogSPDivatt",
    "EcogSPTotal",
    "FLDSTRENG",
    "FSVERSION",
    "FDG",
    "PIB",
    "AV45",
    "FBB",
    "Phase",
    "MOCA",
    "PTETHCAT",
    "ABETA40",
    "COLPROT",
    "ORIGPROT",
    "SITE",
    "VISCODE",
    "subject_date",
    "DX_bl",
    "TAU_adni",
    "PTAU_adni",
    "CDRSB_data",
    "DX",
    "mPACCdigit",
    "mPACCtrailsB",
    "EXAMDATE_bl",
    "CDRSB_bl",
    "ADAS11_bl",
    "ADAS13_bl",
    "ADASQ4_bl",
    "MMSE_bl",
    "RAVLT_immediate_bl",
    "RAVLT_learning_bl",
    "RAVLT_forgetting_bl",
    "RAVLT_perc_forgetting_bl",
    "LDELTOTAL_BL",
    "DIGITSCOR_bl",
    "TRABSCOR_bl",
    "FAQ_bl",
    "mPACCdigit_bl",
    "mPACCtrailsB_bl",
    "FLDSTRENG_bl",
    "FSVERSION_bl",
    "IMAGEUID_bl",
    "Ventricles_bl",
    "Hippocampus_bl",
    "WholeBrain_bl",
    "Entorhinal_bl",
    "Fusiform_bl",
    "MidTemp_bl",
    "ICV_bl",
    "MOCA_bl",
    "EcogPtMem_bl",
    "EcogPtLang_bl",
    "EcogPtVisspat_bl",
    "EcogPtPlan_bl",
    "EcogPtOrgan_bl",
    "EcogPtDivatt_bl",
    "EcogPtTotal_bl",
    "EcogSPMem_bl",
    "EcogSPLang_bl",
    "EcogSPVisspat_bl",
    "EcogSPPlan_bl",
    "EcogSPOrgan_bl",
    "EcogSPDivatt_bl",
    "EcogSPTotal_bl",
    "ABETA_bl",
    "TAU_bl",
    "PTAU_bl",
    "FDG_bl",
    "PIB_bl",
    "AV45_bl",
    "FBB_bl",
    "Years_bl",
    "Month_bl",
    "Month",
    "update_stamp",
    "MMSCORE",
    "FAQ",
    "ABETA42",
    # Drop RID from final output; used only for joining
    "RID",
    "VSHEIGHT",
    "VSHTUNIT",
    "VSWEIGHT",
    "VSWTUNIT",
    "VSTEMP",
    "VSTMPUNT",
    "VSTMPSRC",
    "ADAS11",
    "ADASQ4",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
]


def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace('"', '', regex=False).str.strip().str.lower()


def _norm_viscode2_token(s: pd.Series) -> pd.Series:
    """Normalize VISCODE/VISCODE2 tokens.
    - Lowercase + strip + remove quotes
    - Map 'bl' to 'bl' for a single baseline convention
    """
    t = _norm_str_series(s)
    return t.replace({'bl': 'bl'})


def _drop_ignored_columns_with_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    if not DROP_COLUMNS:
        return df
    cols_to_drop = []
    for col in df.columns:
        base = col
        for suf in ("_from_qc", "_from_psych", "_from_meds", "_x", "_y"):
            if col.endswith(suf):
                base = col[: -len(suf)]
                break
        if base in DROP_COLUMNS:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore')


def _ensure_unique_by(df: pd.DataFrame, keys: List[str], table_name: str) -> pd.DataFrame:
    """Drop exact duplicate rows for the given key(s) to prevent one-to-many expansion.
    No aggregation or feature engineering; keeps the first occurrence.
    """
    before = len(df)
    # Drop rows where any key is missing, as they cannot align by visit
    df = df.dropna(subset=keys)
    df = df.drop_duplicates(subset=keys, keep='first')
    after = len(df)
    if before != after:
        print(f"{table_name}: dropped {before - after} duplicate/unkeyed rows to enforce unique {keys}.")
    return df


def main():
    # 1) Load ADNIMERGE (base table). IMAGEUID can be used to locate MRI files.
    adnimerge_df = pd.read_csv(PATH_ADNIMERGE, low_memory=False)

    # Normalize keys used for joining
    if 'RID' in adnimerge_df.columns:
        adnimerge_df['RID'] = pd.to_numeric(adnimerge_df['RID'], errors='coerce')
    # Keep PTID as-is for interpretability; do not use as join key
    if 'VISCODE' in adnimerge_df.columns:
        adnimerge_df['VISCODE'] = _norm_viscode2_token(adnimerge_df['VISCODE'])

    # Make sure IMAGEUID is an int for downstream tasks
    if 'IMAGEUID' in adnimerge_df.columns:
        adnimerge_df['IMAGEUID'] = pd.to_numeric(adnimerge_df['IMAGEUID'], errors='coerce')

    # Prefer VISCODE2 if present, else fall back to VISCODE
    if 'VISCODE2' in adnimerge_df.columns:
        adnimerge_df['VISCODE2'] = _norm_viscode2_token(adnimerge_df['VISCODE2'])
    else:
        adnimerge_df['VISCODE2'] = _norm_viscode2_token(adnimerge_df.get('VISCODE', pd.Series(index=adnimerge_df.index, dtype=object)))

    # Start merged_df from ADNIMERGE only (no extra preprocessing)
    merged_df = adnimerge_df.copy()
    base_pairs = merged_df[['RID', 'VISCODE2']].dropna().drop_duplicates()
    print(f"ADNIMERGE: {len(merged_df):,} rows | unique (RID,VISCODE2): {len(base_pairs):,}")

    # 2) MRI QC: align strictly by RID+VISCODE2 (keep original columns)
    mri_qc_df = pd.read_csv(PATH_MRI_QC)
    # Our dataset is only concerned with 1.5 T field strength
    mri_qc_df = mri_qc_df[mri_qc_df['FLDSTRENG'] != 3.0]

    # Drop all unneccary columns
    mri_qc_df = mri_qc_df[["RID", "VISCODE2", "PASS"]]

    if 'RID' in mri_qc_df.columns:
        mri_qc_df['RID'] = pd.to_numeric(mri_qc_df['RID'], errors='coerce')
    if 'VISCODE2' in mri_qc_df.columns:
        mri_qc_df['VISCODE2'] = _norm_viscode2_token(mri_qc_df['VISCODE2'])
        right_on = ['RID', 'VISCODE2']
    elif 'VISCODE' in mri_qc_df.columns:
        mri_qc_df['viscode2_from_qc'] = _norm_viscode2_token(mri_qc_df['VISCODE'])
        right_on = ['RID', 'viscode2_from_qc']
    else:
        mri_qc_df['viscode2_from_qc'] = pd.Series(index=mri_qc_df.index, dtype=object)
        right_on = ['RID', 'viscode2_from_qc']

    # Enforce uniqueness on right before merging to avoid multiplicative joins
    if right_on == ['RID', 'VISCODE2']:
        mri_qc_df = _ensure_unique_by(mri_qc_df, ['RID', 'VISCODE2'], 'MRI_QC')
        merged_df = pd.merge(
            merged_df,
            mri_qc_df,
            on=['RID', 'VISCODE2'],
            how='left',
            suffixes=("", "_from_qc")
        )
    else:
        mri_qc_df = _ensure_unique_by(mri_qc_df, right_on, 'MRI_QC')
        merged_df = pd.merge(
            merged_df,
            mri_qc_df,
            left_on=['RID', 'VISCODE2'],
            right_on=right_on,
            how='left',
            suffixes=("", "_from_qc")
        )
    # Drop helper column
    if 'viscode2_from_qc' in merged_df.columns:
        merged_df = merged_df.drop(columns=['viscode2_from_qc'])
    print(f"After MRI_QC merge: {len(merged_df):,} rows")

    # 3) Cognitive composites: ADNI_MEM, ADNI_EF, ADNI_EF2 from UWNPSYCHSUM
    psych_df = pd.read_csv(PATH_UWNPSYCH)[["RID", "VISCODE2", "ADNI_MEM", "ADNI_EF", "ADNI_EF2"]]
    if 'RID' in psych_df.columns:
        psych_df['RID'] = pd.to_numeric(psych_df['RID'], errors='coerce')
    if 'VISCODE2' in psych_df.columns:
        psych_df['VISCODE2'] = _norm_viscode2_token(psych_df['VISCODE2'])
    # Enforce uniqueness on right to prevent one-to-many expansion
    psych_df = _ensure_unique_by(psych_df, ['RID', 'VISCODE2'], 'UWNPSYCH')
    merged_df = pd.merge(merged_df, psych_df, on=['RID', 'VISCODE2'], how='left', suffixes=("", "_from_psych"))
    print(f"After UWNPSYCH merge: {len(merged_df):,} rows")

    # 4) RECCMEDS is intentionally not merged here to preserve 1-row-per-visit shape.
    #    Downstream preprocessing will handle medications from the raw CSV as needed.

    # 5) Merge the MRI file Image IDs from the ADSP-PHC file to fine MRIs later on
    # NOTE: Saving the updated Image ID under IMAGEUID
    # Merge using Subject and Visit to ensure visits align (Visit maps to VISCODE, Subject maps to PTID)
    if os.path.exists(MRI_MAPPING):
        mri_mapping_columns = ['Image Data ID', 'Subject', 'Visit', 'Acq Date']
        # Some historical extracts do not contain the acquisition date; restrict to available columns only
        available_cols = [col for col in mri_mapping_columns if col in pd.read_csv(MRI_MAPPING, nrows=0).columns]
        mri_mapping_df = pd.read_csv(MRI_MAPPING, usecols=available_cols)
    else:
        print(f"Warning: {MRI_MAPPING} not found. Skipping ADSP-PHC mapping.")
        mri_mapping_df = pd.DataFrame()

    if not mri_mapping_df.empty and not {'Image Data ID', 'Subject', 'Visit'}.issubset(mri_mapping_df.columns):
        raise ValueError("ADSP-PHC mapping file must include 'Image Data ID', 'Subject', and 'Visit' columns.")

    if not mri_mapping_df.empty:
        mri_mapping_df = mri_mapping_df.rename(columns={
            'Subject': 'PTID',
            'Visit': 'visit_token',
            'Image Data ID': 'IMAGEUID_from_phc',
            'Acq Date': 'acq_date_raw',
        })

        mri_mapping_df['PTID'] = mri_mapping_df['PTID'].astype(str).str.strip()
        mri_mapping_df['visit_token'] = _norm_str_series(mri_mapping_df['visit_token'])
        if 'acq_date_raw' in mri_mapping_df.columns:
            mri_mapping_df['acq_date'] = pd.to_datetime(mri_mapping_df['acq_date_raw'], errors='coerce')
        else:
            mri_mapping_df['acq_date'] = pd.NaT

        mri_mapping_df = mri_mapping_df.dropna(subset=['PTID', 'visit_token', 'IMAGEUID_from_phc'])

    visit_alignment_df = merged_df[['RID', 'PTID', 'VISCODE2', 'EXAMDATE']].copy()
    visit_alignment_df = visit_alignment_df.dropna(subset=['PTID', 'VISCODE2', 'RID'])
    visit_alignment_df['exam_date'] = pd.to_datetime(visit_alignment_df['EXAMDATE'], errors='coerce')
    visit_alignment_df = visit_alignment_df.dropna(subset=['exam_date'])
    visit_alignment_df = visit_alignment_df.drop_duplicates(subset=['PTID', 'VISCODE2'])

    grouped_visits = visit_alignment_df.groupby('PTID')
    alignment_records = []

    if not mri_mapping_df.empty:
        for ptid, subject_rows in mri_mapping_df.groupby('PTID'):
            if ptid not in grouped_visits.groups:
                continue
            subject_visits = grouped_visits.get_group(ptid)[['VISCODE2', 'exam_date']].sort_values('exam_date')
            if subject_visits.empty:
                continue
            for _, row in subject_rows.iterrows():
                if pd.isna(row.get('acq_date')):
                    continue
                diffs = (subject_visits['exam_date'] - row['acq_date']).abs()
                if diffs.empty:
                    continue
                best_idx = diffs.idxmin()
                best_diff = diffs.loc[best_idx]
                if pd.isna(best_diff) or best_diff > pd.Timedelta(days=45):
                    continue
                alignment_records.append({
                    'PTID': ptid,
                    'VISCODE2': subject_visits.loc[best_idx, 'VISCODE2'],
                    'IMAGEUID_from_phc': row['IMAGEUID_from_phc'],
                    'match_days_offset': int(best_diff.days),
                })

    if alignment_records:
        alignment_df = pd.DataFrame(alignment_records)
        alignment_df = alignment_df.sort_values(['PTID', 'VISCODE2', 'match_days_offset'])
        alignment_df = alignment_df.drop_duplicates(subset=['PTID', 'VISCODE2'], keep='first')

        rid_lookup = visit_alignment_df[['PTID', 'VISCODE2', 'RID']].drop_duplicates()
        alignment_df = alignment_df.merge(rid_lookup, on=['PTID', 'VISCODE2'], how='left')
        alignment_df = alignment_df.dropna(subset=['RID'])

        mapped_count = alignment_df['IMAGEUID_from_phc'].notna().sum()

        alignment_df = alignment_df[['RID', 'VISCODE2', 'IMAGEUID_from_phc']]
        merged_df = merged_df.merge(alignment_df, on=['RID', 'VISCODE2'], how='left')

        if 'IMAGEUID' not in merged_df.columns:
            merged_df['IMAGEUID'] = pd.NA
        # If there is a value for IMAGEUID_from_phc we overwrite the current IMAGEUID value
        # ADNI renames their files after preprocessing and ADSP-PHC has the correct versions.
        merged_df['IMAGEUID'] = merged_df['IMAGEUID_from_phc'].fillna(merged_df['IMAGEUID'])
        merged_df = merged_df.drop(columns=['IMAGEUID_from_phc'])

        # Make sure that IMAGEUID is the same type across the board
        merged_df['IMAGEUID'] = pd.to_numeric(
            merged_df['IMAGEUID'].astype(str).str.lstrip('I'), 
            errors='coerce'
            ).astype('Int64')
        
        print(f"After ADSP-PHC merge: mapped IMAGEUID for {mapped_count:,} visits")
    else:
        print("After ADSP-PHC merge: no ADSP-PHC visits aligned within 45 days; IMAGEUID unchanged")

    # Rename 'M' to 'months_since_bl' if present
    if 'M' in merged_df.columns:
        merged_df.rename(columns={'M': 'months_since_bl'}, inplace=True)

    # Rename Age to subject_age
    if 'AGE' in merged_df.columns:
        merged_df.rename(columns={'AGE': 'subject_age'}, inplace=True)

    # Drop columns flagged to ignore (matching across known merge suffixes)
    merged_df = _drop_ignored_columns_with_suffixes(merged_df)

    # Save to canonical preprocessing path
    out_path = "./preprocessing/ADNI_merged.csv"
    merged_df.to_csv(out_path, index=False)

    # Diagnostics
    print("Merged file saved:")
    print(f" - {out_path}")
    print(f"Rows: {len(merged_df):,} | Columns: {len(merged_df.columns)}")
    # Light diagnostics only


if __name__ == "__main__":
    main()
