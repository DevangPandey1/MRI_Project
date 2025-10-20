import pandas as pd
import numpy as np
from imputation import train_mice_imputer, impute_with_trained_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os
import json
from typing import List, Dict, Tuple, Set
from config import (
    SCRIPT_DIR,
    ROOT_DIR,
    DATA_DIR,
    MODEL_TRAINING_DIR,
    CLINICIAN_POLICY_DIR,
    ALPACA_DIR,
    COLUMNS_TO_DROP,
    CONTINUOUS_VARS,
    ORDINAL_VARS,
    CATEGORICAL_VARS_FOR_IMPUTATION,
    DROP_VARS_FOR_IMPUTATION,
    ACTION_FEATURES,
    BYPASS_IMPUTATION,
    MRI_DATA_ROOT,
    MRI_NII_SUBDIR,
)
from utils import (
    calculate_active_medications,
    calculate_consistent_age,
    filter_subjects_by_visit_count,
    split_data_by_subject,
    get_alpaca_observation_columns,
    impute_missing_values,
    normalize_and_encode_sequence_splits,
    save_artifacts,
    transform_dataset_for_clinician,
    normalize_clinician_splits,
    normalize_ids_post_merge,
)
from clean_med_names import clean_med_names
from mri_preprocess import MRIFinder, add_mri_paths


def summarize_mri_coverage(df: pd.DataFrame) -> Tuple[int, int, float]:
    """Return (resolved_count, total_rows, percent_resolved) for MRI paths."""
    total = len(df)
    if total == 0 or 'mri_path' not in df.columns:
        return 0, total, 0.0
    resolved = df['mri_path'].notna().sum()
    percent = resolved / total * 100 if total else 0.0
    return resolved, total, percent


def log_mri_coverage(label: str, df: pd.DataFrame) -> Tuple[int, int, float]:
    """Log MRI coverage summary and return the underlying stats."""
    resolved, total, percent = summarize_mri_coverage(df)
    if total == 0:
        print(f"MRI coverage [{label}]: 0/0 visits")
    else:
        print(f"MRI coverage [{label}]: {resolved}/{total} visits ({percent:.1f}%)")
    return resolved, total, percent


def coverage_tuple_to_dict(stats: Tuple[int, int, float]) -> Dict[str, float]:
    """Format coverage tuple as a serializable dict."""
    resolved, total, percent = stats
    return {"resolved": int(resolved), "total": int(total), "percent": float(percent)}


# =======================================================================================================================
# MAIN EXECUTION
# =======================================================================================================================

def main():
    """Main preprocessing pipeline."""
    # 1. Load and Initial Cleaning
    df = pd.read_csv(os.path.join(DATA_DIR, 'ADNI_merged.csv'), low_memory=False)

    # Trigger medicatio nname cleaning
    print("Cleaning the  RECCMED dataset")
    clean_med_names()
    
    # Drop any visit without an MRI
    df = df.dropna(subset=['IMAGEUID'])

    # Normalize identifiers: prefer PTID (fallback RID), drop RID, then rename PTID -> subject_id
    df = normalize_ids_post_merge(df)

    # 2. Feature Engineering
    df = calculate_active_medications(df)
    df = calculate_consistent_age(df)
    # Keep subjects with at least 2 visits to ensure pairs exist but allow small test fixtures
    df = filter_subjects_by_visit_count(df, min_visits=2)

    # Resolve MRI paths prior to splitting so splits retain only usable imaging data
    nii_dir = MRI_NII_SUBDIR
    print("MRI NII path: ", MRI_NII_SUBDIR)
    mri_finder = MRIFinder(nii_dir)
    print("Df before paths added: ", df['IMAGEUID'].nunique())
    df['mri_path'] = add_mri_paths(df, finder=mri_finder)
    baseline_mri_coverage = log_mri_coverage("baseline (pre-filter)", df)
    print("Df after paths added: ", df['IMAGEUID'].nunique())

    # Drop visits without resolvable MRI paths and reapply visit-count filtering
    df = df[df['mri_path'].notna()].copy()
    df = df[df['mri_path'].astype(str).str.strip() != '']
    df = filter_subjects_by_visit_count(df, min_visits=2)
    post_filter_mri_coverage = log_mri_coverage("post-MRI filter", df)
    # print("Unique PTID after MRI filter: ", df['PTID'].nunique())

    # 3. Data Splitting (Subject-Aware)
    train_df_raw, val_df_raw, test_df_raw, train_subjects, val_subjects, test_subjects = split_data_by_subject(
        df, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"Subjects -> train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")
    split_mri_coverage = {
        'train': log_mri_coverage("train split (pre-imputation)", train_df_raw),
        'val': log_mri_coverage("val split (pre-imputation)", val_df_raw),
        'test': log_mri_coverage("test split (pre-imputation)", test_df_raw),
    }

    # 4. Imputation
    train_df, val_df, test_df = impute_missing_values(train_df_raw, val_df_raw, test_df_raw)
    imputed_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    imputed_df.drop(columns=['research_group', 'PTMARRY', 'CMFREQNC'], inplace=True, errors='ignore')
    imputed_df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')

    # 4b. Compute next-visit time gap per subject for irregular intervals
    # next_visit_months = months_since_bl(t+1) - months_since_bl(t); 0 for last visit
    imputed_df = imputed_df.sort_values(['subject_id', 'months_since_bl']).reset_index(drop=True)
    imputed_df['next_visit_months'] = (
        imputed_df.groupby('subject_id')['months_since_bl'].shift(-1) - imputed_df['months_since_bl']
    )
    imputed_df['next_visit_months'] = imputed_df['next_visit_months'].fillna(0.0)

    # 4c. Validate MRI coverage persists post-imputation
    post_imputation_coverage = log_mri_coverage("post-imputation overall", imputed_df)

    # 5. Build per-visit sequence datasets (no all-pairs). Keep per-subject splits.
    # Select observation + action features present after imputation
    # Keep absolute time for files/plots, but we will NOT use it as a model input/target
    numeric_obs_cols = [
        c for c in (CONTINUOUS_VARS + ORDINAL_VARS) if c in imputed_df.columns
    ]
    # Include the computed next-visit gap feature
    if 'next_visit_months' in imputed_df.columns:
        numeric_obs_cols = numeric_obs_cols + ['next_visit_months']
    action_cols_present = [c for c in ACTION_FEATURES if c in imputed_df.columns]
    categorical_cols = [c for c in ['PTGENDER', 'PTRACCAT'] if c in imputed_df.columns]

    # Minimal column set for model datasets
    base_cols = ['subject_id']
    # Include passthrough columns that should be preserved but not imputed/scaled
    passthrough_cols = [c for c in BYPASS_IMPUTATION if c in imputed_df.columns and c not in base_cols]
    selected_cols = list(dict.fromkeys(base_cols + passthrough_cols + numeric_obs_cols + categorical_cols + action_cols_present))

    train_seq_raw = imputed_df[imputed_df['subject_id'].isin(train_subjects)][selected_cols].copy()
    val_seq_raw = imputed_df[imputed_df['subject_id'].isin(val_subjects)][selected_cols].copy()
    test_seq_raw = imputed_df[imputed_df['subject_id'].isin(test_subjects)][selected_cols].copy()

    # Normalize + OHE for sequence datasets and build next-visit targets; preserves subject_id
    X_train, X_val, X_test, y_train, y_val, y_test = normalize_and_encode_sequence_splits(
        train_seq_raw, val_seq_raw, test_seq_raw
    )

    # Safety: ensure subject_id exists in all y splits (align with corresponding X rows)
    for X_split, y_split in (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    ):
        if 'subject_id' in X_split.columns and 'subject_id' not in y_split.columns:
            y_split.insert(0, 'subject_id', X_split['subject_id'].values)

    # 6. Create data for the clinician policy model
    alpaca_obs_cols = get_alpaca_observation_columns()
    # Filter to columns that still exist after cleaning/dropping
    clinician_state_features = [
        col for col in alpaca_obs_cols
        if col != 'months_since_bl' and col in imputed_df.columns
    ]
    clinician_x_full, clinician_y_full = transform_dataset_for_clinician(imputed_df, clinician_state_features, ACTION_FEATURES)
    
    clinician_X_train = clinician_x_full[clinician_x_full['subject_id'].isin(train_subjects)].copy()
    clinician_X_val = clinician_x_full[clinician_x_full['subject_id'].isin(val_subjects)].copy()
    clinician_X_test = clinician_x_full[clinician_x_full['subject_id'].isin(test_subjects)].copy()
    clinician_y_train = clinician_y_full[clinician_y_full['subject_id'].isin(train_subjects)].copy()
    clinician_y_val = clinician_y_full[clinician_y_full['subject_id'].isin(val_subjects)].copy()
    clinician_y_test = clinician_y_full[clinician_y_full['subject_id'].isin(test_subjects)].copy()

    # Normalize clinician state features with scaler fit on train only (avoid leakage)
    clinician_X_train, clinician_X_val, clinician_X_test = normalize_clinician_splits(
        clinician_X_train, clinician_X_val, clinician_X_test, save_dir=CLINICIAN_POLICY_DIR
    )

    # 7. Build Schema for sequence-based training
    # Exclude absolute time from model inputs; model conditions on next_visit_months only
    # Exclude bypass columns from model inputs (these are preserved for reference only)
    model_input_cols = [
        c for c in X_train.columns
        if c not in ('subject_id', 'months_since_bl') and c not in BYPASS_IMPUTATION
    ]
    action_cols = [c for c in model_input_cols if c.endswith('_active')]
    # Targets exclude actions, next_visit_months, and absolute time
    observation_cols = [
        c for c in model_input_cols
        if (not c.endswith('_active')) and (c not in ('next_visit_months', 'months_since_bl'))
    ]
    # Target columns from returned y dataset (exclude helpers like subject_id/IMAGEUID)
    y_cols = [c for c in y_train.columns if c not in BYPASS_IMPUTATION]
    y_cont_cols = [c for c in y_cols if c in (CONTINUOUS_VARS + ORDINAL_VARS)]
    y_bin_cols = [c for c in y_cols if c not in y_cont_cols]

    # Group categorical y columns by prefix for convenience
    y_categorical_groups: Dict[str, List[str]] = {}
    for col in y_bin_cols:
        if '_' in col:
            key = col.split('_', 1)[0]
            y_categorical_groups.setdefault(key, []).append(col)
    # Keep only groups with >= 2 columns
    y_categorical_groups = {k: v for k, v in y_categorical_groups.items() if len(v) >= 2}

    mri_coverage_summary = {
        'baseline_pre_filter': coverage_tuple_to_dict(baseline_mri_coverage),
        'post_filter': coverage_tuple_to_dict(post_filter_mri_coverage),
        'post_imputation': coverage_tuple_to_dict(post_imputation_coverage),
        'splits_pre_imputation': {
            split: coverage_tuple_to_dict(stats)
            for split, stats in split_mri_coverage.items()
        },
    }

    schema = {
        "action_cols": action_cols,
        "cont_obs_cols": [c for c in model_input_cols if c in CONTINUOUS_VARS and c != 'months_since_bl'],
        "binary_obs_cols": [c for c in model_input_cols if (not c.endswith('_active')) and c not in (CONTINUOUS_VARS + ORDINAL_VARS)],
        "observation_cols": observation_cols,
        "model_input_cols": model_input_cols,
        # Helper columns available alongside X and y (not part of model inputs/targets)
        "x_helper": [c for c in BYPASS_IMPUTATION if c in X_train.columns],
        "y_helper": [c for c in BYPASS_IMPUTATION if c in y_train.columns],
        "y_cont_cols": y_cont_cols,
        "y_bin_cols": y_bin_cols,
        "y_cols": y_cols,
        "y_categorical_groups": y_categorical_groups,
        "clinician_state_cols": [c for c in clinician_X_train.columns if c != 'subject_id'],
        "clinician_action_cols": [c for c in clinician_y_train.columns if c != 'subject_id'],
        "num_binary_outputs": len(y_bin_cols),
        "num_continuous_outputs": len(y_cont_cols),
        "mri_coverage": mri_coverage_summary,
    }

    # 8. Save sequence datasets and schema
    model_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }
    save_artifacts(model_data, schema, MODEL_TRAINING_DIR, [MODEL_TRAINING_DIR, ALPACA_DIR, CLINICIAN_POLICY_DIR])

    clinician_data = {
        'clinician_X_train': clinician_X_train, 'clinician_X_val': clinician_X_val, 'clinician_X_test': clinician_X_test,
        'clinician_y_train': clinician_y_train, 'clinician_y_val': clinician_y_val, 'clinician_y_test': clinician_y_test,
    }
    save_artifacts(clinician_data, schema, CLINICIAN_POLICY_DIR, [])  # Schema already saved

if __name__ == '__main__':
    main()
