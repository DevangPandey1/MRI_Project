import pandas as pd
import numpy as np
from imputation import train_mice_imputer, impute_with_trained_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os
import json
from typing import List, Dict, Tuple, Set, Optional
from config import ALPACA_DIR, DRUG_CLASS_MAPPING, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION, BYPASS_IMPUTATION, MODEL_TRAINING_DIR, SUBJECT_ID_COL, ROOT_DIR
import config as config_mod

# =======================================================================================================================
# HELPER FUNCTIONS FOR PREPROCESSING
# =======================================================================================================================

def normalize_ids_post_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-merge ID normalization for ADNI:
    - Use PTID as primary identifier; if missing, fall back to RID.
    - After ensuring PTID, drop RID if present.
    - Finally, rename PTID -> subject_id for internal consistency across the pipeline.
    """
    out = df.copy()
    if 'PTID' not in out.columns and 'RID' in out.columns:
        out['PTID'] = out['RID']
    if 'PTID' not in out.columns and 'subject_id' in out.columns:
        out['PTID'] = out['subject_id']
    # Standardize formatting of PTID as string
    if 'PTID' in out.columns:
        out['PTID'] = out['PTID'].astype(str).str.strip()
    # Drop RID if present
    out.drop(columns=['RID'], inplace=True, errors='ignore')
    # Rename PTID -> subject_id if PTID exists
    if 'PTID' in out.columns:
        # Drop any preexisting subject_id to avoid duplicates; PTID is the source of truth
        if 'subject_id' in out.columns:
            out.drop(columns=['subject_id'], inplace=True)
        out = out.rename(columns={'PTID': 'subject_id'})
        # Make sure subject_id is the leading column for downstream convenience
        subject_first = ['subject_id'] + [c for c in out.columns if c != 'subject_id']
        out = out.loc[:, subject_first]
    return out

def get_alpaca_observation_columns() -> List[str]:
    """
    Returns observation columns used for model (state) features.

    Behavior:
    - Binary demographics (OHE groups) are inferred from ALPACA `X_train.csv` if present,
      otherwise a stable fallback list is used to preserve prior behavior.
    - Continuous/ordinal observation columns are sourced dynamically from preprocessing
      configuration (CONTINUOUS_VARS + ORDINAL_VARS), so adding/removing variables in
      config propagates automatically without touching code. `months_since_bl` is kept
      separate but included in the returned list for downstream handling.
    """
    alpaca_training_path = os.path.join(ALPACA_DIR, 'X_train.csv')
    if os.path.exists(alpaca_training_path):
        alpaca_data = pd.read_csv(alpaca_training_path)
        binary_obs_cols = sorted([
            col for col in alpaca_data.columns
            if col.startswith(('PTGENDER_', 'PTRACCAT_', 'PTMARRY_'))
        ])
    else:
        binary_obs_cols = [
            'PTGENDER_Male', 'PTMARRY_Divorced', 'PTMARRY_Married', 'PTMARRY_Never married',
            'PTMARRY_Unknown', 'PTMARRY_Widowed', 'PTRACCAT_Asian', 'PTRACCAT_Black',
            'PTRACCAT_Hawaiian/Other PI', 'PTRACCAT_More than one', 'PTRACCAT_White'
        ]

    # Pull continuous + ordinal observation variables from config dynamically.
    # Exclude months_since_bl here; it is appended explicitly below and handled specially downstream.
    configured_numeric_obs = [
        v for v in (CONTINUOUS_VARS + ORDINAL_VARS)
        if v != 'months_since_bl'
    ]

    # Keep deterministic ordering: numeric obs follow config order; binary obs are sorted for stability.
    cont_ord_obs_cols = configured_numeric_obs

    # Final observation set includes months_since_bl as the final element for downstream processing.
    return cont_ord_obs_cols + binary_obs_cols + ['months_since_bl']

# Intentionally no map_drug_classes here: medication name cleaning is handled
# centrally by preprocessing/clean_med_names.py which produces CMMED_clean.

def calculate_active_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates per-visit active medication flags by reading ADNI RECCMEDS directly.

    Behavior:
    - Loads medication history from './data/tabular/RECCMEDS_10Sep2025.csv'. Requires 'CMMED_clean'
      to be present in that file (produced by preprocessing/clean_med_names.py).
    - Uses CMBGNYR_DRVD as medication start year and CMENDYR_DRVD as end year (missing → ongoing).
    - A medication is marked active for a visit if the visit year falls within [start_year, end_year].
    - Produces one column per medication class with suffix '_active' and a robust 'No Medication_active' flag
      that is 1 only if no other medication classes are active for that visit.
    - Merges flags back to the input per-visit dataframe.
    """
    # --- Normalize keys on input df ---
    df = df.copy()
    # Use subject_id as the canonical identifier throughout the pipeline
    if 'subject_id' not in df.columns:
        if 'PTID' in df.columns:
            df['subject_id'] = df['PTID']
        else:
            raise KeyError("calculate_active_medications expects 'subject_id' in the input dataframe.")
    df['subject_id'] = df['subject_id'].astype(str).str.strip()

    # visit token: prefer existing 'visit'; else derive from VISCODE2 or VISCODE
    if 'visit' not in df.columns:
        token = None
        if 'VISCODE2' in df.columns:
            token = 'VISCODE2'
        elif 'VISCODE' in df.columns:
            token = 'VISCODE'
        if token is None:
            raise KeyError("calculate_active_medications expects 'visit' or 'VISCODE2'/'VISCODE' in the input dataframe.")
        df['visit'] = df[token]

    # visit year from EXAMDATE
    if 'EXAMDATE' not in df.columns:
        raise KeyError("calculate_active_medications expects 'EXAMDATE' in the input dataframe to compute visit year.")
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], errors='coerce')
    df['visit_year'] = df['EXAMDATE'].dt.year

    # Unique per-visit info
    visit_info = df[['subject_id', 'visit', 'visit_year']].dropna().drop_duplicates()

    # --- Load medications (RECCMEDS) if PTID available; otherwise fallback handled below ---
    meds_path = os.path.join(ROOT_DIR, 'data', 'tabular', 'RECCMEDS_10Sep2025.csv')
    if not os.path.exists(meds_path):
        raise FileNotFoundError(f"Medication file not found: {meds_path}")
    # Require preprocessed medication names from CMMED_clean (no fallback to raw CMMED)
    meds_cols = ['PTID', 'CMMED_clean', 'CMBGNYR_DRVD', 'CMENDYR_DRVD']
    try:
        meds = pd.read_csv(meds_path, usecols=meds_cols, low_memory=False)
    except ValueError as e:
        # Provide a clearer error if CMMED_clean is missing
        try:
            header_cols = pd.read_csv(meds_path, nrows=0).columns.tolist()
        except Exception:
            header_cols = []
        if 'CMMED_clean' not in header_cols:
            raise KeyError(
                "RECCMEDS file is missing 'CMMED_clean'. Run preprocessing/clean_med_names.py first to add it."
            ) from e
        raise
    # Rename to subject_id for consistency in-memory
    meds = meds.rename(columns={'PTID': 'subject_id'})
    # Normalize key and numeric year fields
    meds['subject_id'] = meds['subject_id'].astype(str).str.strip()
    meds['CMBGNYR_DRVD'] = pd.to_numeric(meds['CMBGNYR_DRVD'], errors='coerce')
    meds['CMENDYR_DRVD'] = pd.to_numeric(meds['CMENDYR_DRVD'], errors='coerce')

    # Normalize cleaned names for mapping
    meds['CMMED_clean'] = meds['CMMED_clean'].fillna('').astype(str).str.lower().str.strip()
    # Map to fine-grained action names using DRUG_CLASS_MAPPING directly
    meds['med_action'] = meds['CMMED_clean'].map(DRUG_CLASS_MAPPING).fillna('Other_Rare')
    # Exclude explicit 'No_Medication' (handled per-visit later)
    meds = meds[meds['med_action'] != 'No_Medication']

    # Prepare medication timeline rows (exclude explicit 'No Medication' records; handled separately)
    meds = meds[(meds['subject_id'].notna()) & (meds['CMBGNYR_DRVD'].notna()) & (meds['med_action'].notna())]

    # Fill open-ended end year with a far-future sentinel
    meds['CMENDYR_DRVD_filled'] = meds['CMENDYR_DRVD'].fillna(9999)

    # Cross-join per subject: apply each subject's meds across all their visits
    expanded = pd.merge(visit_info, meds[['subject_id', 'CMBGNYR_DRVD', 'CMENDYR_DRVD_filled', 'med_action']],
                        on='subject_id', how='left')

    # Active if visit_year within [start, end]
    active_mask = (
        (expanded['visit_year'] >= expanded['CMBGNYR_DRVD']) &
        (expanded['visit_year'] <= expanded['CMENDYR_DRVD_filled'])
    )
    expanded['is_active'] = active_mask.astype(int)

    # Pivot to get one column per medication class
    # Collapse any actions not in ACTION_FEATURES into 'Other_Rare' on the fly for pivot
    # to make sure the pivoted columns align with ACTION_FEATURES.
    action_features = list(getattr(config_mod, 'ACTION_FEATURES', []))
    # The configured ACTION_FEATURES are expected to be suffixed with '_active'
    # Compose pivot_action with the '_active' suffix and map unknowns to 'Other_Rare_active' if present
    expanded['pivot_action'] = expanded['med_action'] + '_active'
    if 'Other_Rare_active' in action_features:
        expanded['pivot_action'] = expanded['pivot_action'].where(
            expanded['pivot_action'].isin(action_features), 'Other_Rare_active'
        )
    else:
        expanded['pivot_action'] = expanded['pivot_action'].where(
            expanded['pivot_action'].isin(action_features), None
        )

    active_meds_df = expanded.pivot_table(
        index=['subject_id', 'visit', 'visit_year'],
        columns='pivot_action',
        values='is_active',
        fill_value=0,
        aggfunc='max'
    ).reset_index()
    active_meds_df.columns.name = None

    # Ensure all visits are present (even if no meds) for setting No Medication flag
    active_meds_df = visit_info.merge(active_meds_df, on=['subject_id', 'visit', 'visit_year'], how='left')

    # Ensure all configured ACTION_FEATURES columns exist and are int 0/1
    for action in action_features:
        if action not in active_meds_df.columns:
            active_meds_df[action] = 0
    # Restrict to PTID, visit, visit_year + ACTION_FEATURES (order preserved)
    active_meds_df = active_meds_df[['subject_id', 'visit', 'visit_year'] + action_features].copy()
    for action in action_features:
        active_meds_df[action] = active_meds_df[action].fillna(0).astype(int)

    # Robust No_Medication flag (if present): 1 only when all others are 0
    if 'No_Medication_active' in action_features:
        other_cols = [c for c in action_features if c != 'No_Medication_active']
        active_meds_df['No_Medication_active'] = (
            active_meds_df[other_cols].sum(axis=1) == 0
        ).astype(int)

    # Merge flags back to the original dataframe
    output_df = pd.merge(df, active_meds_df, on=['subject_id', 'visit', 'visit_year'], how='left')

    # Ensure columns exist and are integer 0/1, aligned to ACTION_FEATURES
    for col in action_features:
        if col in output_df.columns:
            output_df[col] = output_df[col].fillna(0).astype(int)
        else:
            output_df[col] = 0

    # Ensure we keep subject_id as the canonical identifier
    # Drop PTID if present to avoid confusion downstream
    if 'PTID' in output_df.columns:
        output_df.drop(columns=['PTID'], inplace=True, errors='ignore')
    return output_df

def calculate_consistent_age(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically computes a consistent age for each subject across all visits."""
    df['months_since_bl'] = pd.to_numeric(df['months_since_bl'], errors='coerce')
    df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
    df.dropna(subset=[SUBJECT_ID_COL, 'months_since_bl', 'subject_age'], inplace=True)

    df_sorted = df.sort_values([SUBJECT_ID_COL, 'months_since_bl'])
    earliest_visits = df_sorted.loc[df_sorted.groupby(SUBJECT_ID_COL)['months_since_bl'].idxmin()]
    
    earliest_visits['calculated_true_baseline_age'] = earliest_visits['subject_age'] - (earliest_visits['months_since_bl'] / 12.0)
    
    baseline_age_map = earliest_visits.set_index(SUBJECT_ID_COL)['calculated_true_baseline_age']
    df['calculated_true_baseline_age'] = df[SUBJECT_ID_COL].map(baseline_age_map)
    
    df['subject_age'] = df['calculated_true_baseline_age'] + (df['months_since_bl'] / 12.0)
    df.drop(columns=['calculated_true_baseline_age'], inplace=True)
    
    return df

def filter_subjects_by_visit_count(df: pd.DataFrame, min_visits: int) -> pd.DataFrame:
    """Removes subjects with fewer than a specified number of visits."""
    visit_counts = df.groupby(SUBJECT_ID_COL).size()
    subjects_to_remove = visit_counts[visit_counts < min_visits].index
    return df[~df[SUBJECT_ID_COL].isin(subjects_to_remove)]

def split_data_by_subject(df: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set, Set, Set]:
    """Splits data into train, validation, and test sets based on subject ID."""
    gss_main = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_main.split(df, groups=df[SUBJECT_ID_COL]))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state + 1)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[SUBJECT_ID_COL]))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    train_subjects = set(train_df[SUBJECT_ID_COL].unique())
    val_subjects = set(val_df[SUBJECT_ID_COL].unique())
    test_subjects = set(test_df[SUBJECT_ID_COL].unique())

    return train_df, val_df, test_df, train_subjects, val_subjects, test_subjects

def impute_missing_values(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fits a MICE imputer on the training data and transforms all splits."""
    # Ensure PTID is not part of the imputation design matrix
    drop_vars = list(DROP_VARS_FOR_IMPUTATION)
    # Also exclude any centrally configured bypass columns from the imputation matrix
    for v in BYPASS_IMPUTATION:
        if v not in drop_vars:
            drop_vars.append(v)
    if 'PTID' in train_df.columns or 'PTID' in val_df.columns or 'PTID' in test_df.columns:
        if 'PTID' not in drop_vars:
            drop_vars.append('PTID')

    imputer, imputer_schema = train_mice_imputer(
        train_df.copy(), CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, drop_vars
    )
    
    imputed_train_df = impute_with_trained_imputer(
        train_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, drop_vars, imputer_schema
    )
    imputed_val_df = impute_with_trained_imputer(
        val_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, drop_vars, imputer_schema
    )
    imputed_test_df = impute_with_trained_imputer(
        test_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, drop_vars, imputer_schema
    )
    
    joblib.dump({'schema_columns': imputer_schema}, os.path.join(MODEL_TRAINING_DIR, 'mice_imputer_schema.joblib'))
    
    return imputed_train_df, imputed_val_df, imputed_test_df

def align_columns(train_df, val_df, test_df):
    """Aligns columns of val and test dataframes to match the train dataframe."""
    train_cols = train_df.columns
    val_df = val_df.reindex(columns=train_cols, fill_value=0)
    test_df = test_df.reindex(columns=train_cols, fill_value=0)
    return train_df, val_df, test_df


def normalize_and_encode(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize numeric variables and one-hot encode categoricals for (X, y) splits.

    - Fits scaler_X on X_train numeric features: CONTINUOUS_VARS + ORDINAL_VARS + ['time_delta'] if present.
    - Fits scaler_y on y_train numeric targets: CONTINUOUS_VARS + ORDINAL_VARS (intersection with y columns).
    - Saves scalers to MODEL_TRAINING_DIR as 'scaler_X.joblib' and 'scaler_y.joblib'.
    - One-hot encodes categorical columns in X splits, aligns columns to X_train post-encoding.
    - Preserves SUBJECT_ID_COL for all splits in both X and y.
    """
    # Preserve SUBJECT_ID_COL only if it exists in the given split
    sid_col = SUBJECT_ID_COL
    # Preserve ids for reattachment later (all splits)
    sid_train_X = X_train[sid_col].copy() if sid_col in X_train.columns else None
    sid_val_X = X_val[sid_col].copy() if sid_col in X_val.columns else None
    sid_test_X = X_test[sid_col].copy() if sid_col in X_test.columns else None
    sid_train_y = y_train[sid_col].copy() if sid_col in y_train.columns else None
    sid_val_y = y_val[sid_col].copy() if sid_col in y_val.columns else None
    sid_test_y = y_test[sid_col].copy() if sid_col in y_test.columns else None
    # Drop id before transforms; we'll reinsert later for all splits
    if sid_col in X_train.columns:
        X_train = X_train.drop(columns=[sid_col])
    if sid_col in X_val.columns:
        X_val = X_val.drop(columns=[sid_col])
    if sid_col in X_test.columns:
        X_test = X_test.drop(columns=[sid_col])
    if sid_col in y_train.columns:
        y_train = y_train.drop(columns=[sid_col])
    if sid_col in y_val.columns:
        y_val = y_val.drop(columns=[sid_col])
    if sid_col in y_test.columns:
        y_test = y_test.drop(columns=[sid_col])

    # Identify numeric columns for scalers
    x_numeric = [c for c in (CONTINUOUS_VARS + ORDINAL_VARS) if c in X_train.columns]
    if 'time_delta' in X_train.columns and 'time_delta' not in x_numeric:
        x_numeric = x_numeric + ['time_delta']
    y_numeric = [c for c in (CONTINUOUS_VARS + ORDINAL_VARS) if c in y_train.columns]

    # Fit scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    if x_numeric:
        scaler_X.fit(X_train[x_numeric])
    if y_numeric:
        scaler_y.fit(y_train[y_numeric])
    joblib.dump(scaler_X, os.path.join(MODEL_TRAINING_DIR, 'scaler_X.joblib'))
    joblib.dump(scaler_y, os.path.join(MODEL_TRAINING_DIR, 'scaler_y.joblib'))

    # Apply scaling (in-place on copies)
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_val = y_val.copy()
    y_test = y_test.copy()

    if x_numeric:
        X_train[x_numeric] = scaler_X.transform(X_train[x_numeric])
        if not X_val.empty:
            present = [c for c in x_numeric if c in X_val.columns]
            if present:
                X_val[present] = scaler_X.transform(X_val[present])
        if not X_test.empty:
            present = [c for c in x_numeric if c in X_test.columns]
            if present:
                X_test[present] = scaler_X.transform(X_test[present])
    if y_numeric:
        y_train[y_numeric] = scaler_y.transform(y_train[y_numeric])
        if not y_val.empty:
            present = [c for c in y_numeric if c in y_val.columns]
            if present:
                y_val[present] = scaler_y.transform(y_val[present])
        if not y_test.empty:
            present = [c for c in y_numeric if c in y_test.columns]
            if present:
                y_test[present] = scaler_y.transform(y_test[present])

    # Determine categorical columns for X
    possible_cats = list(CATEGORICAL_VARS_FOR_IMPUTATION)
    cat_cols = [c for c in possible_cats if (c in X_train.columns) or (c in X_val.columns) or (c in X_test.columns)]

    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in cat_cols if c in df.columns]
        if not cols:
            return df
        return pd.get_dummies(df, columns=cols, prefix=cols, dummy_na=False, dtype=int)

    X_train = _encode(X_train)
    X_val = _encode(X_val)
    X_test = _encode(X_test)

    # Align X columns to training
    train_cols = list(X_train.columns)
    X_val = X_val.reindex(columns=train_cols, fill_value=0)
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    # Reattach id for all splits
    if sid_train_X is not None:
        X_train.insert(0, sid_col, sid_train_X.reindex(X_train.index))
    if sid_val_X is not None:
        X_val.insert(0, sid_col, sid_val_X.reindex(X_val.index))
    if sid_test_X is not None:
        X_test.insert(0, sid_col, sid_test_X.reindex(X_test.index))
    if sid_train_y is not None:
        y_train.insert(0, sid_col, sid_train_y.reindex(y_train.index))
    if sid_val_y is not None:
        y_val.insert(0, sid_col, sid_val_y.reindex(y_val.index))
    if sid_test_y is not None:
        y_test.insert(0, sid_col, sid_test_y.reindex(y_test.index))

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_and_encode_sequence_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds next-visit training pairs for sequence modeling and normalizes/encodes features.

    - For each split (train/val/test), for each subject, sorts by `months_since_bl` and creates
      pairs (visit t -> visit t+1). The last visit per subject is dropped as it has no target.
    - X: current-visit features (including actions and optional time-gap features like
      `next_visit_months`). y: next-visit state features excluding any action columns.
    - Numeric columns are standardized with StandardScaler fit on training split
      (X: CONTINUOUS_VARS + ORDINAL_VARS + time-gap extras if present; y: CONTINUOUS_VARS + ORDINAL_VARS
      intersection present in targets). Categorical columns in both X and y are one-hot encoded,
      and val/test are aligned to train columns.
    - Saves `scaler_X.joblib` and `scaler_y.joblib` to `MODEL_TRAINING_DIR`.

    Returns:
      (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    sid_col = SUBJECT_ID_COL

    def build_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy()
        # Sort within subject by time
        df = df.sort_values([sid_col, 'months_since_bl']).reset_index(drop=True)
        # Mask for rows that have a next visit
        has_next = df.groupby(sid_col)['months_since_bl'].shift(-1).notna()
        # X are rows with a following visit
        X_raw = df.loc[has_next].reset_index(drop=True)
        # y are the next-visit rows for selected columns; we'll choose columns later
        y_raw = df.groupby(sid_col).shift(-1).loc[has_next].reset_index(drop=True)
        if sid_col not in y_raw.columns and sid_col in X_raw.columns:
            # Groupby.shift can drop the grouping key; restore it explicitly
            y_raw.insert(0, sid_col, X_raw[sid_col].values)
        return X_raw, y_raw

    # Build raw pairs per split
    X_train_raw, y_train_raw = build_pairs(train_df.copy())
    X_val_raw, y_val_raw = build_pairs(val_df.copy())
    X_test_raw, y_test_raw = build_pairs(test_df.copy())

    # Preserve ids for reattachment later
    sid_train = X_train_raw.get(sid_col, None)
    sid_val = X_val_raw.get(sid_col, None)
    sid_test = X_test_raw.get(sid_col, None)
    sid_train_y = y_train_raw.get(sid_col, None)
    sid_val_y = y_val_raw.get(sid_col, None)
    sid_test_y = y_test_raw.get(sid_col, None)

    # Preserve helper columns (from BYPASS_IMPUTATION) for reattachment to y
    # We exclude SUBJECT_ID_COL here since it's handled separately above
    helper_cols = [h for h in BYPASS_IMPUTATION if h != sid_col]
    y_train_helpers = {h: y_train_raw[h].copy() for h in helper_cols if h in y_train_raw.columns}
    y_val_helpers = {h: y_val_raw[h].copy() for h in helper_cols if h in y_val_raw.columns}
    y_test_helpers = {h: y_test_raw[h].copy() for h in helper_cols if h in y_test_raw.columns}

    # Identify variables
    # Numeric observation variables to scale for both X and y (exclude bypassed)
    numeric_obs_vars = [
        v for v in (CONTINUOUS_VARS + ORDINAL_VARS)
        if v in X_train_raw.columns and v not in BYPASS_IMPUTATION
    ]
    # X numeric may include time-gap extras
    x_numeric = list(numeric_obs_vars)
    for extra in ["next_visit_months", "time_since_prev", "time_delta"]:
        if extra in X_train_raw.columns and extra not in x_numeric:
            x_numeric.append(extra)
    # y numeric are those numeric obs present in y frame
    y_numeric = [v for v in numeric_obs_vars if v in y_train_raw.columns]

    # Determine categorical columns dynamically if not provided
    if categorical_cols is None:
        possible = list(CATEGORICAL_VARS_FOR_IMPUTATION)
        categorical_cols = [
            c for c in possible
            if (c in X_train_raw.columns) or (c in X_val_raw.columns) or (c in X_test_raw.columns)
        ]

    # Drop id before transforms
    if sid_col in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=[sid_col])
    if sid_col in X_val_raw.columns:
        X_val_raw = X_val_raw.drop(columns=[sid_col])
    if sid_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[sid_col])
    if sid_col in y_train_raw.columns:
        y_train_raw = y_train_raw.drop(columns=[sid_col])
    if sid_col in y_val_raw.columns:
        y_val_raw = y_val_raw.drop(columns=[sid_col])
    if sid_col in y_test_raw.columns:
        y_test_raw = y_test_raw.drop(columns=[sid_col])

    # Fit scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    if x_numeric:
        scaler_X.fit(X_train_raw[x_numeric])
    if y_numeric:
        scaler_y.fit(y_train_raw[y_numeric])
    # Persist scalers
    joblib.dump(scaler_X, os.path.join(MODEL_TRAINING_DIR, 'scaler_X.joblib'))
    joblib.dump(scaler_y, os.path.join(MODEL_TRAINING_DIR, 'scaler_y.joblib'))

    # Apply scaling to numeric columns
    def apply_scale(df: pd.DataFrame, cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
        if not cols:
            return df
        present = [c for c in cols if c in df.columns]
        if present:
            df[present] = scaler.transform(df[present])
        return df

    X_train_scaled = X_train_raw.copy()
    X_val_scaled = X_val_raw.copy()
    X_test_scaled = X_test_raw.copy()
    y_train_scaled = y_train_raw.copy()
    y_val_scaled = y_val_raw.copy()
    y_test_scaled = y_test_raw.copy()

    if x_numeric:
        X_train_scaled[x_numeric] = scaler_X.transform(X_train_scaled[x_numeric])
        if not X_val_scaled.empty:
            X_val_scaled = apply_scale(X_val_scaled, x_numeric, scaler_X)
        if not X_test_scaled.empty:
            X_test_scaled = apply_scale(X_test_scaled, x_numeric, scaler_X)
    if y_numeric:
        y_train_scaled[y_numeric] = scaler_y.transform(y_train_scaled[y_numeric])
        if not y_val_scaled.empty:
            y_val_scaled = apply_scale(y_val_scaled, y_numeric, scaler_y)
        if not y_test_scaled.empty:
            y_test_scaled = apply_scale(y_test_scaled, y_numeric, scaler_y)

    # One-hot encode categoricals for both X and y
    def encode_ohe(df: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
        cols = [c for c in cats if c in df.columns]
        if not cols:
            return df
        return pd.get_dummies(df, columns=cols, prefix=cols, dummy_na=False, dtype=int)

    X_train_enc = encode_ohe(X_train_scaled, categorical_cols)
    X_val_enc = encode_ohe(X_val_scaled, categorical_cols)
    X_test_enc = encode_ohe(X_test_scaled, categorical_cols)
    y_train_enc = encode_ohe(y_train_scaled, categorical_cols)
    y_val_enc = encode_ohe(y_val_scaled, categorical_cols)
    y_test_enc = encode_ohe(y_test_scaled, categorical_cols)

    # Align X val/test to X train columns
    x_cols = list(X_train_enc.columns)
    X_val_enc = X_val_enc.reindex(columns=x_cols, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=x_cols, fill_value=0)

    # Determine y columns: exclude actions and absolute/time-gap fields
    def infer_y_cols(df_train: pd.DataFrame) -> List[str]:
        return [
            c for c in df_train.columns
            if (c not in (sid_col, 'months_since_bl', 'next_visit_months', 'time_delta', 'time_since_prev'))
            and (c not in BYPASS_IMPUTATION)
            and (not c.endswith('_active'))
        ]

    y_cols = infer_y_cols(y_train_enc)

    # Align y val/test to y train columns and order
    y_train_enc = y_train_enc.reindex(columns=y_cols, fill_value=0)
    y_val_enc = y_val_enc.reindex(columns=y_cols, fill_value=0)
    y_test_enc = y_test_enc.reindex(columns=y_cols, fill_value=0)

    # Reattach helper columns (subject_id + other BYPASS_IMPUTATION) to X and y
    if sid_train is not None:
        X_train_enc.insert(0, sid_col, sid_train.values)
    if sid_val is not None:
        X_val_enc.insert(0, sid_col, sid_val.values)
    if sid_test is not None:
        X_test_enc.insert(0, sid_col, sid_test.values)
    if sid_train_y is not None:
        y_train_enc.insert(0, sid_col, sid_train_y.values)
    if sid_val_y is not None:
        y_val_enc.insert(0, sid_col, sid_val_y.values)
    if sid_test_y is not None:
        y_test_enc.insert(0, sid_col, sid_test_y.values)

    # Reattach remaining helper columns (e.g., IMAGEUID) to y splits, immediately after subject_id
    def _insert_helpers(df: pd.DataFrame, helpers: Dict[str, pd.Series]):
        # Maintain BYPASS_IMPUTATION order (excluding subject_id)
        pos = 1 if sid_col in df.columns else 0
        for h in [h for h in BYPASS_IMPUTATION if h != sid_col]:
            if h in helpers:
                df.insert(pos, h, helpers[h].values)
                pos += 1

    _insert_helpers(y_train_enc, y_train_helpers)
    _insert_helpers(y_val_enc, y_val_helpers)
    _insert_helpers(y_test_enc, y_test_helpers)

    return X_train_enc, X_val_enc, X_test_enc, y_train_enc, y_val_enc, y_test_enc

def save_artifacts(data_dict: Dict[str, pd.DataFrame], schema: Dict, base_dir: str, schema_dirs: List[str]):
    """Saves dataframes and schema to specified locations."""
    os.makedirs(base_dir, exist_ok=True)
    for name, df in data_dict.items():
        df.to_csv(os.path.join(base_dir, f'{name}.csv'), index=False)
        print(f"Saved {name}.csv to {base_dir} with shape {df.shape}")

    for target_dir in schema_dirs:
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, 'columns_schema.json'), 'w') as f:
            json.dump(schema, f, indent=2)
    print(f"columns_schema.json written to: {', '.join(schema_dirs)}")

def normalize_clinician_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    save_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize clinician state features with a StandardScaler trained on train only.

    - Preserves and returns DataFrames including the configured SUBJECT_ID_COL if present.
    - Fits a StandardScaler on all numeric feature columns present in the training split
      (excluding `subject_id`) and applies to val/test to avoid data leakage.
    - Saves the scaler to `{save_dir}/scaler_clinician_X.joblib`.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Preserve id columns if present
    sid_col = SUBJECT_ID_COL
    sid_train = X_train.get(sid_col, None)
    sid_val = X_val.get(sid_col, None)
    sid_test = X_test.get(sid_col, None)

    # Work on copies to avoid modifying inputs outside
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Drop id for scaling work
    if sid_col in X_train.columns:
        X_train.drop(columns=[sid_col], inplace=True)
    if sid_col in X_val.columns:
        X_val.drop(columns=[sid_col], inplace=True)
    if sid_col in X_test.columns:
        X_test.drop(columns=[sid_col], inplace=True)

    # Determine numeric columns from training split
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        if not X_val.empty:
            present = [c for c in numeric_cols if c in X_val.columns]
            if present:
                X_val[present] = scaler.transform(X_val[present])
        if not X_test.empty:
            present = [c for c in numeric_cols if c in X_test.columns]
            if present:
                X_test[present] = scaler.transform(X_test[present])

    # Save scaler for downstream use
    joblib.dump(scaler, os.path.join(save_dir, 'scaler_clinician_X.joblib'))

    # Reattach id if present
    if sid_train is not None:
        X_train.insert(0, sid_col, sid_train.values)
    if sid_val is not None:
        X_val.insert(0, sid_col, sid_val.values)
    if sid_test is not None:
        X_test.insert(0, sid_col, sid_test.values)

    return X_train, X_val, X_test


def transform_dataset(input_original_data: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function is tested and validated using a test datset in the data_exploration/reformat_data.py file.

    Transforms the original patient visit data into model_x (current state + time_delta)
    and model_y (next state) DataFrames.
    """
    model_x_list = []
    model_y_list = []

    # Calculate the total expected number of pairs
    total_expected_pairs = 0
    # We need to iterate over the groups to calculate this.
    # This will re-group if input_original_data is a DataFrame,
    # or iterate over an existing grouper if input_original_data was already grouped.
    # For robustness, let's assume input_original_data is the DataFrame.
    sid_col = SUBJECT_ID_COL
    temp_grouped_for_check = input_original_data.groupby(sid_col)
    for _, subject_df_for_check in temp_grouped_for_check:
        num_subject_visits = len(subject_df_for_check)
        if num_subject_visits >= 2:
            total_expected_pairs += num_subject_visits * (num_subject_visits - 1) // 2

    # Group by subject for processing
    grouped_data = input_original_data.groupby(sid_col)

    for _, subject_visits_df in grouped_data:
        # Sort visits by time for each subject
        subject_visits_df = subject_visits_df.sort_values(by='months_since_bl').reset_index(drop=True)
        
        num_visits = len(subject_visits_df)
        if num_visits < 2:
            # Not enough visits to create a (current_state, next_state) pair
            continue

        # Iterate through visits to create (current_state, next_state) pairs
        # For each current visit, pair it with all subsequent visits
        for i in range(num_visits - 1):
            current_visit_series = subject_visits_df.iloc[i]
            for j in range(i + 1, num_visits):
                next_visit_series = subject_visits_df.iloc[j]

                time_delta = next_visit_series['months_since_bl'] - current_visit_series['months_since_bl']

                # Construct x_row data dictionary
                x_row_data = {sid_col: current_visit_series[sid_col]}
                x_row_data.update(current_visit_series[feature_columns].to_dict())
                x_row_data['months_since_bl'] = current_visit_series['months_since_bl']
                x_row_data['time_delta'] = time_delta
                model_x_list.append(x_row_data)
                
                # Construct y_row data dictionary
                y_row_data = {sid_col: next_visit_series[sid_col]}
                y_row_data.update(next_visit_series[feature_columns].to_dict())
                y_row_data['months_since_bl'] = next_visit_series['months_since_bl']
                model_y_list.append(y_row_data)

    # --- Integrated Check for Pair Counts ---
    actual_generated_pairs = len(model_x_list)
    if actual_generated_pairs != total_expected_pairs:
        # Detailed error message
        subject_counts_detail = input_original_data.groupby(sid_col).size()
        expected_pairs_detail = {}
        for sid, n_visits in subject_counts_detail.items():
            if n_visits >= 2:
                expected_pairs_detail[sid] = n_visits * (n_visits - 1) // 2
            else:
                expected_pairs_detail[sid] = 0
        
        # Check generated pairs per subject if possible (requires subject_id in model_x_list items)
        generated_pairs_per_subject_detail = {}
        if actual_generated_pairs > 0 and model_x_list and sid_col in model_x_list[0]:
            temp_df_for_generated_counts = pd.DataFrame(model_x_list)
            generated_counts = temp_df_for_generated_counts[sid_col].value_counts().to_dict()
            for sid in subject_counts_detail.keys(): # Iterate over all original subject_ids
                 generated_pairs_per_subject_detail[sid] = generated_counts.get(sid, 0)

        error_message = (
            f"Mismatch in generated pairs within transform_dataset. "
            f"Expected total pairs: {total_expected_pairs}, Got total pairs: {actual_generated_pairs}.\\n"
            f"Details per subject (Visits -> Expected Pairs | Generated Pairs):\\n"
        )
        for sid in subject_counts_detail.keys():
            expected = expected_pairs_detail.get(sid, "N/A")
            generated = generated_pairs_per_subject_detail.get(sid, "N/A")
            visits = subject_counts_detail.get(sid, "N/A")
            error_message += f"  Subject {sid}: ({visits} visits -> Expected: {expected} | Generated: {generated})\\n"

        raise ValueError(error_message)
    # --- End of Integrated Check ---

    # Define column order for model_x to match target_x.csv
    model_x_output_columns = [sid_col] + feature_columns + ['months_since_bl', 'time_delta']
    # Define column order for model_y to match target_y.csv
    model_y_output_columns = [sid_col] + feature_columns + ['months_since_bl']
    
    # Create DataFrames from the lists of dictionaries
    model_x = pd.DataFrame(model_x_list)
    model_y = pd.DataFrame(model_y_list)

    # Ensure correct column order and handle cases where no data pairs are generated
    if not model_x.empty:
        model_x = model_x[model_x_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        model_x = pd.DataFrame(columns=model_x_output_columns)

    if not model_y.empty:
        model_y = model_y[model_y_output_columns] # Use new column order for model_y
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        model_y = pd.DataFrame(columns=model_y_output_columns) # Use new column order for model_y
            
    return model_x, model_y


def transform_dataset_for_clinician(input_original_data: pd.DataFrame, feature_columns: List[str], action_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms the original patient visit data into clinician_x (current state)
    and clinician_y (current action) DataFrames for training a clinician policy model.
    
    This creates a direct mapping from patient state to clinical action: (state) -> (action)
    
    Args:
        input_original_data: DataFrame containing patient visit data
        feature_columns: List of feature column names (excluding actions)
        action_columns: List of action column names (e.g., drug "_active" columns)
    
    Returns:
        tuple: (clinician_x, clinician_y) DataFrames
    """
    clinician_x_list = []
    clinician_y_list = []

    # Calculate the total expected number of records (one per visit)
    total_expected_records = len(input_original_data)

    # Process each visit to create (state) -> (action) pairs
    sid_col = SUBJECT_ID_COL
    for idx, visit_row in input_original_data.iterrows():
        # Construct clinician_x_row data dictionary (current state)
        x_row_data = {sid_col: visit_row[sid_col]}
        # Add current state features
        x_row_data.update(visit_row[feature_columns].to_dict())
        # Include absolute time to satisfy test expectations for clinician state inputs
        if 'months_since_bl' in visit_row.index:
            x_row_data['months_since_bl'] = visit_row['months_since_bl']
        clinician_x_list.append(x_row_data)
        
        # Construct clinician_y_row data dictionary (current action)
        y_row_data = {sid_col: visit_row[sid_col]}
        # Add current actions (what the clinician decided to prescribe at this visit)
        y_row_data.update(visit_row[action_columns].to_dict())
        clinician_y_list.append(y_row_data)

    # --- Validation Check ---
    actual_generated_records = len(clinician_x_list)
    if actual_generated_records != total_expected_records:
        error_message = (
            f"Mismatch in generated records within transform_dataset_for_clinician. "
            f"Expected total records: {total_expected_records}, Got total records: {actual_generated_records}."
        )
        raise ValueError(error_message)
    # --- End of Validation Check ---

    # Define column order for clinician_x (state features + absolute time)
    clinician_x_output_columns = [sid_col] + feature_columns + ['months_since_bl']
    # Define column order for clinician_y (actions only)
    clinician_y_output_columns = [sid_col] + action_columns
    
    # Create DataFrames from the lists of dictionaries
    clinician_x = pd.DataFrame(clinician_x_list)
    clinician_y = pd.DataFrame(clinician_y_list)

    # Ensure correct column order and handle cases where no data pairs are generated
    if not clinician_x.empty:
        clinician_x = clinician_x[clinician_x_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        clinician_x = pd.DataFrame(columns=clinician_x_output_columns)

    if not clinician_y.empty:
        clinician_y = clinician_y[clinician_y_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        clinician_y = pd.DataFrame(columns=clinician_y_output_columns)
            
    return clinician_x, clinician_y
