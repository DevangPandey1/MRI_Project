import pytest
import pandas as pd
import numpy as np
import os
import json
import sys

# Ensure utils finds the correct config module (preprocessing/config.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import preprocessing.config as _config_module
sys.modules['config'] = _config_module

# Add project root to path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    calculate_active_medications,
    calculate_consistent_age,
    filter_subjects_by_visit_count,
    split_data_by_subject,
    align_columns,
    normalize_ids_post_merge,
    save_artifacts,
    transform_dataset,
    transform_dataset_for_clinician,
    get_alpaca_observation_columns,
    impute_missing_values,
    normalize_clinician_splits,
)
from imputation import train_mice_imputer, impute_with_trained_imputer
from mri_preprocess import MRIFinder, add_mri_paths

# Fixtures for sample data
@pytest.fixture
def sample_df():
    data = {
        'subject_id': [1, 1, 1, 2, 2, 3, 4, 4, 4, 4],
        'visit': [1, 2, 3, 1, 2, 1, 1, 2, 3, 4],
        'EXAMDATE': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2020-02-01', '2021-02-01', '2020-03-01', '2020-04-01', '2020-10-01', '2021-04-01', '2021-10-01']),
        'CMMED': ['Aricept', 'Lipitor', 'Donepezil', 'Zocor', 'Aspirin', 'OtherMed', 'No Medication', 'Ibuprofen', 'Lexapro', 'Namenda'],
        'CMBGNYR_DRVD': [2020, 2020, 2021, 2020, 2020, 2020, np.nan, 2020, 2021, 2021],
        'CMENDYR_DRVD': [2020, 2020, 2021, 2020, 2020, 2022, np.nan, 2020, 2021, 2021],
        'months_since_bl': [0, 6, 12, 0, 12, 0, 0, 6, 12, 18],
        'subject_age': [65.0, 65.5, 66.0, 70.0, 71.0, 80.0, 75.0, 75.5, 76.0, 76.5],
        'feature1': [10, 12, 11, 20, 22, 30, 40, 42, 41, 43],
        'PTGENDER': ['Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Female'],
        'PTRACCAT': ['White', 'White', 'White', 'Black', 'Black', 'Asian', 'White', 'White', 'White', 'White'],
        'research_group': ['CN', 'CN', 'CN', 'AD', 'AD', 'MCI', 'CN', 'CN', 'CN', 'CN']
    }
    return pd.DataFrame(data)

@pytest.fixture
def imputation_test_data():
    """Provides sample data for testing the imputation functions."""
    train_df = pd.DataFrame({
        'subject_id': [1, 1, 2, 2],
        'continuous_var': [10.0, 20.0, np.nan, 40.0],
        'ordinal_var': [1, np.nan, 3, 4],
        'PTGENDER': ['Male', 'Female', 'Male', 'Female'],
        'research_group': ['CN', 'AD', 'CN', 'AD']
    })
    val_df = pd.DataFrame({
        'subject_id': [3, 3],
        'continuous_var': [50.0, np.nan],
        'ordinal_var': [5, 6],
        'PTGENDER': ['Male', 'Female'],
        'research_group': ['MCI', 'MCI']
    })
    return train_df, val_df

def test_imputation_functions(imputation_test_data, monkeypatch):
    """
    Tests the train_mice_imputer and impute_with_trained_imputer functions.
    This is an integration test that checks if the imputation process works end-to-end.
    """
    train_df, val_df = imputation_test_data

    import imputation as imputation_mod

    class FakeImputer:
        def __init__(self, estimator=None, **kwargs):
            self.estimator = estimator
            self.means = None
            self.columns = None

        def fit(self, df):
            self.columns = df.columns.tolist()
            self.means = df.mean(numeric_only=True)

        def transform(self, df):
            filled = df.copy()
            for col in self.columns:
                if filled[col].isna().any():
                    filled[col] = filled[col].fillna(self.means[col])
            return filled.to_numpy()

    monkeypatch.setattr(imputation_mod, 'IterativeImputer', FakeImputer, raising=False)
    monkeypatch.setattr(imputation_mod, 'ExtraTreesRegressor', lambda **kwargs: object(), raising=False)

    # Define column types for the imputation
    continuous_vars = ['continuous_var']
    ordinal_vars = ['ordinal_var']
    categorical_vars = ['PTGENDER', 'research_group']
    drop_vars = []

    # 1. Test training the imputer
    imputer, schema = train_mice_imputer(
        train_df, continuous_vars, ordinal_vars, categorical_vars, drop_vars
    )
    assert imputer is not None
    assert isinstance(schema, list)

    # 2. Test imputing with the trained imputer
    imputed_val_df = impute_with_trained_imputer(
        val_df, imputer, continuous_vars, ordinal_vars, categorical_vars, drop_vars, schema
    )

    # Assertions
    assert not imputed_val_df['continuous_var'].isnull().any(), "Continuous column should be imputed."
    assert pd.api.types.is_integer_dtype(imputed_val_df['ordinal_var']), "Ordinal column should be rounded to integer."
    assert imputed_val_df.shape[0] == val_df.shape[0], "Number of rows should be preserved."

def test_clean_med_names_integration_for_actions(tmp_path, monkeypatch):
    # This test validates that, given CMMED_clean already present (produced by clean_med_names.py),
    # calculate_active_medications correctly maps to DRUG_CLASS_MAPPING and emits *_active flags.
    import utils as utils_mod
    import config as config_mod

    test_actions = [
        'AD_Treatment_Donepezil_active',
        'NSAID_Aspirin_active',
        'Other_Rare_active',
        'No_Medication_active',
    ]
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', test_actions, raising=False)

    meds_df = pd.DataFrame({
        'PTID': ['A','A','B','B'],
        'CMMED_clean': ['donepezil','aspirin','other','unknown'],  # 'other' passes through; 'unknown' -> Other_Rare via fillna
        'CMBGNYR_DRVD': [2020, 2020, 2021, 2021],
        'CMENDYR_DRVD': [None, 2020, None, None],
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)

    # Patch path join used inside calculate_active_medications
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    visits = pd.DataFrame({
        'subject_id': ['A','A','B','B'],
        'VISCODE2': ['m06','m18','m06','m12'],
        'EXAMDATE': pd.to_datetime(['2020-06-01','2021-06-01','2020-06-01','2021-06-01']),
        'months_since_bl': [6, 18, 6, 12],
        'subject_age': [70.5, 71.5, 68.5, 69.5],
    })

    out = calculate_active_medications(visits.copy())
    # Columns exist
    for col in test_actions:
        assert col in out.columns
    # Subject A: donepezil active 2020-2021 (open-ended) and aspirin active 2020 only
    a_rows = out[out['subject_id']=='A'].sort_values('EXAMDATE')
    assert a_rows['AD_Treatment_Donepezil_active'].tolist() == [1,1]
    assert a_rows['NSAID_Aspirin_active'].tolist() == [1,0]
    # Subject B: Other_Rare for 'unknown' in 2021 only
    b_rows = out[out['subject_id']=='B'].sort_values('EXAMDATE')
    assert b_rows['Other_Rare_active'].tolist() == [0,1]
    # No_Medication_active complements others
    for sid, rows in [('A', a_rows), ('B', b_rows)]:
        others = [c for c in test_actions if c != 'No_Medication_active']
        expect = (rows[others].sum(axis=1) == 0).astype(int).tolist()
        assert rows['No_Medication_active'].tolist() == expect

def test_calculate_active_medications_from_reccmeds_and_actions(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    # Define action space and mapping for the test
    test_actions = [
        'AD_Treatment_Donepezil_active',
        'Statin_Atorvastatin_active',
        'NSAID_Ibuprofen_active',
        'Other_Rare_active',
        'No_Medication_active',
    ]
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', test_actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {
        'donepezil': 'AD_Treatment_Donepezil',
        'atorvastatin': 'Statin_Atorvastatin',
        'ibuprofen': 'NSAID_Ibuprofen',
        # Any unmapped string should fall into Other_Rare
    }, raising=False)

    # Build a tiny RECCMEDS CSV in a temp path
    meds_df = pd.DataFrame({
        'PTID': ['A', 'A', 'A', 'B'],
        'CMMED_clean': ['donepezil', 'atorvastatin', 'unknownmed', 'ibuprofen'],
        'CMBGNYR_DRVD': [2020, 2020, 2020, 2021],
        'CMENDYR_DRVD': [None, 2021, None, None],  # open-ended for Donepezil/UnknownMed; Atorvastatin ends 2021
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)

    # Monkeypatch the path resolution inside calculate_active_medications to use our temp CSV
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    # Visits for two subjects (PTID A, PTID B), multiple years
    visits = pd.DataFrame({
        'subject_id': ['A','A','B','B'],
        'VISCODE2': ['m06','m18','m06','m12'],
        'EXAMDATE': pd.to_datetime(['2020-06-01','2021-06-01','2020-06-01','2021-06-01']),
        'months_since_bl': [6, 18, 6, 12],
        'subject_age': [70.5, 71.5, 68.5, 69.5],
    })

    out = calculate_active_medications(visits.copy())

    # a) All ACTION_FEATURES columns exist
    for col in test_actions:
        assert col in out.columns, f"Missing action column {col}"

    # b) No leaking between subjects
    # Subject A has no ibuprofen in RECCMEDS
    assert out[out['subject_id']=='A']['NSAID_Ibuprofen_active'].sum() == 0
    # Subject B has ibuprofen only in 2021; A's Donepezil should not appear for B
    assert out[out['subject_id']=='B']['AD_Treatment_Donepezil_active'].sum() == 0

    # c) Mapping correctness
    # Donepezil -> AD_Treatment_Donepezil active for A in 2020 and 2021 (open-ended)
    a_rows = out[out['subject_id']=='A'].sort_values('EXAMDATE')
    assert a_rows['AD_Treatment_Donepezil_active'].tolist() == [1, 1]
    # Atorvastatin active 2020-2021 inclusive; for the two visits (2020, 2021) it's 1,1
    assert a_rows['Statin_Atorvastatin_active'].tolist() == [1, 1]
    # UnknownMed -> Other_Rare (open-ended)
    assert a_rows['Other_Rare_active'].tolist() == [1, 1]

    # d) Overlap: verify multiple meds active simultaneously are all marked 1
    # For A in 2020 and 2021, three meds should be active simultaneously
    for _, row in a_rows.iterrows():
        assert row['AD_Treatment_Donepezil_active'] == 1
        assert row['Statin_Atorvastatin_active'] == 1
        assert row['Other_Rare_active'] == 1
    # For B: ibuprofen only in 2021; in 2020 all actions except No_Medication should be 0
    b_2020 = out[(out['subject_id']=='B') & (out['EXAMDATE'].dt.year==2020)].iloc[0]
    assert int(b_2020['NSAID_Ibuprofen_active']) == 0
    # No_Medication is 1 only when all others are 0
    non_no_cols = [c for c in test_actions if c != 'No_Medication_active']
    expect_no = int(sum(int(b_2020[c]) for c in non_no_cols) == 0)
    assert int(b_2020['No_Medication_active']) == expect_no

def test_calculate_consistent_age(sample_df):
    df = sample_df.copy()
    df.loc[1, 'subject_age'] = 68.0 # Introduce inconsistency
    processed_df = calculate_consistent_age(df)
    ages_subject1 = processed_df[processed_df['subject_id'] == 1]['subject_age']
    baseline_age = ages_subject1.iloc[0]
    assert np.isclose(ages_subject1.iloc[1], baseline_age + 0.5)
    assert np.isclose(ages_subject1.iloc[2], baseline_age + 1.0)

def test_filter_subjects_by_visit_count(sample_df):
    processed_df = filter_subjects_by_visit_count(sample_df, min_visits=3)
    assert 1 in processed_df['subject_id'].unique()
    assert 4 in processed_df['subject_id'].unique()
    assert 2 not in processed_df['subject_id'].unique()
    assert 3 not in processed_df['subject_id'].unique()

def test_split_data_by_subject(sample_df):
    train_df, val_df, test_df, train_subjects, val_subjects, test_subjects = split_data_by_subject(
        sample_df, test_size=0.4, val_size=0.2, random_state=42
    )
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0

def test_align_columns():
    train_df = pd.DataFrame({'a': [1], 'b': [2]})
    val_df = pd.DataFrame({'a': [3], 'c': [4]})
    test_df = pd.DataFrame({'b': [5]})
    _, aligned_val, aligned_test = align_columns(train_df, val_df, test_df)
    assert list(aligned_val.columns) == ['a', 'b']
    assert aligned_val.loc[0, 'a'] == 3
    assert aligned_val.loc[0, 'b'] == 0
    assert list(aligned_test.columns) == ['a', 'b']
    assert aligned_test.loc[0, 'a'] == 0
    assert aligned_test.loc[0, 'b'] == 5


def test_normalize_ids_post_merge_prefers_ptid_and_strips_whitespace():
    df = pd.DataFrame({
        'PTID': ['  S001 ', 'S002'],
        'RID': [101, 202],
        'subject_id': ['legacy-1', 'legacy-2'],
        'value': [1, 2],
    })
    out = normalize_ids_post_merge(df)
    # RID dropped and PTID normalized to subject_id with whitespace removed
    assert 'RID' not in out.columns
    assert 'PTID' not in out.columns
    assert out['subject_id'].tolist() == ['S001', 'S002']
    # Other columns preserved
    assert out['value'].tolist() == [1, 2]


def test_normalize_ids_post_merge_falls_back_to_rid_when_ptid_missing():
    df = pd.DataFrame({
        'RID': [345, 678],
        'feature': [10, 20],
    })
    out = normalize_ids_post_merge(df)
    assert out.columns.tolist()[0] == 'subject_id'
    # RID converted to string representation
    assert out['subject_id'].tolist() == ['345', '678']
    assert out['feature'].tolist() == [10, 20]

def test_transform_dataset_subject_pairing_count():
    """Tests if transform_dataset generates the correct number of pairs."""
    data = {
        'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        'months_since_bl': [0, 6, 12, 18, 24, 0, 6, 12, 0, 6, 0],
        'feature1': [10, 12, 11, 13, 14, 20, 22, 21, 30, 31, 40],
        'feature2': [1, 2, 1, 3, 2, 5, 6, 5, 8, 7, 9]
    }
    input_df = pd.DataFrame(data)
    feature_columns = ['feature1', 'feature2']
    expected_pairs = (5 * 4 // 2) + (3 * 2 // 2) + (2 * 1 // 2) + 0
    model_x, model_y = transform_dataset(input_df, feature_columns)
    assert len(model_x) == expected_pairs
    assert len(model_y) == expected_pairs
    assert 'time_delta' in model_x.columns

def test_transform_dataset_for_clinician():
    """Tests the clinician data transformation."""
    data = {
        'subject_id': [1, 1, 1, 2, 2],
        'months_since_bl': [0, 6, 12, 0, 6],
        'state1': [10, 11, 12, 20, 21],
        'action1_active': [0, 1, 0, 1, 0]
    }
    input_df = pd.DataFrame(data)
    state_cols = ['state1']
    action_cols = ['action1_active']
    x_df, y_df = transform_dataset_for_clinician(input_df, state_cols, action_cols)
    assert len(x_df) == len(input_df)
    assert len(y_df) == len(input_df)
    assert 'action1_active' in y_df.columns
    assert 'state1' in x_df.columns
    assert 'months_since_bl' in x_df.columns


def test_normalize_clinician_splits_scales_train_only(tmp_path):
    import joblib

    X_train = pd.DataFrame({
        'subject_id': ['A', 'B', 'C'],
        'state1': [0.0, 1.0, 2.0],
        'state2': [10.0, 15.0, 20.0],
        'note': ['x', 'y', 'z'],
    })
    X_val = pd.DataFrame({
        'subject_id': ['D', 'E'],
        'state1': [1.0, 1.5],
        'state2': [12.0, 18.0],
        'note': ['q', 'r'],
    })
    X_test = pd.DataFrame({
        'subject_id': ['F'],
        'state1': [0.5],
        'state2': [11.0],
        'note': ['s'],
    })

    scaled_train, scaled_val, scaled_test = normalize_clinician_splits(
        X_train, X_val, X_test, save_dir=str(tmp_path)
    )

    scaler_path = tmp_path / 'scaler_clinician_X.joblib'
    assert scaler_path.exists()
    scaler = joblib.load(scaler_path)
    assert list(scaler.feature_names_in_) == ['state1', 'state2']

    # subject_id preserved as first column
    assert scaled_train.columns.tolist()[0] == 'subject_id'
    assert scaled_val.columns.tolist()[0] == 'subject_id'
    assert scaled_test.columns.tolist()[0] == 'subject_id'

    # Training data should be centered (mean approximately 0)
    np.testing.assert_allclose(scaled_train['state1'].mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled_train['state2'].mean(), 0.0, atol=1e-9)

    # Val/Test scaled using same transform
    expected_val = scaler.transform([[1.0, 12.0]])[0]
    expected_test = scaler.transform([[0.5, 11.0]])[0]
    assert scaled_val['state1'].iloc[0] == pytest.approx(expected_val[0])
    assert scaled_val['state2'].iloc[0] == pytest.approx(expected_val[1])
    assert scaled_test['state1'].iloc[0] == pytest.approx(expected_test[0])
    assert scaled_test['state2'].iloc[0] == pytest.approx(expected_test[1])

def test_save_artifacts(tmp_path):
    """Tests saving of dataframes and schema."""
    data_dict = {'test_df': pd.DataFrame({'a': [1]})}
    schema = {'key': 'value'}
    base_dir = tmp_path / 'output'
    schema_dir = tmp_path / 'schema'
    save_artifacts(data_dict, schema, str(base_dir), [str(schema_dir)])
    assert os.path.exists(base_dir / 'test_df.csv')
    assert os.path.exists(schema_dir / 'columns_schema.json')


def test_preprocess_main_smoke_small_csv(tmp_path, monkeypatch):
    """
    Full integration smoke test with PTID-based meds join. Ensures:
    - No writes outside temp dirs
    - ACTION_FEATURES columns exist in model outputs, with correct overlaps
    """
    import preprocess as preprocess_mod
    import utils as utils_mod
    import config as config_mod

    # 1) Prepare temp directories
    data_dir = tmp_path / "data"
    model_training_dir = tmp_path / "model_training"
    clinician_dir = tmp_path / "clinician"
    alpaca_dir = tmp_path / "alpaca"
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(preprocess_mod, 'clean_med_names', lambda: None, raising=False)
    monkeypatch.setattr(preprocess_mod, 'impute_missing_values', lambda a, b, c: (a.copy(), b.copy(), c.copy()), raising=False)

    # 2) Synthetic ADNI_merged.csv (3 subjects, 2 visits each) with PTID/RID identifiers
    visits_df = pd.DataFrame({
        'PTID': ['S101', 'S101', 'S202', 'S202', 'S303', 'S303'],
        'RID': [101, 101, 202, 202, 303, 303],
        'VISCODE2':   ['m06','m18','m06','m12','m06','m12'],
        'EXAMDATE':   pd.to_datetime(['2020-06-01','2021-06-01','2020-06-01','2021-06-01','2020-06-01','2021-06-01']),
        'months_since_bl': [6,18,6,12,6,12],
        'subject_age': [70.5,71.5,68.5,69.5,72.5,73.5],
        'PTGENDER': ['Male','Male','Female','Female','Male','Male'],
        'PTRACCAT': ['White','White','Black','Black','White','White'],
        'research_group': ['CN','CN','AD','AD','MCI','MCI'],
        'IMAGEUID': [10101,10118,20206,20212,30306,30312],
    })
    (data_dir / 'ADNI_merged.csv').write_text('')  # ensure no overwrite of real file
    visits_df.to_csv(data_dir / 'ADNI_merged.csv', index=False)

    # 2b) Synthetic RECCMEDS
    meds_df = pd.DataFrame({
        'PTID': ['S101','S101','S202'],
        'CMMED_clean': ['donepezil','atorvastatin','unknownmed'],
        'CMBGNYR_DRVD': [2020, 2020, 2021],
        'CMENDYR_DRVD': [None, 2021, None],
    })
    meds_path = data_dir / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)

    # 3) Monkeypatch constants to temp dirs and minimal configs
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)

    test_actions = ['AD_Treatment_Donepezil_active', 'Statin_Atorvastatin_active', 'Other_Rare_active', 'No_Medication_active']
    monkeypatch.setattr(preprocess_mod, 'ACTION_FEATURES', test_actions, raising=False)
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', test_actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {
        'donepezil': 'AD_Treatment_Donepezil',
        'atorvastatin': 'Statin_Atorvastatin',
    }, raising=False)
    # Route meds path
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    # 4) Stubs to keep pipeline deterministic/light
    def fake_split_data_by_subject(df_in, test_size, val_size, random_state):
        subs = sorted(df_in['subject_id'].unique())
        train_subjects, val_subjects, test_subjects = {subs[0]}, {subs[1]}, {subs[2]}
        train_df = df_in[df_in['subject_id'].isin(train_subjects)].copy()
        val_df = df_in[df_in['subject_id'].isin(val_subjects)].copy()
        test_df = df_in[df_in['subject_id'].isin(test_subjects)].copy()
        return train_df, val_df, test_df, train_subjects, val_subjects, test_subjects

    def fake_get_alpaca_observation_columns():
        return ['subject_age', 'months_since_bl']

    def fake_impute_missing_values(train_df, val_df, test_df):
        return train_df.copy(), val_df.copy(), test_df.copy()

    def fake_normalize_and_encode_sequence_splits(X_train, X_val, X_test, *, categorical_cols=None):
        # Return identity X and minimal y with subject_id only
        y_train = X_train[[c for c in X_train.columns if c == 'subject_id']].copy()
        y_val = X_val[[c for c in X_val.columns if c == 'subject_id']].copy()
        y_test = X_test[[c for c in X_test.columns if c == 'subject_id']].copy()
        return X_train, X_val, X_test, y_train, y_val, y_test

    monkeypatch.setattr(preprocess_mod, 'split_data_by_subject', fake_split_data_by_subject, raising=False)
    monkeypatch.setattr(preprocess_mod, 'get_alpaca_observation_columns', fake_get_alpaca_observation_columns, raising=False)
    monkeypatch.setattr(preprocess_mod, 'impute_missing_values', fake_impute_missing_values, raising=False)
    monkeypatch.setattr(preprocess_mod, 'normalize_and_encode_sequence_splits', fake_normalize_and_encode_sequence_splits, raising=False)

    # 5) Run pipeline
    preprocess_mod.main()

    # 6) Artifacts exist only in temp dirs
    for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        assert os.path.exists(model_training_dir / f'{name}.csv')
    for name in ['clinician_X_train', 'clinician_X_val', 'clinician_X_test', 'clinician_y_train', 'clinician_y_val', 'clinician_y_test']:
        assert os.path.exists(clinician_dir / f'{name}.csv')
    for d in [model_training_dir, alpaca_dir, clinician_dir]:
        assert os.path.exists(d / 'columns_schema.json')

    # 7) Validate action columns and values in model X and subject_id in y
    X_train_df = pd.read_csv(model_training_dir / 'X_train.csv')
    y_train_df = pd.read_csv(model_training_dir / 'y_train.csv')
    for col in test_actions:
        assert col in X_train_df.columns
    assert set(X_train_df['subject_id'].unique()) == {'S101'}
    assert 'PTID' not in X_train_df.columns
    assert X_train_df['AD_Treatment_Donepezil_active'].tolist() == [1, 1]
    assert X_train_df['Statin_Atorvastatin_active'].tolist() == [1, 1]
    assert X_train_df['Other_Rare_active'].tolist() == [0, 0]
    assert X_train_df['No_Medication_active'].tolist() == [0, 0]
    assert 'subject_id' in y_train_df.columns


def test_next_visit_months_respects_mri_filter(tmp_path, monkeypatch):
    """
    Ensures next_visit_months is computed after dropping visits without IMAGEUID and
    correctly reflects gaps between remaining visits.

    Scenario: One subject with three scheduled visits at 0, 6, 12 months. The 6-month
    visit is missing IMAGEUID and is dropped. Expected next_visit_months: 12.0 for the
    baseline visit (0 -> 12), and 0.0 for the final visit.
    """
    import preprocess as preprocess_mod

    # 1) Temp dirs
    data_dir = tmp_path / "data"
    model_training_dir = tmp_path / "model_training"
    clinician_dir = tmp_path / "clinician"
    alpaca_dir = tmp_path / "alpaca"
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2) ADNI_merged with IMAGEUID present at 0 and 12 months, missing at 6 months
    visits_df = pd.DataFrame({
        'subject_id': ['S1','S1','S1'],
        'VISCODE2':   ['bl','m06','m12'],
        'EXAMDATE':   pd.to_datetime(['2020-01-01','2020-07-01','2021-01-01']),
        'months_since_bl': [0.0, 6.0, 12.0],
        'subject_age': [70.0, 70.5, 71.0],
        'PTGENDER': ['Male','Male','Male'],
        'PTRACCAT': ['White','White','White'],
        'research_group': ['CN','CN','CN'],
        'IMAGEUID': [12345, np.nan, 67890],
    })
    (data_dir / 'ADNI_merged.csv').write_text('')
    visits_df.to_csv(data_dir / 'ADNI_merged.csv', index=False)

    # 3) Minimal config patches and function stubs to isolate next_visit_months behavior
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)
    # Keep numeric vars minimal but retain months_since_bl for sequencing logic
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    # No actions required for this test
    monkeypatch.setattr(preprocess_mod, 'ACTION_FEATURES', [], raising=False)

    # Stub: do nothing for meds cleaning and meds flags
    monkeypatch.setattr(preprocess_mod, 'clean_med_names', lambda: None, raising=False)
    monkeypatch.setattr(preprocess_mod, 'calculate_active_medications', lambda df: df, raising=False)

    # Stub: split -> all rows to train to make checks simple
    def fake_split_data_by_subject(df_in, test_size, val_size, random_state):
        subjects = set(df_in['subject_id'].unique())
        return df_in.copy(), df_in.iloc[0:0].copy(), df_in.iloc[0:0].copy(), subjects, set(), set()
    monkeypatch.setattr(preprocess_mod, 'split_data_by_subject', fake_split_data_by_subject, raising=False)

    # Stub: no-op imputation
    monkeypatch.setattr(preprocess_mod, 'impute_missing_values', lambda a,b,c: (a.copy(), b.copy(), c.copy()), raising=False)

    # Stub: identity normalization that preserves columns (including next_visit_months)
    def fake_normalize_and_encode_sequence_splits(X_train, X_val, X_test, *, categorical_cols=None):
        # Identity for X; y minimal with subject_id
        y_train = X_train[[c for c in X_train.columns if c == 'subject_id']].copy()
        y_val = X_val[[c for c in X_val.columns if c == 'subject_id']].copy()
        y_test = X_test[[c for c in X_test.columns if c == 'subject_id']].copy()
        return X_train.copy(), X_val.copy(), X_test.copy(), y_train, y_val, y_test
    monkeypatch.setattr(preprocess_mod, 'normalize_and_encode_sequence_splits', fake_normalize_and_encode_sequence_splits, raising=False)

    # 4) Run pipeline
    preprocess_mod.main()

    # 5) Validate next_visit_months on the training output
    X_train = pd.read_csv(model_training_dir / 'X_train.csv')
    # Two visits remain (0 and 12 months)
    assert len(X_train) == 2
    assert 'next_visit_months' in X_train.columns
    # Sorted by subject_id, months_since_bl inside main before computing the gap
    X_train_sorted = X_train.sort_values(['subject_id','months_since_bl']).reset_index(drop=True)
    assert X_train_sorted['months_since_bl'].tolist() == [0.0, 12.0]
    assert X_train_sorted['next_visit_months'].tolist() == [12.0, 0.0]

def test_calculate_active_medications_no_meds_all_visits(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    # Only action we care about here
    actions = ['No_Medication_active']
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)

    # Build visits with PTID but RECCMEDS contains no rows for this PTID
    df = pd.DataFrame({
        "subject_id": [10, 10],
        "PTID": ["X10","X10"],
        "VISCODE2": ["m06","m12"],
        "EXAMDATE": pd.to_datetime(["2020-01-01", "2021-01-01"]),
    })

    # Empty meds CSV so PTID X10 has no medications recorded
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    pd.DataFrame({"PTID": [], "CMMED_clean": [], "CMBGNYR_DRVD": [], "CMENDYR_DRVD": []}).to_csv(meds_path, index=False)
    # Route path inside utils
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    out = calculate_active_medications(df.copy())
    assert 'No_Medication_active' in out.columns
    assert out['No_Medication_active'].tolist() == [1, 1]


def test_active_medications_time_window_boundaries(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    # Configure a single action and No_Medication
    actions = ['A_active', 'No_Medication_active']
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {'a': 'A'}, raising=False)

    # Visits at boundaries and outside
    visits = pd.DataFrame({
        'subject_id': ['S','S','S','S'],
        'VISCODE2': ['m06','m12','m18','m24'],
        'EXAMDATE': pd.to_datetime(['2019-06-01','2020-06-01','2021-06-01','2022-06-01']),
        'months_since_bl': [6,12,18,24],
        'subject_age': [70.5,71.5,72.5,73.5],
    })

    # Med active from 2020 to 2021 inclusive
    meds_df = pd.DataFrame({
        'PTID': ['S'],
        'CMMED_clean': ['a'],
        'CMBGNYR_DRVD': [2020],
        'CMENDYR_DRVD': [2021],
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)
    # Route path
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    out = calculate_active_medications(visits.copy())
    # Expect 0,1,1,0 for A_active across 2019-2022
    assert out.sort_values('EXAMDATE')['A_active'].tolist() == [0,1,1,0]
    # No_Medication_active complement
    assert out.sort_values('EXAMDATE')['No_Medication_active'].tolist() == [1,0,0,1]


def test_active_medications_overlaps_and_duplicates(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    actions = ['A_active', 'No_Medication_active']
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {'a': 'A'}, raising=False)

    visits = pd.DataFrame({
        'subject_id': ['S','S','S','S'],
        'VISCODE2': ['m06','m12','m18','m24'],
        'EXAMDATE': pd.to_datetime(['2019-06-01','2020-06-01','2021-06-01','2022-06-01']),
    })

    # Two overlapping rows for same class A
    meds_df = pd.DataFrame({
        'PTID': ['S','S'],
        'CMMED_clean': ['a','a'],
        'CMBGNYR_DRVD': [2019, 2020],
        'CMENDYR_DRVD': [2020, 2022],
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    out = calculate_active_medications(visits.copy()).sort_values('EXAMDATE')
    assert out['A_active'].tolist() == [1,1,1,1]
    assert out['No_Medication_active'].tolist() == [0,0,0,0]


def test_active_medications_invalid_windows_and_missing_years(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    actions = ['A_active', 'No_Medication_active']
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {'a': 'A'}, raising=False)

    visits = pd.DataFrame({
        'subject_id': ['S','S','S'],
        'VISCODE2': ['m06','m12','m18'],
        'EXAMDATE': pd.to_datetime(['2020-06-01','2021-06-01','2022-06-01']),
    })

    meds_df = pd.DataFrame({
        'PTID': ['S','S'],
        'CMMED_clean': ['a', 'a'],
        'CMBGNYR_DRVD': [2021, None],
        'CMENDYR_DRVD': [2020, 2022],  # first invalid (end < start); second missing start
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    out = calculate_active_medications(visits.copy()).sort_values('EXAMDATE')
    # No activation due to invalid/missing windows
    assert out['A_active'].tolist() == [0,0,0]
    assert out['No_Medication_active'].tolist() == [1,1,1]


def test_unknown_meds_without_other_rare_bucket(tmp_path, monkeypatch):
    import utils as utils_mod
    import config as config_mod

    actions = ['A_active', 'No_Medication_active']  # no Other_Rare_active configured
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {'a': 'A'}, raising=False)

    visits = pd.DataFrame({
        'subject_id': ['S','S'],
        'VISCODE2': ['m06','m12'],
        'EXAMDATE': pd.to_datetime(['2020-06-01','2021-06-01']),
    })

    meds_df = pd.DataFrame({
        'PTID': ['S'],
        'CMMED_clean': ['unknownzzz'],  # not in mapping
        'CMBGNYR_DRVD': [2020],
        'CMENDYR_DRVD': [None],
    })
    meds_path = tmp_path / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    out = calculate_active_medications(visits.copy()).sort_values('EXAMDATE')
    # Unknown meds should not trigger any known action; No_Medication_active fills
    assert out['A_active'].tolist() == [0,0]
    assert out['No_Medication_active'].tolist() == [1,1]

def test_calculate_consistent_age_drops_invalid_and_aligns():

    df = pd.DataFrame({
        "subject_id": [1, 1, 1, 2],
        "months_since_bl": [0, 6, np.nan, 0],  # one invalid row
        "subject_age": [70.0, 70.5, 71.0, np.nan],  # one invalid row
    })
    out = calculate_consistent_age(df)
    # Invalid rows dropped => keep (1,0), (1,6)
    assert set(map(tuple, out[["subject_id","months_since_bl"]].values)) == {(1,0),(1,6)}
    # Consistency: 6 months later = +0.5 years
    s1 = out[out.subject_id == 1].sort_values("months_since_bl")["subject_age"].tolist()
    assert len(s1) == 2 and abs((s1[1] - s1[0]) - 0.5) < 1e-6


def test_normalize_and_encode_subject_id_and_one_hot(tmp_path, monkeypatch):
    import pandas as pd
    import numpy as np
    import utils as utils_mod

    # Force which columns get scaled
    monkeypatch.setattr(utils_mod, "CONTINUOUS_VARS", ["z1"])
    monkeypatch.setattr(utils_mod, "ORDINAL_VARS", ["z2"])

    # Build splits with subject_id in val only, per contract in preprocess.main
    X_train = pd.DataFrame({"z1":[0.0, 2.0], "z2":[1.0, 3.0], "time_delta":[1.0, 2.0], "PTGENDER":["Male","Female"], "PTRACCAT":["White","Black"]})
    X_val   = pd.DataFrame({"subject_id":[10,11], "z1":[1.0, 1.0], "z2":[2.0, 2.0], "time_delta":[1.5, 1.5], "PTGENDER":["Male","Male"], "PTRACCAT":["White","White"]})
    X_test  = pd.DataFrame({"z1":[4.0], "z2":[5.0], "time_delta":[3.0], "PTGENDER":["Female"], "PTRACCAT":["Black"]})

    y_train = pd.DataFrame({"z1":[10.0, 20.0], "z2":[1.0, 2.0]})
    y_val   = pd.DataFrame({"subject_id":[10,11], "z1":[15.0, 25.0], "z2":[1.5, 2.5]})
    y_test  = pd.DataFrame({"z1":[30.0], "z2":[3.0]})

    # Prevent scalers from being written to the real project directory
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    X_tr, X_va, X_te, y_tr, y_va, y_te = utils_mod.normalize_and_encode(X_train, X_val, X_test, y_train, y_val, y_test)

    # subject_id only preserved in val splits
    assert "subject_id" not in X_tr.columns and "subject_id" in X_va.columns
    assert "subject_id" not in y_tr.columns and "subject_id" in y_va.columns

    # One-hot encoding applied (original categorical cols removed)
    assert "PTGENDER" not in X_tr.columns and any(c.startswith("PTGENDER_") for c in X_tr.columns)
    assert "PTRACCAT" not in X_tr.columns and any(c.startswith("PTRACCAT_") for c in X_tr.columns)

    # Scaling happened for X_train: z1, z2, time_delta centered near 0
    assert abs(X_tr["z1"].mean()) < 1e-6
    assert abs(X_tr["z2"].mean()) < 1e-6
    assert abs(X_tr["time_delta"].mean()) < 1e-6

def test_transform_dataset_time_delta_and_alignment():
    import pandas as pd
    from utils import transform_dataset
    df = pd.DataFrame({
        "subject_id":[1,1,1],
        "months_since_bl":[0,6,12],
        "f":[10,11,12]
    })
    X, Y = transform_dataset(df, ["f"])
    # pairs: (0->6),(0->12),(6->12)
    assert X.shape[0] == 3 and Y.shape[0] == 3
    # Check time_deltas exactly
    assert X["time_delta"].tolist() == [6, 12, 6]
    # Check alignment of months_since_bl
    assert X["months_since_bl"].tolist() == [0, 0, 6]
    assert Y["months_since_bl"].tolist() == [6, 12, 12]

def test_transform_dataset_for_clinician_maps_actions():
    import pandas as pd
    from utils import transform_dataset_for_clinician
    df = pd.DataFrame({
        "subject_id":[1,1],
        "months_since_bl":[0,6],
        "s":[5,6],
        "A_active":[0,1],
    })
    X, Y = transform_dataset_for_clinician(df, ["s"], ["A_active"])
    assert X["s"].tolist() == [5,6]
    assert Y["A_active"].tolist() == [0,1]


def test_impute_missing_values_wrapper_saves_schema(tmp_path, monkeypatch):
    import utils as utils_mod
    import numpy as np
    import pandas as pd
    import imputation as imputation_mod

    class FakeImputer:
        def __init__(self, estimator=None, **kwargs):
            self.estimator = estimator
            self.columns = None
            self.means = None

        def fit(self, df):
            self.columns = df.columns.tolist()
            self.means = df.mean(numeric_only=True)

        def transform(self, df):
            filled = df.copy()
            for col in self.columns:
                if filled[col].isna().any():
                    filled[col] = filled[col].fillna(self.means[col])
            return filled.to_numpy()

    monkeypatch.setattr(imputation_mod, 'IterativeImputer', FakeImputer, raising=False)
    monkeypatch.setattr(imputation_mod, 'ExtraTreesRegressor', lambda **kwargs: object(), raising=False)

    # Narrow the variables to those present in our toy data
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['cont'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', ['ord'], raising=False)
    monkeypatch.setattr(utils_mod, 'CATEGORICAL_VARS_FOR_IMPUTATION', ['PTGENDER', 'research_group'], raising=False)
    monkeypatch.setattr(utils_mod, 'DROP_VARS_FOR_IMPUTATION', [], raising=False)
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    # Build simple splits with NaNs
    train_df = pd.DataFrame({
        'subject_id': [1,1,2,2],
        'cont': [1.0, np.nan, 3.0, 4.0],
        'ord': [1, 2, np.nan, 4],
        'PTGENDER': ['Male','Female','Male','Female'],
        'research_group': ['CN','AD','CN','AD']
    })
    val_df = pd.DataFrame({
        'subject_id': [3,3],
        'cont': [np.nan, 6.0],
        'ord': [2, 3],
        'PTGENDER': ['Male','Male'],
        'research_group': ['MCI','MCI']
    })
    test_df = pd.DataFrame({
        'subject_id': [4],
        'cont': [np.nan],
        'ord': [np.nan],
        'PTGENDER': ['Female'],
        'research_group': ['CN']
    })

    imputed_train, imputed_val, imputed_test = utils_mod.impute_missing_values(train_df, val_df, test_df)

    # No NaNs should remain in imputed targets
    for df_out in [imputed_train, imputed_val, imputed_test]:
        assert not df_out['cont'].isna().any()
        assert not df_out['ord'].isna().any()
        # Ordinal should be ints after rounding
        assert pd.api.types.is_integer_dtype(df_out['ord'])

    # Schema file saved
    assert (tmp_path / 'mice_imputer_schema.joblib').exists()


def test_get_alpaca_observation_columns_default(tmp_path, monkeypatch):
    import utils as utils_mod

    # Point ALPACA_DIR to an empty temp dir so fallback list is used
    alp_dir = tmp_path / 'alp'
    alp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(utils_mod, 'ALPACA_DIR', str(alp_dir), raising=False)

    cols = utils_mod.get_alpaca_observation_columns()
    # Ensure key expected defaults are present
    assert 'months_since_bl' in cols
    assert 'PTGENDER_Male' in cols

def test_get_alpaca_observation_columns_from_file(tmp_path, monkeypatch):
    import utils as utils_mod
    # Create an ALPACA_DIR with a minimal X_train.csv including only a subset of OHE binaries
    alp_dir = tmp_path / 'alp_from_file'
    alp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(utils_mod, 'ALPACA_DIR', str(alp_dir), raising=False)

    # Minimal X_train with the needed columns present
    pd.DataFrame({
        'TAU_data': [0.0],
        'subject_age': [70.0],
        'months_since_bl': [0],
        'PTGENDER_Male': [1],
        'PTRACCAT_White': [1],
    }).to_csv(alp_dir / 'X_train.csv', index=False)

    cols = utils_mod.get_alpaca_observation_columns()
    # Should include continuous defaults and only the present OHE binaries from the file
    assert 'PTGENDER_Male' in cols and 'PTRACCAT_White' in cols
    assert 'PTRACCAT_Black' not in cols  # Not present in file
    assert 'months_since_bl' in cols


def test_normalize_and_encode_unseen_category_alignment(tmp_path, monkeypatch):
    import utils as utils_mod
    # Restrict which variables get scaled to make this predictable
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['numeric_feature'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)
    # Prevent scalers from being written to the real project directory
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    train_features = pd.DataFrame({
        'numeric_feature': [0.0, 1.0],
        'time_delta': [1.0, 2.0],
        'PTGENDER': ['Male', 'Male'],  # Train sees only 'Male'
        'PTRACCAT': ['White', 'White'],
    })
    val_features = pd.DataFrame({
        'subject_id': [1, 2],
        'numeric_feature': [2.0, 3.0],
        'time_delta': [1.5, 1.5],
        'PTGENDER': ['Female', 'Female'],  # Unseen category
        'PTRACCAT': ['Black', 'Black'],
    })
    test_features = pd.DataFrame({
        'numeric_feature': [4.0],
        'time_delta': [2.5],
        'PTGENDER': ['Female'],
        'PTRACCAT': ['Black'],
    })
    train_targets = pd.DataFrame({'numeric_feature': [10.0, 20.0]})
    val_targets = pd.DataFrame({'subject_id': [1, 2], 'numeric_feature': [15.0, 25.0]})
    test_targets = pd.DataFrame({'numeric_feature': [30.0]})

    processed_X_train, processed_X_val, processed_X_test, processed_y_train, processed_y_val, processed_y_test = utils_mod.normalize_and_encode(
        train_features.copy(), val_features.copy(), test_features.copy(),
        train_targets.copy(), val_targets.copy(), test_targets.copy()
    )

    # After OHE and align to train, val/test should not contain unseen category columns
    assert not any(c.startswith('PTGENDER_') and 'Female' in c for c in processed_X_val.columns)
    assert any(c == 'PTGENDER_Male' for c in processed_X_val.columns)
    # Rows that were Female should have PTGENDER_Male == 0
    if 'PTGENDER_Male' in processed_X_val.columns:
        assert (processed_X_val['PTGENDER_Male'] == 0).all()
    # subject_id preserved in val
    assert 'subject_id' in processed_X_val.columns and 'subject_id' in processed_y_val.columns


def test_preprocess_main_schema_consistency(tmp_path, monkeypatch):
    import preprocess as preprocess_mod
    import utils as utils_mod
    import config as config_mod
    import json

    # Setup directories
    data_dir = tmp_path / 'data2'
    model_training_dir = tmp_path / 'mt2'
    clinician_dir = tmp_path / 'clin2'
    alpaca_dir = tmp_path / 'alp2'
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(preprocess_mod, 'clean_med_names', lambda: None, raising=False)
    monkeypatch.setattr(preprocess_mod, 'impute_missing_values', lambda a, b, c: (a.copy(), b.copy(), c.copy()), raising=False)

    # Minimal ADNI_merged.csv using subject_id only
    df = pd.DataFrame({
        'subject_id': [1,1,2,2,3,3],
        'VISCODE2': ['m06','m12','m06','m12','m06','m12'],
        'EXAMDATE': pd.to_datetime(['2020-06-01','2021-06-01','2020-06-01','2021-06-01','2020-06-01','2021-06-01']),
        'months_since_bl': [6,12,6,12,6,12],
        'subject_age': [70.5,71.5,75.5,76.5,65.5,66.5],
        'PTGENDER': ['Male','Male','Female','Female','Male','Male'],
        'PTRACCAT': ['White','White','Black','Black','White','White'],
        'research_group': ['CN','CN','AD','AD','MCI','MCI'],
        'IMAGEUID': [11006,11012,22006,22012,33006,33012],
    })
    df.to_csv(data_dir / 'ADNI_merged.csv', index=False)

    # RECCMEDS with one action for P1 only
    meds_df = pd.DataFrame({
        'PTID': ['P1'],
        'CMMED_clean': ['donepezil'],
        'CMBGNYR_DRVD': [2020],
        'CMENDYR_DRVD': [None],
    })
    meds_path = data_dir / 'RECCMEDS_10Sep2025.csv'
    meds_df.to_csv(meds_path, index=False)

    # Configure preprocess to use our dirs and minimal settings
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)
    actions = ['AD_Treatment_Donepezil_active', 'Other_Rare_active', 'No_Medication_active']
    monkeypatch.setattr(preprocess_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(config_mod, 'ACTION_FEATURES', actions, raising=False)
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {'donepezil': 'AD_Treatment_Donepezil'}, raising=False)
    # Route meds path
    orig_join = utils_mod.os.path.join
    def fake_join(*args):
        if len(args) >= 4 and args[-1] == 'RECCMEDS_10Sep2025.csv':
            return str(meds_path)
        return orig_join(*args)
    monkeypatch.setattr(utils_mod.os.path, 'join', fake_join, raising=False)

    # Keep observation columns minimal
    monkeypatch.setattr(preprocess_mod, 'get_alpaca_observation_columns', lambda: ['subject_age', 'months_since_bl'], raising=False)

    # Run full pipeline
    preprocess_mod.main()

    # Load artifacts and schema
    X_train = pd.read_csv(model_training_dir / 'X_train.csv')
    X_val = pd.read_csv(model_training_dir / 'X_val.csv')
    X_test = pd.read_csv(model_training_dir / 'X_test.csv')
    y_train = pd.read_csv(model_training_dir / 'y_train.csv')
    y_val = pd.read_csv(model_training_dir / 'y_val.csv')
    y_test = pd.read_csv(model_training_dir / 'y_test.csv')
    with open(model_training_dir / 'columns_schema.json', 'r') as f:
        schema = json.load(f)

    # Schema consistency checks
    expected_model_input_cols = [
        c for c in X_train.columns
        if c not in ('subject_id', 'months_since_bl') and c not in config_mod.BYPASS_IMPUTATION
    ]
    assert schema['model_input_cols'] == expected_model_input_cols
    # action_cols should be subset of configured actions and present in X_train
    assert all(a in actions for a in schema['action_cols'])
    for a in schema['action_cols']:
        assert a in X_train.columns

    # Model input columns include months_since_bl if present
    assert 'model_input_cols' in schema
    # y files exist and contain subject_id; helpers should not be targets in schema
    assert 'subject_id' in y_train.columns
    assert 'IMAGEUID' in y_train.columns
    assert 'IMAGEUID' not in schema.get('y_cols', [])
    assert 'IMAGEUID' in schema.get('y_helper', [])
    # X should include IMAGEUID as passthrough (not in model_input_cols)
    assert 'IMAGEUID' in X_train.columns


def test_scalers_round_trip_and_expected_columns(tmp_path, monkeypatch):
    """
    Validates that:
    - normalize_and_encode saves scaler_X and scaler_y with expected feature columns
      (X: CONTINUOUS_VARS + ORDINAL_VARS + ['time_delta']; y: CONTINUOUS_VARS + ORDINAL_VARS)
    - Both scalers can round-trip (inverse_transform(transform(data)) ~= data)
    - Expected columns presence matches what rollout_evaluation.py relies on.
    """
    import joblib
    import utils as utils_mod

    # Point MODEL_TRAINING_DIR to temp for saving scalers
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    # Define which variables get scaled
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['continuous_feature'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', ['ordinal_feature'], raising=False)

    # Build tiny splits; include categorical columns and an action-like column
    X_train = pd.DataFrame({
        'continuous_feature': [0.0, 2.0, -1.0],
        'ordinal_feature': [1.0, 3.0, 5.0],
        'time_delta': [1.0, 2.0, 3.0],
        'PTGENDER': ['Male', 'Female', 'Male'],
        'PTRACCAT': ['White', 'Black', 'White'],
        'A_active': [0, 1, 0],
    })
    X_val = pd.DataFrame({
        'subject_id': [10, 11],
        'continuous_feature': [1.0, 1.0],
        'ordinal_feature': [2.0, 2.0],
        'time_delta': [1.5, 1.5],
        'PTGENDER': ['Male', 'Male'],
        'PTRACCAT': ['White', 'White'],
        'A_active': [1, 0],
    })
    X_test = pd.DataFrame({
        'continuous_feature': [4.0],
        'ordinal_feature': [5.0],
        'time_delta': [3.0],
        'PTGENDER': ['Female'],
        'PTRACCAT': ['Black'],
        'A_active': [0],
    })

    y_train = pd.DataFrame({
        'continuous_feature': [10.0, 20.0, 30.0],
        'ordinal_feature': [1.0, 2.0, 3.0],
    })
    y_val = pd.DataFrame({
        'subject_id': [10, 11],
        'continuous_feature': [15.0, 25.0],
        'ordinal_feature': [1.5, 2.5],
    })
    y_test = pd.DataFrame({
        'continuous_feature': [35.0],
        'ordinal_feature': [3.5],
    })

    # Run normalization/encoding to produce scalers
    processed_X_train, processed_X_val, processed_X_test, processed_y_train, processed_y_val, processed_y_test = utils_mod.normalize_and_encode(
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train.copy(), y_val.copy(), y_test.copy()
    )

    # Scalers saved (by normalize_and_encode) to MODEL_TRAINING_DIR as 'scaler_X.joblib'/'scaler_y.joblib'
    scaler_X_path = tmp_path / 'scaler_X.joblib'
    scaler_y_path = tmp_path / 'scaler_y.joblib'
    assert scaler_X_path.exists() and scaler_y_path.exists()

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Expected columns in scalers
    expected_X_cols = ['continuous_feature', 'ordinal_feature', 'time_delta']
    expected_y_cols = ['continuous_feature', 'ordinal_feature']

    assert hasattr(scaler_X, 'feature_names_in_')
    assert hasattr(scaler_y, 'feature_names_in_')
    assert list(scaler_X.feature_names_in_) == expected_X_cols
    assert list(scaler_y.feature_names_in_) == expected_y_cols

    # Round-trip test for X scaler
    original_X_numeric = X_train[expected_X_cols]
    scaled_X = scaler_X.transform(original_X_numeric)
    unscaled_X = scaler_X.inverse_transform(scaled_X)
    np.testing.assert_allclose(unscaled_X, original_X_numeric.values, rtol=1e-6, atol=1e-6)

    # Round-trip test for y scaler
    original_y_numeric = y_train[expected_y_cols]
    scaled_y = scaler_y.transform(original_y_numeric)
    unscaled_y = scaler_y.inverse_transform(scaled_y)
    np.testing.assert_allclose(unscaled_y, original_y_numeric.values, rtol=1e-6, atol=1e-6)

    # Simulate rollout unscaling checks: verify required columns are present
    # Build a predicted trajectory frame containing both target and action features
    pred_df = pd.DataFrame({
        'continuous_feature': scaled_y[:, 0],
        'ordinal_feature': scaled_y[:, 1],
        'A_active': X_train['A_active'].iloc[: scaled_y.shape[0]].values,
        'time_delta': X_train['time_delta'].iloc[: scaled_y.shape[0]].values,
    })
    # Ensure scaler_y columns exist and can be inverse transformed
    pred_y_block = pred_df[expected_y_cols]
    _ = scaler_y.inverse_transform(pred_y_block)
    # Ensure scaler_X columns exist and can be inverse transformed
    pred_x_block = pred_df[expected_X_cols]
    _ = scaler_X.inverse_transform(pred_x_block)

    # Columns that should not be part of any scaler
    assert 'A_active' not in scaler_X.feature_names_in_
    assert 'A_active' not in scaler_y.feature_names_in_
    # Categorical one-hot columns are created after scaling and are not part of scaler_y
    assert 'PTRACCAT_White' not in scaler_y.feature_names_in_


def test_sequence_pair_generation_subjectid_imageuid(tmp_path, monkeypatch):
    import utils as utils_mod

    # Ensure months_since_bl is considered numeric for scaling in X
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)

    # Build a simple training dataframe with two subjects and 3 visits each
    train_df = pd.DataFrame({
        'subject_id': ['A','A','A','B','B','B'],
        'months_since_bl': [0.0, 6.0, 12.0, 0.0, 12.0, 24.0],
        'IMAGEUID': [100, 101, 102, 200, 201, 202],
        'A_active': [0, 1, 0, 0, 0, 1],
        'PTGENDER': ['Male','Male','Male','Female','Female','Female'],
    })
    empty = train_df.iloc[0:0].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = utils_mod.normalize_and_encode_sequence_splits(
        train_df.copy(), empty.copy(), empty.copy()
    )

    # Expect consecutive pairs: A has 2 pairs (0->6, 6->12), B has 2 pairs (0->12, 12->24)
    assert len(X_train) == 4
    assert len(y_train) == 4

    # subject_id present in both X and y
    assert 'subject_id' in X_train.columns
    assert 'subject_id' in y_train.columns

    # y should include IMAGEUID
    assert 'IMAGEUID' in y_train.columns

    # Build expected next IMAGEUID list in row order produced by the function (subject, then time)
    expected_next_uids = []
    for sid, grp in train_df.groupby('subject_id'):
        grp = grp.sort_values('months_since_bl').reset_index(drop=True)
        for i in range(len(grp) - 1):
            expected_next_uids.append(grp.loc[i+1, 'IMAGEUID'])

    assert y_train['IMAGEUID'].tolist() == expected_next_uids
    # subject_ids match row-wise
    assert X_train['subject_id'].tolist() == y_train['subject_id'].tolist()


def test_preprocess_outputs_contain_subject_id_in_all_y_splits(tmp_path, monkeypatch):
    """
    Integration check: after running preprocess.main, all y splits (train/val/test)
    include subject_id and align with the corresponding X split rows.
    """
    import preprocess as preprocess_mod
    import utils as utils_mod
    import config as config_mod

    # Temp dirs
    data_dir = tmp_path / "data"
    model_training_dir = tmp_path / "model_training"
    clinician_dir = tmp_path / "clinician"
    alpaca_dir = tmp_path / "alpaca"
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build dataset with multiple subjects so val/test receive groups deterministically
    subjects = [f"S{i:03d}" for i in range(1, 7)]  # 6 subjects
    rows = []
    uid = 1000
    for s in subjects:
        # exactly two visits per subject to guarantee one pair
        rows.append({
            'subject_id': s,
            'VISCODE2': 'm06',
            'EXAMDATE': pd.to_datetime('2020-06-01'),
            'months_since_bl': 6.0,
            'subject_age': 70.0,
            'PTGENDER': 'Male' if int(s[1:]) % 2 else 'Female',
            'PTRACCAT': 'White',
            'research_group': 'CN',
            'IMAGEUID': uid,
        })
        uid += 1
        rows.append({
            'subject_id': s,
            'VISCODE2': 'm12',
            'EXAMDATE': pd.to_datetime('2021-06-01'),
            'months_since_bl': 12.0,
            'subject_age': 71.0,
            'PTGENDER': 'Male' if int(s[1:]) % 2 else 'Female',
            'PTRACCAT': 'White',
            'research_group': 'CN',
            'IMAGEUID': uid,
        })
        uid += 1
    visits_df = pd.DataFrame(rows)
    visits_df.to_csv(data_dir / 'ADNI_merged.csv', index=False)

    # Minimal meds CSV referenced by calculate_active_medications (empty is fine)
    meds_path = data_dir / 'RECCMEDS_10Sep2025.csv'
    pd.DataFrame({"PTID": [], "CMMED_clean": [], "CMBGNYR_DRVD": [], "CMENDYR_DRVD": []}).to_csv(meds_path, index=False)

    # Monkeypatch dirs and minimal config
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)

    # Run pipeline
    preprocess_mod.main()

    # Load outputs
    X_train = pd.read_csv(model_training_dir / 'X_train.csv')
    X_val = pd.read_csv(model_training_dir / 'X_val.csv')
    X_test = pd.read_csv(model_training_dir / 'X_test.csv')
    y_train = pd.read_csv(model_training_dir / 'y_train.csv')
    y_val = pd.read_csv(model_training_dir / 'y_val.csv')
    y_test = pd.read_csv(model_training_dir / 'y_test.csv')

    # Assert subject_id exists in all y splits
    for split_name, X_df, y_df in (
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test),
    ):
        assert 'subject_id' in y_df.columns, f"subject_id missing in y_{split_name}"
        # If split has rows, subject_id should align with X
        if len(y_df) > 0:
            assert 'subject_id' in X_df.columns
            # Equal length and row-wise alignment
            assert len(y_df) == len(X_df)
            assert X_df['subject_id'].tolist() == y_df['subject_id'].tolist()

def test_mri_finder_matches_exact_tokens(tmp_path):
    subject = '123_S_4567'
    imageuid = 98765
    filename = tmp_path / f'ADNI_{subject}_MR_I{imageuid}.nii'
    filename.write_text('')

    finder = MRIFinder(str(tmp_path))
    path = finder.find(subject, imageuid)

    assert path == str(filename)


def test_mri_finder_ignores_noncanonical_names(tmp_path):
    subject = '123_S_4567'
    imageuid = 98765
    # File lacks the ADNI-subject or I-prefixed IMAGEUID markers
    (tmp_path / f'{subject}_MR_{imageuid}.nii.gz').write_text('')

    finder = MRIFinder(str(tmp_path))

    assert finder.find(subject, imageuid) is None


def test_add_mri_paths_series(tmp_path):
    subject = '123_S_4567'
    imageuid = 98765
    match = tmp_path / f'ADNI_{subject}_T1_I{imageuid}.nii.gz'
    match.write_text('')

    df = pd.DataFrame({
        'subject_id': [subject, '999_S_0000'],
        'IMAGEUID': [imageuid, 1],
    })

    finder = MRIFinder(str(tmp_path))
    mri_paths = add_mri_paths(df, finder=finder)

    assert list(mri_paths) == [str(match), None]
