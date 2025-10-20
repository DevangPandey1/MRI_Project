import os

# =======================================================================================================================
# CONFIGURATION FOR PREPROCESSING
# =======================================================================================================================

# NOTES ON VAR TYPES:
# PASS is binary
# IMAGEUID is just for referencing MRI image later.

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'preprocessing')
MODEL_TRAINING_DIR = os.path.join(ROOT_DIR, 'model_training')
CLINICIAN_POLICY_DIR = os.path.join(ROOT_DIR, 'clinician_policy')
ALPACA_DIR = os.path.join(ROOT_DIR, 'reinforcement_learning', 'ALPACA')

MRI_DATA_ROOT = '/scratch/alpine/nobr3541'
MRI_NII_SUBDIR = os.path.join(MRI_DATA_ROOT, 'nii')
 
# Canonical identifier used internally
SUBJECT_ID_COL = 'subject_id'

# --- Feature Definitions ---
"""
Columns listed in BYPASS_IMPUTATION are preserved in the final datasets
but explicitly excluded from the imputation design matrix. This provides a
single, central place to control which variables should never be imputed
while still flowing through the pipeline (e.g., identifiers or reference IDs).
"""
BYPASS_IMPUTATION = [
    'subject_id',
    'IMAGEUID',
    'mri_path',
]

COLUMNS_TO_DROP = [
    'VSRESP', 'VSPULSE', 'TOTALMOD', 'FAQTOTAL', 'MMSE', 'TRABSCOR', 'ADNI_EF',
    'DIGITSCOR', 'LDELTOTAL', 'RAVLT_immediate', 'GDTOTAL', 'MOCA', 'VSBPDIA', 'GLUCOSE', 'PROTEIN', 'CTRED', 'CTWHITE', 
    'VSBPSYS', 'ADAS13', 'CDRSB_adni', 'PTAU_data', 'FAQTOTAL', 'TOTALMOD', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
    'MidTemp', 'ICV', 'VISCODE', 'VISCODE2'
]

CONTINUOUS_VARS = [
    'ADNI_MEM', 'ADNI_EF', 'ADNI_EF2',
    'VSBPDIA', 'VSPULSE', 'VSRESP', 'GLUCOSE', 'PROTEIN', 'CTRED', 'CTWHITE',
    'TAU_data', 'PTAU_data', 'VSBPSYS', 'subject_age', 'ABETA', 'TRABSCOR',
    'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
    'MidTemp', 'ICV', 'months_since_bl'
]

ORDINAL_VARS = [
    'GDTOTAL', 'TOTALMOD', 'FAQTOTAL', 'PTEDUCAT', 'CDRSB_adni', 'ADAS13',
    'MMSE', 'RAVLT_immediate', 'LDELTOTAL', 'DIGITSCOR', 'MOCA', "APOE4"
]

CATEGORICAL_VARS_FOR_IMPUTATION = ['PTGENDER', 'PTRACCAT', 'PTMARRY', 'research_group', 'PASS']

DROP_VARS_FOR_IMPUTATION = [
    'CMBGNYR_DRVD', 'CMENDYR_DRVD', 'CMENDYR_DRVD_filled', 'visit', 'CMROUTE', 
    'CMREASON', 'CMUNITS', 'CMMED', 'GENOTYPE', 'EXAMDATE', 'DIAGNOSIS', 
    'DXNORM', 'DXMCI', 'DXDEP', 'CMMED_clean', 'med_class', 'visit_year'
]

ACTION_FEATURES = [
    # AD treatments (by molecule)
    "AD_Treatment_Donepezil_active",
    "AD_Treatment_Memantine_active",
    "AD_Treatment_Rivastigmine_active",
    "AD_Treatment_Galantamine_active",

    # Statins (by molecule)
    "Statin_Atorvastatin_active",
    "Statin_Simvastatin_active",
    "Statin_Rosuvastatin_active",
    "Statin_Pravastatin_active",                 # NEW

    # Antihypertensives (by subclass)
    "Antihypertensive_ACE_Inhibitor_active",
    "Antihypertensive_ARB_active",
    "Antihypertensive_Beta_Blocker_active",
    "Antihypertensive_Calcium_Channel_Blocker_active",
    "Antihypertensive_Alpha_Blocker_active",     # e.g., doxazosin

    # Diuretics / Alpha-blocker (urologic)
    "Diuretic_active",                            # e.g., HCTZ (thiazide)
    "Diuretic_Loop_Furosemide_active",            # NEW (Lasix/furosemide)
    "Alpha_Blocker_Tamsulosin_active",            # Flomax / tamsulosin

    # Thyroid / Antithyroid
    "Thyroid_Hormone_Levothyroxine_active",
    "Antithyroid_Hormone_Methimazole_active",     # FIX (_active)

    # Diabetes medications
    "Diabetes_Medication_Metformin_active",
    "Diabetes_Medication_Insulin_active",

    # NSAIDs (by molecule/subtype)
    "NSAID_Aspirin_active",
    "NSAID_Ibuprofen_active",
    "NSAID_Naproxen_active",
    "NSAID_Other_active",                         # Excedrin/other mixed NSAID use
    "NSAID_Meloxicam_active",                     # NEW
    "NSAID_COX2_Celecoxib_active",                # NEW (Celebrex)

    # Analgesic (non-NSAID)
    "Analgesic_Acetaminophen_active",

    # GI / Bone / Steroid
    "PPI_Omeprazole_active",
    "PPI_Esomeprazole_active",                    # NEW (Nexium)
    "GI_Fiber_Psyllium_active",                   # NEW (Metamucil/psyllium)
    "Bone_Health_Alendronate_active",
    "Steroid_Prednisone_active",
    "Steroid_Prednisolone_active",
    "Steroid_Intranasal_Fluticasone_active",      # NEW (Flonase)

    # SSRIs (by molecule)
    "SSRI_Sertraline_active",
    "SSRI_Citalopram_active",
    "SSRI_Escitalopram_active",
    "SSRI_Fluoxetine_active",

    # Other psychotropics
    "Other_Antidepressant_Trazodone_active",
    "Other_Antidepressant_Bupropion_active",
    "Sleep_Aid_active",                           # generic sleep aid bucket
    "Sleep_Aid_Melatonin_active",                 # NEW (melatonin-specific)
    "Stimulant_Methylphenidate_active",
    "Anticonvulsant_Gabapentin_active",           # NEW

    # Cardiovascular antithrombotics
    "Antiplatelet_Clopidogrel_active",            # NEW (Plavix)
    "Anticoagulant_Warfarin_active",              # NEW (Coumadin)

    # Anti-infectives
    "Antibiotic_Penicillin_Amoxicillin_active",   # NEW

    # Supplements (split fine-grained)
    "Supplement_Vitamin_D_active",
    "Supplement_Vitamin_C_active",
    "Supplement_Vitamin_B12_active",
    "Supplement_Vitamin_B9_active",               # FIX (_active)
    "Supplement_Vitamin_E_active",
    "Supplement_Vitamin_B_active",                # NEW (B-complex)
    "Supplement_Vitamin_B7_active",               # NEW (biotin)
    "Supplement_Vitamin_B3_active",               # NEW (niacin)
    "Supplement_Multivitamin_active",
    "Supplement_Calcium_active",
    "Supplement_Magnesium_active",                # NEW
    "Supplement_Zinc_active",
    "Supplement_Iron_active",                     # NEW
    "Supplement_Fish_Oil_active",
    "Supplement_Joint_Glucosamine_active",        # NEW (glucosamine)
    "Supplement_ProbioticHerbal_active",
    "Supplement_Eye_Vitamin_AREDS_active",        # FIX (_active)

    # Catch-alls
    "No_Medication_active",
    "Other_Rare_active",
]

# Updated drug class mapping for more granularity (now with unresolved items + aliases)
DRUG_CLASS_MAPPING = {
    # AD Treatments
    'aricept': 'AD_Treatment_Donepezil',
    'donepezil': 'AD_Treatment_Donepezil',
    'namenda': 'AD_Treatment_Memantine',
    'memantine': 'AD_Treatment_Memantine',
    'exelon': 'AD_Treatment_Rivastigmine',
    'rivastigmine': 'AD_Treatment_Rivastigmine',
    'razadyne': 'AD_Treatment_Galantamine',
    'galantamine': 'AD_Treatment_Galantamine',

    # Statins
    'lipitor': 'Statin_Atorvastatin',
    'atorvastatin': 'Statin_Atorvastatin',
    'simvastatin': 'Statin_Simvastatin',
    'zocor': 'Statin_Simvastatin',
    'crestor': 'Statin_Rosuvastatin',
    'rosuvastatin': 'Statin_Rosuvastatin',
    'pravastatin': 'Statin_Pravastatin',                  # NEW

    # Antihypertensives
    'lisinopril': 'Antihypertensive_ACE_Inhibitor',
    'enalapril': 'Antihypertensive_ACE_Inhibitor',
    'losartan': 'Antihypertensive_ARB',
    'valsartan': 'Antihypertensive_ARB',
    'atenolol': 'Antihypertensive_Beta_Blocker',
    'metoprolol': 'Antihypertensive_Beta_Blocker',
    'amlodipine': 'Antihypertensive_Calcium_Channel_Blocker',
    'norvasc': 'Antihypertensive_Calcium_Channel_Blocker',
    'cardura': 'Antihypertensive_Alpha_Blocker',
    'doxazosin': 'Antihypertensive_Alpha_Blocker',        # generic for Cardura

    # Diuretics
    'hydrochlorothiazide': 'Diuretic',
    'hctz': 'Diuretic',
    'furosemide': 'Diuretic_Loop_Furosemide',             # NEW
    'lasix': 'Diuretic_Loop_Furosemide',                  # NEW

    # Alpha Blocker (urologic)
    'flomax': 'Alpha_Blocker_Tamsulosin',
    'tamsulosin': 'Alpha_Blocker_Tamsulosin',

    # Thyroid / Antithyroid
    'levothyroxine': 'Thyroid_Hormone_Levothyroxine',
    'synthroid': 'Thyroid_Hormone_Levothyroxine',
    'tapazole': 'Antithyroid_Hormone_Methimazole',        # corrected category

    # Diabetes
    'metformin': 'Diabetes_Medication_Metformin',
    'humulin': 'Diabetes_Medication_Insulin',
    'insulin': 'Diabetes_Medication_Insulin',

    # NSAIDs
    'aspirin': 'NSAID_Aspirin',
    'baby aspirin': 'NSAID_Aspirin',
    'low dose aspirin': 'NSAID_Aspirin',
    'asa': 'NSAID_Aspirin',
    'ibuprofen': 'NSAID_Ibuprofen',
    'advil': 'NSAID_Ibuprofen',
    'aleve': 'NSAID_Naproxen',
    'naproxen': 'NSAID_Naproxen',
    'meloxicam': 'NSAID_Meloxicam',                        # NEW
    'celebrex': 'NSAID_COX2_Celecoxib',                    # NEW
    'celecoxib': 'NSAID_COX2_Celecoxib',                   # NEW alias
    'excedrin': 'NSAID_Other',                             # keeping your bucket as-is

    # Analgesics (non-NSAID)
    'tylenol': 'Analgesic_Acetaminophen',
    'acetaminophen': 'Analgesic_Acetaminophen',

    # SSRIs
    'zoloft': 'SSRI_Sertraline',
    'sertraline': 'SSRI_Sertraline',
    'lexapro': 'SSRI_Escitalopram',
    'escitalopram': 'SSRI_Escitalopram',
    'citalopram': 'SSRI_Citalopram',
    'celexa': 'SSRI_Citalopram',
    'prozac': 'SSRI_Fluoxetine',
    'fluoxetine': 'SSRI_Fluoxetine',

    # Other Antidepressants / Stimulants / Sleep aids
    'trazodone': 'Other_Antidepressant_Trazodone',
    'bupropion': 'Other_Antidepressant_Bupropion',
    'alteril': 'Sleep_Aid',
    'melatonin': 'Sleep_Aid_Melatonin',                    # NEW
    'ritalin': 'Stimulant_Methylphenidate',
    'methylphenidate': 'Stimulant_Methylphenidate',
    'gabapentin': 'Anticonvulsant_Gabapentin',             # NEW

    # GI
    'omeprazole': 'PPI_Omeprazole',
    'prilosec': 'PPI_Omeprazole',
    'nexium': 'PPI_Esomeprazole',                          # NEW
    'esomeprazole': 'PPI_Esomeprazole',                    # NEW alias
    'metamucil': 'GI_Fiber_Psyllium',                      # NEW
    'psyllium': 'GI_Fiber_Psyllium',                       # NEW alias

    # Bone Health
    'fosamax': 'Bone_Health_Alendronate',
    'alendronate': 'Bone_Health_Alendronate',

    # Steroids
    'prednisone': 'Steroid_Prednisone',
    'prednisolone': 'Steroid_Prednisolone',
    'flonase': 'Steroid_Intranasal_Fluticasone',           # NEW (brand)
    'fluticasone': 'Steroid_Intranasal_Fluticasone',       # NEW alias

    # Anti-infectives
    'amoxicillin': 'Antibiotic_Penicillin_Amoxicillin',    # NEW

    # Cardiovascular antithrombotics
    'plavix': 'Antiplatelet_Clopidogrel',                  # NEW
    'clopidogrel': 'Antiplatelet_Clopidogrel',             # NEW alias
    'coumadin': 'Anticoagulant_Warfarin',                  # NEW
    'warfarin': 'Anticoagulant_Warfarin',                  # NEW alias

    # Supplements - Vitamins
    'vitamin d': 'Supplement_Vitamin_D',
    'vitamin d3': 'Supplement_Vitamin_D',
    'vitamind': 'Supplement_Vitamin_D',
    'vitamin b12': 'Supplement_Vitamin_B12',
    'cyanocobalamin': 'Supplement_Vitamin_B12',            # NEW alias
    'vitamin b9': 'Supplement_Vitamin_B9',
    'folic acid': 'Supplement_Vitamin_B9',
    'folate': 'Supplement_Vitamin_B9',
    'vitamin c': 'Supplement_Vitamin_C',
    'vitamin e': 'Supplement_Vitamin_E',
    'vitamin b complex': 'Supplement_Vitamin_B',           # NEW
    'biotin': 'Supplement_Vitamin_B7',                     # NEW
    'niacin': 'Supplement_Vitamin_B3',                     # NEW
    'multivitamin': 'Supplement_Multivitamin',
    'menss vitamin': 'Supplement_Multivitamin',
    'centrum silver': 'Supplement_Multivitamin',           # NEW
    'mvi': 'Supplement_Multivitamin',                      # NEW
    'iron': 'Supplement_Iron',                             # NEW

    # Supplements - Minerals
    'calcium': 'Supplement_Calcium',
    'magnesium': 'Supplement_Magnesium',                   # NEW
    'chelated zinc': 'Supplement_Zinc',
    'zinc': 'Supplement_Zinc',

    # Supplements - Oils
    'fish oil': 'Supplement_Fish_Oil',
    'omega-3': 'Supplement_Fish_Oil',                      # NEW alias
    'krill oil': 'Supplement_Fish_Oil',                    # NEW alias

    # Supplements - Probiotics / Herbals / Joint
    'probiotic': 'Supplement_ProbioticHerbal',
    'probiotic formula': 'Supplement_ProbioticHerbal',
    'saccharomyces boulardii': 'Supplement_ProbioticHerbal',
    'green tea': 'Supplement_ProbioticHerbal',
    'mega t green tea': 'Supplement_ProbioticHerbal',
    'cider vinegar': 'Supplement_ProbioticHerbal',
    'focus complex vitamin': 'Supplement_ProbioticHerbal',
    'glucosamine': 'Supplement_Joint_Glucosamine',         # NEW (generic)
    'glucosaminechrondrotin': 'Supplement_ProbioticHerbal',# (left as-is; alt: create *_Glucosamine_Chondroitin)
    'trigosamine': 'Supplement_ProbioticHerbal',
    'icaps': 'Supplement_Eye_Vitamin_AREDS',

    # No Medication
    'no medication': 'No_Medication',

    # Known non-med sentinel / fallback
    '-4': 'Other_Rare',                                    # NEW sink for sentinel
    'other': 'Other_Rare',
}
