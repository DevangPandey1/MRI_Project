# Details on Preprocessing Pipeline

## Files used:
- ADNIMERGE.csv
	- Primary tabular dataset with most variables of interest.
	- Keys: RID, PTID, VISCODE/VISCODE2. IMAGEUID identifies MRI files directly.

- MRINCLUSIO_10Sep2025.csv
	- MRI QC information. Uses columns "PASS", "VISCODE"/"VISCODE2", and identifiers.
	- We keep column names as-is (no renaming). Merge on RID + VISCODE2 after normalizing tokens (e.g., baseline -> m00).

- UWNPSYCHSUM_03Sep2025.csv
	- Cognitive composites (e.g., ADNI_MEM, ADNI_EF, ADNI_EF2).
	- Merge on RID + VISCODE2.

- RECCMEDS_10Sep2025.csv (not merged here)
	- Medication history: "CMMED", "CMBGNYR_DRVD", "CMENDYR_DRVD", "CMBGN", "CMEND", along with RID and PTID.
	- Not merged into `ADNI_merged.csv` to maintain 1-row-per-visit shape. Use directly in `preprocessing/preprocess.py` (join on `['RID','VISCODE2']` when needed).

Notes:
- The merge aligns visits strictly using RID + VISCODE2 across tables that have visit codes. VISCODE/VISCODE2 tokens are normalized to lower-case and baseline is standardized to m00.
- To prevent one-to-many row expansion, right-hand tables are de-duplicated on their join keys before merging (keeps the first row per key; no aggregation or feature derivation).
- We no longer use ADNI_CompleteVisitList_8_22_12.csv in merging; ADNIMERGE's IMAGEUID is sufficient to reference MRI files.
