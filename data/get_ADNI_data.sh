#!/usr/bin/env bash
# ======================================================================
# ADNI DICOM → NIfTI Converter (Path-Derived ADNI-Style Filenames)
#
# Expected DICOM folder structure (one series per leaf "I" directory):
#   /scratch/.../ADNI/<PATIENT_ID>/<SERIES_DESC>/<YYYY-MM-DD_HH_MM_SS.0>/<I#######>/*.dcm
#
# Output filename pattern (per series; multi-echo gets _e1, _e2, ...):
#   ADNI_<PATIENT_ID>_MR_<SERIES_DESC>_Br_<YYYYMMDDHHMMSS0>_<I#######>[_e#].nii.gz
#
# Key behaviors:
#   - Iterates each "I########" series directory and runs dcm2niix there.
#   - Renames outputs to the pattern above (derived purely from path components).
#   - Optionally flattens all outputs into ./nii and (optionally) deletes DICOMs.
#
# Requirements:
#   - dcm2niix (in PATH or available via 'module load dcm2niix' on HPC).
#   - POSIX tools: find, sort, tr, sed, awk (standard on Linux/HPC).
#
# Usage:
#   - Edit CONFIG section below (DICOM_ROOT, flags).
#   - Run: ./convert_adni_dicoms.sh
# ======================================================================

set -Eeuo pipefail

# -----------------------------
# CONFIG — EDIT TO YOUR SETUP
# -----------------------------

# Root of your ADNI DICOM tree (as described above).
DICOM_ROOT="/scratch/alpine/nobr3541/ADNI"
SCRATCH_ROOT="/scratch/alpine/nobr3541"

# If true, move final NIfTIs into a single flat output folder (./nii).
# If false, NIfTIs are renamed in-place inside each I######## folder.
FLATTEN_OUTPUT=true

# If true AND FLATTEN_OUTPUT=true: delete each converted I######## DICOM folder
# after successful conversion & move — this reclaims disk space.
REMOVE_DICOM_AFTER_CONVERSION=false

# Where to put the flattened output (used only if FLATTEN_OUTPUT=true).
FINAL_OUTPUT_DIR="$(SCRATCH_ROOT)/nii"

# Where to put conversion logs (safe to keep with your code).
CONVERSION_LOG_DIR="$(SCRATCH_ROOT)/logs_adni_convert"

# -----------------------------
# END CONFIG
# -----------------------------

# Create output/log directories as needed
mkdir -p "$CONVERSION_LOG_DIR"
if [[ "$FLATTEN_OUTPUT" == true ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR"
fi

# ---- Helper functions ----

# require_cmd: exit with a clear message if a command is missing.
require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $cmd" >&2
    exit 1
  fi
}

# try_load_dcm2niix: on HPCs, dcm2niix may be provided as a module.
try_load_dcm2niix() {
  if ! command -v dcm2niix >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      # Ignore failure; we'll check again after attempting to load.
      module load dcm2niix || true
    fi
  fi
  require_cmd dcm2niix
}

# sanitize_name: make a filesystem-safe token (keep alnum, _, -, . ; replace spaces with _).
sanitize_name() {
  local s="$1"
  s="${s// /_}"
  printf '%s' "$s" | tr -cd '[:alnum:]_.-'
}

# digits_only: strip everything except digits (used to compact the datetime folder name).
digits_only() {
  printf '%s' "$1" | tr -cd '0-9'
}

# convert_series_directory:
#   Given a single I######## directory, run dcm2niix and rename/move outputs.
convert_series_directory() {
  local series_dir="$1"  # .../<PATIENT>/<SERIES>/<DATETIME>/<I#######>

  # Parse expected path components.
  local i_folder datetime_folder series_desc_folder patient_id_folder
  i_folder="$(basename "$series_dir")"                                     # I1619403
  datetime_folder="$(basename "$(dirname "$series_dir")")"                 # 2022-09-09_09_55_29.0
  series_desc_folder="$(basename "$(dirname "$(dirname "$series_dir")")")" # Accelerated_Sagittal_MPRAGE
  patient_id_folder="$(basename "$(dirname "$(dirname "$(dirname "$series_dir")")")")" # 941_S_7106

  # Build cleaned components for the filename pattern.
  local patient_id series_desc dt_compact i_id
  patient_id="$(sanitize_name "$patient_id_folder")"        # 941_S_7106
  series_desc="$(sanitize_name "$series_desc_folder")"      # Accelerated_Sagittal_MPRAGE
  dt_compact="$(digits_only "$datetime_folder")"            # 202209090955290
  i_id="$(sanitize_name "$i_folder")"                       # I1619403

  # Final filename stem (without extension).
  local filename_stem="ADNI_${patient_id}_MR_${series_desc}_Br_${dt_compact}_${i_id}"

  echo "Processing: $series_dir"
  echo "  → Stem: $filename_stem"

  # Quick check: ensure there are DICOM files present.
  if ! find "$series_dir" -maxdepth 1 -type f -iregex '.*\.(dcm|ima)$' | head -n1 >/dev/null; then
    echo "  WARN: No DICOM files found in: $series_dir — skipping."
    return 0
  fi

  # Run dcm2niix INTO the same directory so JSON sidecars start nearby.
  # We'll rename and optionally move immediately after.
  local log_file="${CONVERSION_LOG_DIR}/${patient_id}_${i_id}.log"
  if ! dcm2niix -b y -z y -o "$series_dir" "$series_dir" >"$log_file" 2>&1; then
    echo "  WARN: dcm2niix failed for: $series_dir (see $log_file)"
    return 0
  fi

  # Collect produced NIfTI files (there can be >1 if multi-echo, etc.).
  mapfile -d '' produced_niis < <(find "$series_dir" -maxdepth 1 -type f -iregex '.*\.nii(\.gz)?$' -print0 | sort -z)
  if (( ${#produced_niis[@]} == 0 )); then
    echo "  WARN: No NIfTI produced in: $series_dir (see $log_file)"
    return 0
  fi

  local idx=0
  for nifti_path in "${produced_niis[@]}"; do
    ((idx++))

    # Determine extension and base stem (add _e# if multiple outputs).
    local ext=".nii.gz"
    [[ "$nifti_path" == *.nii ]] && ext=".nii"

    local this_stem="$filename_stem"
    if (( ${#produced_niis[@]} > 1 )); then
      this_stem="${filename_stem}_e${idx}"
    fi

    # Destination (flatten or in-place).
    local dest_nii dest_json
    if [[ "$FLATTEN_OUTPUT" == true ]]; then
      dest_nii="${FINAL_OUTPUT_DIR}/${this_stem}${ext}"
      dest_json="${FINAL_OUTPUT_DIR}/${this_stem}.json"
    else
      dest_nii="${series_dir}/${this_stem}${ext}"
      dest_json="${series_dir}/${this_stem}.json"
    fi

    # Avoid accidental overwrite by suffixing _dupN if needed.
    if [[ -e "$dest_nii" ]]; then
      local n=1
      while [[ -e "${dest_nii%${ext}}_dup${n}${ext}" ]]; do ((n++)); done
      dest_nii="${dest_nii%${ext}}_dup${n}${ext}"
      dest_json="${dest_nii%${ext}}.json"
    fi

    # Move/rename the NIfTI.
    mv -f "$nifti_path" "$dest_nii"

    # Move/rename the JSON sidecar if present (same stem).
    local json_src="${nifti_path%.*}.json"
    if [[ -f "$json_src" ]]; then
      mv -f "$json_src" "$dest_json"
    fi

    echo "  → $(basename "$dest_nii")"
  done

  # Optionally delete the source DICOM series directory to reclaim space.
  if [[ "$FLATTEN_OUTPUT" == true && "$REMOVE_DICOM_AFTER_CONVERSION" == true ]]; then
    rm -rf "$series_dir"
  fi
}

# ---- Main flow ----

# Ensure dcm2niix is available (try module load if not in PATH).
try_load_dcm2niix

# Find all series directories that match the expected "I########" leaf folders.
# Using a POSIX ERE to catch any .../I<digits> path.
mapfile -d '' series_dirs < <(find "$DICOM_ROOT" -type d -regextype posix-extended -regex '.*/I[0-9]+' -print0 | sort -z)

echo "Found ${#series_dirs[@]} series directories under: $DICOM_ROOT"
echo

# Convert each series directory
count=0
for series_dir in "${series_dirs[@]}"; do
  ((count++))
  echo "[$count/${#series_dirs[@]}]"
  convert_series_directory "$series_dir"
done

if [[ "$FLATTEN_OUTPUT" == true ]]; then
  echo
  echo "All done. Final NIfTIs are in: $FINAL_OUTPUT_DIR"
else
  echo
  echo "All done. NIfTIs were written/renamed in-place under each I######## directory."
fi
