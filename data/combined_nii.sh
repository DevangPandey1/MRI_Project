#!/usr/bin/env bash
set -euo pipefail

SRC="./ADNI"
DEST="./nii"

mkdir -p "$DEST"

# 1) Count all .nii files under $SRC (robust to spaces/newlines)
total=$(find "$SRC" -type f -iname '*.nii' -print0 | tr -cd '\0' | wc -c)
echo "Found $total .nii files under $SRC"

# Optional: how many would be newly copied (i.e., basename doesn't already exist in $DEST)
to_copy=$(find "$SRC" -type f -iname '*.nii' -print0 \
  | while IFS= read -r -d '' f; do
      base="${f##*/}"
      [[ -e "$DEST/$base" ]] || printf '%s\0' "$f"
    done | tr -cd '\0' | wc -c)
echo "Will copy $to_copy new files to $DEST (skipping existing basenames)"

# 2) Copy without overwriting existing files (avoids duplicates by name)
find "$SRC" -type f -iname '*.nii' -exec cp -n -t "$DEST" {} +

# Final count in destination
final=$(find "$DEST" -maxdepth 1 -type f -iname '*.nii' -print0 | tr -cd '\0' | wc -c)
echo "Now $final .nii files in $DEST"
