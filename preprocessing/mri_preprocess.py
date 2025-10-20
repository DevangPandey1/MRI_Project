import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd

from config import MRI_DATA_ROOT, MRI_NII_SUBDIR

NII_SUFFIXES = (".nii", ".nii.gz")


def _normalize_token(value) -> str:
    """Convert identifiers to stable string tokens for matching."""
    if value is None:
        return ""
    if isinstance(value, str):
        token = value.strip()
        if token.endswith(".0"):
            try:
                token = str(int(float(token)))
            except ValueError:
                pass
        return token
    if pd.isna(value):  # Handles pd.NA / np.nan
        return ""
    if isinstance(value, (int, )):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    token = str(value).strip()
    if token.endswith(".0"):
        try:
            token = str(int(float(token)))
        except ValueError:
            pass
    return token


@dataclass
class MRIFinder:
    """Utility to resolve MRI file paths by subject and IMAGEUID tokens."""

    nii_dir: str
    recursive: bool = False

    def __post_init__(self) -> None:
        resolved_dir = os.path.abspath(self.nii_dir)
        if not os.path.isdir(resolved_dir):
            raise FileNotFoundError(f"MRI NIfTI directory not found: {resolved_dir}")
        self.nii_dir = resolved_dir
        self._files = self._collect_files()
        self._cache: Dict[tuple[str, str], Optional[str]] = {}

    def _collect_files(self) -> Iterable[str]:
        if self.recursive:
            collected = []
            for root, _, files in os.walk(self.nii_dir):
                for name in files:
                    if name.lower().endswith(NII_SUFFIXES):
                        collected.append(os.path.join(root, name))
            return collected
        return [
            os.path.join(self.nii_dir, name)
            for name in os.listdir(self.nii_dir)
            if name.lower().endswith(NII_SUFFIXES)
        ]

    def find(self, subject_id, imageuid, *, verbose: bool = False) -> Optional[str]:
        subject_token = _normalize_token(subject_id)
        image_token = _normalize_token(imageuid)

        if not subject_token or not image_token:
            if verbose:
                print(f"Skipping lookup: subject_id={subject_id!r}, IMAGEUID={imageuid!r}")
            return None

        key = (subject_token, image_token)
        if key in self._cache:
            return self._cache[key]

        subject_marker = f"ADNI_{subject_token}_"
        image_marker = f"I{image_token}"

        for path in self._files:
            name = os.path.basename(path)
            if subject_marker in name and image_marker in name:
                assert subject_marker in name, (
                    f"MRI filename missing subject marker: expected '{subject_marker}' in '{name}'"
                )
                assert image_marker in name, (
                    f"MRI filename missing IMAGEUID marker: expected '{image_marker}' in '{name}'"
                )
                self._cache[key] = path
                if verbose:
                    print(f"Match: subject_id={subject_token}, IMAGEUID={image_token} -> {path}")
                return path

        if verbose:
            print(f"No MRI located for subject_id={subject_token}, IMAGEUID={image_token}")
        self._cache[key] = None
        return None


def add_mri_paths(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    image_col: str = "IMAGEUID",
    finder: Optional[MRIFinder] = None,
    nii_dir: Optional[str] = None,
    recursive: bool = False,
    verbose: bool = False,
) -> pd.Series:
    """Return a Series of MRI paths matched to the provided dataframe."""

    if subject_col not in df.columns or image_col not in df.columns:
        missing = [c for c in (subject_col, image_col) if c not in df.columns]
        raise KeyError(f"Dataframe is missing required columns: {missing}")

    if finder is None:
        base_dir = nii_dir or os.path.join(MRI_DATA_ROOT, MRI_NII_SUBDIR)
        finder = MRIFinder(base_dir, recursive=recursive)

    def _lookup(row):
        return finder.find(row[subject_col], row[image_col], verbose=verbose)

    return df.apply(_lookup, axis=1)


__all__ = ["MRIFinder", "add_mri_paths"]
