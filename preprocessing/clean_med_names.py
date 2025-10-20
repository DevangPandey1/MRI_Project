#!/usr/bin/env python3
"""
Normalize CMMED into canonical medication names (keys), add CMMED_clean, and save.

- CMMED_clean is the normalized medication name (e.g., "vit c" -> "vitamin c")
- No extra rows are created; only a single new column is added.
- Loosened fuzzy matching + rich alias table to maximize coverage.
- Prefers generics when both brand & generic present in a combo.

Usage:
  python clean_cmmed.py --input RECCMEDS_10Sep2025.csv \
                        --output RECCMEDS_10Sep2025_with_CMMED_clean.csv \
                        --unresolved-out CMMED_unresolved_values_with_counts.csv
Options:
  --fuzzy-cutoff 0.80      # loosen/tighten fuzzy matching (default 0.80)
  --strict-exact-only      # disable fuzzy matching entirely
"""

import argparse
import pandas as pd
import numpy as np
import re
import difflib
from collections import Counter
from typing import List, Optional
from config import DRUG_CLASS_MAPPING



# Canonical keys come strictly from DRUG_CLASS_MAPPING to keep config as ground truth
EXTRA_CANONICAL = set()

CANONICAL_KEYS = list(DRUG_CLASS_MAPPING.keys())
CANONICAL_ALL = CANONICAL_KEYS + list(EXTRA_CANONICAL)

# Prefer generics (if both brand/generic show up in a combo)
GENERIC_PREFERENCE = {
    'donepezil','memantine','rivastigmine','galantamine',
    'atorvastatin','simvastatin','rosuvastatin',
    'lisinopril','enalapril','losartan','valsartan','atenolol','metoprolol','amlodipine',
    'hydrochlorothiazide','tamsulosin',
    'levothyroxine','metformin','insulin',
    'ibuprofen','naproxen','acetaminophen',
    'sertraline','escitalopram','citalopram','fluoxetine',
    'trazodone','bupropion',
    'omeprazole','alendronate',
    'prednisone','prednisolone',
    'vitamin d','vitamin d3','vitamin b12','vitamin b complex','vitamin c','vitamin e','multivitamin','menss vitamin',
    'calcium','zinc','chelated zinc','fish oil',
    'probiotic', 'probiotic formula','saccharomyces boulardii','green tea','cider vinegar',
    'focus complex vitamin','glucosaminechrondrotin','trigosamine','icaps','hctz'
}

# Rich alias table -> canonical key (medication name)
ALIAS_TO_KEY = {
    # vitamins & minerals
    'ascorbic acid': 'vitamin c', 'C': 'vitamin c', 'c': 'vitamin c',
    'vit c': 'vitamin c', 'vit-c': 'vitamin c', 'vit. c': 'vitamin c',
    'vit b 12': 'vitamin b12', 'vit b-12': 'vitamin b12', 'vitamin b 12': 'vitamin b12',
    'b 12': 'vitamin b12', 'b-12': 'vitamin b12', 'b12': 'vitamin b12', 'cyanocobalamin': 'vitamin b12',
    'folic acid': 'vitamin b9',
    'B-12': 'vitamin b12', 'B-6': 'vitamin b6',
    'alpha tocopherol': 'vitamin e', 'alpha-tocopherol': 'vitamin e',
    'cholecalciferol': 'vitamin d3', 'ergocalciferol': 'vitamin d',
    'vit d': 'vitamin d', 'vit d3': 'vitamin d3', 'vit d-3': 'vitamin d3',
    'vit. d': 'vitamin d', 'vit. d3': 'vitamin d3', 'vitamind': 'vitamin d',
    'multi vitamin': 'multivitamin', 'multi-vitamin': 'multivitamin',
    'mv': 'multivitamin', 'one a day': 'multivitamin', 'one-a-day': 'multivitamin', 'centrum': 'multivitamin',
    "men's vitamin": 'menss vitamin', 'mens vitamin': 'menss vitamin', 'mens multivitamin': 'menss vitamin',
    'zinc gluconate': 'zinc', 'zinc oxide': 'zinc', 'zinc sulfate': 'zinc',
    'tums': 'calcium', 'caltrate': 'calcium',

    # vitamin B family (map to existing config keys)
    'vitamin b': 'vitamin b complex', 'vit b': 'vitamin b complex',
    'b complex': 'vitamin b complex', 'vitamin b complex': 'vitamin b complex',
    'vit b complex': 'vitamin b complex', 'b-complex': 'vitamin b complex',
    'niacin': 'niacin', 'nicotinic acid': 'niacin',
    'biotin': 'biotin',

    # probiotics/herbals
    'pro-biotic': 'probiotic', 'pro biotic': 'probiotic',
    'acidophilus': 'probiotic', 'lactobacillus': 'probiotic',
    'culturelle': 'probiotic', 'align': 'probiotic',
    'florastor': 'saccharomyces boulardii',
    'green tea extract': 'green tea', 'apple cider vinegar': 'cider vinegar',

    # NSAIDs & analgesics (with misspellings)
    'motrin': 'ibuprofen', 'naprosyn': 'naproxen',
    'asa': 'aspirin',
    'paracetamol': 'acetaminophen',
    'baby aspirin': 'aspirin', 'baby asprin': 'aspirin', 'baby aspriin': 'aspirin',
    'asprin': 'aspirin', 'aspirn': 'aspirin', 'aspriin': 'aspirin',
    'bayer aspirin': 'aspirin',
    'tylenol extra strength': 'tylenol', 'excedrin migraine': 'excedrin',

    # diabetes
    'glucophage': 'metformin',
    'humalog': 'insulin', 'novolog': 'insulin', 'lantus': 'insulin', 'basaglar': 'insulin', 'tresiba': 'insulin',

    # thyroid
    'levothyroxine sodium': 'levothyroxine', 'levoxyl': 'levothyroxine',
    'eltroxin': 'levothyroxine', 'eltroxin sodium': 'levothyroxine', 'synthroid': 'synthroid',

    # antihypertensives
    'cozaar': 'losartan', 'diovan': 'valsartan', 'lopressor': 'metoprolol',
    'toprol': 'metoprolol', 'toprol xl': 'metoprolol',
    'norvasc': 'norvasc', 'cardura': 'cardura',

    # diuretic
    'hct': 'hctz', 'hctz': 'hctz',
    'hydro-chloro-thiazide': 'hydrochlorothiazide', 'hydroclorothiazide': 'hydrochlorothiazide',

    # alpha blocker
    'flomax': 'flomax', 'tamsulosin hcl': 'tamsulosin',

    # gi
    'omeprazole dr': 'omeprazole', 'omeprazole delayed release': 'omeprazole',
    'prilosec otc': 'prilosec', 'losec': 'omeprazole',

    # bone health
    'alendronate sodium': 'alendronate', 'fosamax': 'fosamax',

    # steroids
    'prednizone': 'prednisone', 'prednison': 'prednisone', 'prednisolone sodium': 'prednisolone',

    # psych
    'sertraline hcl': 'sertraline', 'escitalopram oxalate': 'escitalopram',
    'citalopram hydrobromide': 'citalopram', 'fluoxetine hcl': 'fluoxetine',
    'bupropion hcl': 'bupropion', 'bupropion sr': 'bupropion', 'bupropion xl': 'bupropion',
    'trazodone hcl': 'trazodone', 'ritalin la': 'ritalin', 'ritalin sr': 'ritalin', 'concerta': 'methylphenidate',

    # ad treatments
    'donepezil hcl': 'donepezil', 'memantine hcl': 'memantine',
    'rivastigmine tartrate': 'rivastigmine', 'exelon patch': 'exelon', 'razadyne er': 'razadyne',

    # statins
    'atorvastatin calcium': 'atorvastatin', 'simvastatin': 'simvastatin',
    'rosuvastatin calcium': 'rosuvastatin', 'zocor': 'zocor', 'lipitor': 'lipitor', 'crestor': 'crestor',
}

def normalize_text(text: str) -> str:
    s = text.lower()
    s = re.sub(r'[\u2010-\u2015]', '-', s)  # normalize dashes
    s = re.sub(r'[^\w\s\-/+&.,]', ' ', s)  # keep separators
    s = s.replace("’", "'")
    # remove release/forms & frequencies
    s = re.sub(r'\b(otc|dr|sr|er|xr|la|cr|xl|prn|qd|qod|bid|tid|qhs)\b', ' ', s)
    # remove common form/route descriptors
    s = re.sub(r'\b(tab|tablet|cap|capsule|po|oral|sublingual|sl|chewable|softgel|enteric)\b', ' ', s)
    # remove units
    s = re.sub(r'\b(mg|mcg|iu|g|gram|ml|l|cc|units?)\b', ' ', s)
    # remove standalone numbers, but keep B-vitamin numbers (e.g., B12, B-12)
    # preserves digits when immediately preceded by 'b' (case-insensitive handled earlier)
    s = re.sub(r'(?<!b)\b\d+(\.\d+)?\b', ' ', s)
    s = re.sub(r'[,.;]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def segments_from(text: str) -> List[str]:
    segs = re.split(r'\bwith\b|&|/|\+|\band\b', text)
    return [re.sub(r'\s+', ' ', s).strip() for s in segs if s and s.strip()]

def resolve_one(segment: str,
                canonical_all: List[str],
                alias_to_key: dict,
                fuzzy_cutoff: float,
                strict_exact_only: bool) -> Optional[str]:
    # alias exact
    if segment in alias_to_key:
        return alias_to_key[segment]
    # canonical exact
    if segment in canonical_all:
        return segment
    # substring contains check (word-boundaries) for canonical keys
    for key in canonical_all:
        if re.search(rf"\b{re.escape(key)}\b", segment):
            return key
    # substring contains check for aliases (map to canonical)
    for ak in alias_to_key.keys():
        if re.search(rf"\b{re.escape(ak)}\b", segment):
            return alias_to_key[ak]
    if strict_exact_only:
        return None
    # fuzzy to canonical_all (looser by default)
    cand = difflib.get_close_matches(segment, canonical_all, n=1, cutoff=fuzzy_cutoff)
    if cand:
        return cand[0]
    # fuzzy to aliases (slightly tighter to avoid wild jumps)
    alias_keys = list(alias_to_key.keys())
    cand2 = difflib.get_close_matches(segment, alias_keys, n=1, cutoff=min(0.86, max(0.50, fuzzy_cutoff + 0.06)))
    if cand2:
        return alias_to_key[cand2[0]]
    return None

def choose_best(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    generics = [c for c in candidates if c in GENERIC_PREFERENCE]
    if generics:
        return generics[0]
    return candidates[0]

def normalize_to_key(text,
                     canonical_all: List[str],
                     alias_to_key: dict,
                     fuzzy_cutoff: float,
                     strict_exact_only: bool) -> Optional[str]:
    if pd.isna(text):
        return None

    raw = str(text).strip()
    raw_l = raw.lower()

    # Early-handle known sentinel values that should pass through unchanged
    if raw_l in {"-4", "other"}:
        return raw_l

    # Perform vitamin expansions BEFORE general normalization so numbers like "12" are preserved
    # Normalize common variants to a single token first
    tmp = raw_l
    tmp = re.sub(r"\bvit(?:amin)?\s*b[\s\-\.]?12\b", "vitamin b12", tmp)
    tmp = re.sub(r"\bb[\s\-\.]?12\b", "vitamin b12", tmp)
    tmp = re.sub(r"\bvit\s*c\b", "vitamin c", tmp)
    tmp = re.sub(r"\bvit\s*d3\b", "vitamin d3", tmp)
    tmp = re.sub(r"\bvit\s*d\b", "vitamin d", tmp)
    tmp = tmp.replace('vitamind', 'vitamin d')

    s = normalize_text(tmp)

    if not s:
        return None

    # full-string alias direct
    if s in alias_to_key:
        s = alias_to_key[s]

    # direct canonical
    if s in canonical_all:
        return s

    # segment for combos (e.g., losartan/hctz 100-25 mg)
    segs = segments_from(s)
    cands = []
    for seg in segs:
        r = resolve_one(seg, canonical_all, alias_to_key, fuzzy_cutoff, strict_exact_only)
        if r:
            cands.append(r)
    if cands:
        return choose_best(cands)

    # last resort: fuzzy on full string
    return resolve_one(s, canonical_all, alias_to_key, fuzzy_cutoff, strict_exact_only)

def clean_med_names():
    ap = argparse.ArgumentParser()
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    INPUT_DIR = os.path.join(project_root, "data", "tabular", "RECCMEDS_10Sep2025.csv")
    # ap.add_argument("--input", default="../data/tabular/RECCMEDS_10Sep2025.csv", required=True, help="Path to input CSV with CMMED column")
    # ap.add_argument("--output", required=True, help="Path to write CSV with CMMED_clean added")
    ap.add_argument("--unresolved-out", default=None, help="Optional path to write unresolved values + counts")
    ap.add_argument("--fuzzy-cutoff", type=float, default=0.80, help="Fuzzy cutoff (0-1). Lower is looser.")
    ap.add_argument("--strict-exact-only", action="store_true", help="Disable fuzzy matching")
    args = ap.parse_args()

    df = pd.read_csv(INPUT_DIR)
    if "CMMED" not in df.columns:
        raise KeyError("The input file does not contain a 'CMMED' column.")

    original_len = len(df)

    # Apply resolver
    resolved = df["CMMED"].apply(
        lambda x: normalize_to_key(
            x, CANONICAL_ALL, ALIAS_TO_KEY, args.fuzzy_cutoff, args.strict_exact_only
        )
    )
    df["CMMED_clean"] = resolved

    # Safety: no fabricated rows
    assert len(df) == original_len, "Row count changed during processing!"

    # Diagnostics
    mask_nn = df["CMMED"].notna()
    resolved_count = df["CMMED_clean"].notna().sum()
    unresolved_count = mask_nn.sum() - resolved_count
    print(f"Total rows: {original_len}")
    print(f"Non-null CMMED: {mask_nn.sum()}")
    print(f"Resolved to key: {resolved_count}")
    print(f"Unresolved: {unresolved_count}")
    # Collect and print top 20 unresolved input values (normalized)
    unresolved_values = (
        df.loc[mask_nn & df["CMMED_clean"].isna(), "CMMED"]
        .astype(str)
        .apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    )
    if len(unresolved_values) > 0:
        vc = unresolved_values.value_counts().head(20)
        print("Top 20 unresolved (normalized):")
        for val, cnt in vc.items():
            print(f"  {cnt:6d}  {val}")
    else:
        print("Top 20 unresolved (normalized): none")

    # Mark unresolved as 'other' in CMMED_clean
    df["CMMED_clean"] = df["CMMED_clean"].fillna("other")
    # Persist with unresolved marked as 'other'
    df.to_csv(INPUT_DIR, index=False)

    # if args.unresolved_out:
    #     unresolved_values = (
    #         df.loc[mask_nn & df["CMMED_clean"].isna(), "CMMED"]
    #         .astype(str)
    #         .apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    #     )
    #     cnt = Counter(unresolved_values)
    #     pd.DataFrame(cnt.most_common(), columns=["unresolved_value_lower", "count"]).to_csv(
    #         args.unresolved_out, index=False
    #     )
    #     print(f"Wrote unresolved list to: {args.unresolved_out}")

if __name__ == "__main__":
    clean_med_names()
