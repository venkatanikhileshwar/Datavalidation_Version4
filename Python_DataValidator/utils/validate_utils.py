# utils/validate_utils.py
from __future__ import annotations
from typing import Dict, List, Optional
import re
import pandas as pd


# ---------- Null checks ----------
def null_checks(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return rows where any of the chosen columns are null/NA.
    If cols is None, checks all columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    use_cols = list(cols) if cols else list(df.columns)
    mask = df[use_cols].isna().any(axis=1)
    out = df.loc[mask, use_cols].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# ---------- Type checks (light heuristics) ----------
def _is_mostly_numeric(s: pd.Series, thr: float = 0.8) -> bool:
    if s is None or s.empty:
        return False
    num = pd.to_numeric(s, errors="coerce")
    return num.notna().mean() >= thr

def _is_date_like_name(colname: str) -> bool:
    lc = str(colname).lower()
    return "date" in lc or lc.endswith("_dt") or lc.endswith("dt")

def type_checks(df: pd.DataFrame, spec: Dict[str, str]) -> pd.DataFrame:
    """
    spec: mapping column -> expected type ('number' | 'date')
    Returns rows where the column doesn't conform (very light check).
    """
    if df is None or df.empty or not spec:
        return pd.DataFrame()

    issues = []
    for col, expected in spec.items():
        if col not in df.columns:
            continue
        s = df[col]
        if expected == "number":
            if not _is_mostly_numeric(s):
                bad = pd.to_numeric(s, errors="coerce").isna()
                if bad.any():
                    temp = df.loc[bad, [col]].copy()
                    temp["__reason__"] = "expected number"
                    issues.append(temp)
        elif expected == "date":
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            bad = parsed.isna() & s.notna()
            if bad.any():
                temp = df.loc[bad, [col]].copy()
                temp["__reason__"] = "expected date"
                issues.append(temp)

    if not issues:
        return pd.DataFrame()
    out = pd.concat(issues, axis=0, ignore_index=True)
    return out


# ---------- Presence checks (anti-joins) ----------
def anti_join(left: pd.DataFrame, right: pd.DataFrame, on_cols: List[str], side: str) -> pd.DataFrame:
    """
    side='left'  -> rows in left not present in right (on ALL on_cols)
    side='right' -> rows in right not present in left
    Returns just the join columns, deduplicated.
    """
    if left is None or right is None or not on_cols:
        return pd.DataFrame()

    if side == "left":
        m = left.merge(right, how="left", on=on_cols, indicator=True)
        out = m.loc[m["_merge"] == "left_only", on_cols].drop_duplicates().reset_index(drop=True)
        return out
    else:
        m = right.merge(left, how="left", on=on_cols, indicator=True)
        out = m.loc[m["_merge"] == "left_only", on_cols].drop_duplicates().reset_index(drop=True)
        return out


# ---------- Key-based mismatches ----------
def compute_mismatches_by_key(file_df: pd.DataFrame, db_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Assumes BOTH DataFrames are normalized.
    - Inner-join on key.
    - For every non-key column, compare with NA==NA treated as equal.
    - Report only rows with any mismatch. Also skip conflicting-keys from file.
    """
    if (
        file_df is None or db_df is None or
        file_df.empty or db_df.empty or
        key_col not in file_df.columns or key_col not in db_df.columns
    ):
        return pd.DataFrame()

    compare_cols = [c for c in file_df.columns if c != key_col]
    if not compare_cols:
        return pd.DataFrame()

    # detect keys with conflicting rows in file
    grp = file_df.groupby(key_col, dropna=False)
    conflict_keys = grp.apply(lambda g: len(g.drop_duplicates()) > 1)
    conflict_keys = set(conflict_keys[conflict_keys].index.astype(str))

    # unique file rows for fair comparison
    file_unique = file_df.drop_duplicates(subset=[key_col] + compare_cols, keep="first")
    if conflict_keys:
        file_unique = file_unique[~file_unique[key_col].astype(str).isin(conflict_keys)]

    merged = file_unique.merge(db_df, how="inner", on=key_col, suffixes=("_file", "_db"))
    if merged.empty:
        return pd.DataFrame()

    for c in compare_cols:
        cf, cd = f"{c}_file", f"{c}_db"
        eq = (merged[cf].isna() & merged[cd].isna()) | (merged[cf] == merged[cd])
        merged[f"{c}__mismatch"] = ~eq

    flags = [f"{c}__mismatch" for c in compare_cols]
    cols_out = [key_col]
    for c in compare_cols:
        cols_out.extend([f"{c}_file", f"{c}_db", f"{c}__mismatch"])

    out = merged.loc[merged[flags].any(axis=1), cols_out].reset_index(drop=True)
    return out
