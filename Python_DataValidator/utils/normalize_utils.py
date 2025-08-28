# utils/normalize_utils.py
from __future__ import annotations
from typing import List, Tuple, Optional
import re
import pandas as pd

# -------------------- Defaults --------------------
_DEFAULT_BLANKS = ["", "-", "_", "NA", "N/A", "None", "null"]
_NUM_CLEAN_RE = re.compile(r"[,\s]")   # commas + spaces
_CURR_CHARS = "$€£¥₹"


# -------------------- Row/Column cleanup --------------------
def drop_completely_empty_rows(df: pd.DataFrame, blank_equivalents=None) -> pd.DataFrame:
    """
    Drop rows that are empty across all columns (treating configured blanks as empty).
    Also drops fully-empty columns.
    """
    if df is None or df.empty:
        return df

    blanks = set((blank_equivalents or _DEFAULT_BLANKS))
    blanks = {str(x).lower() for x in blanks}

    # drop all-NaN fast path
    df2 = df.dropna(how="all").copy()
    if df2.empty:
        return df2

    # treat configured blanks as empty strings for row-emptiness check
    s = df2.astype(str).apply(lambda col: col.str.strip())
    s = s.mask(s.applymap(lambda x: str(x).lower() in blanks), "")

    row_is_empty = s.eq("").all(axis=1)
    out = df2.loc[~row_is_empty].copy()

    # drop fully-empty columns too
    col_is_empty = s.eq("").all(axis=0)
    if col_is_empty.any():
        out = out.loc[:, ~col_is_empty].copy()

    return out


# -------------------- Primitive normalizers --------------------
def _clean_number_like(s: pd.Series, cfg: dict) -> pd.Series:
    cs = "".join(cfg.get("currency_symbols", list(_CURR_CHARS)))
    s2 = s.str.replace("\u00A0", " ", regex=False).str.replace("\u200B", "", regex=False)
    # remove currency symbols and percent sign
    s2 = s2.str.replace(f"[{re.escape(cs)}%]", "", regex=True)
    if cfg.get("strip_commas_and_spaces", True):
        s2 = s2.str.replace(_NUM_CLEAN_RE, "", regex=True)
    if cfg.get("remove_underscores_in_numbers", False):
        s2 = s2.str.replace("_", "", regex=False)
    # (123.45) -> -123.45
    if cfg.get("treat_parentheses_as_negative", True):
        s2 = s2.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    return s2


def _to_numeric(series: pd.Series, cfg: dict) -> pd.Series:
    s = series.astype("string")
    s = _clean_number_like(s, cfg)
    num = pd.to_numeric(s, errors="coerce")

    if num.dropna().empty:
        return num

    # keep integer dtype if everything integral
    if (num.dropna() == num.dropna().astype(int)).all():
        return num.astype("Int64")
    return num.astype(float)


def _to_string(series: pd.Series, cfg: dict) -> pd.Series:
    s = series.astype("string")
    # normalize exotic whitespace
    s = s.str.replace("\u00A0", " ", regex=False).str.replace("\u200B", "", regex=False)
    s = s.str.strip()

    if cfg.get("collapse_internal_spaces", True):
        s = s.str.replace(r"\s+", " ", regex=True)

    # blanks -> <NA>
    blanks = {b.lower() for b in cfg.get("blank_equivalents", _DEFAULT_BLANKS)}
    s = s.mask(s.str.lower().isin(blanks), pd.NA)

    case = str(cfg.get("string_case", "lower")).lower()
    if case == "lower":
        s = s.str.casefold()
    elif case == "upper":
        s = s.str.upper()
    # else: leave as-is
    return s


def _parse_dates_to_dt(series: pd.Series) -> pd.Series:
    """
    Parse many date forms to datetime64[ns].
    Handles:
      - YYYYMMDD (8 digits)
      - ISO (YYYY-MM-DD)
      - Slashed formats
      - Excel serial numbers
    """
    as_str = series.astype("string").str.strip()

    # If exactly 8 digits, convert YYYYMMDD -> YYYY-MM-DD for better parsing
    eight = as_str.str.match(r"^\d{8}$", na=False)
    if eight.any():
        as_str.loc[eight] = as_str.loc[eight].str.replace(
            r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True
        )

    parsed = pd.to_datetime(as_str, errors="coerce", infer_datetime_format=True)

    # Excel serials fallback (only integers make sense here)
    bad = parsed.isna() & as_str.str.match(r"^\d+$", na=False)
    if bad.any():
        ser = pd.to_numeric(as_str[bad], errors="coerce")
        # Excel serial origin=1899-12-30
        parsed.loc[bad] = pd.to_datetime(ser, unit="D", origin="1899-12-30", errors="coerce")

    return parsed


def _to_date_string_from_any(series: pd.Series, fmt: str) -> pd.Series:
    parsed = _parse_dates_to_dt(series)
    return parsed.dt.strftime(fmt)


# -------------------- Type detection (value-based) --------------------
def _detect_date_ratio(series: pd.Series) -> Tuple[pd.Series, float]:
    """
    Try to parse as dates; return (parsed_dt_series, ratio_of_valid_dates_in_nonblank).
    """
    s = series
    # consider only non-empty values
    mask_valid_input = s.notna() & (s.astype(str).str.strip() != "")
    if not mask_valid_input.any():
        parsed = _parse_dates_to_dt(s)
        return parsed, 0.0

    parsed = _parse_dates_to_dt(s)
    ratio = (parsed.notna() & mask_valid_input).sum() / mask_valid_input.sum()
    return parsed, float(ratio)


def _is_mostly_numeric(series: pd.Series, cfg: dict, thr: float) -> Tuple[pd.Series, float]:
    """
    Return (numeric_series, ratio_of_numeric_in_nonblank) after cleaning.
    """
    s = _to_string(series, cfg)  # ensure blanks removed and trimmed
    s_clean = _clean_number_like(s.fillna(""), cfg)
    num = pd.to_numeric(s_clean, errors="coerce")
    mask_input = s.notna() & (s.astype(str).str.strip() != "")
    if not mask_input.any():
        return num, 0.0
    ratio = num.notna().sum() / mask_input.sum()
    return num, float(ratio)


def _name_hints_date(colname: str) -> bool:
    """Loose hints from column name."""
    c = str(colname).lower()
    return ("date" in c) or c.endswith(("dt", "dat"))


# -------------------- DataFrame-level normalization --------------------
def normalize_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Column-wise normalization with auto type detection:
      - Trim/whitespace normalize strings; canonicalize case; blanks -> <NA>
      - Strip currency/percent/commas/underscores; parentheses negative
      - Auto-detect dates by values: if ≥ date_detect_threshold (default 0.6) parse → cfg['date_format']
      - Else coerce mostly-numeric columns to numeric dtype (Int64/float) using numeric_majority_threshold
      - Drop fully-empty rows/columns
    """
    if df is None or df.empty:
        return df

    nc = cfg or {}
    fmt = nc.get("date_format", "%Y-%m-%d")
    num_thr = float(nc.get("numeric_majority_threshold", 0.8))
    date_thr = float(nc.get("date_detect_threshold", 0.6))  # NEW: configurable

    out = drop_completely_empty_rows(df, nc.get("blank_equivalents")).copy()

    for col in out.columns:
        raw = out[col]

        # always have a normalized string view (trimming/blanks/case)
        s_str = _to_string(raw, nc)

        # try value-based detection
        parsed_dates, date_ratio = _detect_date_ratio(raw)
        num_series, num_ratio = _is_mostly_numeric(raw, nc, num_thr)

        # choose date if strong signal OR name hints date and moderate signal
        if (date_ratio >= date_thr) or (_name_hints_date(col) and date_ratio >= 0.4 and date_ratio >= num_ratio):
            out[col] = parsed_dates.dt.strftime(fmt)
            continue

        # otherwise numeric if strong numeric signal
        if num_ratio >= num_thr:
            # if truly integral, keep Int64; else float
            if num_series.dropna().empty:
                out[col] = num_series
            elif (num_series.dropna() == num_series.dropna().astype(int)).all():
                out[col] = num_series.astype("Int64")
            else:
                out[col] = num_series.astype(float)
            continue

        # fallback: normalized text
        out[col] = s_str

    return out


# -------------------- Key harmonization for joins --------------------
def harmonize_keys(file_df: pd.DataFrame, db_df: pd.DataFrame, keys: List[str], cfg: dict) -> None:
    """
    Ensure join keys have same dtype/content on both sides (in-place).
    Auto-detects date/numeric/string using values (not just names).
    """
    if file_df is None or db_df is None or not keys:
        return

    fmt = (cfg or {}).get("date_format", "%Y-%m-%d")
    num_thr = float((cfg or {}).get("numeric_majority_threshold", 0.8))
    date_thr = float((cfg or {}).get("date_detect_threshold", 0.6))

    for k in keys:
        if k not in file_df.columns:
            continue

        f_raw = file_df[k]
        d_raw = db_df[k] if k in db_df.columns else None

        # detect by values on each side
        f_dates, f_date_ratio = _detect_date_ratio(f_raw)
        f_nums,  f_num_ratio  = _is_mostly_numeric(f_raw, cfg, num_thr)

        if d_raw is not None:
            d_dates, d_date_ratio = _detect_date_ratio(d_raw)
            d_nums,  d_num_ratio  = _is_mostly_numeric(d_raw, cfg, num_thr)
        else:
            # align indices for safe operations
            d_dates = pd.Series(pd.NaT, index=file_df.index, dtype="datetime64[ns]")
            d_nums  = pd.Series(pd.NA, index=file_df.index, dtype="Float64")
            d_date_ratio = 0.0
            d_num_ratio  = 0.0

        # If both sides look like dates (or one is strong date and the other moderate), use date
        if (f_date_ratio >= date_thr and d_date_ratio >= 0.4) or (d_date_ratio >= date_thr and f_date_ratio >= 0.4):
            file_df[k] = f_dates.dt.strftime(fmt)
            if k in db_df.columns:
                db_df[k] = d_dates.dt.strftime(fmt)
            continue

        # Else if both mostly numeric, use numeric
        if (f_num_ratio >= 0.7) and (d_num_ratio >= 0.7):
            file_df[k] = f_nums
            if k in db_df.columns:
                db_df[k] = d_nums
            continue

        # Else, canonical strings on both sides
        file_df[k] = _to_string(f_raw, cfg)
        if k in db_df.columns:
            db_df[k] = _to_string(d_raw, cfg)
