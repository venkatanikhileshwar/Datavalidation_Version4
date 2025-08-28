# utils/normalize_utils.py
from __future__ import annotations
import re
from typing import List, Optional
import pandas as pd

# -------------------- Defaults --------------------
_DEFAULT_BLANKS = ["", "-", "_", "NA", "N/A", "None", "null"]
_NUM_CLEAN_RE = re.compile(r"[,\s]")   # commas + spaces
_CURR_CHARS = "$€£¥₹"


# -------------------- Row/Column cleanup --------------------
def drop_completely_empty_rows(df: pd.DataFrame, blank_equivalents=None) -> pd.DataFrame:
    """
    Drop rows that are empty across all columns (treating configured blanks as empty).
    Also drops fully empty columns.
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


def _to_date_string(series: pd.Series, fmt: str) -> pd.Series:
    """
    Parse many date forms (YYYYMMDD, ISO, slashed, Excel serials) -> formatted string.
    """
    s = series
    as_str = s.astype("string").str.strip()

    # If exactly 8 digits, convert YYYYMMDD -> YYYY-MM-DD for better parsing
    eight = as_str.str.match(r"^\d{8}$", na=False)
    as_str.loc[eight] = as_str.loc[eight].str.replace(
        r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True
    )

    parsed = pd.to_datetime(as_str, errors="coerce", infer_datetime_format=True)

    # Excel serials fallback
    bad = parsed.isna() & as_str.str.match(r"^\d+$", na=False)
    if bad.any():
        ser = pd.to_numeric(as_str[bad], errors="coerce")
        parsed.loc[bad] = pd.to_datetime(ser, unit="D", origin="1899-12-30", errors="coerce")

    return parsed.dt.strftime(fmt)


# -------------------- DataFrame-level normalization --------------------
def normalize_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Column-wise normalization:
      - Trim/whitespace normalize strings; canonicalize case; blanks -> <NA>
      - Strip currency/percent/commas/underscores; parentheses negative
      - Coerce mostly-numeric columns to numeric dtype (Int64/float)
      - Parse date-like columns by name to configured format (default %Y-%m-%d)
      - Drop fully-empty rows/columns
    """
    if df is None or df.empty:
        return df

    nc = cfg or {}
    fmt = nc.get("date_format", "%Y-%m-%d")
    thresh = float(nc.get("numeric_majority_threshold", 0.8))

    out = drop_completely_empty_rows(df, nc.get("blank_equivalents")).copy()

    for col in out.columns:
        raw = out[col]

        # always have a string view for blanks/case/space handling
        s_str = _to_string(raw, nc)

        # try numeric
        s_num = _to_numeric(s_str, nc)

        # choose numeric if majority numeric
        if s_num.notna().mean() >= thresh:
            # if name hints date, parse as date string instead of numeric
            if str(col).lower().find("date") != -1 or str(col).lower().endswith(("_dt", "dt")):
                out[col] = _to_date_string(raw, fmt)  # parse from raw to keep real dates
            else:
                out[col] = s_num
        else:
            # non-numeric; if date-like name, parse date
            if str(col).lower().find("date") != -1 or str(col).lower().endswith(("_dt", "dt")):
                out[col] = _to_date_string(raw, fmt)
            else:
                out[col] = s_str

    return out


# -------------------- Key harmonization for joins --------------------
def harmonize_keys(file_df: pd.DataFrame, db_df: pd.DataFrame, keys: List[str], cfg: dict) -> None:
    """
    Ensure join keys have same dtype/content on both sides (in-place).
    - If key name looks date-like: format both sides to cfg['date_format']
    - Else: if both mostly numeric -> numeric on both; otherwise canonical strings
    """
    if file_df is None or db_df is None or not keys:
        return

    fmt = (cfg or {}).get("date_format", "%Y-%m-%d")

    for k in keys:
        if k not in file_df.columns:
            continue

        # string baseline on file side
        f = _to_string(file_df[k], cfg)

        # db side may not have the column (rare), handle safely
        d = _to_string(db_df[k], cfg) if k in db_df.columns else None

        # date-like by name
        if str(k).lower().find("date") != -1 or str(k).lower().endswith(("_dt", "dt")):
            file_df[k] = _to_date_string(file_df[k], fmt)
            if d is not None:
                db_df[k] = _to_date_string(db_df[k], fmt)
            continue

        # try numeric harmonization (explicitly avoid "or" on Series!)
        fn = pd.to_numeric(_clean_number_like(f.fillna(""), cfg), errors="coerce")

        if d is None:
            # create an aligned empty series for metrics; no cast on db side
            d_src = pd.Series(dtype="string", index=file_df.index)
        else:
            d_src = d

        dn = pd.to_numeric(_clean_number_like(d_src.fillna(""), cfg), errors="coerce")

        if (fn.notna().mean() > 0.7) and (dn.notna().mean() > 0.7):
            file_df[k] = fn
            if d is not None:
                db_df[k] = dn
        else:
            file_df[k] = f
            if d is not None:
                db_df[k] = _to_string(db_df[k], cfg)
