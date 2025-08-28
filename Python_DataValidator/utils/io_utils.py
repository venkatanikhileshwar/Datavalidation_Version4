# utils/io_utils.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import io
import os
import pandas as pd

from .normalize_utils import drop_completely_empty_rows


# -------- Detect file kind by name ----------
def detect_type(filename: str) -> str:
    if not filename:
        return "other"
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        return "excel"
    if ext == ".csv":
        return "csv"
    if ext in (".txt", ".tsv", ".dat"):
        return "txt"
    if ext == ".pdf":
        return "pdf"
    return "other"


# -------- NEW: encoding + delimiter sniffer ----------
_DELIMS = [",", "|", "\t", ";", "~", "^"]

def _decode_bytes(b: bytes) -> str:
    """Try common encodings; return text."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    # last resort
    return b.decode("utf-8", errors="replace")

def _sniff_delimiter(text: str) -> Optional[str]:
    """Pick the delimiter with the highest count on header line."""
    if not text:
        return None
    first_line = text.splitlines()[0] if text.splitlines() else text
    scores = {d: first_line.count(d) for d in _DELIMS}
    best, best_n = None, 0
    for d, n in scores.items():
        if n > best_n:
            best, best_n = d, n
    return best if best_n > 0 else None


# -------- Robust text-like reader (CSV/TXT/DAT/TSV) ----------
def _read_text_like(file) -> pd.DataFrame:
    """
    Read text files with robust delimiter sniffing.
    Handles ',', '|', tab, ';', '~', '^' and common encodings.
    """
    # get raw bytes
    try:
        file.seek(0)
    except Exception:
        pass
    raw = file.read()
    # if stream gives str, convert to bytes first
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")

    text = _decode_bytes(raw)
    sep = _sniff_delimiter(text)  # None -> let pandas try flexible parse

    # Use StringIO for pandas
    fh = io.StringIO(text)

    if sep is None:
        # Let pandas' python engine auto-figure mixed separators if possible
        df = pd.read_csv(fh, dtype=object, sep=None, engine="python")
    else:
        df = pd.read_csv(fh, dtype=object, sep=sep, engine="python")

    return df


# -------- Header loaders ----------
def load_headers(file, file_kind: str) -> Dict[str, List[str]]:
    if file_kind == "excel":
        xls = pd.ExcelFile(file)
        scopes = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=0, dtype=object)
            scopes[sheet] = list(df.columns)
        return scopes

    if file_kind in ("csv", "txt"):
        df = _read_text_like(file)
        df = drop_completely_empty_rows(df)  # ensure no fake blank-rows
        return {"Data": list(df.columns)}

    return {"Data": []}


# -------- Reader ----------
def read_sheet_df(file, file_kind: str, sheet_name: str = None) -> pd.DataFrame:
    if file_kind == "excel":
        df = pd.read_excel(file, sheet_name=sheet_name, dtype=object)
        return drop_completely_empty_rows(df)

    if file_kind in ("csv", "txt"):
        df = _read_text_like(file)
        return drop_completely_empty_rows(df)

    # for pdf, app.py will read the temporary CSV itself
    return pd.DataFrame()
