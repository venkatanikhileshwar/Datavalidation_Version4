# app.py — Streamlit UI/controller only
# Relies on utils/: io_utils, sql_utils, normalize_utils, validate_utils, report_utils, pdf_utils

import json
import time
import random
import sqlalchemy
import pandas as pd
import streamlit as st

# ---- project utils ----
from utils.io_utils import detect_type, load_headers, read_sheet_df
from utils.sql_utils import get_engine, run_query_preview, run_query_full, get_probe_sql
from utils.normalize_utils import normalize_dataframe, harmonize_keys, drop_completely_empty_rows
from utils.validate_utils import null_checks, type_checks, anti_join, compute_mismatches_by_key
from utils.report_utils import export_excel
from utils.pdf_utils import convert_uploaded_pdf_to_csv_temp

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Data Validation", layout="wide")
st.title("Data Validation")

# ---------------- Reset helpers ----------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = random.randint(1_000, 9_999)

def reset_app():
    for k in [
        "engine", "engine_url", "sql_text", "db_headers",
        "mapping", "require_key", "__pdf_csv_path__",
        "db_ok", "sql_ok", "mapping_ok"
    ]:
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] = random.randint(1_000, 9_999)
    st.rerun()

c_head1, c_head2 = st.columns([6, 1])
with c_head2:
    if st.button("Reset", use_container_width=True, help="Clear all inputs and start fresh"):
        reset_app()

# ---------------- Config ----------------
cfg = json.load(open("config/appconfig.json"))

db_options     = cfg["defaults"]["db"]["dropdown"]
default_db     = cfg["defaults"]["db"]["default"]
connections    = cfg["connections"]
valids_default = cfg["defaults"]["validations"]
norm_cfg       = cfg.get("normalization", {
    "date_format": "%Y-%m-%d",
    "string_case": "lower",
    "collapse_internal_spaces": True,
    "currency_symbols": ["$", "€", "£", "¥", "₹"],
    "strip_percent": True,
    "treat_parentheses_as_negative": True,
    "blank_equivalents": ["", "-", "_", "NA", "N/A", "None", "null"],
    "numeric_majority_threshold": 0.8,
    "strip_commas_and_spaces": True,
    "remove_underscores_in_numbers": False
})

# ---------------- Sidebar checklist ----------------
st.sidebar.header("Run checklist")
checklist = {"file": False, "db": False, "sql_ok": False, "mapping_ok": False}
for k in ("db_ok", "sql_ok", "mapping_ok"):
    st.session_state.setdefault(k, False)

def _show(label: str, ok: bool) -> None:
    if ok:
        st.sidebar.success(f"{label} ✓")
    else:
        st.sidebar.warning(f"{label} ⏳")


# ---------------- Tiny UI helpers ----------------
MAX_UI_ROWS = 1000
MAX_STYLER_CELLS = 250_000

def show_df_with_notice(name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    total = len(df)
    if total > MAX_UI_ROWS:
        st.warning(f"{name} has {total:,} rows. Showing first {MAX_UI_ROWS:,}. Download the Excel for full details.")
    else:
        st.caption(f"{name}: showing all {total:,} rows.")
    st.dataframe(df.head(MAX_UI_ROWS), use_container_width=True, height=420)

# ---------------- Step 1: Upload File ----------------
st.subheader("1) Upload File")
file = st.file_uploader(
    "Choose a file",
    type=["xlsx", "xls", "csv", "txt", "pdf"],
    key=f"uploader_{st.session_state['uploader_key']}",
    help="Excel shows the Sheet scope selector. CSV/TXT/PDF are treated as a single dataset."
)

file_kind = None
headers_by_scope = {}
chosen_scope = None

if file is not None:
    file_kind = detect_type(file.name)

    if file_kind == "pdf":
        with st.spinner("Converting PDF to CSV…"):
            csv_path = convert_uploaded_pdf_to_csv_temp(file)
        if not csv_path:
            st.error("No table detected in the PDF. Please upload a CSV/Excel converted from this PDF.")
            st.stop()
        st.success("PDF converted to CSV. Proceeding…")
        st.session_state["__pdf_csv_path__"] = csv_path

        # read the converted CSV and drop empty rows immediately
        df_tmp = pd.read_csv(csv_path, dtype=object)
        df_tmp = drop_completely_empty_rows(df_tmp, norm_cfg.get("blank_equivalents"))
        headers_by_scope = {"Data": list(df_tmp.columns)}
        chosen_scope = ["Data"]
        file_kind = "csv"
        checklist["file"] = True

    elif file_kind == "excel":
        headers_by_scope = load_headers(file, file_kind)
        checklist["file"] = True
        st.markdown("**Sheet scope (Excel only)**")
        scope = st.radio("Which sheets to compare?", ["All sheets","Single sheet","Select specific"], index=0, horizontal=True)
        if scope == "All sheets":
            chosen_scope = list(headers_by_scope.keys())
        elif scope == "Single sheet":
            one = st.selectbox("Pick a sheet", list(headers_by_scope.keys()))
            chosen_scope = [one] if one else []
        else:
            chosen_scope = st.multiselect("Pick specific sheets", list(headers_by_scope.keys()))
        if not chosen_scope:
            st.warning("Select at least one sheet.")
        else:
            st.info(f"Selected sheets: {', '.join(chosen_scope)}")

    elif file_kind in ("csv", "txt"):
        headers_by_scope = load_headers(file, file_kind)
        chosen_scope = list(headers_by_scope.keys())
        checklist["file"] = True

    else:
        st.error("Supported types: Excel, CSV, TXT, PDF.")

# ---------------- Step 2: Database ----------------
st.subheader("2) Database")
col1, col2 = st.columns([2,1])
with col1:
    db_choice = st.selectbox("Database", db_options, index=db_options.index(default_db))
with col2:
    test_click = st.button("Test Connection", use_container_width=True, help="Uses credentials from config/appconfig.json.")

if test_click:
    try:
        if db_choice not in connections:
            raise ValueError(f"No connection details found for {db_choice}")
        url = connections[db_choice]["url"]
        engine = get_engine(url)
        with engine.connect() as conn:
            # Dialect-aware probe (Oracle uses FROM DUAL)
            conn.execute(sqlalchemy.text(get_probe_sql(str(engine.url))))
        st.success(f"Connected to {db_choice}")
        st.session_state["engine"] = engine
        st.session_state["engine_url"] = url
        st.session_state["db_ok"] = True
        checklist["db"] = True
    except Exception as e:
        st.error(f"Could not connect: {e}")
        checklist["db"] = False
else:
    if "engine" in st.session_state:
        try:
            with st.session_state["engine"].connect() as _:
                checklist["db"] = True
        except Exception:
            checklist["db"] = False

# ---------------- Step 3: SQL ----------------
st.subheader("3) SQL")
st.caption("Enter your SQL (any valid query, typically SELECT). The preview loader is dialect-aware across databases.")
sql = st.text_area(
    "Enter your SQL:",
    height=120,
    placeholder="SELECT record_id, name, amount, status FROM sales WHERE status='ACTIVE' ORDER BY record_id;"
)
run_preview_btn = st.button("Run Query")

if run_preview_btn:
    if not checklist["db"] or "engine" not in st.session_state:
        st.error("Connect to the database first.")
    else:
        try:
            db_headers = run_query_preview(st.session_state["engine"], sql, limit=1000)
            if not db_headers:
                st.warning("Query returned no rows/columns.")
            else:
                st.success(f"Query OK. Columns: {', '.join(db_headers)}")
                st.session_state["sql_text"]   = sql
                st.session_state["db_headers"] = db_headers
                st.session_state["sql_ok"]     = True
                checklist["sql_ok"]            = True
        except Exception as e:
            st.error(f"Query failed: {e}")

# ---------------- Step 4: Header Mapping ----------------
st.subheader("4) Header Mapping (File ↔ DB)")
if not checklist["file"]:
    st.info("Upload a file to view its headers.")
elif "db_headers" not in st.session_state:
    st.info("Run the SQL query to load DB columns.")
else:
    db_headers = st.session_state["db_headers"]
    if not db_headers:
        st.info("Run the SQL query to load DB columns.")
        st.stop()

    require_key_name = st.selectbox(
        "Pick the DB key column (must exist in your SELECT):",
        options=db_headers,
        index=0,
        help="Used for key-based mismatches."
    )

    # Build file header list (use scope::header)
    if file_kind == 'excel':
        file_headers = [f"{s}::{h}" for s in (chosen_scope or []) for h in headers_by_scope[s]]
    else:
        first_scope = next(iter(headers_by_scope.keys()), "Data")
        file_headers = [f"{first_scope}::{h}" for h in headers_by_scope.get(first_scope, [])]

    options = ["— Ignore —"] + db_headers
    mapping = {}
    cols_map = st.columns(2)
    for i, fh in enumerate(file_headers):
        with cols_map[i % 2]:
            mapping[fh] = st.selectbox(
                fh, options, index=0, key=f"map_{i}",
                help="Map this file header to a DB column, or choose '— Ignore —'."
            )

    if require_key_name not in mapping.values():
        st.warning(f"Map at least one file header to the DB key column: '{require_key_name}'.")
    else:
        st.success("Key mapping OK.")
        st.session_state["mapping"]     = mapping
        st.session_state["require_key"] = require_key_name
        st.session_state["mapping_ok"]  = True
        checklist["mapping_ok"]         = True

# ---------------- Sidebar status ----------------
checklist["file"]       = bool(file is not None and file_kind != 'other')
checklist["db"]         = bool(st.session_state.get("db_ok", False))
checklist["sql_ok"]     = bool(st.session_state.get("sql_ok", False))
checklist["mapping_ok"] = bool(st.session_state.get("mapping_ok", False))

st.sidebar.subheader("Checklist status")
_show("1) File uploaded", checklist["file"])
_show("2) DB connected", checklist["db"])
_show("3) SQL loaded", checklist["sql_ok"])
_show("4) Key mapped", checklist["mapping_ok"])

# ---------------- Step 5: Validations ----------------
st.subheader("5) Validations")
val_dup = st.checkbox("Duplicates (row-wise)", value=valids_default.get("keyUniqueness", True))
val_nulls = st.checkbox("Null checks", value=valids_default.get("nullChecks", True))
val_types = st.checkbox("Type checks", value=valids_default.get("typeChecks", True))
val_mismatch = st.checkbox(
    "Value mismatches (Actual vs Expected) — by KEY",
    value=valids_default.get("valueMismatches", True)
)

ready = all(checklist.values())
btn = st.button("Run Validation", disabled=not ready, type="primary", use_container_width=True)

if btn and ready:
    t0 = time.perf_counter()
    try:
        # 1) Load file data (respect sheet scope)
        if file_kind == 'excel':
            parts = []
            for s in (chosen_scope or []):
                df = read_sheet_df(file, file_kind, sheet_name=s)  # drops empty rows inside
                df.columns = [f"{s}::{c}" for c in df.columns]
                parts.append(df)
            file_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        else:
            if "__pdf_csv_path__" in st.session_state:
                tmp_path = st.session_state["__pdf_csv_path__"]
                file_df = pd.read_csv(tmp_path, dtype=object)
                file_df = drop_completely_empty_rows(file_df, norm_cfg.get("blank_equivalents"))
                first_scope = "Data"
                file_df.columns = [f"{first_scope}::{c}" for c in file_df.columns]
            else:
                file_df = read_sheet_df(file, file_kind)  # drops empty rows inside
                first_scope = next(iter(headers_by_scope.keys()), "Data")
                file_df.columns = [f"{first_scope}::{c}" for c in file_df.columns]

        # 2) Apply mapping → keep only mapped cols, rename to DB names
        remap = {k: v for k, v in st.session_state["mapping"].items() if v and v != "— Ignore —"}

        # warn on duplicate target mappings & keep first
        if remap:
            targets = list(remap.values())
            dup_targets = sorted({t for t in targets if targets.count(t) > 1})
            if dup_targets:
                st.warning("Multiple file columns mapped to: " + ", ".join(dup_targets) + ". Keeping the first occurrence.")

        file_df_renamed = file_df.rename(columns=lambda x: remap.get(x, None))
        file_df_renamed = file_df_renamed[[c for c in file_df_renamed.columns if c is not None]]
        if file_df_renamed.columns.duplicated().any():
            file_df_renamed = file_df_renamed.loc[:, ~file_df_renamed.columns.duplicated(keep="first")]

        key_col = st.session_state["require_key"]
        if key_col not in file_df_renamed.columns:
            st.error("Key column missing after mapping. Fix header mapping.")
            st.stop()

        # 3) Fetch DB (full) and keep only mapped columns (plus key)
        engine = st.session_state["engine"]
        db_df = run_query_full(engine, st.session_state["sql_text"])
        keep_cols = sorted(set([key_col] + [c for c in file_df_renamed.columns if c != key_col]))
        db_df = db_df[[c for c in keep_cols if c in db_df.columns]]
        if db_df.columns.duplicated().any():
            db_df = db_df.loc[:, ~db_df.columns.duplicated(keep="first")]

        # 4) Normalize both sides
        file_norm = normalize_dataframe(file_df_renamed, norm_cfg)
        db_norm   = normalize_dataframe(db_df,       norm_cfg)

        # 5) Harmonize key dtype/content (date/numeric/string) for safe joins
        harmonize_keys(file_norm, db_norm, [key_col], norm_cfg)

        # 6) Validations
        results = {}

        # 6A) Row-wise duplicates
        if val_dup and not file_norm.empty:
            dup_mask = file_norm.duplicated(subset=list(file_norm.columns), keep=False)
            results["Duplicates"] = file_norm.loc[dup_mask].copy()

        # 6B) Presence checks on ALL mapped columns (row-wise)
        compare_cols_all = list(file_norm.columns)
        results["Missing_in_DB"]   = anti_join(file_norm, db_norm, compare_cols_all, side="left")
        results["Missing_in_File"] = anti_join(file_norm, db_norm, compare_cols_all, side="right")

        # 6C) Null checks
        if val_nulls:
            results["Null_Issues"] = null_checks(file_norm, compare_cols_all)

        # 6D) Type checks (simple heuristics)
        if val_types:
            spec = {}
            for c in file_norm.columns:
                lc = str(c).lower()
                if 'date' in lc or lc.endswith('_dt') or lc.endswith('dt'):
                    spec[c] = 'date'
                elif any(x in lc for x in ['amt','amount','total','rate','price','cost','percent','pct']):
                    spec[c] = 'number'
            results["Type_Issues"] = type_checks(file_norm, spec)

        # 6E) Key-based value mismatches
        if val_mismatch:
            results["Value_Mismatches"] = compute_mismatches_by_key(file_norm, db_norm, key_col)

        # 7) Summary & metrics
        duration = time.perf_counter() - t0
        mism_df = results.get("Value_Mismatches", pd.DataFrame())
        miss_db = results.get("Missing_in_DB", pd.DataFrame())
        miss_fi = results.get("Missing_in_File", pd.DataFrame())
        results["Summary"] = pd.DataFrame([
            ("File rows (mapped)", len(file_norm)),
            ("DB rows (selected)", len(db_norm)),
            ("Duplicates (flat)", len(results.get("Duplicates", pd.DataFrame()))),
            ("Missing in DB", len(miss_db)),
            ("Missing in File", len(miss_fi)),
            ("Null issues", 0 if results.get("Null_Issues") is None else len(results.get("Null_Issues"))),
            ("Value mismatches", 0 if mism_df is None else len(mism_df)),
            ("Validation time (sec)", round(duration, 2)),
        ], columns=["Metric","Value"])

        st.success(f"Validation complete in {round(duration,2)}s. See results below.")
        st.caption("Normalized for: dates (yyyy-mm-dd), currency/percent, case & whitespace, blanks, and dtype-safe joins.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("File rows", len(file_norm))
        c2.metric("DB rows", len(db_norm))
        c3.metric("Missing in DB", len(miss_db))
        c4.metric("Mismatches", 0 if mism_df is None else len(mism_df))

        # 8) Drilldowns (UI-capped)
        for name in ["Value_Mismatches", "Missing_in_DB", "Missing_in_File", "Null_Issues", "Type_Issues", "Duplicates"]:
            df_show = results.get(name)
            if isinstance(df_show, pd.DataFrame) and not df_show.empty:
                with st.expander(name, expanded=(name == "Value_Mismatches")):
                    cells = df_show.shape[0] * df_show.shape[1]
                    if cells > MAX_STYLER_CELLS or name != "Value_Mismatches":
                        show_df_with_notice(name, df_show)
                    else:
                        st.dataframe(df_show.head(MAX_UI_ROWS), use_container_width=True, height=420)

        # 9) Export full results
        outpath = "validation_report.xlsx"
        export_excel(outpath, results)
        with open(outpath, "rb") as f:
            st.download_button("Download Excel report", f, file_name="validation_report.xlsx", use_container_width=True)

    except Exception as e:
        st.error(f"Validation failed: {e}")
