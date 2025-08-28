# utils/sql_utils.py
from __future__ import annotations
from typing import List
import sqlalchemy
import pandas as pd


# ---------- Engine ----------
def get_engine(url: str) -> sqlalchemy.Engine:
    """
    Create a SQLAlchemy engine from the given URL.
    """
    if not url:
        raise ValueError("Empty DB URL.")
    return sqlalchemy.create_engine(url, pool_pre_ping=True)


# ---------- Dialect-aware probe ----------
def get_probe_sql(url: str) -> str:
    """
    Return a minimal probe SQL that works for the given database URL.
    Oracle needs 'FROM DUAL'; most others accept 'SELECT 1'.
    """
    low = (url or "").lower()
    if "oracle" in low or "cx_oracle" in low:
        return "SELECT 1 FROM DUAL"
    return "SELECT 1"


def test_connection(engine: sqlalchemy.Engine) -> None:
    """
    Execute a tiny probe query appropriate for the engine's dialect.
    Raises on failure.
    """
    probe = get_probe_sql(str(engine.url))
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(probe))


# ---------- Dialect-aware preview limiter ----------
def _apply_limit_sql(sql: str, limit: int, url: str) -> str:
    """
    Wrap arbitrary SELECT as subquery and apply a dialect-appropriate limit.
    Works for SQL Server, Oracle, Snowflake, Postgres, MySQL, SQLite.
    """
    s = (sql or "").strip().rstrip(";")
    if not s.lower().startswith("select"):
        # Caller may pass a complex query; still wrap it.
        pass

    low = (url or "").lower()
    n = int(limit)

    # SQL Server (ODBC/pyodbc/mssql)
    if "mssql" in low or "odbc" in low or "pyodbc" in low:
        return f"SELECT TOP {n} * FROM ({s}) AS _lim_"

    # Oracle
    if "oracle" in low or "cx_oracle" in low:
        return f"SELECT * FROM ({s}) _lim_ FETCH FIRST {n} ROWS ONLY"

    # Snowflake
    if "snowflake" in low:
        return f"SELECT * FROM ({s}) AS _lim_ LIMIT {n}"

    # Postgres / MySQL / MariaDB / SQLite
    if "postgres" in low or "pgsql" in low or "mysql" in low or "mariadb" in low or "sqlite" in low:
        return f"SELECT * FROM ({s}) AS _lim_ LIMIT {n}"

    # Fallback: try LIMIT
    return f"SELECT * FROM ({s}) AS _lim_ LIMIT {n}"


# ---------- Preview (columns only) ----------
def run_query_preview(engine: sqlalchemy.Engine, sql: str, limit: int = 1000) -> List[str]:
    """
    Executes a small, dialect-aware limited version of the query and
    returns list of column names.
    """
    if not sql or not isinstance(sql, str):
        raise ValueError("SQL text is required.")
    url = str(engine.url)
    lim_sql = _apply_limit_sql(sql, limit, url)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(lim_sql))
        return list(result.keys())


# ---------- Full fetch ----------
def run_query_full(engine: sqlalchemy.Engine, sql: str) -> pd.DataFrame:
    """
    Execute the user query fully (no limit). Returns a pandas DataFrame.
    """
    if not sql or not isinstance(sql, str):
        raise ValueError("SQL text is required.")
    with engine.connect() as conn:
        df = pd.read_sql_query(sqlalchemy.text(sql), conn)
    return df
