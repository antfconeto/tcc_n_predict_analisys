"""Banco de dados local (SQLite) para dados brutos, resultados, relatórios e gráficos.

Toda execução de script é registrada em `runs`. Tabelas de resultado recebem a
coluna `run_id`, o que preserva o histórico: uma nova execução acrescenta linhas
em vez de sobrescrever as anteriores. `load_df` devolve, por padrão, a execução
mais recente.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    script      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    params      TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS catalog (
    table_name  TEXT PRIMARY KEY,
    description TEXT,
    script      TEXT,
    last_run_id INTEGER,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    report_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER REFERENCES runs(run_id),
    name       TEXT NOT NULL,
    format     TEXT NOT NULL DEFAULT 'markdown',
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(run_id),
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dbc_layout (
    ponto      INTEGER PRIMARY KEY,
    bloco      TEXT NOT NULL,
    tratamento TEXT NOT NULL,
    dose       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS falker_readings (
    reading_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    medicao         REAL,
    ponto           REAL,
    hora            TEXT,
    data            TEXT,
    latitude        REAL,
    longitude       REAL,
    clorofila_a     REAL,
    clorofila_b     REAL,
    clorofila_total REAL
);

CREATE TABLE IF NOT EXISTS images (
    image_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    date_folder TEXT NOT NULL,
    data        TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    ponto       INTEGER,
    bloco       TEXT,
    tratamento  TEXT,
    dose        REAL,
    photo_time  TEXT,
    UNIQUE (date_folder, file_name)
);

CREATE INDEX IF NOT EXISTS idx_reports_name ON reports(name);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
"""

_current_run_id: int | None = None
_current_script: str | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema() -> None:
    with connect() as conn:
        conn.executescript(CORE_SCHEMA)


def table_exists(name: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    return row is not None


def list_tables() -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


def query(sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ─────────────────────────────────────────────────────────────────────────────
# Execuções (runs)
# ─────────────────────────────────────────────────────────────────────────────
def _script_label(script: str | None) -> str:
    path = Path(script or sys.argv[0]).resolve()
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return path.name


def start_run(script: str | None = None, params: dict | None = None, notes: str | None = None) -> int:
    global _current_run_id, _current_script
    init_schema()
    label = _script_label(script)
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (script, started_at, status, params, notes) VALUES (?, ?, 'running', ?, ?)",
            (label, _now(), json.dumps(params, default=str) if params else None, notes),
        )
        _current_run_id = cur.lastrowid
    _current_script = label
    return _current_run_id


def finish_run(status: str = "finished") -> None:
    global _current_run_id, _current_script
    if _current_run_id is None:
        return
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",
            (_now(), status, _current_run_id),
        )
    _current_run_id = None
    _current_script = None


def current_run_id() -> int:
    if _current_run_id is None:
        start_run()
    return _current_run_id


def run_main(main_fn, params: dict | None = None, script: str | None = None):
    """Executa `main_fn` dentro de uma execução registrada (status finished/failed)."""
    start_run(script=script or getattr(sys.modules.get(main_fn.__module__), "__file__", None), params=params)
    print(f"[db] run_id={_current_run_id}  banco={config.DB_PATH}")
    try:
        result = main_fn()
    except BaseException:
        traceback.print_exc()
        finish_run("failed")
        raise
    finish_run("finished")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tabelas de resultado
# ─────────────────────────────────────────────────────────────────────────────
def _sqlite_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        s = out[col]
        if s.dtype == bool:
            out[col] = s.astype(int)
        elif s.dtype == object or str(s.dtype) in ("string", "str"):
            out[col] = s.map(
                lambda v: v if v is None or isinstance(v, (str, int, float, np.integer, np.floating))
                else (None if (isinstance(v, float) and np.isnan(v)) else str(v))
            )
    return out


def _existing_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def save_df(
    df: pd.DataFrame,
    table: str,
    description: str | None = None,
    run_id: int | None = None,
) -> int:
    """Acrescenta `df` à tabela `table` marcando as linhas com o run_id atual."""
    run_id = run_id if run_id is not None else current_run_id()
    data = _sqlite_ready(df)
    data.insert(0, "run_id", run_id)

    with connect() as conn:
        if table_exists(table):
            existing = _existing_columns(conn, table)
            for col in data.columns:
                if col not in existing:
                    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}"')
        data.to_sql(table, conn, if_exists="append", index=False)
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_run" ON "{table}"(run_id)'
        )
        conn.execute(
            """INSERT INTO catalog (table_name, description, script, last_run_id, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(table_name) DO UPDATE SET
                 description = COALESCE(excluded.description, catalog.description),
                 script = excluded.script,
                 last_run_id = excluded.last_run_id,
                 updated_at = excluded.updated_at""",
            (table, description, _current_script, run_id, _now()),
        )
    return len(data)


def latest_run_id(table: str, filters: dict | None = None) -> int | None:
    if not table_exists(table):
        return None
    where, params = _where(filters)
    with connect() as conn:
        row = conn.execute(f'SELECT MAX(run_id) FROM "{table}"{where}', params).fetchone()
    return row[0] if row else None


def _where(filters: dict | None, extra: str | None = None):
    clauses, params = [], []
    for k, v in (filters or {}).items():
        clauses.append(f'"{k}" = ?')
        params.append(v)
    if extra:
        clauses.append(extra)
    return ((" WHERE " + " AND ".join(clauses)) if clauses else ""), params


def load_df(
    table: str,
    filters: dict | None = None,
    run_id: int | str | None = "latest",
    keep_run_id: bool = False,
) -> pd.DataFrame:
    """Lê uma tabela de resultado.

    run_id="latest" (padrão) → apenas a execução mais recente que satisfaz `filters`;
    run_id=None → todas as execuções; run_id=<int> → execução específica.
    """
    if not table_exists(table):
        raise LookupError(
            f"Tabela '{table}' não existe em {config.DB_PATH}. Rode o script que a gera primeiro."
        )
    filters = dict(filters or {})
    if run_id == "latest":
        run_id = latest_run_id(table, filters)
        if run_id is None:
            raise LookupError(f"Nenhuma linha em '{table}' para {filters}.")
    if run_id is not None:
        filters["run_id"] = run_id
    where, params = _where(filters)
    with connect() as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"{where}', conn, params=params)
    if not keep_run_id and "run_id" in df.columns:
        df = df.drop(columns=["run_id"])
    return df


def has_rows(table: str, filters: dict | None = None) -> bool:
    return latest_run_id(table, filters) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Relatórios e gráficos
# ─────────────────────────────────────────────────────────────────────────────
def save_report(name: str, content: str, fmt: str = "markdown", run_id: int | None = None) -> Path:
    """Grava o relatório no banco e escreve uma cópia em outputs/reports/ para leitura."""
    run_id = run_id if run_id is not None else current_run_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO reports (run_id, name, format, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, name, fmt, content, _now()),
        )
    ext = ".md" if fmt == "markdown" else ".txt"
    out = config.REPORTS_DIR / f"{name}{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


@contextmanager
def report_writer(name: str, fmt: str = "markdown"):
    """Substituto de `open(path, "w")` para relatórios: grava no banco ao fechar."""
    buf = io.StringIO()
    yield buf
    save_report(name, buf.getvalue(), fmt=fmt)


def load_report(name: str, run_id: int | None = None) -> str:
    sql = "SELECT content FROM reports WHERE name = ?"
    params: list = [name]
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    sql += " ORDER BY report_id DESC LIMIT 1"
    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
    if row is None:
        raise LookupError(f"Relatório '{name}' não encontrado.")
    return row[0]


def register_artifact(path: str | Path, kind: str = "plot", run_id: int | None = None) -> None:
    run_id = run_id if run_id is not None else current_run_id()
    path = Path(path).resolve()
    try:
        label = str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        label = str(path)
    with connect() as conn:
        conn.execute(
            "INSERT INTO artifacts (run_id, kind, path, created_at) VALUES (?, ?, ?, ?)",
            (run_id, kind, label, _now()),
        )


def plot_path(name: str) -> str:
    """Caminho para salvar um gráfico em outputs/plots/ (registrado no banco)."""
    out = config.PLOTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    register_artifact(out, kind="plot")
    return str(out)
