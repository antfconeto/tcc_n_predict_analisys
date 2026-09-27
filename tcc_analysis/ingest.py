"""Carga inicial do banco: dados brutos e importação da análise anterior (legada)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import config, db
from .data import BIOMASS_TABLE, FALKER_COLUMNS, HEIGHT_WEIGHT_TABLE
from .indices import get_exif_time

LEGACY_SCRIPT = "legacy_import"


def parse_falker_csv(file_path: Path) -> pd.DataFrame:
    """Lê o CSV transposto exportado pelo Falker Leaf (mesma lógica dos scripts antigos)."""
    skip_count = 0
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Medição" in line:
                skip_count = i
                break

    df = pd.read_csv(file_path, skiprows=skip_count, header=None)
    df = df.set_index(0).transpose()
    df.columns.name = None
    for col in ["Medição", "Ponto", "Latitude", "Longitude", "Clorofila A", "Clorofila B", "Clorofila Total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # A primeira linha transposta é a coluna vazia do cabeçalho, não uma medição
    return df.dropna(subset=["Medição"]).reset_index(drop=True)


def ingest_raw(verbose: bool = True) -> None:
    """(Re)carrega as tabelas de dados brutos a partir de data/raw e da pasta de imagens."""
    log = print if verbose else (lambda *a, **k: None)
    db.init_schema()
    run_id = db.current_run_id()

    with db.connect() as conn:
        conn.execute("DELETE FROM dbc_layout")
        conn.executemany(
            "INSERT INTO dbc_layout (ponto, bloco, tratamento, dose) VALUES (?, ?, ?, ?)",
            [(p, v["Bloco"], v["Tratamento"], v["Dose"]) for p, v in config.PONTO_TO_DBC.items()],
        )

        conn.execute("DELETE FROM falker_readings")
        inverse = {v: k for k, v in FALKER_COLUMNS.items()}
        for source, path in config.FALKER_FILES.items():
            df = parse_falker_csv(path)
            df = df.rename(columns=inverse)[list(FALKER_COLUMNS)]
            df.insert(0, "source", source)
            df.to_sql("falker_readings", conn, if_exists="append", index=False)
            log(f"  falker_readings ← {path.name}: {len(df)} leituras")

    # Tabelas de campo: cópia fiel dos CSVs (texto como está no arquivo)
    for table, path in ((HEIGHT_WEIGHT_TABLE, config.HEIGHT_WEIGHT_CSV), (BIOMASS_TABLE, config.BIOMASS_CSV)):
        df = pd.read_csv(path)
        with db.connect() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        db.save_df(df, table, description=f"Dados brutos de campo ({path.name})", run_id=run_id)
        log(f"  {table} ← {path.name}: {len(df)} linhas")

    ingest_images(verbose=verbose)


def ingest_images(verbose: bool = True) -> None:
    """Cataloga as imagens por data/ponto (os arquivos continuam em disco)."""
    log = print if verbose else (lambda *a, **k: None)
    root = config.IMAGE_ROOT
    if not root.exists():
        log(f"  [aviso] pasta de imagens não encontrada: {root} (defina TCC_IMAGE_ROOT)")
        return
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = re.match(r"^[Pp](\d+)", img.name)
            ponto = int(m.group(1)) if m else None
            layout = config.PONTO_TO_DBC.get(ponto, {}) if folder.name != "01-06" else {}
            rows.append((
                folder.name, f"{folder.name}-{config.YEAR}", img.name,
                str(img.relative_to(root)), ponto,
                layout.get("Bloco"), layout.get("Tratamento"), layout.get("Dose"),
                get_exif_time(str(img)),
            ))
    with db.connect() as conn:
        conn.execute("DELETE FROM images")
        conn.executemany(
            """INSERT INTO images (date_folder, data, file_name, rel_path, ponto, bloco,
                                   tratamento, dose, photo_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    log(f"  images ← {root}: {len(rows)} arquivos")


# ─────────────────────────────────────────────────────────────────────────────
# Análise anterior (projeto original) → tabelas legacy__*
# ─────────────────────────────────────────────────────────────────────────────
def _legacy_table_name(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[0] in ("images-infos", "vegetation-analysis"):
        parts = parts[1:]
    slug = "__".join(parts)
    return "legacy__" + re.sub(r"[^0-9A-Za-z_]+", "_", slug)


def legacy_imported() -> bool:
    if not db.table_exists("runs"):
        return False
    df = db.query("SELECT run_id FROM runs WHERE script = ? AND status = 'finished'", (LEGACY_SCRIPT,))
    return not df.empty


def drop_legacy() -> None:
    with db.connect() as conn:
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'legacy\\_\\_%' ESCAPE '\\'"
        ).fetchall():
            conn.execute(f'DROP TABLE "{name}"')
        conn.execute("DELETE FROM catalog WHERE table_name LIKE 'legacy\\_\\_%' ESCAPE '\\'")
        legacy_runs = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE script = ?", (LEGACY_SCRIPT,))]
        for rid in legacy_runs:
            conn.execute("DELETE FROM reports WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (rid,))


def import_legacy(legacy_root: Path | None = None, verbose: bool = True) -> dict:
    """Importa todos os resultados do projeto original sem alterar os arquivos de origem."""
    log = print if verbose else (lambda *a, **k: None)
    legacy_root = Path(legacy_root or config.LEGACY_ROOT)
    output_root = legacy_root / "data" / "output"
    if not output_root.exists():
        raise FileNotFoundError(f"Pasta de resultados legados não encontrada: {output_root}")

    db.init_schema()
    run_id = db.start_run(
        script=LEGACY_SCRIPT,
        params={"legacy_root": str(legacy_root)},
        notes="Resultados da análise anterior (scripts 01–17 e another_scripts), importados como estavam.",
    )
    counts = {"tables": 0, "reports": 0, "artifacts": 0}

    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_root)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            table = _legacy_table_name(rel)
            df = pd.read_csv(path)
            db.save_df(df, table, description=f"[legado] data/output/{rel}", run_id=run_id)
            counts["tables"] += 1
            log(f"  {table:<70} {len(df):>5} linhas")
        elif suffix in (".md", ".txt"):
            db.save_report(
                f"legacy/{rel.with_suffix('')}", path.read_text(encoding="utf-8"),
                fmt="markdown" if suffix == ".md" else "text", run_id=run_id,
            )
            counts["reports"] += 1
        else:
            # .xlsx (cópias das mesmas tabelas em CSV), imagens etc.: apenas referenciados
            db.register_artifact(path, kind=f"legacy{suffix}", run_id=run_id)
            counts["artifacts"] += 1

    for png in sorted((legacy_root / "plots").rglob("*.png")):
        db.register_artifact(png, kind="legacy_plot", run_id=run_id)
        counts["artifacts"] += 1

    manifest = legacy_root / "data" / "export_app_validation" / "manifest.json"
    if manifest.exists():
        images = json.loads(manifest.read_text())["images"]
        df = pd.json_normalize(images)
        db.save_df(df, "legacy__app_validation_manifest",
                   description="[legado] manifest.json do dataset de validação do app", run_id=run_id)
        counts["tables"] += 1

    for doc in sorted((legacy_root / "docs").glob("*.md")):
        db.save_report(f"legacy/docs/{doc.stem}", doc.read_text(encoding="utf-8"), run_id=run_id)
        counts["reports"] += 1

    db.finish_run("finished")
    return counts
