"""Carga de dados a partir do banco (substitui a leitura direta dos CSVs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import db
from .config import PONTO_TO_DBC

FALKER_COLUMNS = {
    "medicao": "Medição",
    "ponto": "Ponto",
    "hora": "Hora",
    "data": "Data",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "clorofila_a": "Clorofila A",
    "clorofila_b": "Clorofila B",
    "clorofila_total": "Clorofila Total",
}

HEIGHT_WEIGHT_TABLE = "campo_altura_peso"
BIOMASS_TABLE = "campo_massa_verde_seca"
INDICES_TABLE = "image_indices"


# ─────────────────────────────────────────────────────────────────────────────
# Clorofila medida (Falker)
# ─────────────────────────────────────────────────────────────────────────────
def load_falker(source: str = "cloro") -> pd.DataFrame:
    """Leituras Falker de um arquivo de origem ('cloro' = 18/05–26/05, 'test' = 01/06)."""
    df = db.query(
        "SELECT * FROM falker_readings WHERE source = ? ORDER BY reading_id", (source,)
    )
    if df.empty:
        raise LookupError(
            f"Sem leituras Falker para '{source}'. Rode pipeline/00_configuracao/01_init_db.py."
        )
    return df[list(FALKER_COLUMNS)].rename(columns=FALKER_COLUMNS).reset_index(drop=True)


def load_falker_all() -> pd.DataFrame:
    return pd.concat([load_falker("cloro"), load_falker("test")], ignore_index=True)


def map_dbc(ponto, field: str):
    if pd.notna(ponto) and int(ponto) in PONTO_TO_DBC:
        return PONTO_TO_DBC[int(ponto)][field]
    return None


def add_dbc_columns(df: pd.DataFrame, fields=("Tratamento", "Bloco")) -> pd.DataFrame:
    for field in fields:
        df[field] = df["Ponto"].map(lambda x, f=field: map_dbc(x, f))
    return df


def load_chlorophyll_with_layout() -> pd.DataFrame:
    """Todas as leituras (cloro + test) com Tratamento/Bloco e sem Clorofila Total nula."""
    df = add_dbc_columns(load_falker_all())
    return df.dropna(subset=["Clorofila Total"]).copy()


def build_merged_dataset(df_cloro: pd.DataFrame, df_indices: pd.DataFrame) -> pd.DataFrame:
    """Média de clorofila por parcela (Data, Ponto) unida aos índices da imagem."""
    df_chlo_plot = (
        df_cloro.dropna(subset=["Tratamento"])
        .groupby(["Data", "Ponto"])
        .agg({
            "Clorofila Total": "mean",
            "Bloco": "first",
            "Tratamento": "first",
        })
        .reset_index()
    )
    return pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])


# ─────────────────────────────────────────────────────────────────────────────
# Índices extraídos das imagens
# ─────────────────────────────────────────────────────────────────────────────
def save_indices(df: pd.DataFrame, block_size: int, variant: str = "default") -> None:
    out = df.copy()
    out.insert(0, "Block_Size", int(block_size))
    out.insert(0, "Variant", variant)
    db.save_df(
        out, INDICES_TABLE,
        description="Estatísticas dos índices espectrais por imagem (variante de pré-processamento × grade)",
    )


def indices_available(block_size: int, variant: str = "default") -> bool:
    return db.has_rows(INDICES_TABLE, {"Variant": variant, "Block_Size": int(block_size)})


def load_indices(block_size: int, variant: str = "default") -> pd.DataFrame:
    df = db.load_df(INDICES_TABLE, {"Variant": variant, "Block_Size": int(block_size)})
    return df.drop(columns=["Variant", "Block_Size"])


# ─────────────────────────────────────────────────────────────────────────────
# Dados de campo (altura, massa verde/seca) — 26/05/2026
# ─────────────────────────────────────────────────────────────────────────────
def load_height_weight_raw() -> pd.DataFrame:
    """Equivalente a pd.read_csv('dados-altura-peso.csv')."""
    return db.load_df(HEIGHT_WEIGHT_TABLE, run_id=None)


def load_biomass_raw() -> pd.DataFrame:
    """Equivalente a pd.read_csv('massa_verde_seca_compilada.csv')."""
    return db.load_df(BIOMASS_TABLE, run_id=None)


def clean_block(b):
    if pd.isna(b):
        return None
    b_str = str(b).strip()
    if b_str.startswith("Bl "):
        return "Bloco " + b_str.replace("Bl ", "").strip()
    return b_str


def parse_heights(h_str):
    if pd.isna(h_str):
        return np.nan
    h_str = str(h_str).replace('"', '').strip()
    parts = [float(x.strip()) for x in h_str.split(",") if x.strip()]
    return np.mean(parts) if parts else np.nan


def load_heights() -> pd.DataFrame:
    """Alturas/pesos por parcela com Bloco normalizado e Altura_Media calculada."""
    df_alt_raw = load_height_weight_raw().dropna(subset=["Bloco"])
    df_alt_raw["Bloco"] = df_alt_raw["Bloco"].apply(clean_block)
    df_alt_raw["Tratamento"] = df_alt_raw["Tratamento"].str.strip()
    df_alt_raw["Altura_Media"] = df_alt_raw["Alturas Anotadas (cm)"].apply(parse_heights)
    return df_alt_raw


def load_vegetation_plots(date: str = "26-05-2026", verbose: bool = True) -> pd.DataFrame:
    """Clorofila observada + altura + biomassa por parcela, com rendimentos em t/ha.

    Mesma lógica de `load_and_merge_data` dos antigos scripts 11, 12 e 13.
    """
    log = print if verbose else (lambda *a, **k: None)

    log("  [Data Loading] Parsing chlorophyll data...")
    df_cloro = load_falker("cloro")
    df_cloro_day = df_cloro[df_cloro["Data"] == date].copy()
    if df_cloro_day.empty:
        raise ValueError(f"No chlorophyll data found for the date {date}!")

    df_chlo_ponto = (
        df_cloro_day.groupby("Ponto")[["Clorofila A", "Clorofila B", "Clorofila Total"]]
        .mean().reset_index()
    )
    for field in ("Bloco", "Tratamento", "Dose"):
        df_chlo_ponto[field] = df_chlo_ponto["Ponto"].map(lambda x, f=field: map_dbc(x, f))

    log("  [Data Loading] Loading height-weight data...")
    df_alt_raw = load_heights()
    cols_to_float = [
        "Peso Matéria Total - Peso MT (kg)",
        "Peso Amostra Detalhada (kg)",
        "Peso do Caule - C (kg)",
        "Peso da Folha/Raiz - F/P (kg)",
    ]
    for col in cols_to_float:
        if col in df_alt_raw.columns:
            df_alt_raw[col] = df_alt_raw[col].astype(str).str.replace(",", ".").astype(float)

    log("  [Data Loading] Loading compiled green/dry weights...")
    df_mass = load_biomass_raw()
    df_mass["Bloco"] = df_mass["Bloco"].str.strip()
    df_mass["Tratamento"] = df_mass["Tratamento"].str.strip()
    df_mass_pivoted = df_mass.pivot_table(
        index=["Bloco", "Tratamento"],
        columns="Componente",
        values=["Massa_Verde", "Massa_Seca"],
    )
    df_mass_pivoted.columns = [f"{val}_{comp}" for val, comp in df_mass_pivoted.columns]
    df_mass_pivoted = df_mass_pivoted.reset_index()

    log("  [Data Loading] Merging datasets...")
    df = pd.merge(df_chlo_ponto, df_alt_raw, on=["Bloco", "Tratamento"])
    df = pd.merge(df, df_mass_pivoted, on=["Bloco", "Tratamento"])

    # Rendimentos por parcela extrapolados para 1 hectare (t/ha)
    df["Massa_Verde_Total_ha"] = df["Peso Matéria Total - Peso MT (kg)"] * 10.0
    df["Massa_Seca_Total_ha"] = df["Massa_Verde_Total_ha"] * (df["Massa_Seca_Amostra"] / df["Massa_Verde_Amostra"])
    df["Massa_Verde_Folha_ha"] = df["Massa_Verde_Total_ha"] * (df["Massa_Verde_Folha"] / df["Massa_Verde_Amostra"])
    df["Massa_Seca_Folha_ha"] = df["Massa_Verde_Total_ha"] * (df["Massa_Seca_Folha"] / df["Massa_Verde_Amostra"])
    df["Massa_Verde_Colmo_ha"] = df["Massa_Verde_Total_ha"] * (df["Massa_Verde_Colmo"] / df["Massa_Verde_Amostra"])
    df["Massa_Seca_Colmo_ha"] = df["Massa_Verde_Total_ha"] * (df["Massa_Seca_Colmo"] / df["Massa_Verde_Amostra"])

    # Relação folha/colmo
    df["Rel_Folha_Colmo_Verde"] = df["Massa_Verde_Folha_ha"] / df["Massa_Verde_Colmo_ha"]
    df["Rel_Folha_Colmo_Seco"] = df["Massa_Seca_Folha_ha"] / df["Massa_Seca_Colmo_ha"]
    return df
