"""Ferramentas comuns às análises complementares (pipeline/07_analises_complementares)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from . import config, db
from .data import falker_plot_means

# Medidas da foto usadas nas análises: os índices mais consistentes nas três coletas e na classificação
# (proporção de azul em relação ao vermelho e saturação), sempre a mediana dos blocos de vegetação.
PHOTO_MEASURES = ["Median_IPCA", "Median_b_over_r", "Median_SI"]
MEASURE_LABELS = {
    "SPAD": "SPAD (Falker)",
    "Median_IPCA": "Foto: IPCA (mediana)",
    "Median_b_over_r": "Foto: b/r (mediana)",
    "Median_SI": "Foto: SI (mediana)",
    "SPAD_rede": "Foto: rede neural (SPAD previsto)",
}


def photo_units() -> pd.DataFrame:
    """Tabela sci_photo_units (foto inteira = Recorte −1; recortes 0..8), com bloco, tratamento e dose."""
    df = db.load_df("sci_photo_units")
    df["Ponto"] = df.Ponto.astype(int)
    for field in ("Bloco", "Tratamento", "Dose"):
        df[field] = [None if d == config.TEST_DATE else config.PONTO_TO_DBC.get(int(p), {}).get(field)
                     for d, p in zip(df.Data, df.Ponto)]
    return df


def plot_table() -> pd.DataFrame:
    """Uma linha por parcela × data do experimento: Falker (média, desvio, A, B), índices da foto inteira e a rede."""
    fk = falker_plot_means()
    ph = photo_units()
    ph = ph[(ph.Recorte == -1) & ph.Data.isin(config.EXPERIMENTAL_DATES)]
    keep = ["Data", "Ponto", "Cobertura"] + [c for c in ph.columns if c.startswith(("DP_", "IQR_", "CV_"))] + \
        [c for c in ph.columns if c.split("_")[0] in ("Median", "Mean", "P75", "P90") and not c.endswith("_rgb")]
    out = fk.merge(ph[keep], on=["Data", "Ponto"], how="left")
    pred = db.load_df("definitive_model_predictions")[["Data", "Ponto", "Predicao_MLP"]].copy()
    pred["Ponto"] = pred.Ponto.astype(int)
    out = out.merge(pred.rename(columns={"Predicao_MLP": "SPAD_rede"}), on=["Data", "Ponto"], how="left")
    out["Tem_foto"] = out.Median_IPCA.notna()
    return out


def orientation(df: pd.DataFrame, col: str) -> float:
    """+1 se a medida cresce com o SPAD (todas as datas juntas), −1 se decresce."""
    sub = df[[col, "SPAD"]].dropna()
    return 1.0 if np.corrcoef(sub[col], sub.SPAD)[0, 1] >= 0 else -1.0


def dbc_anova(df: pd.DataFrame, y: str) -> dict:
    """ANOVA em blocos (y ~ bloco + dose): F, p e η² da dose, e o quadrado médio do resíduo."""
    d = df[[y, "Bloco", "Dose"]].dropna().rename(columns={y: "y"})
    if d.Dose.nunique() < 2 or d.Bloco.nunique() < 2 or len(d) < 6:
        return {}
    fit = ols("y ~ C(Bloco) + C(Dose)", data=d).fit()
    tab = sm.stats.anova_lm(fit, typ=2)
    ss_total = float(((d.y - d.y.mean()) ** 2).sum())
    return {"n": len(d), "F_dose": float(tab.loc["C(Dose)", "F"]), "p_dose": float(tab.loc["C(Dose)", "PR(>F)"]),
            "eta2_dose": float(tab.loc["C(Dose)", "sum_sq"] / ss_total) if ss_total else np.nan,
            "F_bloco": float(tab.loc["C(Bloco)", "F"]), "p_bloco": float(tab.loc["C(Bloco)", "PR(>F)"]),
            "QM_residuo": float(fit.mse_resid), "GL_residuo": float(fit.df_resid)}
