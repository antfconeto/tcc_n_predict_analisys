"""
Foto do dossel × leitura da folha: a foto carrega informação de estrutura além da clorofila?

O Falker mede a clorofila de uma folha; a foto vê o dossel (cor das folhas, densidade, sombras, solo). Em 26/05, única
data com corte (12 parcelas), para a massa seca total, a de folha e a altura:
  - R² com o SPAD sozinho, com a medida da foto sozinha e com os dois juntos;
  - correlação parcial da foto com a produção descontando o SPAD (informação que o Falker não tem) e a recíproca;
  - o mesmo para a cobertura vegetal da foto (estrutura pura, sem cor).
Medidas da foto: IPCA, b/r e SI (medianas), a cobertura e a rede neural (SPAD previsto).

Tabela: ext_canopy_leaf.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.data import load_vegetation_plots
from tcc_analysis.science import MEASURE_LABELS, PHOTO_MEASURES, plot_table

DATE = "26-05-2026"
YS = {"Massa_Seca_Total_ha": "Massa seca total", "Massa_Seca_Folha_ha": "Massa seca de folha", "Altura_Media": "Altura média"}


def r2_ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2), y - X @ b


def partial(y, x, z):
    """Correlação parcial de x com y controlando z (e o p com n − 3 graus de liberdade)."""
    _, ry = r2_ols(y, [z])
    _, rx = r2_ols(x, [z])
    r = np.corrcoef(rx, ry)[0, 1]
    t = r * np.sqrt((len(y) - 3) / (1 - r ** 2))
    return float(r), float(2 * stats.t.sf(abs(t), len(y) - 3))


def main():
    df = plot_table()
    df = df[df.Data == DATE].copy()
    v = load_vegetation_plots(verbose=False)
    v["Ponto"] = v.Ponto.astype(int)
    df = df.merge(v[["Ponto"] + [c for c in YS if c not in df.columns]], on="Ponto")
    rows = []
    measures = PHOTO_MEASURES + ["Cobertura", "SPAD_rede"]
    labels = {**MEASURE_LABELS, "Cobertura": "Foto: cobertura vegetal (%)"}
    for ycol, ylab in YS.items():
        y = df[ycol].to_numpy(float)
        s = df.SPAD.to_numpy(float)
        r2_s, _ = r2_ols(y, [s])
        for m in measures:
            x = df[m].to_numpy(float)
            r2_x, _ = r2_ols(y, [x])
            r2_sx, _ = r2_ols(y, [s, x])
            pr_x, p_x = partial(y, x, s)
            pr_s, p_s = partial(y, s, x)
            rows.append({"Producao": ylab, "Medida_foto": labels[m], "n": len(y), "R2_SPAD": r2_s, "R2_foto": r2_x, "R2_juntos": r2_sx,
                         "Ganho_foto_sobre_SPAD": r2_sx - r2_s, "Ganho_SPAD_sobre_foto": r2_sx - r2_x,
                         "Parcial_foto_descontando_SPAD": pr_x, "p_foto": p_x, "Parcial_SPAD_descontando_foto": pr_s, "p_SPAD": p_s,
                         "r_foto_SPAD": float(np.corrcoef(x, s)[0, 1])})
    out = pd.DataFrame(rows)
    db.save_df(out, "ext_canopy_leaf", description="Foto do dossel × folha (Falker): R² e correlações parciais com a produção em 26/05")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
