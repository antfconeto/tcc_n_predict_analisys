"""
Repetibilidade: a foto mede a parcela com menos ruído que o Falker?

Dentro de cada data, cada parcela tem várias medidas: 7 leituras do Falker (folhas) e 9 recortes da foto.
Com a ANOVA de um fator (parcela), calcula-se (Shrout; Fleiss, 1979):
  - ICC(1): correlação intraclasse de UMA medida (uma folha, um recorte);
  - ICC(1,k): confiabilidade da MÉDIA da parcela (7 folhas, 9 recortes) — a medida que de fato é usada;
  - confiabilidade meia × meia: correlação, entre parcelas, das médias de duas metades independentes das medidas
    (3 × 3 folhas; 4 × 4 recortes), corrigida para o total por Spearman-Brown; média de 2000 sorteios.
E, com a ANOVA em blocos das médias por parcela: F e η² da dose (quanto da variação entre parcelas é tratamento).
Só entram as parcelas com foto na data (7 em 18/05; 12 em 21/05 e 26/05), para comparar o mesmo conjunto.

Tabela: sci_repeatability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all
from tcc_analysis.science import MEASURE_LABELS, PHOTO_MEASURES, dbc_anova, photo_units

N_SPLITS = 2000


def icc_oneway(groups):
    """ICC(1) e ICC(1,k) de Shrout e Fleiss (1979) para grupos de tamanhos possivelmente diferentes."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    n = len(groups)
    k = np.mean([len(g) for g in groups])
    grand = np.mean(np.concatenate(groups))
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (n - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (sum(len(g) for g in groups) - n)
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icck = (msb - msw) / msb
    return icc1, icck, k


def split_half(groups, rng):
    """Correlação entre as médias de duas metades sorteadas, corrigida por Spearman-Brown."""
    rs = []
    for _ in range(N_SPLITS):
        a, b = [], []
        for g in groups:
            g = rng.permutation(np.asarray(g, float))
            h = len(g) // 2
            a.append(g[:h].mean())
            b.append(g[h:2 * h].mean())
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.nanmean(rs))
    return r, 2 * r / (1 + r)


def main():
    rng = np.random.default_rng(42)
    f = load_falker_all()
    f = f[f.Data.isin(config.EXPERIMENTAL_DATES)].copy()
    f["Ponto"] = f.Ponto.astype(int)
    units = photo_units()
    tiles = units[(units.Recorte >= 0) & units.Data.isin(config.EXPERIMENTAL_DATES)]
    whole = units[(units.Recorte == -1) & units.Data.isin(config.EXPERIMENTAL_DATES)]

    rows = []
    for d in config.EXPERIMENTAL_DATES:
        plots = sorted(whole[whole.Data == d].Ponto.unique())
        fd = f[(f.Data == d) & f.Ponto.isin(plots)]
        measures = [("SPAD", "folhas", [fd[fd.Ponto == p]["Clorofila Total"].to_numpy() for p in plots],
                     fd.groupby("Ponto")["Clorofila Total"].mean())]
        for m in PHOTO_MEASURES:
            td = tiles[tiles.Data == d]
            measures.append((m, "recortes", [td[td.Ponto == p][m].to_numpy() for p in plots],
                             whole[whole.Data == d].set_index("Ponto")[m]))
        for name, unit, groups, plot_vals in measures:
            icc1, icck, k = icc_oneway(groups)
            r_half, r_full = split_half(groups, rng)
            dfp = pd.DataFrame({"Ponto": plots, "y": plot_vals.loc[plots].to_numpy()})
            for field in ("Bloco", "Dose"):
                dfp[field] = dfp.Ponto.map(lambda p, fl=field: config.PONTO_TO_DBC[p][fl])
            an = dbc_anova(dfp, "y")
            rows.append({"Data": d, "Medida": name, "Rotulo": MEASURE_LABELS[name], "Unidade": unit, "n_parcelas": len(plots),
                         "k_por_parcela": k, "ICC_uma_medida": icc1, "Confiabilidade_media": icck,
                         "Meia_x_meia_r": r_half, "Meia_x_meia_corrigida": r_full,
                         "F_dose": an.get("F_dose"), "p_dose": an.get("p_dose"), "eta2_dose": an.get("eta2_dose")})

    out = pd.DataFrame(rows)
    # média das datas com todas as parcelas fotografadas (21/05 e 26/05)
    full = out[out.Data.isin(["21-05-2026", "26-05-2026"])]
    mean = full.groupby(["Medida", "Rotulo", "Unidade"], as_index=False)[
        ["k_por_parcela", "ICC_uma_medida", "Confiabilidade_media", "Meia_x_meia_r", "Meia_x_meia_corrigida", "F_dose", "eta2_dose"]].mean()
    mean["Data"] = "Média 21/05 e 26/05"
    mean["n_parcelas"] = 12
    out = pd.concat([out, mean], ignore_index=True)
    db.save_df(out, "sci_repeatability", description="Repetibilidade (ICC, meia × meia) e sinal da dose: Falker × índices da foto")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
