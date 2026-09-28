"""
Como as leituras do Falker se distribuem dentro da parcela: simétricas ou com cauda? Média ou mediana?

  - Todas as parcelas do experimento (7 leituras): desvios de cada leitura em relação à média da sua parcela,
    padronizados pelo desvio da parcela e juntados (252 valores): assimetria, curtose e Shapiro-Wilk.
  - Pontos de teste de 01/06 (20 leituras cada): assimetria e normalidade por ponto.
  - Média × mediana como resumo da parcela: reamostrando k leituras de cada ponto de teste (k = 5, 7, 10), qual dos dois
    varia menos (erro padrão empírico) e qual fica mais perto da média das 20 leituras.
  - Leituras discrepantes: fora de 1,5 × IQR da própria parcela.

Tabelas: ext_reading_shape, ext_reading_mean_median, ext_reading_residuals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all

N_BOOT = 5000


def shape(x):
    return {"n": len(x), "Assimetria": float(stats.skew(x)), "Curtose_excesso": float(stats.kurtosis(x)),
            "Shapiro_W": float(stats.shapiro(x)[0]), "Shapiro_p": float(stats.shapiro(x)[1])}


def main():
    rng = np.random.default_rng(42)
    f = load_falker_all()
    f["Ponto"] = f.Ponto.astype(int)
    exp = f[f.Data.isin(config.EXPERIMENTAL_DATES)].copy()
    g = exp.groupby(["Data", "Ponto"])["Clorofila Total"]
    exp["z"] = (exp["Clorofila Total"] - g.transform("mean")) / g.transform("std")
    exp["Desvio_SPAD"] = exp["Clorofila Total"] - g.transform("mean")
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    exp["Discrepante"] = (exp["Clorofila Total"] < q1 - 1.5 * (q3 - q1)) | (exp["Clorofila Total"] > q3 + 1.5 * (q3 - q1))
    rows = [{"Conjunto": "Experimento: desvios padronizados (7 leituras × 36 parcelas)", **shape(exp.z.to_numpy()),
             "Discrepantes_pct": 100 * exp.Discrepante.mean(),
             "Discrepantes_abaixo": int((exp.Discrepante & (exp.z < 0)).sum()), "Discrepantes_acima": int((exp.Discrepante & (exp.z > 0)).sum())}]
    by_plot = exp.groupby(["Data", "Ponto"])["Clorofila Total"].apply(lambda s: stats.skew(s))
    rows.append({"Conjunto": "Experimento: assimetria por parcela (média de 36)", "n": len(by_plot), "Assimetria": float(by_plot.mean()),
                 "Prop_parcelas_cauda_esquerda": float((by_plot < 0).mean())})
    test = f[f.Data == config.TEST_DATE]
    mm = []
    for p, s in test.groupby("Ponto"):
        x = s["Clorofila Total"].to_numpy(float)
        q1, q3 = np.percentile(x, [25, 75])
        rows.append({"Conjunto": f"Ponto de teste {p} (01/06)", **shape(x), "Media": float(x.mean()), "Mediana": float(np.median(x)),
                     "Discrepantes_pct": 100 * float(np.mean((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))))})
        for k in (5, 7, 10):
            means = np.array([rng.choice(x, k, replace=False).mean() for _ in range(N_BOOT)])
            meds = np.array([np.median(rng.choice(x, k, replace=False)) for _ in range(N_BOOT)])
            mm.append({"Ponto": int(p), "Leituras": k, "EP_media": float(means.std()), "EP_mediana": float(meds.std()),
                       "Erro_medio_media_vs_20": float(np.mean(np.abs(means - x.mean()))),
                       "Erro_medio_mediana_vs_20": float(np.mean(np.abs(meds - x.mean())))})
    mm = pd.DataFrame(mm)
    mm_sum = mm.groupby("Leituras")[["EP_media", "EP_mediana", "Erro_medio_media_vs_20", "Erro_medio_mediana_vs_20"]].mean().reset_index()
    db.save_df(pd.DataFrame(rows), "ext_reading_shape", description="Forma da distribuição das leituras do Falker dentro da parcela")
    db.save_df(pd.concat([mm.assign(Nivel="ponto"), mm_sum.assign(Nivel="média dos pontos")], ignore_index=True),
               "ext_reading_mean_median", description="Média × mediana como resumo da parcela (reamostragem dos pontos de teste)")
    db.save_df(exp[["Data", "Ponto", "Clorofila Total", "Desvio_SPAD", "z", "Discrepante"]], "ext_reading_residuals",
               description="Desvio de cada leitura em relação à média da parcela")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(mm_sum.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
