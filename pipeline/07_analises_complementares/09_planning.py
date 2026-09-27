"""
Planejamento da próxima coleta, com a variação medida neste experimento.

1. Leituras do Falker por parcela: quantas para a média ficar a ±E SPAD (IC 95 %), com o desvio típico entre folhas;
   e o teto da classificação (3 classes) em função do número de leituras (simulação: mesma parcela medida duas vezes).
2. Blocos (repetições) para detectar uma diferença entre duas doses (ANOVA em blocos com 4 doses, α = 5 %,
   poder de 80 %, t não central). Em SPAD, para o Falker: o desvio residual entre parcelas é separado em variação
   real entre parcelas e ruído das folhas, σ²(k leituras) = σ²_parcela + σ²_folha / k. Sem unidade, para comparar
   Falker e foto: diferença de 25 e 50 kg N/ha, com o efeito = inclinação da resposta × diferença de dose e o ruído
   = desvio residual de cada medida (21/05 e 26/05, todas as parcelas fotografadas).
3. Datas de coleta: desempenho de uma reta índice → SPAD treinada com 1 data e com 2 datas, testada numa data que
   não entrou no treino (R² e acerto das 3 classes). Com só 3 datas é uma indicação da tendência, não uma curva.

Tabelas: sci_plan_readings, sci_plan_ceiling, sci_plan_blocks, sci_plan_sensitivity, sci_plan_dates.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import dbc_anova, orientation, plot_table

READINGS = [7, 10, 14, 20, 28, 40]
DELTAS = [1.0, 2.0, 3.0, 4.0]
PHOTO_BEST = "Median_IPCA"


def classify(v, limits):
    v = np.asarray(v, float)
    return np.where(v < limits[0], 0, np.where(v <= limits[1], 1, 2))


def blocks_needed(sigma, delta, n_trt=4, alpha=0.05, power=0.8, max_b=60):
    """Menor nº de blocos com poder ≥ 80 % para a diferença entre duas doses (contraste com t não central)."""
    for b in range(2, max_b + 1):
        df = (b - 1) * (n_trt - 1)
        ncp = delta / (sigma * np.sqrt(2 / b))
        tcrit = stats.t.ppf(1 - alpha / 2, df)
        pw = 1 - stats.nct.cdf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)
        if pw >= power:
            return b, float(pw)
    return np.nan, np.nan


def main():
    rng = np.random.default_rng(42)
    df = plot_table()
    limits = db.load_df("chl_class_scheme").sort_values("Ordem").SPAD_max.dropna().tolist()
    sd_leaf = float(np.sqrt((df.SPAD_dp ** 2).mean()))

    # 1. leituras por parcela
    reads = []
    for e in [0.5, 1.0, 1.5, 2.0, 2.5]:
        n = 2
        while stats.t.ppf(0.975, n - 1) * sd_leaf / np.sqrt(n) > e and n < 1000:
            n += 1
        reads.append({"Margem_SPAD": e, "Leituras_necessarias": n})
    ceiling = []
    for k in READINGS:
        se = df.SPAD_dp.to_numpy() / np.sqrt(k)
        ex, ks = [], []
        for _ in range(3000):
            a = classify(df.SPAD + rng.normal(0, se), limits)
            b = classify(df.SPAD + rng.normal(0, se), limits)
            ex.append(np.mean(a == b))
        tcrit = stats.t.ppf(0.975, k - 1)
        lo, hi = classify(df.SPAD - tcrit * se, limits), classify(df.SPAD + tcrit * se, limits)
        ceiling.append({"Leituras": k, "IC95_meia_largura": float(np.mean(tcrit * se)), "Teto_classe_exata": float(np.mean(ex)),
                        "Parcelas_com_classe_segura": float(np.mean(lo == hi))})

    # 2. blocos para detectar Δ
    anovas = {d: dbc_anova(df[df.Data == d], "SPAD") for d in config.EXPERIMENTAL_DATES}
    mse = float(np.mean([a["QM_residuo"] for a in anovas.values()]))
    var_plot = max(mse - sd_leaf ** 2 / 7, 0.05)
    blocks = []
    for k in (7, 14, 28):
        sigma = float(np.sqrt(var_plot + sd_leaf ** 2 / k))
        for delta in DELTAS:
            b, pw = blocks_needed(sigma, delta)
            blocks.append({"Cenario": f"Falker, {k} leituras", "Unidade": "SPAD", "Sigma_residual": sigma,
                           "Diferenca": f"{delta:g} SPAD", "Blocos_necessarios": b, "Poder": pw})
    # sem unidade: diferença de dose detectável. Efeito = inclinação da resposta × diferença de dose; ruído = resíduo.
    # Datas com todas as parcelas fotografadas, para comparar o mesmo conjunto.
    sens = []
    for m in ["SPAD", PHOTO_BEST]:
        sl, sg = [], []
        for d in ["21-05-2026", "26-05-2026"]:
            sub = df[(df.Data == d) & df.Tem_foto]
            sl.append(abs(np.polyfit(sub.Dose, sub[m], 1)[0]))
            sg.append(np.sqrt(dbc_anova(sub, m)["QM_residuo"]))
        slope, sigma = float(np.mean(sl)), float(np.mean(sg))
        sens.append({"Medida": m, "Inclinacao_por_kg": slope, "Sigma_residual": sigma, "Razao_sinal_ruido_100kg": slope * 100 / sigma})
        label = "Falker, 7 leituras" if m == "SPAD" else f"Foto ({PHOTO_BEST})"
        for dd in (25, 50):
            b, pw = blocks_needed(sigma, slope * dd)
            blocks.append({"Cenario": label, "Unidade": "dose", "Sigma_residual": sigma,
                           "Diferenca": f"{dd} kg N/ha", "Blocos_necessarios": b, "Poder": pw})

    # 3. datas de treino
    s = orientation(df, PHOTO_BEST)
    ph = df[df.Tem_foto].copy()
    x = s * ph[PHOTO_BEST].to_numpy()
    y = ph.SPAD.to_numpy()
    dates = ph.Data.to_numpy()
    date_rows = []
    for n_train in (1, 2):
        r2s, accs, maes = [], [], []
        for train in itertools.combinations(config.EXPERIMENTAL_DATES, n_train):
            for test in [d for d in config.EXPERIMENTAL_DATES if d not in train]:
                tr, te = np.isin(dates, train), dates == test
                b, a = np.polyfit(x[tr], y[tr], 1)
                p = a + b * x[te]
                r2s.append(1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
                maes.append(float(np.mean(np.abs(y[te] - p))))
                accs.append(float(np.mean(classify(p, limits) == classify(y[te], limits))))
        date_rows.append({"Datas_no_treino": n_train, "Combinacoes": len(r2s), "R2_na_data_nova_medio": float(np.mean(r2s)),
                          "Erro_medio_SPAD": float(np.mean(maes)), "Acerto_classe_medio": float(np.mean(accs))})

    db.save_df(pd.DataFrame(reads).assign(DP_entre_folhas=sd_leaf), "sci_plan_readings", description="Leituras do Falker por parcela para cada margem de erro")
    db.save_df(pd.DataFrame(ceiling), "sci_plan_ceiling", description="Teto da classificação e incerteza em função do nº de leituras")
    db.save_df(pd.DataFrame(blocks).assign(Variancia_entre_parcelas=var_plot, Variancia_folhas=sd_leaf ** 2),
               "sci_plan_blocks", description="Blocos necessários para detectar diferenças entre doses (poder 80 %)")
    db.save_df(pd.DataFrame(sens), "sci_plan_sensitivity", description="Sensibilidade à dose: inclinação, ruído residual e razão sinal/ruído")
    db.save_df(pd.DataFrame(date_rows), "sci_plan_dates", description="Generalização para uma data nova com 1 ou 2 datas no treino")
    print(pd.DataFrame(reads).to_string(index=False))
    print(pd.DataFrame(ceiling).round(3).to_string(index=False))
    print(f"σ folha {sd_leaf:.2f}  QM resíduo {mse:.2f}  var parcela {var_plot:.2f}")
    print(pd.DataFrame(sens).round(4).to_string(index=False))
    print(pd.DataFrame(blocks).round(2).to_string(index=False))
    print(pd.DataFrame(date_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
