"""
Concordância foto × Falker pelo método de Bland e Altman (1986).

Para cada parcela × data com foto: média das duas medidas (eixo x) e diferença foto − Falker (eixo y).
  - viés: diferença média, com IC 95 % (t);
  - limites de concordância (LoA): viés ± 1,96 × DP das diferenças, com IC 95 % aproximado (±t·√(3s²/n));
  - viés proporcional: inclinação da diferença em função da média;
  - viés por data e LoA depois de tirar o viés de cada data (o que sobraria com uma referência fotografada no dia);
  - referência: LoA esperada entre duas medições do próprio Falker na mesma parcela (7 leituras cada).
Métodos (SPAD previsto sempre fora da amostra, com a data testada fora do treino): rede neural definitiva, reta no
melhor índice, reta no índice escolhido sem a data testada e a rede pequena da busca aninhada (tabela chl_class_plots).

Tabelas: sci_bland_altman, sci_bland_altman_points.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db

METHODS = [("Rede neural definitiva", "SPAD_rede"), ("Reta no melhor índice", "SPAD_melhor_indice"),
           ("Reta no índice escolhido sem a data", "SPAD_indice_aninhado"), ("Rede pequena (busca aninhada)", "SPAD_mlp")]


def ba(diff, mean):
    n = len(diff)
    bias, sd = diff.mean(), diff.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    se_bias, se_loa = sd / np.sqrt(n), np.sqrt(3 * sd ** 2 / n)
    slope, _, _, p_slope, _ = stats.linregress(mean, diff)
    return {"n": n, "Vies": bias, "Vies_IC_min": bias - t * se_bias, "Vies_IC_max": bias + t * se_bias, "DP_diferencas": sd,
            "LoA_inf": bias - 1.96 * sd, "LoA_sup": bias + 1.96 * sd,
            "LoA_inf_IC_min": bias - 1.96 * sd - t * se_loa, "LoA_sup_IC_max": bias + 1.96 * sd + t * se_loa,
            "Vies_proporcional_inclinacao": slope, "Vies_proporcional_p": p_slope}


def main():
    p = db.load_df("chl_class_plots")
    p["Ponto"] = p.Ponto.astype(int)
    rows, pts = [], []
    se = p.SPAD_dp / np.sqrt(p.n_leituras)
    falker_sd = float(np.sqrt(np.mean(2 * se ** 2)))
    rows.append({"Metodo": "Falker × Falker (duas medições de 7 leituras)", "n": len(p), "Vies": 0.0, "DP_diferencas": falker_sd,
                 "LoA_inf": -1.96 * falker_sd, "LoA_sup": 1.96 * falker_sd})
    for name, col in METHODS:
        sub = p[p[col].notna()].copy()
        if sub.empty:
            continue
        diff = (sub[col] - sub.SPAD).to_numpy()
        mean = ((sub[col] + sub.SPAD) / 2).to_numpy()
        rec = {"Metodo": name, **ba(diff, mean)}
        for d in config.EXPERIMENTAL_DATES:
            m = (sub.Data == d).to_numpy()
            rec[f"Vies_{d[:5]}"] = float(diff[m].mean())
        # sem o viés de cada data: o que resta de desacordo dentro do dia
        resid = diff - pd.Series(diff).groupby(sub.Data.to_numpy()).transform("mean").to_numpy()
        sd_w = float(np.sqrt((resid ** 2).sum() / (len(resid) - sub.Data.nunique())))
        rec.update({"DP_dentro_da_data": sd_w, "LoA_dentro_da_data": 1.96 * sd_w})
        rows.append(rec)
        for (_, r), dv, mv in zip(sub.iterrows(), diff, mean):
            pts.append({"Metodo": name, "Data": r.Data, "Ponto": int(r.Ponto), "Dose": r.Dose, "Falker": r.SPAD,
                        "Foto": r[col], "Media": mv, "Diferenca": dv})
    out = pd.DataFrame(rows)
    db.save_df(out, "sci_bland_altman", description="Bland-Altman foto × Falker: viés, limites de concordância e viés por data")
    db.save_df(pd.DataFrame(pts), "sci_bland_altman_points", description="Pontos do gráfico de Bland-Altman (média × diferença)")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
