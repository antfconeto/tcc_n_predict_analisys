"""
Nitrogênio, biomassa e crescimento (26/05, a única data com corte; 12 parcelas em blocos).

1. Partição e teor de matéria seca: fração de folha na massa (verde e seca) e teor de matéria seca (seca ÷ verde) de
   folha, colmo e da amostra inteira, por dose (ANOVA em blocos e tendência linear com a dose).
2. A clorofila satura antes da produção? Para o SPAD e para a massa seca total e de folha: fração do ganho de 0 a
   100 kg alcançada com 50 kg (e com 75 kg), e o termo quadrático (curvatura) da resposta. IC por bootstrap.
3. Eficiência de uso do N: kg de massa seca ganhos por kg de N em relação à parcela sem N do mesmo bloco, e o retorno
   marginal entre doses consecutivas (médias).
4. Uniformidade do crescimento: CV das 5 alturas de cada parcela × dose, e × a variação de cor da foto (CV de b/r).

Tabelas: ext_partition, ext_saturation, ext_nue, ext_height_uniformity, ext_biomass_plots.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.data import load_vegetation_plots
from tcc_analysis.science import dbc_anova, plot_table

warnings.filterwarnings("ignore")
DATE = "26-05-2026"
N_BOOT = 4000


def trend(df, y):
    r = stats.spearmanr(df.Dose, df[y])
    b = np.polyfit(df.Dose, df[y], 1)[0]
    return {"Inclinacao_por_100kg": 100 * b, "rho_dose": r.correlation, "p_rho_dose": r.pvalue}


def main():
    rng = np.random.default_rng(42)
    v = load_vegetation_plots(verbose=False)
    v["Ponto"] = v.Ponto.astype(int)
    pt = plot_table()
    pt = pt[pt.Data == DATE][["Ponto", "SPAD", "CV_b_over_r", "Median_b_over_r"]]
    v = v.merge(pt, on="Ponto", how="left")
    v["Fracao_folha_verde"] = v.Massa_Verde_Folha / (v.Massa_Verde_Folha + v.Massa_Verde_Colmo)
    v["Fracao_folha_seca"] = v.Massa_Seca_Folha / (v.Massa_Seca_Folha + v.Massa_Seca_Colmo)
    v["MS_folha_pct"] = 100 * v.Massa_Seca_Folha / v.Massa_Verde_Folha
    v["MS_colmo_pct"] = 100 * v.Massa_Seca_Colmo / v.Massa_Verde_Colmo
    v["MS_amostra_pct"] = 100 * v.Massa_Seca_Amostra / v.Massa_Verde_Amostra
    hs = v["Alturas Anotadas (cm)"].astype(str).str.replace(";", ",").str.split(",")
    v["Alturas"] = hs.map(lambda xs: [float(x) for x in xs if x.strip()])
    v["Altura_DP"] = v.Alturas.map(lambda a: float(np.std(a, ddof=1)))
    v["Altura_CV"] = 100 * v.Altura_DP / v.Altura_Media
    v["Altura_amplitude"] = v.Alturas.map(lambda a: max(a) - min(a))

    # 1. partição e teor de matéria seca
    part = []
    labels = {"Fracao_folha_verde": "Fração de folha (massa verde)", "Fracao_folha_seca": "Fração de folha (massa seca)",
              "MS_folha_pct": "Matéria seca da folha (%)", "MS_colmo_pct": "Matéria seca do colmo (%)",
              "MS_amostra_pct": "Matéria seca da amostra (%)"}
    for col, lab in labels.items():
        means = v.groupby("Dose")[col].mean()
        part.append({"Variavel": col, "Rotulo": lab, **{f"Media_{int(d)}kg": m for d, m in means.items()},
                     **{k: x for k, x in dbc_anova(v, col).items() if k in ("F_dose", "p_dose", "eta2_dose", "F_bloco", "p_bloco")},
                     **trend(v, col)})

    # 2. saturação: fração do ganho 0→100 kg atingida com 50 e 75 kg
    sat = []
    groups = {d: v.index[v.Dose == d].to_numpy() for d in sorted(v.Dose.unique())}
    for col, lab in [("SPAD", "SPAD (clorofila da folha)"), ("Massa_Seca_Total_ha", "Massa seca total"),
                     ("Massa_Seca_Folha_ha", "Massa seca de folha"), ("Altura_Media", "Altura média")]:
        def stats_of(idx_by_dose):
            m = {d: v.loc[i, col].mean() for d, i in idx_by_dose.items()}
            gain = m[100] - m[0]
            q = np.polyfit(np.concatenate([[d] * len(i) for d, i in idx_by_dose.items()]),
                           np.concatenate([v.loc[i, col].to_numpy() for i in idx_by_dose.values()]), 2)
            return (m[50] - m[0]) / gain if gain else np.nan, (m[75] - m[0]) / gain if gain else np.nan, q[0] * 1e4 / (abs(gain) or 1)
        f50, f75, curv = stats_of(groups)
        boot = np.array([stats_of({d: rng.choice(i, len(i), replace=True) for d, i in groups.items()}) for _ in range(N_BOOT)])
        x = np.concatenate([[d] * len(i) for d, i in groups.items()])
        yv = np.concatenate([v.loc[i, col].to_numpy() for i in groups.values()])
        X = np.c_[np.ones_like(x), x, x ** 2]
        beta, res, *_ = np.linalg.lstsq(X, yv, rcond=None)
        sigma2 = ((yv - X @ beta) ** 2).sum() / (len(yv) - 3)
        se_c = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[2, 2])
        sat.append({"Variavel": col, "Rotulo": lab, "Ganho_0_100": float(v[v.Dose == 100][col].mean() - v[v.Dose == 0][col].mean()),
                    "Fracao_do_ganho_50kg": f50, "IC50_min": np.nanpercentile(boot[:, 0], 2.5), "IC50_max": np.nanpercentile(boot[:, 0], 97.5),
                    "Fracao_do_ganho_75kg": f75, "IC75_min": np.nanpercentile(boot[:, 1], 2.5), "IC75_max": np.nanpercentile(boot[:, 1], 97.5),
                    "Curvatura_relativa": curv, "p_termo_quadratico": float(2 * stats.t.sf(abs(beta[2] / se_c), len(yv) - 3))})
    # diferença de saturação SPAD − massa (bootstrap pareado: as mesmas parcelas sorteadas para as duas)
    diffs = []
    for _ in range(N_BOOT):
        idx = {d: rng.choice(i, len(i), replace=True) for d, i in groups.items()}
        fr = {}
        for col in ("SPAD", "Massa_Seca_Total_ha"):
            m = {d: v.loc[i, col].mean() for d, i in idx.items()}
            fr[col] = (m[50] - m[0]) / (m[100] - m[0]) if m[100] != m[0] else np.nan
        diffs.append(fr["SPAD"] - fr["Massa_Seca_Total_ha"])
    diffs = np.array(diffs)
    sat.append({"Variavel": "diferenca", "Rotulo": "SPAD − massa seca total (fração do ganho com 50 kg)",
                "Fracao_do_ganho_50kg": float(np.nanmedian(diffs)), "IC50_min": np.nanpercentile(diffs, 2.5),
                "IC50_max": np.nanpercentile(diffs, 97.5), "Prob_SPAD_satura_antes": float(np.nanmean(diffs > 0))})

    # 3. eficiência de uso do N
    zero = v[v.Dose == 0].set_index("Bloco")
    nue = []
    for _, r in v[v.Dose > 0].iterrows():
        for col in ("Massa_Seca_Total_ha", "Massa_Seca_Folha_ha"):
            nue.append({"Ponto": int(r.Ponto), "Bloco": r.Bloco, "Dose": r.Dose, "Producao": col,
                        "Ganho_t_ha": r[col] - zero.loc[r.Bloco, col],
                        "Eficiencia_kg_MS_por_kg_N": 1000 * (r[col] - zero.loc[r.Bloco, col]) / r.Dose})
    nue = pd.DataFrame(nue)
    means = v.groupby("Dose")[["Massa_Seca_Total_ha", "Massa_Seca_Folha_ha"]].mean()
    marg = []
    doses = sorted(means.index)
    for a, b in zip(doses[:-1], doses[1:]):
        for col in means.columns:
            marg.append({"Producao": col, "De": a, "Para": b,
                         "Retorno_marginal_kg_MS_por_kg_N": 1000 * (means.loc[b, col] - means.loc[a, col]) / (b - a)})
    nue_summary = nue.groupby(["Producao", "Dose"]).Eficiencia_kg_MS_por_kg_N.agg(["mean", "std", "count"]).reset_index()
    nue_summary.columns = ["Producao", "Dose", "Eficiencia_media", "Eficiencia_dp", "n"]
    nue_out = pd.concat([nue_summary.assign(Tipo="eficiência agronômica (vs. 0 kg do mesmo bloco)"),
                         pd.DataFrame(marg).assign(Tipo="retorno marginal entre doses")], ignore_index=True)
    an_nue = dbc_anova(nue[nue.Producao == "Massa_Seca_Total_ha"].assign(Bloco=lambda d: d.Bloco), "Eficiencia_kg_MS_por_kg_N")

    # 4. uniformidade da altura
    hu = [{"Medida": "Altura_CV", "Rotulo": "CV das 5 alturas (%)", **trend(v, "Altura_CV"),
           **{k: x for k, x in dbc_anova(v, "Altura_CV").items() if k in ("F_dose", "p_dose")},
           **{f"Media_{int(d)}kg": m for d, m in v.groupby("Dose").Altura_CV.mean().items()}},
          {"Medida": "Altura_amplitude", "Rotulo": "Amplitude das 5 alturas (cm)", **trend(v, "Altura_amplitude"),
           **{k: x for k, x in dbc_anova(v, "Altura_amplitude").items() if k in ("F_dose", "p_dose")},
           **{f"Media_{int(d)}kg": m for d, m in v.groupby("Dose").Altura_amplitude.mean().items()}}]
    r = stats.spearmanr(v.Altura_CV, v.CV_b_over_r)
    hu.append({"Medida": "Altura_CV_x_foto", "Rotulo": "CV das alturas × CV de cor da foto (b/r)", "rho_dose": r.correlation, "p_rho_dose": r.pvalue})

    keep = ["Ponto", "Bloco", "Tratamento", "Dose", "SPAD", "Altura_Media", "Altura_DP", "Altura_CV", "Altura_amplitude",
            "Massa_Seca_Total_ha", "Massa_Seca_Folha_ha", "Massa_Verde_Total_ha", *labels, "CV_b_over_r"]
    db.save_df(pd.DataFrame(part), "ext_partition", description="Partição folha/colmo e teor de matéria seca por dose (26/05)")
    db.save_df(pd.DataFrame(sat), "ext_saturation", description="A clorofila satura antes da produção? Fração do ganho atingida com 50 e 75 kg")
    db.save_df(nue_out.assign(F_dose_eficiencia=an_nue.get("F_dose"), p_dose_eficiencia=an_nue.get("p_dose")),
               "ext_nue", description="Eficiência de uso do N e retorno marginal entre doses (26/05)")
    db.save_df(pd.DataFrame(hu), "ext_height_uniformity", description="Uniformidade da altura × dose e × variação de cor da foto")
    db.save_df(v[keep], "ext_biomass_plots", description="Por parcela (26/05): SPAD, alturas, massas, partição e matéria seca")
    print(pd.DataFrame(part).round(3).to_string(index=False))
    print(pd.DataFrame(sat).round(3).to_string(index=False))
    print(nue_out.round(2).to_string(index=False)); print("ANOVA eficiência:", {k: round(x, 3) for k, x in an_nue.items()})
    print(pd.DataFrame(hu).round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
