"""
Dinâmica entre as coletas: a clorofila cai igual em todas as doses?

  - Modelo misto de medidas repetidas no SPAD por parcela (Falker, 3 datas × 12 parcelas):
        SPAD ~ dose (fator) × data (fator) + bloco, com intercepto aleatório por parcela (statsmodels MixedLM, ML).
    A interação dose × data é testada por razão de verossimilhança contra o modelo sem interação.
  - Taxa de variação por parcela (SPAD por dia, reta de 18/05 a 26/05) e ANOVA em blocos da taxa pela dose.
  - Relação da taxa com a massa em 26/05: queda maior onde a planta cresceu mais (diluição do N na biomassa)?
  - Foto: variação 21/05 → 26/05 de cada medida (todas as parcelas fotografadas nas duas datas), padronizada.

Tabelas: sci_temporal_means, sci_temporal_tests, sci_temporal_slopes.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_vegetation_plots
from tcc_analysis.science import MEASURE_LABELS, PHOTO_MEASURES, dbc_anova, orientation, plot_table

warnings.filterwarnings("ignore")


def main():
    df = plot_table()
    day0 = pd.to_datetime(config.EXPERIMENTAL_DATES[0], format="%d-%m-%Y")
    df["Dia"] = (pd.to_datetime(df.Data, format="%d-%m-%Y") - day0).dt.days

    means = []
    for (d, dose), g in df.groupby(["Data", "Dose"]):
        means.append({"Data": d, "Dia": int(g.Dia.iloc[0]), "Dose": dose, "Medida": "SPAD", "Media": g.SPAD.mean(),
                      "EP": g.SPAD.std(ddof=1) / np.sqrt(len(g)), "n": len(g)})

    d = df[["SPAD", "Dose", "Data", "Bloco", "Ponto"]].rename(columns={"SPAD": "y"})
    full = smf.mixedlm("y ~ C(Dose) * C(Data) + C(Bloco)", d, groups=d.Ponto).fit(reml=False)
    red = smf.mixedlm("y ~ C(Dose) + C(Data) + C(Bloco)", d, groups=d.Ponto).fit(reml=False)
    nodate = smf.mixedlm("y ~ C(Dose) + C(Bloco)", d, groups=d.Ponto).fit(reml=False)
    nodose = smf.mixedlm("y ~ C(Data) + C(Bloco)", d, groups=d.Ponto).fit(reml=False)
    lr = lambda big, small: (2 * (big.llf - small.llf), int(len(big.fe_params) - len(small.fe_params)))  # noqa: E731
    tests = []
    for name, (stat, gl) in [("Interação dose × data", lr(full, red)), ("Efeito da data", lr(red, nodate)),
                             ("Efeito da dose", lr(red, nodose))]:
        tests.append({"Teste": name, "Qui2": stat, "GL": gl, "p": float(stats.chi2.sf(stat, gl))})
    tests.append({"Teste": "Variância entre parcelas (intercepto aleatório)", "Qui2": float(full.cov_re.iloc[0, 0]), "GL": None, "p": None})
    tests.append({"Teste": "Variância residual", "Qui2": float(full.scale), "GL": None, "p": None})

    # taxa por parcela e relação com a massa em 26/05
    slopes = []
    for p, g in df.groupby("Ponto"):
        b = np.polyfit(g.Dia, g.SPAD, 1)[0]
        slopes.append({"Ponto": int(p), "Bloco": g.Bloco.iloc[0], "Dose": g.Dose.iloc[0], "Taxa_SPAD_dia": b,
                       "Variacao_total": float(g.sort_values("Dia").SPAD.iloc[-1] - g.sort_values("Dia").SPAD.iloc[0])})
    slopes = pd.DataFrame(slopes)
    veg = load_vegetation_plots(verbose=False)[["Ponto", "Massa_Seca_Total_ha", "Altura_Media"]]
    veg["Ponto"] = veg.Ponto.astype(int)
    slopes = slopes.merge(veg, on="Ponto")
    an = dbc_anova(slopes, "Taxa_SPAD_dia")
    tests.append({"Teste": "Taxa de variação (SPAD/dia) × dose (ANOVA em blocos)", "Qui2": an.get("F_dose"), "GL": None, "p": an.get("p_dose")})
    for col, label in [("Massa_Seca_Total_ha", "massa seca total"), ("Altura_Media", "altura")]:
        r = stats.spearmanr(slopes.Taxa_SPAD_dia, slopes[col])
        tests.append({"Teste": f"Taxa × {label} em 26/05 (Spearman)", "Qui2": float(r.correlation), "GL": None, "p": float(r.pvalue)})

    # foto: 21/05 → 26/05
    both = df[df.Data.isin(["21-05-2026", "26-05-2026"]) & df.Tem_foto]
    for m in PHOTO_MEASURES:
        s = orientation(df, m)
        wide = both.pivot(index="Ponto", columns="Data", values=m) * s
        sd = both[m].std(ddof=1)
        ch = ((wide["26-05-2026"] - wide["21-05-2026"]) / sd).rename("Variacao_z").reset_index()
        ch["Dose"] = ch.Ponto.map(lambda p: config.PONTO_TO_DBC[p]["Dose"])
        ch["Bloco"] = ch.Ponto.map(lambda p: config.PONTO_TO_DBC[p]["Bloco"])
        an = dbc_anova(ch, "Variacao_z")
        tests.append({"Teste": f"{MEASURE_LABELS[m]}: variação 21/05 → 26/05 × dose", "Qui2": an.get("F_dose"), "GL": None, "p": an.get("p_dose")})
        for (dd, dose), g in both.groupby(["Data", "Dose"]):
            means.append({"Data": dd, "Dia": int(g.Dia.iloc[0]), "Dose": dose, "Medida": m, "Media": float((s * g[m]).mean()),
                          "EP": float(g[m].std(ddof=1) / np.sqrt(len(g))), "n": len(g)})
    ch_spad = both.pivot(index="Ponto", columns="Data", values="SPAD")
    ch_spad = (ch_spad["26-05-2026"] - ch_spad["21-05-2026"]).rename("v").reset_index()
    ch_spad["Dose"] = ch_spad.Ponto.map(lambda p: config.PONTO_TO_DBC[p]["Dose"])
    ch_spad["Bloco"] = ch_spad.Ponto.map(lambda p: config.PONTO_TO_DBC[p]["Bloco"])
    an = dbc_anova(ch_spad, "v")
    tests.append({"Teste": "SPAD (Falker): variação 21/05 → 26/05 × dose", "Qui2": an.get("F_dose"), "GL": None, "p": an.get("p_dose")})

    tests = pd.DataFrame(tests).rename(columns={"Qui2": "Estatistica"})
    db.save_df(pd.DataFrame(means), "sci_temporal_means", description="SPAD e medidas da foto por data e dose")
    db.save_df(tests, "sci_temporal_tests", description="Dose × data: modelo misto, taxa de variação e relação com a massa")
    db.save_df(slopes, "sci_temporal_slopes", description="Taxa de variação do SPAD por parcela (18/05 → 26/05)")
    print(tests.round(4).to_string(index=False))
    print(slopes.groupby("Dose")[["Taxa_SPAD_dia", "Variacao_total"]].mean().round(3))


if __name__ == "__main__":
    db.run_main(main)
