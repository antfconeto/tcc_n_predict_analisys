"""
Heterogeneidade dentro da parcela: parcelas sem N são mais desuniformes? A foto vê a mesma variação que o Falker?

  - Foto: dispersão entre os blocos 10×10 de vegetação de cada foto (desvio padrão e CV de b/r, IPCA e g).
  - Falker: desvio padrão e CV das 7 leituras da parcela.
Por data: ANOVA em blocos da dose sobre cada medida de heterogeneidade e correlação de Spearman com a dose; e a
correlação, entre parcelas, da heterogeneidade da foto com a do Falker (todas as datas com foto juntas e por data).

Tabelas: sci_heterogeneity_plots, sci_heterogeneity_tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import dbc_anova, plot_table

MEASURES = {"SPAD_dp": "Falker: desvio das 7 leituras", "SPAD_cv": "Falker: CV das 7 leituras (%)",
            "DP_b_over_r": "Foto: desvio de b/r entre blocos", "CV_b_over_r": "Foto: CV de b/r entre blocos (%)",
            "DP_IPCA": "Foto: desvio de IPCA entre blocos", "DP_g": "Foto: desvio de g entre blocos",
            "CV_g": "Foto: CV de g entre blocos (%)"}


def main():
    df = plot_table()
    df["SPAD_cv"] = 100 * df.SPAD_dp / df.SPAD
    keep = ["Data", "Ponto", "Bloco", "Dose", "SPAD", "Cobertura", "Tem_foto"] + list(MEASURES)
    plots = df[keep].copy()

    tests = []
    for m, label in MEASURES.items():
        for d in config.EXPERIMENTAL_DATES + ["Todas"]:
            sub = plots if d == "Todas" else plots[plots.Data == d]
            sub = sub[sub[m].notna()]
            if len(sub) < 6:
                continue
            rho = spearmanr(sub.Dose, sub[m])
            rec = {"Medida": m, "Rotulo": label, "Data": d, "n": len(sub),
                   "rho_dose": rho.correlation, "p_rho_dose": rho.pvalue,
                   "Media_0kg": sub[sub.Dose == 0][m].mean(), "Media_100kg": sub[sub.Dose == 100][m].mean()}
            if d != "Todas":
                rec.update({k: v for k, v in dbc_anova(sub, m).items() if k in ("F_dose", "p_dose", "eta2_dose")})
            if m.startswith(("DP_", "CV_")):
                both = sub[sub.SPAD_dp.notna()]
                r = spearmanr(both[m], both.SPAD_dp)
                rec.update({"rho_com_Falker_dp": r.correlation, "p_rho_com_Falker_dp": r.pvalue})
                rc = spearmanr(both[m], both.Cobertura)
                rec.update({"rho_com_cobertura": rc.correlation})
            tests.append(rec)
    tests = pd.DataFrame(tests)
    db.save_df(plots, "sci_heterogeneity_plots", description="Heterogeneidade por parcela e data: Falker (7 leituras) e foto (blocos)")
    db.save_df(tests, "sci_heterogeneity_tests", description="Heterogeneidade × dose e heterogeneidade da foto × do Falker")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
