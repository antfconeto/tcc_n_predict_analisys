"""
Clorofila a × b: a dose muda a proporção entre elas? A foto acompanha mais a clorofila a ou a b?

O Falker separa o índice de clorofila a e b (estimativas ópticas do aparelho, não extração em laboratório).
  - Por data: ANOVA em blocos da dose para a, b, total e a razão a/b (médias por parcela).
  - Médias por dose de a, b e a/b.
  - Correlação de cada índice da foto (117) com a e com b dentro de cada data; resumo de quantos acompanham mais
    a b do que a a, e a correlação entre a e b nas parcelas (se forem quase a mesma informação, separar não adianta).

Tabelas: sci_ab_anova, sci_ab_means, sci_ab_index_corr, sci_ab_summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import dbc_anova, plot_table


def main():
    df = plot_table()
    df["Razao_ab"] = df.Clorofila_A / df.Clorofila_B
    vars_ = {"Clorofila_A": "Clorofila a", "Clorofila_B": "Clorofila b", "SPAD": "Total", "Razao_ab": "Razão a/b"}

    anova, means = [], []
    for d in config.EXPERIMENTAL_DATES:
        sub = df[df.Data == d]
        for v, label in vars_.items():
            anova.append({"Data": d, "Variavel": label, **dbc_anova(sub, v)})
            for dose, g in sub.groupby("Dose"):
                means.append({"Data": d, "Variavel": label, "Dose": dose, "Media": g[v].mean(),
                              "EP": g[v].std(ddof=1) / np.sqrt(len(g))})

    feats = [c for c in df.columns if c.split("_")[0] in ("Median", "Mean", "P75", "P90") and df[c].notna().any()]
    corr = []
    for d in config.EXPERIMENTAL_DATES:
        sub = df[(df.Data == d) & df.Tem_foto]
        for f in feats:
            x = sub[f].to_numpy(float)
            if np.std(x) < 1e-12:
                continue
            ra = np.corrcoef(x, sub.Clorofila_A)[0, 1]
            rb = np.corrcoef(x, sub.Clorofila_B)[0, 1]
            corr.append({"Data": d, "Variavel": f, "n": len(sub), "r_a": ra, "r_b": rb, "Dif_abs": abs(rb) - abs(ra)})
    corr = pd.DataFrame(corr)
    summary = []
    for d in config.EXPERIMENTAL_DATES + ["Todas"]:
        c = corr if d == "Todas" else corr[corr.Data == d]
        sub = df if d == "Todas" else df[df.Data == d]
        top_a = c.loc[c.r_a.abs().idxmax()]
        top_b = c.loc[c.r_b.abs().idxmax()]
        summary.append({"Data": d, "r_a_b_parcelas": float(np.corrcoef(sub.Clorofila_A, sub.Clorofila_B)[0, 1]),
                        "Mediana_abs_r_a": float(c.r_a.abs().median()), "Mediana_abs_r_b": float(c.r_b.abs().median()),
                        "Prop_indices_mais_b": float((c.Dif_abs > 0).mean()),
                        "Melhor_indice_a": top_a.Variavel, "Melhor_r_a": top_a.r_a,
                        "Melhor_indice_b": top_b.Variavel, "Melhor_r_b": top_b.r_b,
                        "Razao_ab_media": float(sub.Razao_ab.mean())})

    db.save_df(pd.DataFrame(anova), "sci_ab_anova", description="ANOVA em blocos da dose para clorofila a, b, total e a razão a/b")
    db.save_df(pd.DataFrame(means), "sci_ab_means", description="Clorofila a, b e razão a/b por dose e data")
    db.save_df(corr, "sci_ab_index_corr", description="Correlação de cada índice da foto com a clorofila a e com a b, por data")
    db.save_df(pd.DataFrame(summary), "sci_ab_summary", description="Resumo: a foto acompanha mais a clorofila a ou a b?")
    print(pd.DataFrame(anova)[["Data", "Variavel", "F_dose", "p_dose", "eta2_dose"]].round(3).to_string(index=False))
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
