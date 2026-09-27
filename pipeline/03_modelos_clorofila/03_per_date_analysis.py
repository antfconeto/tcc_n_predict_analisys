"""
Análise por coleta: cada data é analisada sozinha, sem comparar datas entre si.

Para cada data (18/05, 21/05, 26/05 e, com ressalvas, 01/06):
  1. Correlação de cada índice × estatística (pré-processamento definitivo, blocos 10×10) com o SPAD
     médio da parcela: Pearson r, p, Spearman ρ, R² e q (Benjamini-Hochberg dentro da data).
  2. Validação dentro da data deixando uma parcela de fora (regressão linear simples): R²_LOO.
  3. ANOVA em blocos (índice ~ Bloco + Tratamento) dos 5 melhores índices e do SPAD: a imagem
     distingue as doses naquela data?
  4. Desempenho do modelo definitivo dentro da data (predições fora da amostra já gravadas).
  5. Consistência: correlação de cada índice nas três datas do experimento.

Tabelas: per_date_summary, per_date_correlations, per_date_anova, per_date_points,
per_date_consistency.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import build_merged_dataset, load_chlorophyll_with_layout, load_indices
from tcc_analysis.metrics import compute_mae, compute_r2

warnings.filterwarnings("ignore")

STAT_PREFIXES = {"Median": "Mediana", "Mean": "Média", "P75": "P75", "P90": "P90"}
MIN_N = 6            # abaixo disso a data não tem pontos suficientes para inferência
TOP_ANOVA = 5
TOP_POINTS = 8


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def loo_linear(x, y):
    """Predição de cada parcela por uma reta ajustada nas demais parcelas da mesma data."""
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        if np.std(x[m]) == 0:
            pred[i] = y[m].mean()
            continue
        b, a = np.polyfit(x[m], y[m], 1)
        pred[i] = a + b * x[i]
    return pred


def split_name(col):
    stat, _, idx = col.partition("_")
    return idx, STAT_PREFIXES.get(stat, stat)


def main():
    variant = config.BEST_PREPROCESS_VARIANT
    df = build_merged_dataset(load_chlorophyll_with_layout(), load_indices(10, variant))
    df = df.rename(columns={"Bloco_x": "Bloco", "Tratamento_x": "Tratamento"})
    dose = {int(p): v["Dose"] for p, v in config.PONTO_TO_DBC.items()}
    df["Dose"] = [None if d == config.TEST_DATE else dose.get(int(p)) for d, p in zip(df.Data, df.Ponto)]
    counts = db.query("SELECT data AS Data, COUNT(*) AS n FROM falker_readings GROUP BY data").set_index("Data").n

    features = [c for c in df.columns if c.split("_")[0] in STAT_PREFIXES and pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in features if not c.endswith("_rgb") and df[c].std() > 1e-9]

    pred = db.load_df("definitive_model_predictions") if db.has_rows("definitive_model_predictions") else None

    summary, corr_rows, anova_rows, point_rows = [], [], [], []
    dates = config.EXPERIMENTAL_DATES + [config.TEST_DATE]
    for d in dates:
        sub = df[df.Data == d].reset_index(drop=True)
        y = sub["Clorofila Total"].to_numpy()
        n = len(sub)
        if n < 3:
            continue
        rows = []
        for col in features:
            x = sub[col].to_numpy(dtype=float)
            if np.std(x) < 1e-12:
                continue
            r, p = pearsonr(x, y)
            rho = spearmanr(x, y).correlation
            loo = loo_linear(x, y) if n >= MIN_N else np.full(n, np.nan)
            b, a = np.polyfit(x, y, 1)
            idx, stat = split_name(col)
            rows.append({"Data": d, "Variavel": col, "Indice": idx, "Estatistica": stat, "n": n,
                         "r": r, "p": p, "rho": rho, "R2": r * r,
                         "R2_LOO": compute_r2(y, loo) if n >= MIN_N else np.nan,
                         "MAE_LOO": compute_mae(y, loo) if n >= MIN_N else np.nan,
                         "Inclinacao": b, "Intercepto": a})
        c = pd.DataFrame(rows)
        c["q"] = bh_qvalues(c.p)
        c["Confiavel"] = n >= MIN_N
        c = c.sort_values("R2_LOO" if n >= MIN_N else "R2", ascending=False).reset_index(drop=True)
        c["Rank"] = np.arange(1, len(c) + 1)
        corr_rows.append(c)
        best = c.iloc[0]

        # pontos das melhores variáveis (para os gráficos do site)
        for col in c.Variavel.head(TOP_POINTS):
            x = sub[col].to_numpy(dtype=float)
            loo = loo_linear(x, y) if n >= MIN_N else np.full(n, np.nan)
            for i, row in sub.iterrows():
                point_rows.append({"Data": d, "Variavel": col, "Ponto": int(row.Ponto), "Bloco": row.Bloco,
                                   "Tratamento": row.Tratamento, "Dose": row.Dose, "SPAD": y[i], "Valor": x[i],
                                   "Previsto_LOO": loo[i]})

        # ANOVA em blocos: SPAD e os melhores índices distinguem as doses?
        if d != config.TEST_DATE and n >= MIN_N and sub.Bloco.notna().all():
            for col in ["Clorofila Total"] + list(c.Variavel.head(TOP_ANOVA)):
                data = pd.DataFrame({"v": sub[col].astype(float), "Bloco": sub.Bloco, "Tratamento": sub.Tratamento})
                if data.Bloco.nunique() < 2 or data.Tratamento.nunique() < 2:
                    continue
                try:
                    tab = sm.stats.anova_lm(ols("v ~ C(Bloco) + C(Tratamento)", data=data).fit(), typ=2)
                except Exception:
                    continue
                anova_rows.append({
                    "Data": d, "Variavel": "SPAD (Falker, média da parcela)" if col == "Clorofila Total" else col,
                    "F_tratamento": tab.loc["C(Tratamento)", "F"], "p_tratamento": tab.loc["C(Tratamento)", "PR(>F)"],
                    "F_bloco": tab.loc["C(Bloco)", "F"], "p_bloco": tab.loc["C(Bloco)", "PR(>F)"],
                    "GL_residuo": tab.loc["Residual", "df"],
                    "Tratamentos_com_foto": data.Tratamento.nunique(), "Parcelas": len(data),
                })

        rec = {"Data": d, "n_fotos": n, "n_leituras": int(counts.get(d, 0)),
               "SPAD_media": y.mean(), "SPAD_min": y.min(), "SPAD_max": y.max(), "SPAD_dp": y.std(ddof=1),
               "Confiavel": n >= MIN_N, "Melhor_variavel": best.Variavel, "Melhor_r": best.r, "Melhor_R2": best.R2,
               "Melhor_R2_LOO": best.R2_LOO, "Melhor_MAE_LOO": best.MAE_LOO,
               "Variaveis_q05": int((c.q < 0.05).sum()), "Variaveis_testadas": len(c)}
        rb = c[c.Variavel == "Median_r_over_b"]
        if len(rb):
            rec.update({"rb_r": rb.r.iloc[0], "rb_R2_LOO": rb.R2_LOO.iloc[0]})
        if pred is not None:
            pp = pred[pred.Data == d]
            if len(pp) >= 3:
                rec.update({
                    "Modelo_R2_na_data": compute_r2(pp["Clorofila Total"], pp.Predicao_MLP),
                    "Modelo_r_na_data": np.corrcoef(pp["Clorofila Total"], pp.Predicao_MLP)[0, 1],
                    "Modelo_spearman": spearmanr(pp["Clorofila Total"], pp.Predicao_MLP).correlation,
                    "Modelo_MAE": compute_mae(pp["Clorofila Total"], pp.Predicao_MLP),
                    "Modelo_vies": float((pp.Predicao_MLP - pp["Clorofila Total"]).mean()),
                })
        summary.append(rec)
        print(f"{d}: n={n:2d}  melhor {best.Variavel:<22} r={best.r:+.3f}  R²={best.R2:.3f}  "
              f"R²_LOO={best.R2_LOO if n >= MIN_N else float('nan'):.3f}  q<0,05: {rec['Variaveis_q05']}/{len(c)}")

    corr = pd.concat(corr_rows, ignore_index=True)
    # consistência nas três datas do experimento
    exp = corr[corr.Data.isin(config.EXPERIMENTAL_DATES)]
    piv_r = exp.pivot(index="Variavel", columns="Data", values="r")
    piv_loo = exp.pivot(index="Variavel", columns="Data", values="R2_LOO")
    cons = pd.DataFrame({
        "Variavel": piv_r.index,
        **{f"r_{d[:5]}": piv_r[d].values for d in config.EXPERIMENTAL_DATES if d in piv_r},
        "Media_abs_r": piv_r.abs().mean(axis=1).values,
        "Min_abs_r": piv_r.abs().min(axis=1).values,
        "Mesmo_sinal": (np.sign(piv_r).nunique(axis=1) == 1).values,
        "R2_LOO_medio": piv_loo.mean(axis=1).values,
        "R2_LOO_min": piv_loo.min(axis=1).values,
    })
    cons[["Indice", "Estatistica"]] = [split_name(v) for v in cons.Variavel]
    cons = cons.sort_values("R2_LOO_min", ascending=False).reset_index(drop=True)

    db.save_df(pd.DataFrame(summary), "per_date_summary", description="Resumo por coleta: SPAD, melhor índice e modelo dentro da data")
    db.save_df(corr, "per_date_correlations", description="Correlação de cada índice com o SPAD dentro de cada coleta")
    db.save_df(pd.DataFrame(anova_rows), "per_date_anova", description="ANOVA em blocos dentro de cada coleta (SPAD e melhores índices)")
    db.save_df(pd.DataFrame(point_rows), "per_date_points", description="Pontos (parcelas) das melhores variáveis de cada coleta")
    db.save_df(cons, "per_date_consistency", description="Consistência de cada índice nas três coletas do experimento")

    print("\nMais consistentes nas três datas (maior R²_LOO mínimo):")
    print(cons.head(10)[["Variavel", "r_18-05", "r_21-05", "r_26-05", "R2_LOO_min", "R2_LOO_medio"]].round(3).to_string(index=False))
    print("\nANOVA dentro da data:")
    print(pd.DataFrame(anova_rows)[["Data", "Variavel", "F_tratamento", "p_tratamento"]].round(4).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
