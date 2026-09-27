"""
A variação de cor dentro da parcela, combinada com os índices, prevê melhor o SPAD numa data nova?

Entradas por foto (31 fotos das datas do experimento):
  - nível: mediana, média, P75 e P90 de cada índice nos blocos de vegetação (117 variáveis);
  - variação: desvio padrão, CV e intervalo interquartil de cada índice entre os blocos (≈ 90 variáveis).
Famílias de modelos lineares (mínimos quadrados), todos avaliados deixando uma data de fora (LODO):
  A. um índice de nível;           B. uma medida de variação;
  C. nível + variação (todos os pares);   D. nível + nível (controle: uma entrada a mais sem ser variação).
Para cada família: o melhor candidato pelo LODO (otimista, a escolha olhou as três datas) e a escolha aninhada
(para cada data testada, o candidato é escolhido só com as outras duas: treina numa, avalia na outra e vice-versa).
Também o acerto das 3 classes (limites de chl_class_scheme) com as predições aninhadas.

Tabelas: color_var_summary, color_var_top, color_var_nested.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import plot_table

TOP = 30


def classify(v, limits):
    v = np.asarray(v, float)
    return np.where(v < limits[0], 0, np.where(v <= limits[1], 1, 2))


def weighted_kappa(a, b, k=3):
    a, b = np.asarray(a, int), np.asarray(b, int)
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    return 1 - (w * obs).sum() / (w * exp).sum()


def fit_predict(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    b = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ b


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    t0 = time.time()
    df = plot_table()
    df = df[df.Tem_foto].reset_index(drop=True)
    limits = db.load_df("chl_class_scheme").sort_values("Ordem").SPAD_max.dropna().tolist()
    y, dates = df.SPAD.to_numpy(float), df.Data.to_numpy()
    level = [c for c in df.columns if c.split("_")[0] in ("Median", "Mean", "P75", "P90") and not c.endswith("_rgb")
             and df[c].notna().all() and df[c].std() > 1e-9]
    var = [c for c in df.columns if c.startswith(("DP_", "CV_", "IQR_")) and df[c].notna().all() and np.isfinite(df[c]).all()
           and df[c].std() > 1e-9]
    X = {c: df[c].to_numpy(float) for c in level + var}
    families = {
        "A. um índice de nível": [(c,) for c in level],
        "B. uma medida de variação": [(c,) for c in var],
        "C. nível + variação": list(itertools.product(level, var)),
        "D. nível + nível (controle)": list(itertools.combinations(level, 2)),
    }
    print(f"{len(level)} de nível, {len(var)} de variação; candidatos: " + ", ".join(f"{k[:2]} {len(v)}" for k, v in families.items()))
    udates = config.EXPERIMENTAL_DATES
    masks = {d: dates == d for d in udates}

    def mat(cand, m):
        return np.column_stack([X[c][m] for c in cand])

    summary, top_rows, nested_rows = [], [], []
    for fam, cands in families.items():
        # LODO simples de cada candidato
        scores = np.empty(len(cands))
        for i, cand in enumerate(cands):
            p = np.empty(len(y))
            for d in udates:
                te = masks[d]
                p[te] = fit_predict(mat(cand, ~te), y[~te], mat(cand, te))
            scores[i] = r2(y, p)
        order = np.argsort(-scores)
        best = cands[order[0]]
        p_best = np.empty(len(y))
        for d in udates:
            te = masks[d]
            p_best[te] = fit_predict(mat(best, ~te), y[~te], mat(best, te))
        for rank, i in enumerate(order[:TOP], 1):
            top_rows.append({"Familia": fam, "Rank": rank, "Variaveis": " + ".join(cands[i]), "R2_LODO": scores[i]})
        # escolha aninhada
        p_nest = np.empty(len(y))
        for d in udates:
            te = masks[d]
            a, b = [x for x in udates if x != d]
            inner = np.empty(len(cands))
            for i, cand in enumerate(cands):
                pa = fit_predict(mat(cand, masks[b]), y[masks[b]], mat(cand, masks[a]))
                pb = fit_predict(mat(cand, masks[a]), y[masks[a]], mat(cand, masks[b]))
                yy, pp = np.r_[y[masks[a]], y[masks[b]]], np.r_[pa, pb]
                inner[i] = r2(yy, pp)
            ch = cands[int(np.argmax(inner))]
            p_nest[te] = fit_predict(mat(ch, ~te), y[~te], mat(ch, te))
            nested_rows.append({"Familia": fam, "Data_testada": d, "Escolhido": " + ".join(ch), "R2_na_escolha": float(inner.max()),
                                "R2_data_testada": r2(y[te], p_nest[te]), "MAE_data_testada": float(np.mean(np.abs(y[te] - p_nest[te])))})
        yc, pc = classify(y, limits), classify(p_nest, limits)
        summary.append({"Familia": fam, "Candidatos": len(cands), "Melhor_da_busca": " + ".join(best),
                        "R2_melhor_da_busca": float(scores[order[0]]), "R2_aninhado": r2(y, p_nest),
                        "MAE_aninhado": float(np.mean(np.abs(y - p_nest))),
                        "Classe_exata_aninhado": float(np.mean(yc == pc)), "Kappa_aninhado": weighted_kappa(yc, pc),
                        "Classe_exata_melhor_da_busca": float(np.mean(yc == classify(p_best, limits)))})
        print(f"  {fam:<30} busca R²={scores[order[0]]:.3f} ({' + '.join(best)})  aninhado R²={r2(y, p_nest):.3f}  "
              f"classe {np.mean(yc == pc):.0%}  [{time.time() - t0:.0f}s]")

    db.save_df(pd.DataFrame(summary), "color_var_summary", description="Variação de cor + índices: melhor da busca e escolha aninhada por família de modelos")
    db.save_df(pd.DataFrame(top_rows), "color_var_top", description="Os melhores candidatos de cada família (R² deixando uma data de fora)")
    db.save_df(pd.DataFrame(nested_rows), "color_var_nested", description="Escolha aninhada: candidato escolhido sem a data testada e o resultado nela")
    print(pd.DataFrame(nested_rows).round(3).to_string(index=False))
    print(pd.DataFrame(top_rows).groupby("Familia").head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
