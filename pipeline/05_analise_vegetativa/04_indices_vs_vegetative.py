"""
Índices da foto × altura e biomassa (26/05, 12 parcelas) e busca de MLP.

1. Correlação de cada índice × estatística (pré-processamento definitivo, blocos 10×10) e da
   cobertura vegetal com cada variável de campo: Pearson r, p, Spearman ρ, R², q (Benjamini-Hochberg
   dentro da variável) e R² de validação deixando uma parcela de fora (reta simples).
2. Referências: SPAD medido (Falker) e SPAD previsto pelo modelo definitivo (fora da amostra).
3. Busca de MLP para as variáveis principais. Com 12 parcelas, o melhor R² de uma busca grande é
   otimista, então cada configuração é avaliada de dois jeitos:
     - LOO simples: R² de validação de cada configuração (o máximo é o "melhor R² da busca");
     - LOO aninhado: para cada parcela deixada de fora, a busca inteira é refeita só nas outras 11
       (LOO interno) e a configuração vencedora prevê a parcela. Esse é o desempenho esperado.
   A seleção de índices (top-k por |r|) é sempre feita dentro do treino de cada dobra.

Tabelas: veg_index_correlations, veg_index_summary, veg_points, veg_mlp_search,
veg_model_summary, veg_model_predictions.
"""

import itertools
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_indices, load_vegetation_plots
from tcc_analysis.metrics import compute_mae, compute_r2

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # processos do joblib (lbfgs sem convergir em redes pequenas)

DATE = "26-05-2026"
STAT_PREFIXES = {"Median": "Mediana", "Mean": "Média", "P75": "P75", "P90": "P90"}
TARGETS = {
    "Altura_Media": "Altura média (cm)",
    "Massa_Verde_Folha_ha": "Massa verde de folha (t/ha)",
    "Massa_Seca_Folha_ha": "Massa seca de folha (t/ha)",
    "Massa_Verde_Colmo_ha": "Massa verde de colmo (t/ha)",
    "Massa_Seca_Colmo_ha": "Massa seca de colmo (t/ha)",
    "Massa_Verde_Total_ha": "Massa verde total (t/ha)",
    "Massa_Seca_Total_ha": "Massa seca total (t/ha)",
    "Rel_Folha_Colmo_Seco": "Relação folha/colmo (seca)",
}
MLP_TARGETS = ["Altura_Media", "Massa_Verde_Folha_ha", "Massa_Seca_Folha_ha", "Massa_Verde_Total_ha", "Massa_Seca_Total_ha"]
TOP_POINTS = 6

HIDDEN = [(1,), (2,), (3,), (5,), (8,)]
ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0]
ACTIVATIONS = ["relu", "tanh"]
# conjuntos de entrada: ("modelo", None) = 24 entradas do modelo de clorofila; ("top", k) = k índices de maior |r|
# no treino; ("top+cob", k) = idem + cobertura vegetal
FEATURE_SETS = [("modelo", None), ("top", 1), ("top", 2), ("top", 4), ("top", 8), ("top+cob", 2), ("top+cob", 4)]
N_BAG = 5


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
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        b, a = np.polyfit(x[m], y[m], 1)
        pred[i] = a + b * x[i]
    return pred


def split_name(col):
    stat, _, idx = col.partition("_")
    return idx, STAT_PREFIXES.get(stat, stat)


def fs_label(fs):
    kind, k = fs
    return "24 entradas do modelo" if kind == "modelo" else f"{k} índice{'s' if k > 1 else ''} de maior |r|" + (" + cobertura" if kind == "top+cob" else "")


def cfg_label(cfg):
    fs, hidden, alpha, act = cfg
    return f"{fs_label(fs)} · {hidden[0]} neurônio{'s' if hidden[0] > 1 else ''} {act} · α={alpha:g}"


# ─────────────────────────────────────────────────────────────────────────────
# ajuste
# ─────────────────────────────────────────────────────────────────────────────
def select_columns(fs, X, y, names, cover_idx):
    """Índices das colunas usadas pela configuração, escolhidos só com (X, y) de treino."""
    kind, k = fs
    if kind == "modelo":
        return [names.index(c) for c in config.MODEL_FEATURES]
    xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
    r = np.abs(xc.T @ yc) / np.where(denom == 0, np.inf, denom)
    r[cover_idx] = -1  # a cobertura só entra quando pedida
    cols = list(np.argsort(-r)[:k])
    if kind == "top+cob":
        cols.append(cover_idx)
    return cols


def fit_predict(cfg, Xtr, ytr, Xte, names, cover_idx, seeds=(0,)):
    fs, hidden, alpha, act = cfg
    cols = select_columns(fs, Xtr, ytr, names, cover_idx)
    sx = StandardScaler().fit(Xtr[:, cols])
    my, sy = ytr.mean(), ytr.std() or 1.0
    preds = []
    for seed in seeds:
        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation=act, solver="lbfgs", alpha=alpha,
                           max_iter=2000, random_state=seed)
        mlp.fit(sx.transform(Xtr[:, cols]), (ytr - my) / sy)
        preds.append(mlp.predict(sx.transform(Xte[:, cols])) * sy + my)
    return np.mean(preds, axis=0)


def loo_predictions(cfg, X, y, names, cover_idx, seeds=(0,)):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        pred[i] = fit_predict(cfg, X[m], y[m], X[i:i + 1], names, cover_idx, seeds)[0]
    return pred


def flat_search_task(cfg, X, y, names, cover_idx):
    pred = loo_predictions(cfg, X, y, names, cover_idx)
    return cfg, compute_r2(y, pred), compute_mae(y, pred)


def nested_task(i, configs, X, y, names, cover_idx):
    """Busca refeita sem a parcela i (LOO interno nas outras 11); a vencedora prevê a parcela i."""
    m = np.ones(len(y), bool)
    m[i] = False
    Xin, yin = X[m], y[m]
    best, best_r2 = None, -np.inf
    for cfg in configs:
        r2 = compute_r2(yin, loo_predictions(cfg, Xin, yin, names, cover_idx))
        if r2 > best_r2:
            best, best_r2 = cfg, r2
    pred = fit_predict(best, Xin, yin, X[i:i + 1], names, cover_idx, seeds=range(N_BAG))[0]
    # referências aninhadas: melhor índice único (reta) e Ridge nas 24 entradas
    xc = Xin - Xin.mean(0)
    r = np.abs(xc.T @ (yin - yin.mean())) / np.sqrt((xc ** 2).sum(0) * ((yin - yin.mean()) ** 2).sum() + 1e-300)
    r[cover_idx] = -1
    j = int(np.argmax(r))
    b, a = np.polyfit(Xin[:, j], yin, 1)
    lin = a + b * X[i, j]
    cols = [names.index(c) for c in config.MODEL_FEATURES]
    ridge_best, ridge_r2 = None, -np.inf
    for al in [0.1, 1, 10, 100]:
        p = np.empty(len(yin))
        for t in range(len(yin)):
            mm = np.ones(len(yin), bool)
            mm[t] = False
            sc = StandardScaler().fit(Xin[mm][:, cols])
            p[t] = Ridge(alpha=al).fit(sc.transform(Xin[mm][:, cols]), yin[mm]).predict(sc.transform(Xin[t:t + 1, cols]))[0]
        r2 = compute_r2(yin, p)
        if r2 > ridge_r2:
            ridge_best, ridge_r2 = al, r2
    sc = StandardScaler().fit(Xin[:, cols])
    ridge = Ridge(alpha=ridge_best).fit(sc.transform(Xin[:, cols]), yin).predict(sc.transform(X[i:i + 1, cols]))[0]
    return i, best, best_r2, pred, names[j], lin, ridge


# ─────────────────────────────────────────────────────────────────────────────
def main():
    veg = load_vegetation_plots(DATE, verbose=False)
    idx = load_indices(10, config.BEST_PREPROCESS_VARIANT)
    idx = idx[idx.Data == DATE].drop(columns=["Bloco", "Tratamento", "Dose"], errors="ignore")
    df = veg.merge(idx, on="Ponto").sort_values("Ponto").reset_index(drop=True)
    pred = db.load_df("definitive_model_predictions")
    pred = pred[pred.Data == DATE][["Ponto", "Predicao_MLP"]].rename(columns={"Predicao_MLP": "SPAD_previsto"})
    df = df.merge(pred, on="Ponto", how="left")
    n = len(df)
    print(f"{n} parcelas em {DATE}")

    features = [c for c in df.columns if c.split("_")[0] in STAT_PREFIXES and pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in features if not c.endswith("_rgb") and df[c].std() > 1e-9]
    predictors = features + ["Canopy_Cover"]
    refs = {"Clorofila Total": "SPAD medido (Falker)", "SPAD_previsto": "SPAD previsto pela foto (modelo)",
            "Dose": "Dose de N"}

    # 1. correlações ───────────────────────────────────────────────────────────
    corr_rows, summary, point_rows = [], [], []
    for yc, ylabel in TARGETS.items():
        y = df[yc].to_numpy(float)
        rows = []
        for col in predictors + list(refs):
            x = df[col].to_numpy(float)
            r, p = pearsonr(x, y)
            loo = loo_linear(x, y)
            is_ref = col in refs
            ind, stat = ("Cobertura vegetal", "–") if col == "Canopy_Cover" else (refs[col], "referência") if is_ref else split_name(col)
            rows.append({"Variavel_campo": yc, "Rotulo_campo": ylabel, "Variavel": col, "Indice": ind, "Estatistica": stat,
                         "Referencia": is_ref, "n": n, "r": r, "p": p, "rho": spearmanr(x, y).correlation, "R2": r * r,
                         "R2_LOO": compute_r2(y, loo), "MAE_LOO": compute_mae(y, loo)})
        c = pd.DataFrame(rows)
        img = ~c.Referencia
        c.loc[img, "q"] = bh_qvalues(c.loc[img, "p"])
        c = c.sort_values(["Referencia", "R2_LOO"], ascending=[True, False]).reset_index(drop=True)
        c.loc[~c.Referencia, "Rank"] = np.arange(1, img.sum() + 1)
        corr_rows.append(c)
        best = c[~c.Referencia].iloc[0]
        ref = c[c.Referencia].set_index("Variavel")
        cov = c[c.Variavel == "Canopy_Cover"].iloc[0]
        summary.append({"Variavel_campo": yc, "Rotulo_campo": ylabel, "n": n, "Media": y.mean(), "Min": y.min(), "Max": y.max(),
                        "Melhor_indice": best.Variavel, "Melhor_r": best.r, "Melhor_R2_LOO": best.R2_LOO, "Melhor_MAE_LOO": best.MAE_LOO,
                        "Indices_q05": int((c.q < 0.05).sum()), "Indices_testados": int(img.sum()),
                        "Cobertura_r": cov.r, "Cobertura_R2_LOO": cov.R2_LOO,
                        "SPAD_medido_r": ref.loc["Clorofila Total", "r"], "SPAD_medido_R2_LOO": ref.loc["Clorofila Total", "R2_LOO"],
                        "SPAD_previsto_r": ref.loc["SPAD_previsto", "r"], "SPAD_previsto_R2_LOO": ref.loc["SPAD_previsto", "R2_LOO"],
                        "Dose_r": ref.loc["Dose", "r"], "Dose_R2_LOO": ref.loc["Dose", "R2_LOO"]})
        for col in list(c[~c.Referencia].Variavel.head(TOP_POINTS)) + ["Clorofila Total", "SPAD_previsto", "Canopy_Cover"]:
            x = df[col].to_numpy(float)
            for i, row in df.iterrows():
                point_rows.append({"Variavel_campo": yc, "Variavel": col, "Ponto": int(row.Ponto), "Bloco": row.Bloco,
                                   "Tratamento": row.Tratamento, "Dose": row.Dose, "Y": y[i], "Valor": x[i]})
        print(f"{ylabel:<30} melhor {best.Variavel:<20} r={best.r:+.3f} R²_LOO={best.R2_LOO:.3f}  "
              f"q<0,05: {int((c.q < 0.05).sum())}/{int(img.sum())}  SPAD medido R²_LOO={ref.loc['Clorofila Total', 'R2_LOO']:.3f}")

    corr = pd.concat(corr_rows, ignore_index=True)

    # 2. busca de MLP ───────────────────────────────────────────────────────────
    X = df[predictors].to_numpy(float)
    names = list(predictors)
    cover_idx = names.index("Canopy_Cover")
    configs = list(itertools.product(FEATURE_SETS, HIDDEN, ALPHAS, ACTIVATIONS))
    print(f"\nBusca de MLP: {len(configs)} configurações × {len(MLP_TARGETS)} variáveis")

    search_rows, model_rows, pred_rows = [], [], []
    with Parallel(n_jobs=-1) as par:
        for yc in MLP_TARGETS:
            t0 = time.time()
            y = df[yc].to_numpy(float)
            flat = par(delayed(flat_search_task)(cfg, X, y, names, cover_idx) for cfg in configs)
            flat.sort(key=lambda t: -t[1])
            for rank, (cfg, r2, mae) in enumerate(flat, 1):
                fs, hidden, alpha, act = cfg
                search_rows.append({"Variavel_campo": yc, "Rank": rank, "Configuracao": cfg_label(cfg), "Entradas": fs_label(fs),
                                    "Neuronios": hidden[0], "Ativacao": act, "Alpha": alpha, "R2_LOO": r2, "MAE_LOO": mae})
            best_cfg = flat[0][0]
            flat_pred = loo_predictions(best_cfg, X, y, names, cover_idx, seeds=range(N_BAG))

            nested = par(delayed(nested_task)(i, configs, X, y, names, cover_idx) for i in range(n))
            nested.sort(key=lambda t: t[0])
            p_mlp = np.array([t[3] for t in nested])
            p_lin = np.array([t[5] for t in nested])
            p_ridge = np.array([t[6] for t in nested])
            chosen = pd.Series([cfg_label(t[1]) for t in nested]).value_counts()
            chosen_idx = pd.Series([t[4] for t in nested]).value_counts()

            spad = df["Clorofila Total"].to_numpy(float)
            spad_pred = df["SPAD_previsto"].to_numpy(float)
            approaches = [
                ("MLP: melhor da busca (LOO simples, otimista)", flat_pred, cfg_label(best_cfg), False),
                ("MLP: busca aninhada (esperado em dados novos)", p_mlp, f"configuração mais escolhida: {chosen.index[0]} ({chosen.iloc[0]} de {n})", True),
                ("Reta no melhor índice (escolhido no treino)", p_lin, f"índice mais escolhido: {chosen_idx.index[0]} ({chosen_idx.iloc[0]} de {n})", True),
                ("Ridge nas 24 entradas (α escolhido no treino)", p_ridge, "", True),
                ("Reta no SPAD medido (Falker)", loo_linear(spad, y), "precisa do clorofilômetro", True),
                ("Reta no SPAD previsto pela foto", loo_linear(spad_pred, y), "modelo de clorofila definitivo", True),
            ]
            for name, p, detail, honest in approaches:
                model_rows.append({"Variavel_campo": yc, "Rotulo_campo": TARGETS[yc], "Abordagem": name, "Detalhe": detail,
                                   "Honesto": honest, "R2": compute_r2(y, p), "MAE": compute_mae(y, p),
                                   "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))})
            for i, row in df.iterrows():
                pred_rows.append({"Variavel_campo": yc, "Ponto": int(row.Ponto), "Bloco": row.Bloco, "Tratamento": row.Tratamento,
                                  "Dose": row.Dose, "Observado": y[i], "MLP_busca": flat_pred[i], "MLP_aninhada": p_mlp[i],
                                  "Reta_melhor_indice": p_lin[i], "Ridge": p_ridge[i]})
            print(f"  {TARGETS[yc]:<30} busca R²={flat[0][1]:.3f} ({cfg_label(best_cfg)})  aninhada R²={compute_r2(y, p_mlp):.3f}  "
                  f"reta aninhada R²={compute_r2(y, p_lin):.3f}  [{time.time() - t0:.0f}s]")

    db.save_df(corr, "veg_index_correlations", description="Índices da foto × altura e biomassa (26/05): correlação e R² de validação")
    db.save_df(pd.DataFrame(summary), "veg_index_summary", description="Resumo por variável de campo: melhor índice, cobertura, SPAD e dose")
    db.save_df(pd.DataFrame(point_rows), "veg_points", description="Pontos (parcelas) dos melhores índices para cada variável de campo")
    db.save_df(pd.DataFrame(search_rows), "veg_mlp_search", description="Busca de MLP para altura e biomassa: R² LOO de cada configuração")
    db.save_df(pd.DataFrame(model_rows), "veg_model_summary", description="Altura e biomassa: MLP (busca e aninhada) e referências")
    db.save_df(pd.DataFrame(pred_rows), "veg_model_predictions", description="Predições fora da amostra de altura e biomassa por parcela")

    print("\nResumo:")
    print(pd.DataFrame(model_rows)[["Rotulo_campo", "Abordagem", "R2", "MAE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
