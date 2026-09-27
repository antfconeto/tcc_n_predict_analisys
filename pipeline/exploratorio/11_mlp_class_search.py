"""
Busca de uma MLP que classifique melhor a clorofila (classes de Beaufils sobre o SPAD, lidas de chl_class_scheme).

A rede prevê o SPAD e a classe sai pelos mesmos limites da página Classes (tabela chl_class_scheme). O alvo de
cada foto é o SPAD médio Falker da parcela na data.

Configurações testadas:
  - dados: foto inteira (1 exemplo por foto) ou recortes 3×3 (até 9 exemplos por foto, mesmo SPAD; a previsão da
    foto é a média dos recortes). Os recortes só aumentam o treino; a avaliação é sempre por foto;
  - entradas: 24 do modelo definitivo; 4 ou 8 índices de maior |r| com o SPAD no treino; 5 ou 10 componentes
    principais dos 117 índices (ajustados no treino);
  - rede: 2, 4, 8 ou 8+4 neurônios, ReLU ou tanh, α = 0,1, 1 ou 10; média de 3 sementes.

Validação (nunca usa a data testada para escolher nada):
  - LODO simples: cada configuração é treinada em duas datas e testada na terceira; a melhor configuração nesse
    critério dá o "melhor da busca" (otimista, porque a escolha olhou as três datas);
  - LODO aninhado: para cada data testada, a configuração é escolhida só com as outras duas (treina numa, avalia
    na outra, e vice-versa); a vencedora é treinada nas duas e prevê a data testada.
As fotos de 01/06 (pontos de teste, fora do delineamento) entram só no treino, como no modelo definitivo.

Tabelas: mlp_class_search, mlp_class_nested, mlp_class_summary, mlp_class_predictions.
"""

import itertools
import os
import re
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all
from tcc_analysis.indices import INDEX_STATISTICS, calculate_all_indices
from tcc_analysis.preprocess import build_vegetation_mask, merge_preprocess_config, preprocess_image

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

TILES = 3
MIN_VEG_BLOCKS = 30
N_SEEDS = 3

DATA_MODES = ["foto", "recortes"]
INPUTS = [("modelo", None), ("top", 4), ("top", 8), ("pca", 5), ("pca", 10)]
HIDDEN = [(2,), (4,), (8,), (8, 4)]
ACTIVATIONS = ["relu", "tanh"]
ALPHAS = [0.1, 1.0, 10.0]


# ─────────────────────────────────────────────────────────────────────────────
# extração: foto inteira e recortes 3×3, com o pré-processamento definitivo
# ─────────────────────────────────────────────────────────────────────────────
def stats_row(indices, mask):
    row = {}
    for key, arr in indices.items():
        vals = arr[mask] if mask.any() else np.array([0.0])
        for stat_name, fn in INDEX_STATISTICS.items():
            row[f"{stat_name}_{key}"] = fn(vals)
    return row


def extract(block_size=10):
    pp = merge_preprocess_config(config.BEST_PREPROCESS_CONFIG)
    root = str(config.IMAGE_ROOT)
    rows = []
    for folder in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        date = f"{folder}-{config.YEAR}"
        for fn in sorted(os.listdir(os.path.join(root, folder))):
            m = re.match(r"^[Pp](\d+)", fn)
            if not m or not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(root, folder, fn))
            if img is None:
                continue
            f, _ = preprocess_image(img, pp)
            h, w, _ = f.shape
            nh, nw = h // block_size, w // block_size
            with np.errstate(invalid="ignore"):
                down = np.nanmean(f[:nh * block_size, :nw * block_size].reshape(nh, block_size, nw, block_size, 3), axis=(1, 3))
            ind = calculate_all_indices(down[:, :, 2], down[:, :, 1], down[:, :, 0])
            mask = build_vegetation_mask(ind, pp)
            base = {"Data": date, "Ponto": int(m.group(1))}
            rows.append({**base, "Recorte": -1, "Veg_blocks": int(mask.sum()), **stats_row(ind, mask)})
            ys = np.linspace(0, nh, TILES + 1).astype(int)
            xs = np.linspace(0, nw, TILES + 1).astype(int)
            for i in range(TILES):
                for j in range(TILES):
                    sl = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
                    tm = mask[sl]
                    if tm.sum() < MIN_VEG_BLOCKS:
                        continue
                    rows.append({**base, "Recorte": i * TILES + j, "Veg_blocks": int(tm.sum()),
                                 **stats_row({k: v[sl] for k, v in ind.items()}, tm)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# modelo
# ─────────────────────────────────────────────────────────────────────────────
def fit_predict(cfg, tr, te, feats, y_col="SPAD"):
    mode, (kind, k), hidden, act, alpha = cfg
    tr = tr[tr.Recorte >= 0] if mode == "recortes" else tr[tr.Recorte == -1]
    te_rows = te[te.Recorte >= 0] if mode == "recortes" else te[te.Recorte == -1]
    Xtr, ytr = tr[feats].to_numpy(float), tr[y_col].to_numpy(float)
    Xte = te_rows[feats].to_numpy(float)
    if kind == "modelo":
        cols = [feats.index(c) for c in config.MODEL_FEATURES]
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    elif kind == "top":
        xc = Xtr - Xtr.mean(0)
        r = np.abs(xc.T @ (ytr - ytr.mean())) / (np.sqrt((xc ** 2).sum(0) * ((ytr - ytr.mean()) ** 2).sum()) + 1e-300)
        cols = np.argsort(-r)[:k]
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    sx = StandardScaler().fit(Xtr)
    Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)
    if kind == "pca":
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    my, sy = ytr.mean(), ytr.std() or 1.0
    preds = []
    for seed in range(N_SEEDS):
        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation=act, solver="lbfgs", alpha=alpha, max_iter=3000, random_state=seed)
        mlp.fit(Xtr, (ytr - my) / sy)
        preds.append(mlp.predict(Xte) * sy + my)
    out = te_rows[["Data", "Ponto"]].assign(pred=np.mean(preds, axis=0))
    return out.groupby(["Data", "Ponto"]).pred.mean()


def to_class(v, limits):
    """Mesma regra de 03_modelos_clorofila/04_chlorophyll_classes.py: limites internos fechados na classe central."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape, dtype=int)
    for i, lim in enumerate(limits):
        out = np.where(v > lim if i >= len(limits) / 2 else v >= lim, i + 1, out)
    return out


def weighted_kappa(a, b, k=None):
    k = k or int(max(np.max(a), np.max(b)) + 1)
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    d = (w * exp).sum()
    return 1 - (w * obs).sum() / d if d > 0 else np.nan


def metrics(y, p, limits):
    a, b = to_class(y, limits), to_class(p, limits)
    return {"n": len(y), "Acerto_exato": float(np.mean(a == b)), "Acerto_1_classe": float(np.mean(np.abs(a - b) <= 1)),
            "Kappa_ponderado": weighted_kappa(a, b, len(limits) + 1), "R2_SPAD": 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2),
            "MAE_SPAD": float(np.mean(np.abs(y - p)))}


def cfg_label(cfg):
    mode, (kind, k), hidden, act, alpha = cfg
    inp = {"modelo": "24 entradas do modelo", "top": f"{k} índices de maior |r|", "pca": f"{k} componentes principais"}[kind]
    return f"{'recortes 3×3' if mode == 'recortes' else 'foto inteira'} · {inp} · {'+'.join(map(str, hidden))} neurônios {act} · α={alpha:g}"


def lodo_task(cfg, data, feats, eval_dates):
    """Treina sem cada data de eval_dates (01/06 sempre no treino) e prevê essa data."""
    out = []
    for d in eval_dates:
        out.append(fit_predict(cfg, data[data.Data != d], data[data.Data == d], feats))
    return cfg, pd.concat(out)


def inner_task(cfg, data, feats, pair, limits):
    """Escolha sem a data testada: treina numa das duas datas restantes (+ 01/06), avalia na outra, e vice-versa."""
    ks = []
    for d_fit, d_eval in [pair, pair[::-1]]:
        tr = data[data.Data.isin([d_fit, config.TEST_DATE])]
        p = fit_predict(cfg, tr, data[data.Data == d_eval], feats)
        y = data[(data.Data == d_eval) & (data.Recorte == -1)].set_index(["Data", "Ponto"]).SPAD.loc[p.index]
        ks.append(weighted_kappa(to_class(y, limits), to_class(p, limits), len(limits) + 1))
    return cfg, float(np.nanmean(ks))


def main():
    t0 = time.time()
    scheme = db.load_df("chl_class_scheme").sort_values("Ordem")
    limits = scheme.SPAD_max.dropna().tolist()
    CLASSES = scheme.Classe.tolist()

    feats_df = extract()
    f = load_falker_all()
    spad = f.groupby(["Data", "Ponto"])["Clorofila Total"].mean().rename("SPAD").reset_index()
    spad["Ponto"] = spad.Ponto.astype(int)
    data = feats_df.merge(spad, on=["Data", "Ponto"])
    feats = [c for c in data.columns if c.split("_")[0] in ("Median", "Mean", "P75", "P90")
             and not c.endswith("_rgb") and data[c].std() > 1e-9]
    n_photos = (data.Recorte == -1).sum()
    print(f"{n_photos} fotos, {(data.Recorte >= 0).sum()} recortes, {len(feats)} variáveis  [{time.time() - t0:.0f}s]")

    exp_dates = config.EXPERIMENTAL_DATES
    photos = data[(data.Recorte == -1) & data.Data.isin(exp_dates)].set_index(["Data", "Ponto"]).sort_index()
    y_all = photos.SPAD

    configs = list(itertools.product(DATA_MODES, INPUTS, HIDDEN, ACTIVATIONS, ALPHAS))
    print(f"{len(configs)} configurações")

    with Parallel(n_jobs=-1) as par:
        # LODO simples
        flat = par(delayed(lodo_task)(cfg, data, feats, exp_dates) for cfg in configs)
        search, flat_preds = [], {}
        for cfg, p in flat:
            p = p.loc[y_all.index]
            flat_preds[cfg] = p
            search.append({"Configuracao": cfg_label(cfg), "Dados": cfg[0], "Entradas": cfg_label(cfg).split(" · ")[1],
                           "Neuronios": "+".join(map(str, cfg[2])), "Ativacao": cfg[3], "Alpha": cfg[4],
                           **metrics(y_all.to_numpy(), p.to_numpy(), limits)})
        search = pd.DataFrame(search).sort_values(["Kappa_ponderado", "Acerto_exato", "R2_SPAD"], ascending=False).reset_index(drop=True)
        search["Rank"] = np.arange(1, len(search) + 1)
        print(f"busca simples pronta [{time.time() - t0:.0f}s]")

        # LODO aninhado
        nested_pred = pd.Series(index=y_all.index, dtype=float)
        nested_rows = []
        for d in exp_dates:
            pair = tuple(x for x in exp_dates if x != d)
            inner = par(delayed(inner_task)(cfg, data, feats, pair, limits) for cfg in configs)
            best_cfg, best_k = max(inner, key=lambda t: (t[1] if np.isfinite(t[1]) else -9))
            p = fit_predict(best_cfg, data[data.Data != d], data[data.Data == d], feats)
            nested_pred.loc[p.index] = p.values
            yd = y_all.loc[p.index]
            nested_rows.append({"Data_testada": d, "Configuracao_escolhida": cfg_label(best_cfg), "Kappa_na_escolha": best_k,
                                **metrics(yd.to_numpy(), p.to_numpy(), limits)})
            print(f"  {d}: {cfg_label(best_cfg)}  kappa escolha={best_k:.3f}  [{time.time() - t0:.0f}s]")
    nested_rows.append({"Data_testada": "Todas as datas", "Configuracao_escolhida": "", "Kappa_na_escolha": np.nan,
                        **metrics(y_all.to_numpy(), nested_pred.to_numpy(), limits)})
    nested = pd.DataFrame(nested_rows)

    # comparação com o que já existe (mesmas 31 fotos)
    prev = db.load_df("chl_class_agreement")
    prev = prev[prev.Escopo == "Todas as datas"]
    best_label = search.Configuracao.iloc[0]
    best_cfg = next(c for c in configs if cfg_label(c) == best_label)
    summary = [
        {"Abordagem": "MLP: melhor da busca (otimista)", "Detalhe": best_label,
         **metrics(y_all.to_numpy(), flat_preds[best_cfg].to_numpy(), limits)},
        {"Abordagem": "MLP: busca aninhada (esperado em coleta nova)", "Detalhe": "configuração escolhida sem a data testada",
         **metrics(y_all.to_numpy(), nested_pred.to_numpy(), limits)},
    ]
    for _, r in prev.iterrows():
        summary.append({"Abordagem": r.Metodo, "Detalhe": "página Classes", "n": r.n, "Acerto_exato": r.Acerto_exato,
                        "Acerto_1_classe": r.Acerto_1_classe, "Kappa_ponderado": r.Kappa_ponderado})
    summary = pd.DataFrame(summary)

    preds = photos.reset_index()[["Data", "Ponto", "SPAD"]].copy()
    preds["Classe"] = [CLASSES[i] for i in to_class(preds.SPAD, limits)]
    preds["SPAD_busca"] = flat_preds[best_cfg].values
    preds["Classe_busca"] = [CLASSES[i] for i in to_class(preds.SPAD_busca, limits)]
    preds["SPAD_aninhada"] = nested_pred.values
    preds["Classe_aninhada"] = [CLASSES[i] for i in to_class(preds.SPAD_aninhada, limits)]

    db.save_df(search, "mlp_class_search", description="Busca de MLP para as classes de clorofila: cada configuração com validação deixando uma data de fora")
    db.save_df(nested, "mlp_class_nested", description="Busca de MLP aninhada: configuração escolhida sem a data testada")
    db.save_df(summary, "mlp_class_summary", description="Classes de clorofila: MLP da busca, MLP aninhada e abordagens da página Classes")
    db.save_df(preds, "mlp_class_predictions", description="SPAD e classe previstos por foto pela MLP da busca e pela aninhada")

    print("\nMelhores da busca simples:")
    print(search.head(12)[["Rank", "Configuracao", "Acerto_exato", "Acerto_1_classe", "Kappa_ponderado", "R2_SPAD"]].round(3).to_string(index=False))
    print("\nAninhada:")
    print(nested.round(3).to_string(index=False))
    print("\nResumo:")
    print(summary[["Abordagem", "Acerto_exato", "Acerto_1_classe", "Kappa_ponderado"]].round(3).to_string(index=False))
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    db.run_main(main)
