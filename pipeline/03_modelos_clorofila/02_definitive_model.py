"""
Modelo definitivo (v2): MLP com 24 entradas × melhor índice único, validação por data.

1. Carrega a clorofila média de cada parcela e as entradas extraídas com o pré-processamento
   definitivo (filtro bilateral, gamma 0,8, recorte 20%, máscara HSV de sombra/reflexo, blocos
   10×10, vegetação ExG > 0,15).
2. Avalia a MLP campeã (24 entradas: mediana, média, P75 e P90 de r, g, b, rg, rb, gb;
   3 neurônios ReLU, α = 8, L-BFGS, média de 10 sementes) com GroupKFold por data.
   As colunas *_rgb da versão anterior foram removidas: (r+g+b)/3 vale sempre 1/3 e o resíduo
   de arredondamento de Mean_rgb inflava o R² (ver README).
3. Escolhe o melhor índice único (regressão linear, mesma validação).
4. Grava predições e métricas no banco, a referência da versão anterior sem o artefato e os
   pesos da rede treinada com todas as fotos (app v2) + docs/modelo_app_v2.md.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.config import MODEL_FEATURES, RGB28_FEATURES
from tcc_analysis.db import plot_path
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse
from tcc_analysis.modeling import (
    BLOCK_SIZE,
    CHAMP_MLP_PARAMS,
    GROUP_COL,
    N_BAG_SEEDS,
    RANDOM_STATE,
    group_kfold_oof_lr,
    group_kfold_oof_mlp,
    load_model_dataset,
    make_mlp_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)
DOCS = Path(__file__).resolve().parents[2] / "docs"


def export_app_weights(X, y):
    """Rede treinada com todas as fotos (semente 42, sem média de sementes), como o app v1."""
    pipe = make_mlp_pipeline(**CHAMP_MLP_PARAMS)
    pipe.fit(X, y)
    scaler, mlp = pipe.named_steps["scaler"], pipe.named_steps["mlp"]
    weights = {
        "features": MODEL_FEATURES,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "W1": mlp.coefs_[0].tolist(),
        "b1": mlp.intercepts_[0].tolist(),
        "W2": mlp.coefs_[1].reshape(-1).tolist(),
        "b2": float(mlp.intercepts_[1][0]),
        "preprocess": config.BEST_PREPROCESS_CONFIG,
        "hidden_layer_sizes": list(CHAMP_MLP_PARAMS["hidden_layer_sizes"]),
        "alpha": CHAMP_MLP_PARAMS["alpha"],
        "random_state": RANDOM_STATE,
    }
    fit_r2 = compute_r2(y, pipe.predict(X))
    db.save_df(pd.DataFrame([{"Versao": "app_v2", "Pesos_JSON": json.dumps(weights), "R2_ajuste": fit_r2}]),
               "app_model_weights", description="Pesos da MLP para o app (treino com todas as fotos)")
    return weights, fit_r2


def write_app_doc(weights, metrics):
    fmt = lambda arr: ",\n  ".join(", ".join(f"{v:.10g}" for v in arr[i:i + 6]) for i in range(0, len(arr), 6))
    w1 = np.array(weights["W1"])
    cols = "\n\n".join(f"final w1Col{j + 1} = <double>[\n  {fmt(w1[:, j].tolist())}\n];" for j in range(w1.shape[1]))
    feats = "\n".join(f"{i + 1}. `{f}`" for i, f in enumerate(weights["features"]))
    text = f"""# Modelo de predição de clorofila — app v2

Gerado por `pipeline/03_modelos_clorofila/02_definitive_model.py`. Substitui a versão 1
(`definitive_models_doc.md`), cujas 28 entradas incluíam as colunas `*_rgb`.

## Por que mudou

O canal `rgb` = (r+g+b)/3 vale sempre 1/3. Na versão 1, o resíduo de arredondamento de `Mean_rgb`
(float32) virava uma entrada grande depois da padronização, e o R² de 0,753 dependia dele. No app,
calculado em float64, essa entrada fica constante. A versão 2 remove as 4 colunas `*_rgb` e passa a
descartar pixels de sombra e reflexo.

## Desempenho (validação por data, fotos de datas fora do treino)

| Modelo | R² | RMSE (SPAD) | MAE (SPAD) |
| :--- | :---: | :---: | :---: |
| MLP v2 (24 entradas, máscara HSV) | {metrics['mlp_r2']:.3f} | {metrics['mlp_rmse']:.2f} | {metrics['mlp_mae']:.2f} |
| Melhor índice único (`{metrics['idx_name']}`) | {metrics['idx_r2']:.3f} | {metrics['idx_rmse']:.2f} | {metrics['idx_mae']:.2f} |
| MLP v1 sem o artefato (28 → 24 entradas, sem máscara) | {metrics['v1_r2']:.3f} | – | – |

Para escala: uma única leitura do Falker se afasta em média cerca de 2 SPAD da média da parcela.

## Pré-processamento (tem de ser idêntico no app)

1. Filtro bilateral: `d = 9`, `sigmaColor = 75`, `sigmaSpace = 75` (imagem BGR original).
2. Gamma 0,8: `I = 255 · (I_raw / 255) ^ (1 / 0,8)`, por canal, tabela de 256 valores truncada para inteiro.
3. Recorte de 20% de cada borda.
4. **Máscara HSV (novo)**, em cada pixel da imagem já com gamma (HSV do OpenCV: H 0–179, S e V 0–255):
   o pixel é descartado se `V < 45`, `V > 240`, `S < 25` ou (`V > 240` e `S < 20`).
5. Blocos de 10×10 px: média de B, G e R **só dos pixels válidos**; bloco sem pixel válido é descartado.
6. Por bloco: `S = R + G + B` (1 se 0); `r = R/S`, `g = G/S`, `b = B/S`;
   `rg = (r+g)/2`, `rb = (r+b)/2`, `gb = (g+b)/2`.
7. Vegetação: blocos com `ExG = 2g − r − b > 0,15`.
8. Estatísticas nos blocos de vegetação: mediana, média, percentil 75 e percentil 90
   (percentis com interpolação linear, como `numpy.percentile`).

## Entradas, na ordem

{feats}

## Rede

`x_pad[i] = (x[i] − means[i]) / scales[i]` → `h_j = max(0, b1[j] + Σ x_pad[i] · W1[i][j])` (3 neurônios)
→ `SPAD = b2 + Σ h_j · W2[j]`.

```dart
final means = <double>[
  {fmt(weights['means'])}
];

final scales = <double>[
  {fmt(weights['scales'])}
];

{cols}

final b1 = <double>[{', '.join(f'{v:.10g}' for v in weights['b1'])}];
final w2 = <double>[{', '.join(f'{v:.10g}' for v in weights['W2'])}];
const b2 = {weights['b2']:.10g};
```

Rede treinada com todas as {metrics['n']} fotos (semente {weights['random_state']}); R² de ajuste
nas próprias fotos: {metrics['fit_r2']:.3f} (otimista — o valor a citar é o da validação por data).
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "modelo_app_v2.md").write_text(text, encoding="utf-8")


def main():
    variant = config.BEST_PREPROCESS_VARIANT
    print("=" * 80)
    print("MODELO DEFINITIVO v2 — MLP (24 entradas) × melhor índice único")
    print(f"  Pré-processamento: {variant} {config.BEST_PREPROCESS_CONFIG} · blocos {BLOCK_SIZE}x{BLOCK_SIZE}")
    print("=" * 80)

    df_model = load_model_dataset(variant)
    X_mlp = df_model[MODEL_FEATURES].values
    y = df_model["Clorofila Total"].values
    groups = df_model[GROUP_COL].values

    pipe_mlp = make_mlp_pipeline(**CHAMP_MLP_PARAMS)
    y_pred_mlp = group_kfold_oof_mlp(X_mlp, y, groups, pipe_mlp, n_bag=N_BAG_SEEDS)
    mlp_r2, mlp_rmse, mlp_mae = compute_r2(y, y_pred_mlp), compute_rmse(y, y_pred_mlp), compute_mae(y, y_pred_mlp)

    # referência: versão 1 (bilateral sem HSV) com as colunas *_rgb removidas
    df_v1 = load_model_dataset(config.APP_V1_VARIANT, RGB28_FEATURES)
    v1_feats = [c for c in RGB28_FEATURES if not c.endswith("_rgb")]
    y_v1 = df_v1["Clorofila Total"].values
    v1_pred = group_kfold_oof_mlp(df_v1[v1_feats].values, y_v1, df_v1[GROUP_COL].values, pipe_mlp, n_bag=N_BAG_SEEDS)
    v1_r2 = compute_r2(y_v1, v1_pred)

    # melhor índice único
    stat_prefixes = ("Median_", "Mean_", "P75_", "P90_")
    candidates = [c for c in df_model.columns if c.startswith(stat_prefixes) and df_model[c].std() > 1e-6]
    best_idx_name, best_idx_r2, best_idx_y_pred = None, -np.inf, None
    for col in candidates:
        pred = group_kfold_oof_lr(df_model[col].values, y, groups)
        r2 = compute_r2(y, pred)
        if r2 > best_idx_r2:
            best_idx_name, best_idx_r2, best_idx_y_pred = col, r2, pred
    idx_rmse, idx_mae = compute_rmse(y, best_idx_y_pred), compute_mae(y, best_idx_y_pred)

    lr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(df_model[[best_idx_name]].values, y)
    coef, sc = lr.named_steps["lr"].coef_[0], lr.named_steps["scaler"]
    raw_slope = coef / sc.scale_[0]
    raw_intercept = lr.named_steps["lr"].intercept_ - coef * sc.mean_[0] / sc.scale_[0]
    formula_str = f"Clorofila = {raw_slope:.5f} * {best_idx_name} + ({raw_intercept:.5f})"

    df_output = df_model[[GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]].rename(
        columns={"Bloco_x": "Bloco", "Tratamento_x": "Tratamento"})
    df_output["Predicao_MLP"] = y_pred_mlp
    df_output["Predicao_Melhor_Indice"] = best_idx_y_pred
    df_output["Erro_Absoluto_MLP"] = np.abs(df_output["Clorofila Total"] - y_pred_mlp)
    df_output["Erro_Absoluto_Indice"] = np.abs(df_output["Clorofila Total"] - best_idx_y_pred)
    db.save_df(df_output, "definitive_model_predictions",
               description="Predições fora da amostra por foto: MLP v2 (24 entradas) × melhor índice único")
    db.save_df(pd.DataFrame([
        {"Modelo": "MLP 24 entradas", "R2_OOF": mlp_r2, "RMSE_OOF": mlp_rmse, "MAE_OOF": mlp_mae, "Formula": None},
        {"Modelo": best_idx_name, "R2_OOF": best_idx_r2, "RMSE_OOF": idx_rmse, "MAE_OOF": idx_mae, "Formula": formula_str},
    ]), "definitive_model_summary", description="Métricas fora da amostra da MLP v2 e do melhor índice")
    db.save_df(pd.DataFrame([
        {"Modelo": "MLP v1 com Mean_rgb (artefato numérico)", "R2_OOF": 0.7534, "Observacao": "valor antigo; depende do arredondamento float32 de Mean_rgb"},
        {"Modelo": "MLP v1 sem *_rgb (bilateral, sem máscara HSV)", "R2_OOF": v1_r2, "Observacao": "mesma rede, 24 entradas"},
        {"Modelo": "MLP v2 (bilateral + máscara HSV, 24 entradas)", "R2_OOF": mlp_r2, "Observacao": "modelo definitivo"},
    ]), "model_version_history", description="Evolução do modelo e a correção do artefato de Mean_rgb")

    weights, fit_r2 = export_app_weights(X_mlp, y)
    write_app_doc(weights, {"mlp_r2": mlp_r2, "mlp_rmse": mlp_rmse, "mlp_mae": mlp_mae, "idx_name": best_idx_name,
                            "idx_r2": best_idx_r2, "idx_rmse": idx_rmse, "idx_mae": idx_mae, "v1_r2": v1_r2,
                            "n": len(y), "fit_r2": fit_r2})

    print(f"{'':<14} {'MLP v2 (24)':>14} {best_idx_name:>22}")
    print(f"{'R² (OOF)':<14} {mlp_r2:>14.4f} {best_idx_r2:>22.4f}")
    print(f"{'RMSE':<14} {mlp_rmse:>14.4f} {idx_rmse:>22.4f}")
    print(f"{'MAE':<14} {mlp_mae:>14.4f} {idx_mae:>22.4f}")
    print(f"MLP v1 sem *_rgb (referência): R² {v1_r2:.4f}")
    print(f"Fórmula do índice: {formula_str}")
    print(f"Pesos do app v2: tabela app_model_weights e docs/modelo_app_v2.md (R² de ajuste {fit_r2:.3f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    lims = [min(y.min(), y_pred_mlp.min(), best_idx_y_pred.min()) - 2, max(y.max(), y_pred_mlp.max(), best_idx_y_pred.max()) + 2]
    for ax, pred, color, title in ((ax1, y_pred_mlp, "#2a78d6", f"MLP v2 (24 entradas)\nR² = {mlp_r2:.3f} | RMSE = {mlp_rmse:.2f}"),
                                   (ax2, best_idx_y_pred, "#eb6834", f"{best_idx_name}\nR² = {best_idx_r2:.3f} | RMSE = {idx_rmse:.2f}")):
        ax.scatter(y, pred, color=color, alpha=0.85, edgecolors="white", s=50)
        ax.plot(lims, lims, color="#8a8983", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("SPAD medido (Falker)"); ax.set_ylabel("SPAD previsto (fora da amostra)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path("comparison_mlp_vs_best_index.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    db.run_main(main)
