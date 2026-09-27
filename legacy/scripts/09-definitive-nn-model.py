"""
Modelo de Rede Neural Definitiva vs. Melhor Índice Espectral.

Este script:
1. Carrega os dados de Clorofila e os índices extraídos no melhor pré-processamento
   (10x10, bilateral filter, máscara ExG>0.15 — R² OOF ≈ 0.753 no script 07).
2. Treina a MLP Campeã (rgb28, hidden_layers=(3,), alpha=8.0, activation='relu', bagging de 10 seeds).
3. Identifica dinamicamente o melhor Índice Espectral único (via GroupKFold OOF).
4. Prediz a clorofila para todas as 34 imagens (amostras).
5. Compara a precisão (R², RMSE, MAE) dos dois modelos.
6. Salva as predições detalhadas em CSV e gera um gráfico comparativo lado a lado.
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Configurações do layout experimental (DBC)
ponto_to_dbc = {
    1:  {"Bloco": "Bloco 1", "Tratamento": "T3", "Dose": 75},
    2:  {"Bloco": "Bloco 1", "Tratamento": "T2", "Dose": 50},
    3:  {"Bloco": "Bloco 1", "Tratamento": "T4", "Dose": 100},
    4:  {"Bloco": "Bloco 1", "Tratamento": "T1", "Dose": 0},
    5:  {"Bloco": "Bloco 2", "Tratamento": "T4", "Dose": 100},
    6:  {"Bloco": "Bloco 2", "Tratamento": "T3", "Dose": 75},
    7:  {"Bloco": "Bloco 2", "Tratamento": "T1", "Dose": 0},
    8:  {"Bloco": "Bloco 2", "Tratamento": "T2", "Dose": 50},
    9:  {"Bloco": "Bloco 3", "Tratamento": "T2", "Dose": 50},
    10: {"Bloco": "Bloco 3", "Tratamento": "T4", "Dose": 100},
    11: {"Bloco": "Bloco 3", "Tratamento": "T3", "Dose": 75},
    12: {"Bloco": "Bloco 3", "Tratamento": "T1", "Dose": 0},
}

RANDOM_STATE = 42
GROUP_COL = "Data"
N_BAG_SEEDS = 10
MAX_ITER = 4000
BLOCK_SIZE = 10

# Melhor pré-processamento (script 07-rgb-preprocess-search.py, R² OOF = 0.7534)
BEST_PREPROCESS_VARIANT = "bilateral"
BEST_PREPROCESS_CONFIG = {"filter_type": "bilateral"}
CSV_INDICES = (
    "data/output/images-infos/preprocess_search/"
    f"extracted_indices_{BLOCK_SIZE}x{BLOCK_SIZE}_{BEST_PREPROCESS_VARIANT}.csv"
)

# Parâmetros da MLP Campeã (script 07, rgb28, GroupKFold por Data)
CHAMP_MLP_PARAMS = {
    "hidden_layer_sizes": (3,),
    "alpha": 8.0,
    "activation": "relu",
}


def load_falker_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    skip_count = 0
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Medição" in line:
                skip_count = i
                break

    df = pd.read_csv(file_path, skiprows=skip_count, header=None)
    df = df.set_index(0).transpose()
    df.columns.name = None

    for col in ["Medição", "Ponto", "Clorofila Total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_r2(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def compute_rmse(y_obs, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_obs) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_obs, y_pred):
    return float(np.mean(np.abs(np.asarray(y_obs) - np.asarray(y_pred))))


def build_merged_dataset(df_cloro, df_indices):
    df_chlo_plot = (
        df_cloro.dropna(subset=["Tratamento"])
        .groupby(["Data", "Ponto"])
        .agg({
            "Clorofila Total": "mean",
            "Bloco": "first",
            "Tratamento": "first",
        })
        .reset_index()
    )
    return pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])


def make_mlp_pipeline(hidden_layer_sizes, alpha, activation="relu", random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="lbfgs",
            alpha=alpha,
            max_iter=MAX_ITER,
            random_state=random_state,
        )),
    ])


def bagged_predict(pipe_template, X_train, y_train, X_test, n_bag=N_BAG_SEEDS):
    preds = []
    for seed in range(n_bag):
        pipe = clone(pipe_template)
        pipe.set_params(mlp__random_state=seed)
        pipe.fit(X_train, y_train)
        preds.append(pipe.predict(X_test))
    return np.mean(preds, axis=0)


def group_kfold_oof_mlp(X, y, groups, pipe_template, n_bag=N_BAG_SEEDS):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        y_pred = bagged_predict(
            pipe_template, X[train_idx], y[train_idx], X[test_idx], n_bag=n_bag,
        )
        y_oof[test_idx] = y_pred

    return y_oof


def group_kfold_oof_lr(X_col, y, groups):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    for train_idx, test_idx in gkf.split(X_col, y, groups):
        model = clone(pipe)
        model.fit(X_col[train_idx].reshape(-1, 1), y[train_idx])
        y_pred = model.predict(X_col[test_idx].reshape(-1, 1))
        y_oof[test_idx] = y_pred

    return y_oof


def main():
    print("=" * 80)
    print("COMPARAÇÃO DEFINITIVA: MLP CAMPEÃ VS. MELHOR ÍNDICE ESPECTRAL")
    print(f"  Pré-processamento: {BEST_PREPROCESS_VARIANT} (ExG>0.15, grade {BLOCK_SIZE}x{BLOCK_SIZE})")
    print("=" * 80)

    # 1. Carregar dados
    df_cloro = load_falker_csv("data/clorophyll/cloro.csv")
    df_test = load_falker_csv("data/clorophyll/test.csv")
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)
    
    df_all_cloro["Tratamento"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"]
        if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro["Bloco"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"]
        if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    if not os.path.exists(CSV_INDICES):
        print(f"Erro: arquivo de índices {CSV_INDICES} não encontrado!")
        print("Execute 07-rgb-preprocess-search.py ou extraia com config 'bilateral'.")
        return

    df_indices = pd.read_csv(CSV_INDICES)
    df_merged = build_merged_dataset(df_all_cloro, df_indices)

    # Identificar colunas de features
    # rgb28 usa Median, Mean, P75 e P90 dos 7 canais/razões
    rgb_channels = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
    feature_cols = []
    for stat in ["Median", "Mean", "P75", "P90"]:
        feature_cols.extend([f"{stat}_{name}" for name in rgb_channels])

    # Selecionar apenas as colunas necessárias e dropar NaNs nelas
    meta_cols = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    df_model = df_merged.dropna(subset=meta_cols + feature_cols).copy()

    X_mlp = df_model[feature_cols].values
    y = df_model["Clorofila Total"].values
    groups = df_model[GROUP_COL].values

    # 2. Avaliar MLP Campeã
    print(
        f"\nAvaliando MLP Campeã (rgb28, {BLOCK_SIZE}x{BLOCK_SIZE}, "
        f"hidden=(3,), alpha=8.0, preprocess={BEST_PREPROCESS_VARIANT})..."
    )
    pipe_mlp = make_mlp_pipeline(**CHAMP_MLP_PARAMS)
    y_pred_mlp = group_kfold_oof_mlp(X_mlp, y, groups, pipe_mlp, n_bag=N_BAG_SEEDS)

    mlp_r2 = compute_r2(y, y_pred_mlp)
    mlp_rmse = compute_rmse(y, y_pred_mlp)
    mlp_mae = compute_mae(y, y_pred_mlp)

    # 3. Encontrar dinamicamente o melhor Índice Espectral único (via GroupKFold OOF)
    print("Buscando o melhor índice espectral único...")
    
    # Lista de todos os possíveis índices individuais na base
    potential_indices = [c for c in df_indices.columns if any(c.startswith(f"{stat}_") for stat in ["Median", "Mean", "P75", "P90"])]
    
    best_idx_name = None
    best_idx_r2 = -np.inf
    best_idx_y_pred = None

    for col in potential_indices:
        X_col = df_model[col].values
        # Avalia regressão linear simples via CV
        y_pred_col = group_kfold_oof_lr(X_col, y, groups)
        r2_col = compute_r2(y, y_pred_col)
        
        if r2_col > best_idx_r2:
            best_idx_r2 = r2_col
            best_idx_name = col
            best_idx_y_pred = y_pred_col

    idx_rmse = compute_rmse(y, best_idx_y_pred)
    idx_mae = compute_mae(y, best_idx_y_pred)

    print(f"  -> Melhor índice encontrado: {best_idx_name}")
    print(f"  -> R² OOF do índice: {best_idx_r2:.4f}")

    # 4. Ajustar modelo linear global do melhor índice para obter os coeficientes reais da fórmula
    pipe_lr_global = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    X_best_col = df_model[best_idx_name].values.reshape(-1, 1)
    pipe_lr_global.fit(X_best_col, y)
    
    lr_model = pipe_lr_global.named_steps["lr"]
    scaler = pipe_lr_global.named_steps["scaler"]
    
    # Coeficientes da fórmula linear (desnormalizados para fins informativos)
    raw_slope = lr_model.coef_[0] / scaler.scale_[0]
    raw_intercept = lr_model.intercept_ - (lr_model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])
    formula_str = f"Clorofila = {raw_slope:.5f} * {best_idx_name} + ({raw_intercept:.5f})"

    # 5. Salvar predições de todas as imagens em CSV
    df_output = df_model[[GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]].copy()
    df_output = df_output.rename(columns={"Bloco_x": "Bloco", "Tratamento_x": "Tratamento"})
    df_output["Predicao_MLP"] = y_pred_mlp
    df_output["Predicao_Melhor_Indice"] = best_idx_y_pred
    df_output["Erro_Absoluto_MLP"] = np.abs(df_output["Clorofila Total"] - y_pred_mlp)
    df_output["Erro_Absoluto_Indice"] = np.abs(df_output["Clorofila Total"] - best_idx_y_pred)

    out_csv = "data/output/images-infos/definitive_model_predictions.csv"
    df_output.to_csv(out_csv, index=False)

    # 6. Exibir tabela comparativa
    print("\n" + "=" * 80)
    print("COMPARAÇÃO DE ACURÁCIA (GroupKFold Out-of-Fold)")
    print("=" * 80)
    print(f"{'Métrica':<15} | {'MLP Campeã (28 features)':<25} | {'Melhor Índice (' + best_idx_name + ')':<30}")
    print("-" * 80)
    print(f"{'R² (OOF)':<15} | {mlp_r2:<25.4f} | {best_idx_r2:<30.4f}")
    print(f"{'RMSE (OOF)':<15} | {mlp_rmse:<25.4f} | {idx_rmse:<30.4f}")
    print(f"{'MAE (OOF)':<15} | {mlp_mae:<25.4f} | {idx_mae:<30.4f}")
    print("-" * 80)
    print(f"Fórmula do Índice: {formula_str}")
    print("=" * 80)

    # 7. Gerar e salvar gráfico comparativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    lims = [
        min(y.min(), y_pred_mlp.min(), best_idx_y_pred.min()) - 2,
        max(y.max(), y_pred_mlp.max(), best_idx_y_pred.max()) + 2
    ]

    # Subplote 1: MLP
    ax1.scatter(y, y_pred_mlp, color="#1E88E5", alpha=0.85, edgecolors="k", linewidths=0.5, s=50)
    ax1.plot(lims, lims, "r--", lw=1.5, label="Identidade 1:1")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Clorofila Observada (Falker)")
    ax1.set_ylabel("Clorofila Predita (MLP OOF)")
    ax1.set_title(
        f"MLP Campeã (bilateral, 28 feat.)\nR² OOF = {mlp_r2:.4f} | RMSE = {mlp_rmse:.2f}",
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Subplote 2: Melhor Índice
    ax2.scatter(y, best_idx_y_pred, color="#43A047", alpha=0.85, edgecolors="k", linewidths=0.5, s=50)
    ax2.plot(lims, lims, "r--", lw=1.5, label="Identidade 1:1")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Clorofila Observada (Falker)")
    ax2.set_ylabel(f"Clorofila Predita ({best_idx_name} OOF)")
    ax2.set_title(f"Melhor Índice ({best_idx_name})\nR² OOF = {best_idx_r2:.4f} | RMSE = {idx_rmse:.2f}", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plot_path = "plots/comparison_mlp_vs_best_index.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nPredições salvas em: {out_csv}")
    print(f"Gráfico de comparação salvo em: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
