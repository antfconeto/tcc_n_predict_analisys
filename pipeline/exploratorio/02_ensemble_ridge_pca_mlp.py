"""
Ensemble triplo para prever Clorofila Total (GroupKFold por Data).

Para cada estatística de bloco vegetado (Mediana, Média, P75, P90):
  1. Regressão Ridge — índices vegetativos
  2. PCA + Ridge — todos os índices espectrais dessa estatística
  3. MLP — StandardScaler → PCA → MLP(8)
  Ensemble: média (Reg + PCA + MLP), predições out-of-fold

Requer 01-extract-indicies.py (colunas Median_, Mean_, P75_, P90_).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.config import PONTO_TO_DBC as ponto_to_dbc
from tcc_analysis.data import build_merged_dataset, indices_available, load_falker, load_indices
from tcc_analysis.db import plot_path
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse

RANDOM_STATE = 42
N_PCA_COMPONENTS = 6
GROUP_COL = "Data"

VEGETATION_INDEX_NAMES = [
    "GLI", "HI", "NGRDI", "SI", "VARI", "HUE",
    "MGRVI", "MPRI", "ExG", "RGBVI", "SCI",
]
STAT_METHODS = {
    "median": {"prefix": "Median", "label": "Mediana"},
    "mean":   {"prefix": "Mean",   "label": "Média"},
    "p75":    {"prefix": "P75",    "label": "Percentil 75"},
    "p90":    {"prefix": "P90",    "label": "Percentil 90"},
}


def get_spectral_columns(df_indices, prefix):
    return [c for c in df_indices.columns if c.startswith(f"{prefix}_")]


def get_regression_columns(df_indices, prefix):
    cols = [f"{prefix}_{name}" for name in VEGETATION_INDEX_NAMES]
    return [c for c in cols if c in df_indices.columns]


def n_pca_components(n_train, n_features):
    return max(2, min(N_PCA_COMPONENTS, n_train - 1, n_features))


def make_pca_regression_pipeline(n_train, n_features):
    """PCA nos índices espectrais + Ridge nas componentes (ramo linear separado)."""
    n_pca = n_pca_components(n_train, n_features)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]), n_pca


def make_mlp_pipeline(n_train, n_features):
    n_pca = n_pca_components(n_train, n_features)
    use_early_stop = n_train >= 30

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="adam",
            alpha=0.1,
            learning_rate_init=0.005,
            max_iter=1500,
            batch_size=min(16, max(4, n_train // 2)),
            random_state=RANDOM_STATE,
            early_stopping=use_early_stop,
            validation_fraction=0.2 if use_early_stop else 0.1,
            n_iter_no_change=25,
            tol=1e-4,
        )),
    ]), n_pca


def make_regression_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def group_kfold_evaluate_ensemble(X_reg, X_mlp, y, groups, n_mlp_features):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)

    y_reg_oof = np.full(len(y), np.nan)
    y_pca_oof = np.full(len(y), np.nan)
    y_mlp_oof = np.full(len(y), np.nan)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_reg, y, groups), start=1):
        held_out = ", ".join(sorted(set(groups[test_idx])))

        reg_pipe = make_regression_pipeline()
        reg_pipe.fit(X_reg[train_idx], y[train_idx])
        y_pred_reg = reg_pipe.predict(X_reg[test_idx])

        pca_pipe, n_pca = make_pca_regression_pipeline(len(train_idx), n_mlp_features)
        pca_pipe.fit(X_mlp[train_idx], y[train_idx])
        y_pred_pca = pca_pipe.predict(X_mlp[test_idx])
        var_explained = float(np.sum(pca_pipe.named_steps["pca"].explained_variance_ratio_))

        mlp_pipe, _ = make_mlp_pipeline(len(train_idx), n_mlp_features)
        mlp_pipe.fit(X_mlp[train_idx], y[train_idx])
        y_pred_mlp = mlp_pipe.predict(X_mlp[test_idx])

        y_pred_ens = (y_pred_reg + y_pred_pca + y_pred_mlp) / 3.0
        y_reg_oof[test_idx] = y_pred_reg
        y_pca_oof[test_idx] = y_pred_pca
        y_mlp_oof[test_idx] = y_pred_mlp

        fold_rows.append({
            "Fold": fold_idx,
            "Held_Out_Date": held_out,
            "N_Train": len(train_idx),
            "N_Test": len(test_idx),
            "N_PCA_Components": n_pca,
            "PCA_Var_Explained": var_explained,
            "R2_Reg": compute_r2(y[test_idx], y_pred_reg),
            "R2_PCA": compute_r2(y[test_idx], y_pred_pca),
            "R2_MLP": compute_r2(y[test_idx], y_pred_mlp),
            "R2_Ensemble": compute_r2(y[test_idx], y_pred_ens),
            "RMSE_Reg": compute_rmse(y[test_idx], y_pred_reg),
            "RMSE_PCA": compute_rmse(y[test_idx], y_pred_pca),
            "RMSE_MLP": compute_rmse(y[test_idx], y_pred_mlp),
            "RMSE_Ensemble": compute_rmse(y[test_idx], y_pred_ens),
            "MAE_Ensemble": compute_mae(y[test_idx], y_pred_ens),
        })

    y_ens_oof = (y_reg_oof + y_pca_oof + y_mlp_oof) / 3.0
    return y_reg_oof, y_pca_oof, y_mlp_oof, y_ens_oof, pd.DataFrame(fold_rows), n_splits


def fit_evaluate_ensemble(df_merged, mlp_cols, reg_cols, block_size, stat_key, stat_label):
    meta_cols = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    # reg_cols ⊆ mlp_cols — evita duplicar colunas no PCA/MLP
    df_model = df_merged[meta_cols + mlp_cols].dropna().copy()
    n = len(df_model)

    if n < 10:
        print(f"  [{block_size}x{block_size}] Amostras insuficientes ({n}). Pulando.")
        return None, None, None

    if len(reg_cols) < 2:
        print(f"  [{block_size}x{block_size}] Índices vegetativos insuficientes. Pulando.")
        return None, None, None

    n_groups = df_model[GROUP_COL].nunique()
    if n_groups < 2:
        print(f"  [{block_size}x{block_size}] Menos de 2 datas. Pulando.")
        return None, None, None

    X_mlp = df_model[mlp_cols].values
    X_reg = df_model[reg_cols].values
    y = df_model["Clorofila Total"].values
    groups = df_model[GROUP_COL].values

    y_reg_oof, y_pca_oof, y_mlp_oof, y_ens_oof, df_folds, n_splits = group_kfold_evaluate_ensemble(
        X_reg, X_mlp, y, groups, len(mlp_cols),
    )

    metrics = {
        "Statistic": stat_key,
        "Stat_Label": stat_label,
        "Block_Size": f"{block_size}x{block_size}",
        "N_Total": n,
        "N_Groups": n_groups,
        "CV_Folds": n_splits,
        "Group_By": GROUP_COL,
        "N_MLP_Features": len(mlp_cols),
        "N_Regression_Features": len(reg_cols),
        "N_PCA_Components": int(df_folds["N_PCA_Components"].iloc[0]),
        "PCA_Var_Explained_mean": df_folds["PCA_Var_Explained"].mean(),
        "R2_Reg_OOF": compute_r2(y, y_reg_oof),
        "R2_PCA_OOF": compute_r2(y, y_pca_oof),
        "R2_MLP_OOF": compute_r2(y, y_mlp_oof),
        "R2_Ensemble_OOF": compute_r2(y, y_ens_oof),
        "RMSE_Reg_OOF": compute_rmse(y, y_reg_oof),
        "RMSE_PCA_OOF": compute_rmse(y, y_pca_oof),
        "RMSE_MLP_OOF": compute_rmse(y, y_mlp_oof),
        "RMSE_Ensemble_OOF": compute_rmse(y, y_ens_oof),
        "MAE_Ensemble_OOF": compute_mae(y, y_ens_oof),
        "R2_Reg_Fold_mean": df_folds["R2_Reg"].mean(),
        "R2_PCA_Fold_mean": df_folds["R2_PCA"].mean(),
        "R2_MLP_Fold_mean": df_folds["R2_MLP"].mean(),
        "R2_Ensemble_Fold_mean": df_folds["R2_Ensemble"].mean(),
        "R2_Ensemble_Fold_std": df_folds["R2_Ensemble"].std(),
    }

    df_preds = df_model.copy()
    df_preds["Split"] = "oof"
    df_preds["Clorofila_Regressao"] = y_reg_oof
    df_preds["Clorofila_PCA"] = y_pca_oof
    df_preds["Clorofila_MLP"] = y_mlp_oof
    df_preds["Clorofila_Ensemble"] = y_ens_oof
    df_preds["Clorofila_Predita"] = y_ens_oof
    df_preds["Fold"] = np.nan

    for _, fold in df_folds.iterrows():
        mask = df_preds[GROUP_COL].isin(fold["Held_Out_Date"].split(", "))
        df_preds.loc[mask, "Fold"] = fold["Fold"]

    df_folds["Block_Size"] = f"{block_size}x{block_size}"
    return metrics, df_preds, df_folds


def save_pca_loadings(df_model, mlp_cols, block_size):
    """Ajusta PCA+Ridge em todas as amostras e exporta cargas e variância explicada."""
    X = df_model[mlp_cols].values
    y = df_model["Clorofila Total"].values
    pipe, n_pca = make_pca_regression_pipeline(len(y), len(mlp_cols))
    pipe.fit(X, y)

    pca = pipe.named_steps["pca"]
    df_load = pd.DataFrame(
        pca.components_.T,
        index=mlp_cols,
        columns=[f"PC{i + 1}" for i in range(n_pca)],
    )
    df_load["Block_Size"] = f"{block_size}x{block_size}"
    df_var = pd.DataFrame({
        "Block_Size": [f"{block_size}x{block_size}"],
        "N_Components": [n_pca],
        **{f"Var_PC{i + 1}": [pca.explained_variance_ratio_[i]] for i in range(n_pca)},
        "Var_Total_Explained": [float(np.sum(pca.explained_variance_ratio_))],
    })
    return df_load, df_var


def plot_predictions(df_preds, block_size, metrics, stat_label):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    panels = [
        ("Clorofila_Regressao", "R2_Reg_OOF", "RMSE_Reg_OOF",
         "Regressão vegetativa"),
        ("Clorofila_PCA", "R2_PCA_OOF", "RMSE_PCA_OOF",
         f"PCA→Ridge ({metrics['N_PCA_Components']} comp.)"),
        ("Clorofila_MLP", "R2_MLP_OOF", "RMSE_MLP_OOF",
         "MLP (PCA interno)"),
        ("Clorofila_Ensemble", "R2_Ensemble_OOF", "RMSE_Ensemble_OOF",
         "Ensemble (média 3 modelos)"),
    ]

    for ax, (pred_col, r2_key, rmse_key, title) in zip(axes, panels):
        y_obs = df_preds["Clorofila Total"]
        y_pred = df_preds[pred_col]

        ax.scatter(y_pred, y_obs, color="#1976D2", alpha=0.7, edgecolors="white", s=40)
        lo = min(y_obs.min(), y_pred.min()) - 2
        hi = max(y_obs.max(), y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Clorofila predita")
        ax.set_ylabel("Clorofila observada")
        ax.set_title(title, fontsize=9)
        ax.text(
            0.05, 0.95,
            f"R² OOF = {metrics[r2_key]:.3f}\nRMSE OOF = {metrics[rmse_key]:.2f}",
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.suptitle(
        f"Ensemble — {stat_label} — {block_size}x{block_size}  |  "
        f"GroupKFold {GROUP_COL} ({metrics['CV_Folds']} folds)  |  "
        f"var. expl. PCA ≈ {metrics['PCA_Var_Explained_mean']:.0%}",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    stat_slug = metrics["Statistic"]
    out_path = (
        plot_path(f"mlp_regression_ensemble_{block_size}x{block_size}_{stat_slug}_groupkfold_date.png")
    )
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    print("=" * 80)
    print("ENSEMBLE: REGRESSÃO + PCA→RIDGE + MLP")
    print(f"  Validação: GroupKFold por '{GROUP_COL}' (leave-one-date-out)")
    print(f"  (1) Regressão Ridge — {len(VEGETATION_INDEX_NAMES)} índices vegetativos")
    print(f"  (2) PCA({N_PCA_COMPONENTS} comp.) + Ridge — índices espectrais por estatística")
    print(f"  Estatísticas: {', '.join(m['label'] for m in STAT_METHODS.values())}")
    print(f"  (3) MLP: StandardScaler → PCA → MLP(8)")
    print(f"  Ensemble: média dos 3 modelos (OOF)")
    print("=" * 80)


    df_cloro = load_falker("cloro")
    df_test = load_falker("test")
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


    block_sizes = [10, 5, 2]
    all_metrics = []
    all_preds = []
    all_folds = []
    all_loadings = []
    all_pca_var = []

    for bs in block_sizes:
        indices_label = f"image_indices (grade {bs}x{bs})"
        if not indices_available(bs):
            print(f"\nArquivo não encontrado: {indices_label}")
            continue

        print(f"\n--- Grade {bs}x{bs} ---")
        df_indices = load_indices(bs)
        df_merged = build_merged_dataset(df_all_cloro, df_indices)
        meta_cols = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]

        for stat_key, stat_info in STAT_METHODS.items():
            prefix = stat_info["prefix"]
            label = stat_info["label"]
            print(f"\n  [{label}]")
            mlp_cols = get_spectral_columns(df_indices, prefix)
            reg_cols = get_regression_columns(df_indices, prefix)

            if not mlp_cols:
                print(f"    Colunas {prefix}_* ausentes. Rode 01-extract-indicies.py.")
                continue

            df_model_full = df_merged[meta_cols + mlp_cols].dropna().copy()
            result = fit_evaluate_ensemble(
                df_merged, mlp_cols, reg_cols, bs, stat_key, label,
            )
            if result[0] is None:
                continue

            metrics, df_preds, df_folds = result
            df_load, df_var = save_pca_loadings(df_model_full, mlp_cols, bs)
            df_load = df_load.reset_index(names="Feature")
            df_load["Statistic"] = stat_key
            df_var["Statistic"] = stat_key
            all_loadings.append(df_load)
            all_pca_var.append(df_var)
            all_metrics.append(metrics)
            df_preds["Block_Size"] = f"{bs}x{bs}"
            df_preds["Statistic"] = stat_key
            all_preds.append(df_preds)
            df_folds["Statistic"] = stat_key
            all_folds.append(df_folds)

            fig_path = plot_predictions(df_preds, bs, metrics, label)
            print(f"    Amostras: {metrics['N_Total']}  |  Datas: {metrics['N_Groups']}  |  "
                  f"Folds: {metrics['CV_Folds']}")
            print(f"    OOF — Reg: R²={metrics['R2_Reg_OOF']:.4f}  "
                  f"PCA: R²={metrics['R2_PCA_OOF']:.4f}  "
                  f"MLP: R²={metrics['R2_MLP_OOF']:.4f}  "
                  f"Ens: R²={metrics['R2_Ensemble_OOF']:.4f}  "
                  f"RMSE={metrics['RMSE_Ensemble_OOF']:.3f}")
            print(f"    Gráfico: {fig_path}")

    if not all_metrics:
        print("\nNenhum modelo foi treinado.")
        return

    df_summary = pd.DataFrame(all_metrics)
    db.save_df(df_summary, "mlp_regression_ensemble_summary",
               description="Ensemble Ridge + PCA→Ridge + MLP (GroupKFold por Data)")
    db.save_df(pd.concat(all_preds, ignore_index=True), "mlp_regression_ensemble_predictions")
    db.save_df(pd.concat(all_folds, ignore_index=True), "mlp_regression_ensemble_folds")
    db.save_df(pd.concat(all_loadings, ignore_index=True), "pca_loadings_spectral_indices")
    db.save_df(pd.concat(all_pca_var, ignore_index=True), "pca_variance_explained")

    print("\n" + "=" * 80)
    print("RESUMO — GroupKFold por Data (OOF)")
    print("=" * 80)
    cols = [
        "Statistic", "Stat_Label", "Block_Size", "N_Groups", "CV_Folds",
        "R2_Reg_OOF", "R2_PCA_OOF", "R2_MLP_OOF", "R2_Ensemble_OOF", "RMSE_Ensemble_OOF",
    ]
    print(df_summary[cols].to_string(index=False))
    print("\nTabelas salvas: mlp_regression_ensemble_{summary,predictions,folds}, "
          "pca_loadings_spectral_indices, pca_variance_explained")


if __name__ == "__main__":
    db.run_main(main)
