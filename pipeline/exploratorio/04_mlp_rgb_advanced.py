"""
MLP RGB Avançada para prever Clorofila Total.

Novas técnicas implementadas:
- Pipeline flexível com PCA (redução de dimensionalidade).
- BaggingRegressor nativo do scikit-learn com Bootstrap real de amostras.
- Busca de hiperparâmetros expandida (ativações relu/tanh/identity, alphas elevados e uso de PCA).
- Validação cruzada rigorosa por Data (GroupKFold).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.config import PONTO_TO_DBC as ponto_to_dbc
from tcc_analysis.data import build_merged_dataset, indices_available, load_falker, load_indices
from tcc_analysis.db import plot_path
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
GROUP_COL = "Data"
TARGET_R2 = 0.75

RGB_CHANNEL_NAMES = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
RGB_CORE_NAMES = ["r", "g", "b"]

STAT_METHODS = {
    "median": {"prefix": "Median", "label": "Mediana"},
    "mean":   {"prefix": "Mean",   "label": "Média"},
    "p75":    {"prefix": "P75",    "label": "Percentil 75"},
    "p90":    {"prefix": "P90",    "label": "Percentil 90"},
}

FEATURE_MODES = {
    "rgb3": {
        "label": "RGB (r, g, b)",
        "names": RGB_CORE_NAMES,
        "stats": ["median"],
    },
    "rgb7": {
        "label": "RGB + razões (7 canais)",
        "names": RGB_CHANNEL_NAMES,
        "stats": ["median"],
    },
    "rgb3_chrom": {
        "label": "RGB + cromáticos normalizados",
        "names": RGB_CORE_NAMES,
        "stats": ["median"],
        "chromatic": True,
    },
    "rgb12": {
        "label": "RGB (r,g,b) × 4 estatísticas",
        "names": RGB_CORE_NAMES,
        "stats": list(STAT_METHODS.keys()),
    },
    "rgb28": {
        "label": "RGB completo × 4 estatísticas",
        "names": RGB_CHANNEL_NAMES,
        "stats": list(STAT_METHODS.keys()),
    },
}

# Grids de Busca
ACTIVATION_GRID = ["relu", "tanh", "identity"]
HIDDEN_LAYERS_GRID = [(2,), (3,), (4,), (2, 2)]
ALPHA_GRID = [1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 200.0]
PCA_GRID = [
    {"use_pca": False, "n_components": None},
    {"use_pca": True, "n_components": 3},
    {"use_pca": True, "n_components": 5},
]

N_ESTIMATORS_BAG = 30
MAX_ITER = 4000

DEFAULT_PARAMS = {
    "hidden_layer_sizes": (2,),
    "alpha": 10.0,
    "activation": "tanh",
    "use_pca": False,
    "n_components": None
}


def get_rgb_columns(prefix, channel_names):
    return [f"{prefix}_{name}" for name in channel_names]


def build_feature_matrix(df_model, feature_cols, chromatic=False):
    X = df_model[feature_cols].values.astype(float)
    if not chromatic:
        return X

    r_idx = next(i for i, c in enumerate(feature_cols) if c.endswith("_r"))
    g_idx = next(i for i, c in enumerate(feature_cols) if c.endswith("_g"))
    b_idx = next(i for i, c in enumerate(feature_cols) if c.endswith("_b"))
    r, g, b = X[:, r_idx], X[:, g_idx], X[:, b_idx]
    tot = r + g + b + 1e-9
    chrom = np.column_stack([r / tot, g / tot, b / tot])
    return np.column_stack([X, chrom])


def resolve_feature_columns(feature_mode, stat_prefixes):
    mode = FEATURE_MODES[feature_mode]
    cols = []
    for stat_key in mode["stats"]:
        prefix = STAT_METHODS[stat_key]["prefix"]
        cols.extend(get_rgb_columns(prefix, mode["names"]))
    return cols, bool(mode.get("chromatic", False))


def make_mlp_pipeline(hidden_layer_sizes, alpha, activation="relu", use_pca=False, n_components=None, random_state=RANDOM_STATE):
    steps = [("scaler", StandardScaler())]
    if use_pca and n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    steps.append(("mlp", MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="lbfgs",
        alpha=alpha,
        max_iter=MAX_ITER,
        random_state=random_state,
    )))
    return Pipeline(steps)


def make_bagged_estimator(base_pipeline, n_estimators=N_ESTIMATORS_BAG, random_state=RANDOM_STATE):
    return BaggingRegressor(
        estimator=base_pipeline,
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1
    )


def group_kfold_oof_bagged(X, y, groups, base_pipeline, n_estimators=N_ESTIMATORS_BAG):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)
    fold_rows = []

    bagged_model = make_bagged_estimator(base_pipeline, n_estimators=n_estimators)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model = clone(bagged_model)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_oof[test_idx] = y_pred
        
        held_out = ", ".join(sorted(set(groups[test_idx])))
        fold_rows.append({
            "Fold": fold_idx,
            "Held_Out_Date": held_out,
            "N_Train": len(train_idx),
            "N_Test": len(test_idx),
            "R2": compute_r2(y[test_idx], y_pred),
            "RMSE": compute_rmse(y[test_idx], y_pred),
            "MAE": compute_mae(y[test_idx], y_pred),
        })

    return y_oof, pd.DataFrame(fold_rows), n_splits


def search_hyperparameters(X, y, groups):
    best_r2 = -np.inf
    best_params = dict(DEFAULT_PARAMS)

    # Para acelerar, se X tiver poucas colunas, limitamos o PCA
    num_feats = X.shape[1]

    for hidden in HIDDEN_LAYERS_GRID:
        for alpha in ALPHA_GRID:
            for activation in ACTIVATION_GRID:
                for pca_opt in PCA_GRID:
                    use_pca = pca_opt["use_pca"]
                    n_comp = pca_opt["n_components"]

                    if use_pca and n_comp >= num_feats:
                        continue # Pula PCA se os componentes forem maiores que as features

                    pipe = make_mlp_pipeline(
                        hidden_layer_sizes=hidden,
                        alpha=alpha,
                        activation=activation,
                        use_pca=use_pca,
                        n_components=n_comp
                    )
                    
                    # Para busca rápida usamos bagging menor
                    y_oof, _, _ = group_kfold_oof_bagged(X, y, groups, pipe, n_estimators=10)
                    r2 = compute_r2(y, y_oof)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {
                            "hidden_layer_sizes": hidden,
                            "alpha": alpha,
                            "activation": activation,
                            "use_pca": use_pca,
                            "n_components": n_comp
                        }

    return best_params, best_r2


def fit_evaluate_advanced_mlp(
    df_merged,
    feature_mode,
    block_size,
    tune_hyperparams=True,
    mlp_params=None,
):
    stat_prefixes = FEATURE_MODES[feature_mode]["stats"]
    feature_cols, chromatic = resolve_feature_columns(feature_mode, stat_prefixes)
    stat_label = " + ".join(STAT_METHODS[s]["label"] for s in stat_prefixes)

    meta_cols = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    df_model = df_merged[meta_cols + feature_cols].dropna().copy()

    n = len(df_model)
    if n < 10:
        return None, None, None

    n_groups = df_model[GROUP_COL].nunique()
    if n_groups < 2:
        return None, None, None

    X = build_feature_matrix(df_model, feature_cols, chromatic=chromatic)
    y = df_model["Clorofila Total"].values
    groups = df_model[GROUP_COL].values

    if mlp_params is not None:
        best_params = dict(mlp_params)
        search_r2 = np.nan
    elif tune_hyperparams:
        best_params, search_r2 = search_hyperparameters(X, y, groups)
    else:
        best_params = dict(DEFAULT_PARAMS)
        search_r2 = np.nan

    pipe = make_mlp_pipeline(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        alpha=best_params["alpha"],
        activation=best_params["activation"],
        use_pca=best_params["use_pca"],
        n_components=best_params["n_components"]
    )
    y_oof, df_folds, n_splits = group_kfold_oof_bagged(X, y, groups, pipe, n_estimators=N_ESTIMATORS_BAG)

    metrics = {
        "Feature_Mode": feature_mode,
        "Feature_Label": FEATURE_MODES[feature_mode]["label"],
        "Stat_Label": stat_label,
        "Block_Size": f"{block_size}x{block_size}",
        "N_Features": X.shape[1],
        "N_Total": n,
        "N_Groups": n_groups,
        "CV_Folds": n_splits,
        "Group_By": GROUP_COL,
        "Activation": best_params["activation"],
        "Hidden_Layers": str(best_params["hidden_layer_sizes"]),
        "Alpha": best_params["alpha"],
        "Use_PCA": best_params["use_pca"],
        "PCA_Components": best_params["n_components"],
        "N_Bag": N_ESTIMATORS_BAG,
        "Search_R2_OOF": search_r2,
        "R2_OOF": compute_r2(y, y_oof),
        "RMSE_OOF": compute_rmse(y, y_oof),
        "MAE_OOF": compute_mae(y, y_oof),
        "R2_Fold_mean": df_folds["R2"].mean(),
        "R2_Fold_std": df_folds["R2"].std(),
        "Target_R2": TARGET_R2,
        "Meets_Target": compute_r2(y, y_oof) >= TARGET_R2,
    }

    df_preds = df_model.copy()
    df_preds["Clorofila_Predita"] = y_oof
    df_preds["Split"] = "oof"
    df_preds["Fold"] = np.nan
    for _, fold in df_folds.iterrows():
        mask = df_preds[GROUP_COL].isin(fold["Held_Out_Date"].split(", "))
        df_preds.loc[mask, "Fold"] = fold["Fold"]

    df_folds["Block_Size"] = f"{block_size}x{block_size}"
    df_folds["Feature_Mode"] = feature_mode
    return metrics, df_preds, df_folds, best_params


def plot_predictions(df_preds, metrics):
    fig, ax = plt.subplots(figsize=(6, 6))
    y_obs = df_preds["Clorofila Total"].values
    y_pred = df_preds["Clorofila_Predita"].values

    ax.scatter(y_obs, y_pred, alpha=0.85, edgecolors="k", linewidths=0.4)
    lims = [
        min(y_obs.min(), y_pred.min()) - 1,
        max(y_obs.max(), y_pred.max()) + 1,
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Clorofila Total observada")
    ax.set_ylabel("Clorofila Total predita (OOF)")
    
    pca_str = f"PCA={metrics['PCA_Components']}" if metrics['Use_PCA'] else "NoPCA"
    ax.set_title(
        f"MLP RGB Avançada — {metrics['Feature_Label']} — {metrics['Block_Size']}\n"
        f"R² OOF = {metrics['R2_OOF']:.3f}  (meta: {TARGET_R2:.2f})",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.05, 0.95,
        f"RMSE = {metrics['RMSE_OOF']:.2f}\n"
        f"MAE  = {metrics['MAE_OOF']:.2f}\n"
        f"MLP{metrics['Hidden_Layers']}, f={metrics['Activation']}, α={metrics['Alpha']}\n"
        f"{pca_str}, bag={metrics['N_Bag']}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    slug = metrics["Feature_Mode"]
    out_path = plot_path(f"mlp_rgb_advanced_{metrics['Block_Size']}_{slug}_groupkfold_date.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    print("=" * 80)
    print("MLP RGB AVANÇADA — previsão de Clorofila Total")
    print(f"  Validação: GroupKFold por '{GROUP_COL}' (leave-one-date-out)")
    print(f"  Meta: R² OOF ≥ {TARGET_R2:.2f}")
    print(f"  Otimizações: PCA, Bootstrap Bagging, Ativações Alternativas, Alpha Estendido")
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
    feature_modes = ["rgb3", "rgb7", "rgb3_chrom", "rgb12", "rgb28"]
    all_metrics = []
    all_preds = []
    all_folds = []
    tuned_params = {}

    for bs in block_sizes:
        indices_label = f"image_indices (grade {bs}x{bs})"
        if not indices_available(bs):
            print(f"\nArquivo não encontrado: {indices_label}")
            continue

        print(f"\n--- Grade {bs}x{bs} ---")
        df_indices = load_indices(bs)
        df_merged = build_merged_dataset(df_all_cloro, df_indices)

        for mode in feature_modes:
            label = FEATURE_MODES[mode]["label"]
            do_tune = bs == 10 and mode in {"rgb3", "rgb7", "rgb28"}
            if do_tune:
                print(f"\n  [{label}] — busca avançada de hiperparâmetros...")
            else:
                print(f"\n  [{label}] — hiperparâmetros {'padrão' if mode not in tuned_params else 'reutilizados'}...")
            
            result = fit_evaluate_advanced_mlp(
                df_merged,
                mode,
                bs,
                tune_hyperparams=do_tune,
                mlp_params=tuned_params.get(mode),
            )
            if result[0] is None:
                print(f"    Dados insuficientes. Pulando.")
                continue

            metrics, df_preds, df_folds, best_params = result
            if do_tune:
                tuned_params[mode] = best_params
                if mode == "rgb3":
                    tuned_params.setdefault("rgb3_chrom", best_params)
                    tuned_params.setdefault("rgb12", best_params)
                if mode == "rgb7":
                    # rgb28 tem busca própria porque tem muito mais features
                    pass
            all_metrics.append(metrics)
            df_preds["Block_Size"] = f"{bs}x{bs}"
            df_preds["Feature_Mode"] = mode
            all_preds.append(df_preds)
            all_folds.append(df_folds)

            fig_path = plot_predictions(df_preds, metrics)
            status = "✓" if metrics["Meets_Target"] else "✗"
            pca_info = f"PCA={metrics['PCA_Components']}" if metrics['Use_PCA'] else "NoPCA"
            print(f"    Amostras: {metrics['N_Total']}  |  Features: {metrics['N_Features']}  |  "
                  f"Folds: {metrics['CV_Folds']}")
            print(f"    MLP{metrics['Hidden_Layers']}, f={metrics['Activation']}, α={metrics['Alpha']}, {pca_info}  |  "
                  f"R² OOF = {metrics['R2_OOF']:.4f}  {status} meta {TARGET_R2:.2f}  |  "
                  f"RMSE = {metrics['RMSE_OOF']:.3f}")
            print(f"    Gráfico: {fig_path}")

    if not all_metrics:
        print("\nNenhum modelo foi treinado.")
        return

    df_summary = pd.DataFrame(all_metrics).sort_values("R2_OOF", ascending=False)
    db.save_df(df_summary, "mlp_rgb_advanced_summary",
               description="MLP RGB com PCA, bagging bootstrap e ativações alternativas")
    db.save_df(pd.concat(all_preds, ignore_index=True), "mlp_rgb_advanced_predictions")
    db.save_df(pd.concat(all_folds, ignore_index=True), "mlp_rgb_advanced_folds")

    best = df_summary.iloc[0]
    print("\n" + "=" * 80)
    print("RESUMO — MLP RGB AVANÇADA (GroupKFold por Data, OOF)")
    print("=" * 80)
    cols = [
        "Feature_Mode", "Feature_Label", "Block_Size", "N_Features",
        "Activation", "Hidden_Layers", "Alpha", "Use_PCA", "PCA_Components", "R2_OOF", "RMSE_OOF", "Meets_Target",
    ]
    print(df_summary[cols].to_string(index=False))

    print(f"\nMelhor configuração: {best['Feature_Label']} — {best['Block_Size']}  |  "
          f"R² OOF = {best['R2_OOF']:.4f}")
    if best["R2_OOF"] < TARGET_R2:
        gap = TARGET_R2 - best["R2_OOF"]
        print(f"\nMeta R² = {TARGET_R2:.2f} não atingida (faltam {gap:.3f}).")
    else:
        print(f"\nMeta R² = {TARGET_R2:.2f} atingida! ✓")

    print("\nTabelas salvas: mlp_rgb_advanced_summary, mlp_rgb_advanced_predictions, mlp_rgb_advanced_folds")


if __name__ == "__main__":
    db.run_main(main)
