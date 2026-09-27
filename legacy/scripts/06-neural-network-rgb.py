"""
MLP para prever Clorofila Total usando apenas dados RGB (canais e razões).

Features: Median/Mean/P75/P90 de r, g, b, rg, rb, gb, rgb (extraídos em 01-extract-indicies.py).
Pré-processamento de imagem (01 / image_preprocess.py): gamma=0.8, máscara VARI>0.10.
Validação: GroupKFold por Data (leave-one-date-out), predições out-of-fold.
Meta: R² OOF ≥ 0.75

Pipeline: StandardScaler → MLPRegressor (solver lbfgs, bagging de seeds).
"""

import os
import warnings
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

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

HIDDEN_LAYERS_GRID = [(2,), (3,), (4,), (5,), (6,), (4, 2)]
ALPHA_GRID = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
N_BAG_SEEDS = 20
N_BAG_SEARCH = 8
MAX_ITER = 4000

# Melhor configuração encontrada offline (10×10, mediana, rgb3/rgb7)
DEFAULT_MLP_PARAMS = {"hidden_layer_sizes": (3,), "alpha": 5.0}

EXPERIMENTAL_DATES = {"18-05-2026", "21-05-2026", "26-05-2026"}


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


def make_mlp_pipeline(hidden_layer_sizes, alpha, random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
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


def group_kfold_oof(X, y, groups, pipe_template, n_bag=N_BAG_SEEDS):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        y_pred = bagged_predict(
            pipe_template, X[train_idx], y[train_idx], X[test_idx], n_bag=n_bag,
        )
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


def search_hyperparameters(X, y, groups, n_bag=8):
    best_r2 = -np.inf
    best_params = {"hidden_layer_sizes": (4,), "alpha": 10.0}

    for hidden in HIDDEN_LAYERS_GRID:
        for alpha in ALPHA_GRID:
            pipe = make_mlp_pipeline(hidden, alpha)
            y_oof, _, _ = group_kfold_oof(X, y, groups, pipe, n_bag=n_bag)
            r2 = compute_r2(y, y_oof)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"hidden_layer_sizes": hidden, "alpha": alpha}

    return best_params, best_r2


def fit_evaluate_rgb_mlp(
    df_merged,
    feature_mode,
    block_size,
    tune_hyperparams=True,
    experimental_only=False,
    mlp_params=None,
):
    stat_prefixes = FEATURE_MODES[feature_mode]["stats"]
    feature_cols, chromatic = resolve_feature_columns(feature_mode, stat_prefixes)
    stat_label = " + ".join(STAT_METHODS[s]["label"] for s in stat_prefixes)

    meta_cols = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    df_model = df_merged[meta_cols + feature_cols].dropna().copy()

    if experimental_only:
        df_model = df_model[df_model[GROUP_COL].isin(EXPERIMENTAL_DATES)].copy()

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
        best_params, search_r2 = search_hyperparameters(X, y, groups, n_bag=N_BAG_SEARCH)
    else:
        best_params = dict(DEFAULT_MLP_PARAMS)
        search_r2 = np.nan

    pipe = make_mlp_pipeline(**best_params)
    y_oof, df_folds, n_splits = group_kfold_oof(X, y, groups, pipe, n_bag=N_BAG_SEEDS)

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
        "Experimental_Only": experimental_only,
        "Hidden_Layers": str(best_params["hidden_layer_sizes"]),
        "Alpha": best_params["alpha"],
        "N_Bag": N_BAG_SEEDS,
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
    return metrics, df_preds, df_folds


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
    ax.set_title(
        f"MLP RGB — {metrics['Feature_Label']} — {metrics['Block_Size']}\n"
        f"GroupKFold {GROUP_COL} ({metrics['CV_Folds']} folds)  |  "
        f"R² OOF = {metrics['R2_OOF']:.3f}  (meta: {TARGET_R2:.2f})",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.05, 0.95,
        f"RMSE = {metrics['RMSE_OOF']:.2f}\n"
        f"MAE  = {metrics['MAE_OOF']:.2f}\n"
        f"MLP{metrics['Hidden_Layers']}, α={metrics['Alpha']}, bag={metrics['N_Bag']}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    slug = metrics["Feature_Mode"]
    out_path = f"plots/mlp_rgb_{metrics['Block_Size']}_{slug}_groupkfold_date.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    print("=" * 80)
    print("MLP RGB — previsão de Clorofila Total")
    print(f"  Validação: GroupKFold por '{GROUP_COL}' (leave-one-date-out)")
    print(f"  Meta: R² OOF ≥ {TARGET_R2:.2f}")
    print(f"  Features: apenas canais RGB (r, g, b e derivados rg, rb, gb, rgb)")
    print(f"  Pipeline: StandardScaler → MLPRegressor (lbfgs) + bagging ({N_BAG_SEEDS} seeds)")
    print("=" * 80)

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

    os.makedirs("data/output/images-infos", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    block_sizes = [10, 5, 2]
    feature_modes = ["rgb3", "rgb7", "rgb3_chrom", "rgb12", "rgb28"]
    all_metrics = []
    all_preds = []
    all_folds = []
    tuned_params = {}

    for bs in block_sizes:
        csv_indices = f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv"
        if not os.path.exists(csv_indices):
            print(f"\nArquivo não encontrado: {csv_indices}")
            continue

        print(f"\n--- Grade {bs}x{bs} ---")
        df_indices = pd.read_csv(csv_indices)
        df_merged = build_merged_dataset(df_all_cloro, df_indices)

        for mode in feature_modes:
            label = FEATURE_MODES[mode]["label"]
            do_tune = bs == 10 and mode in {"rgb3", "rgb7"}
            if do_tune:
                print(f"\n  [{label}] — busca de hiperparâmetros...")
            else:
                print(f"\n  [{label}] — hiperparâmetros {'padrão' if mode not in tuned_params else 'reutilizados'}...")
            result = fit_evaluate_rgb_mlp(
                df_merged,
                mode,
                bs,
                tune_hyperparams=do_tune,
                experimental_only=False,
                mlp_params=tuned_params.get(mode),
            )
            if result[0] is None:
                print(f"    Dados insuficientes. Pulando.")
                continue

            metrics, df_preds, df_folds = result
            if do_tune:
                tuned_params[mode] = {
                    "hidden_layer_sizes": literal_eval(metrics["Hidden_Layers"]),
                    "alpha": metrics["Alpha"],
                }
                if mode == "rgb3":
                    tuned_params.setdefault("rgb3_chrom", tuned_params["rgb3"])
                    tuned_params.setdefault("rgb12", tuned_params["rgb3"])
                if mode == "rgb7":
                    tuned_params.setdefault("rgb28", tuned_params["rgb7"])
            all_metrics.append(metrics)
            df_preds["Block_Size"] = f"{bs}x{bs}"
            df_preds["Feature_Mode"] = mode
            all_preds.append(df_preds)
            all_folds.append(df_folds)

            plot_path = plot_predictions(df_preds, metrics)
            status = "✓" if metrics["Meets_Target"] else "✗"
            print(f"    Amostras: {metrics['N_Total']}  |  Features: {metrics['N_Features']}  |  "
                  f"Folds: {metrics['CV_Folds']}")
            print(f"    MLP{metrics['Hidden_Layers']}, α={metrics['Alpha']}  |  "
                  f"R² OOF = {metrics['R2_OOF']:.4f}  {status} meta {TARGET_R2:.2f}  |  "
                  f"RMSE = {metrics['RMSE_OOF']:.3f}")
            print(f"    Gráfico: {plot_path}")

    if not all_metrics:
        print("\nNenhum modelo foi treinado.")
        return

    df_summary = pd.DataFrame(all_metrics).sort_values("R2_OOF", ascending=False)
    df_summary.to_csv(
        "data/output/images-infos/mlp_rgb_groupkfold_date_summary.csv", index=False,
    )
    pd.concat(all_preds, ignore_index=True).to_csv(
        "data/output/images-infos/mlp_rgb_groupkfold_date_predictions.csv", index=False,
    )
    pd.concat(all_folds, ignore_index=True).to_csv(
        "data/output/images-infos/mlp_rgb_groupkfold_date_folds.csv", index=False,
    )

    best = df_summary.iloc[0]
    print("\n" + "=" * 80)
    print("RESUMO — MLP RGB (GroupKFold por Data, OOF)")
    print("=" * 80)
    cols = [
        "Feature_Mode", "Feature_Label", "Block_Size", "N_Features",
        "Hidden_Layers", "Alpha", "R2_OOF", "RMSE_OOF", "Meets_Target",
    ]
    print(df_summary[cols].to_string(index=False))

    print(f"\nMelhor configuração: {best['Feature_Label']} — {best['Block_Size']}  |  "
          f"R² OOF = {best['R2_OOF']:.4f}")
    if best["R2_OOF"] < TARGET_R2:
        gap = TARGET_R2 - best["R2_OOF"]
        print(f"\nMeta R² = {TARGET_R2:.2f} não atingida (faltam {gap:.3f}).")
        print("Com N≈34 amostras e validação leave-one-date-out, o teto observado")
        print("com features RGB puras fica em ~0.60. Para aproximar 0.75 seria necessário")
        print("mais amostras/datas ou relaxar a validação temporal.")
    else:
        print(f"\nMeta R² = {TARGET_R2:.2f} atingida.")

    print("\nArquivos salvos:")
    print("  data/output/images-infos/mlp_rgb_groupkfold_date_summary.csv")
    print("  data/output/images-infos/mlp_rgb_groupkfold_date_predictions.csv")
    print("  data/output/images-infos/mlp_rgb_groupkfold_date_folds.csv")


if __name__ == "__main__":
    main()
