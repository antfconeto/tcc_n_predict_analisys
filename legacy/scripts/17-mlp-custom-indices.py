import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Disable warnings for clean output
warnings.filterwarnings("ignore")

# DBC layout mapping
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
    12: {"Bloco": "Bloco 3", "Tratamento": "T1", "Dose": 0}
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

def make_mlp_pipeline(hidden_layer_sizes, alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            solver="lbfgs",
            max_iter=4000,
            random_state=42,
        )),
    ])

def bagged_oof(X, y, groups, hidden_layer_sizes, alpha, n_bag=10):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    y_oof = np.zeros(len(y))
    pipe_template = make_mlp_pipeline(hidden_layer_sizes, alpha)

    for train_idx, test_idx in gkf.split(X, y, groups):
        preds = []
        for seed in range(n_bag):
            pipe = clone(pipe_template)
            pipe.set_params(mlp__random_state=seed)
            pipe.fit(X[train_idx], y[train_idx])
            preds.append(pipe.predict(X[test_idx]))
        y_oof[test_idx] = np.mean(preds, axis=0)

    return y_oof

def main():
    print("=" * 100)
    print("NEURAL NETWORK WITH DISCOVERED SPECTRAL INDICES & HUE")
    print("  -> Validation: GroupKFold by Date (leave-one-date-out, bagged MLP)")
    print("=" * 100)

    # 1. Load chlorophyll data
    cloro_csv = "data/clorophyll/cloro.csv"
    test_csv = "data/clorophyll/test.csv"

    print("\nLoading chlorophyll data...")
    df_cloro = load_falker_csv(cloro_csv)
    df_test = load_falker_csv(test_csv)
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)

    # Add treatment and block mapping
    df_all_cloro["Tratamento"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro["Bloco"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    # Aggregate to plot-level averages
    df_chlo_plot = (df_all_cloro
                    .groupby(["Data", "Ponto"])
                    .agg({"Clorofila Total": "mean"})
                    .reset_index())

    block_sizes = [10, 5, 2]
    aggregations = ["Mean", "Median"]
    
    # Feature sets definition
    feature_set_options = {
        "custom_only": ["idx_custom1", "idx_custom2"],
        "custom_baseline_hue": ["idx_custom1", "idx_custom2", "r_over_b", "b_over_r", "HUE"],
        "baseline_hue": ["r_over_b", "b_over_r", "HUE"],
        "idx2_baseline_hue": ["idx_custom2", "r_over_b", "b_over_r", "HUE"],
        "custom_baseline_hue_h": ["idx_custom1", "idx_custom2", "r_over_b", "b_over_r", "HUE", "H_HSL"]
    }

    # First pass: Use fixed default MLP parameters for extremely fast search
    default_hidden = (3,)
    default_alpha = 8.0

    all_results = []

    print("\nSearching best combination of Block Size, Aggregation, and Feature Set...")
    for bs in block_sizes:
        csv_indices = f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv"
        if not os.path.exists(csv_indices):
            continue

        df_indices = pd.read_csv(csv_indices)
        df_merged = pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])

        for stat in aggregations:
            r_col = f"{stat}_r"
            g_col = f"{stat}_g"
            b_col = f"{stat}_b"
            rob_col = f"{stat}_r_over_b"
            bor_col = f"{stat}_b_over_r"
            hue_col = f"{stat}_HUE"
            h_hsl_col = f"{stat}_H"

            if not all(col in df_merged.columns for col in [r_col, g_col, b_col]):
                continue

            # Compute features in a separate dataframe to avoid fragmentation warning
            df_features = pd.DataFrame(index=df_merged.index)
            df_features["idx_custom1"] = (df_merged[r_col] + df_merged[b_col]) / (df_merged[g_col] * df_merged[r_col] * df_merged[b_col] + 1e-8)
            df_features["idx_custom2"] = df_merged[r_col] / (df_merged[g_col] * df_merged[b_col] + 1e-8)
            df_features["r_over_b"] = df_merged[rob_col]
            df_features["b_over_r"] = df_merged[bor_col]
            df_features["HUE"] = df_merged[hue_col] if hue_col in df_merged.columns else np.nan
            df_features["H_HSL"] = df_merged[h_hsl_col] if h_hsl_col in df_merged.columns else np.nan

            df_model = pd.concat([df_merged[["Data", "Ponto", "Clorofila Total"]], df_features], axis=1)
            df_model = df_model.dropna(subset=["Clorofila Total", "idx_custom1", "idx_custom2"]).copy()

            if len(df_model) < 10:
                continue

            groups = df_model["Data"].values
            y = df_model["Clorofila Total"].values

            for fs_name, fs_cols in feature_set_options.items():
                available_cols = [c for c in fs_cols if c in df_model.columns and not df_model[c].isna().all()]
                if not available_cols:
                    continue

                X = df_model[available_cols].values
                y_oof = bagged_oof(X, y, groups, default_hidden, default_alpha, n_bag=10)
                r2 = compute_r2(y, y_oof)
                rmse = compute_rmse(y, y_oof)
                print(
                    {
                    "Block_Size": bs,
                    "Aggregation": stat,
                    "Feature_Set": fs_name,
                    "Features_Used": ", ".join(available_cols),
                    "N_Features": len(available_cols),
                    "R2_OOF": r2,
                    "RMSE_OOF": rmse
                }
                )
                all_results.append({
                    "Block_Size": bs,
                    "Aggregation": stat,
                    "Feature_Set": fs_name,
                    "Features_Used": ", ".join(available_cols),
                    "N_Features": len(available_cols),
                    "R2_OOF": r2,
                    "RMSE_OOF": rmse
                })

    # Convert results to DataFrame and find the best combination
    df_results = pd.DataFrame(all_results).sort_values("R2_OOF", ascending=False)
    best_combo = df_results.iloc[0]

    print(f"\nBest configuration found:")
    print(f"  Block Size:   {best_combo['Block_Size']}x{best_combo['Block_Size']}")
    print(f"  Aggregation:  {best_combo['Aggregation']}")
    print(f"  Feature Set:  {best_combo['Feature_Set']}")
    print(f"  Features:     {best_combo['Features_Used']}")
    print(f"  Initial R²:   {best_combo['R2_OOF']:.4f}")

    # Step 2: Run a fine-tuned grid search *only* on the best configuration
    print("\nRunning hyperparameter optimization for the best configuration...")
    bs = best_combo["Block_Size"]
    stat = best_combo["Aggregation"]
    fs_name = best_combo["Feature_Set"]
    fs_cols = best_combo["Features_Used"].split(", ")

    csv_indices = f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv"
    df_indices = pd.read_csv(csv_indices)
    df_merged = pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])
    
    r_col = f"{stat}_r"
    g_col = f"{stat}_g"
    b_col = f"{stat}_b"
    rob_col = f"{stat}_r_over_b"
    bor_col = f"{stat}_b_over_r"
    hue_col = f"{stat}_HUE"
    h_hsl_col = f"{stat}_H"

    df_features = pd.DataFrame(index=df_merged.index)
    df_features["idx_custom1"] = (df_merged[r_col] + df_merged[b_col]) / (df_merged[g_col] * df_merged[r_col] * df_merged[b_col] + 1e-8)
    df_features["idx_custom2"] = df_merged[r_col] / (df_merged[g_col] * df_merged[b_col] + 1e-8)
    df_features["r_over_b"] = df_merged[rob_col]
    df_features["b_over_r"] = df_merged[bor_col]
    df_features["HUE"] = df_merged[hue_col] if hue_col in df_merged.columns else np.nan
    df_features["H_HSL"] = df_merged[h_hsl_col] if h_hsl_col in df_merged.columns else np.nan

    df_model = pd.concat([df_merged[["Data", "Ponto", "Clorofila Total"]], df_features], axis=1)
    df_model = df_model.dropna(subset=["Clorofila Total"] + fs_cols).copy()

    X = df_model[fs_cols].values
    y = df_model["Clorofila Total"].values
    groups = df_model["Data"].values

    hidden_layer_grid = [(2,), (3,), (4,), (5,), (3, 3), (4, 2)]
    alpha_grid = [0.1, 1.0, 5.0, 8.0, 10.0, 15.0]

    best_r2 = -np.inf
    best_rmse = np.inf
    best_params = None
    best_y_oof = None

    for hidden in hidden_layer_grid:
        for alpha in alpha_grid:
            y_oof = bagged_oof(X, y, groups, hidden, alpha, n_bag=15)
            r2 = compute_r2(y, y_oof)
            rmse = compute_rmse(y, y_oof)

            if r2 > best_r2:
                best_r2 = r2
                best_rmse = rmse
                best_params = (hidden, alpha)
                best_y_oof = y_oof

    # Save detailed summary table
    df_results["Fine_Tuned"] = False
    df_results.loc[df_results.index[0], "R2_OOF"] = best_r2
    df_results.loc[df_results.index[0], "RMSE_OOF"] = best_rmse
    df_results.loc[df_results.index[0], "Hidden_Layers"] = str(best_params[0])
    df_results.loc[df_results.index[0], "Alpha"] = best_params[1]
    df_results.loc[df_results.index[0], "Fine_Tuned"] = True

    output_csv = "data/output/images-infos/mlp_custom_indices_summary.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll validation results successfully saved to {output_csv}")

    # Display Top 10 configurations
    print("\n" + "=" * 115)
    print("  TOP 10 MLP REGRESSION CONFIGURATIONS WITH CUSTOM SPECTRAL INDICES & HUE (OOF GroupKFold by Date)")
    print("=" * 115)
    print(df_results[[
        "Block_Size", "Aggregation", "Feature_Set", "R2_OOF", "RMSE_OOF"
    ]].head(10).to_string(index=False))

    # Output details of best model and save plots
    print(f"\nBest Overall Model (Fine-Tuned): Block={bs}x{bs} | Aggregation={stat} | FeatureSet={fs_name}")
    print(f"  -> MLP Structure: {best_params[0]} | Alpha: {best_params[1]}")
    print(f"  -> R² OOF: {best_r2:.4f}  |  RMSE OOF: {best_rmse:.4f}")

    # Save predictions of the best model
    df_model["Predicted_Chlorophyll"] = best_y_oof
    pred_csv = "data/output/images-infos/mlp_custom_indices_predictions.csv"
    df_model.to_csv(pred_csv, index=False)
    print(f"Best model predictions saved to {pred_csv}")

    # Plot predictions vs observations
    plt.figure(figsize=(7, 7))
    y_obs = y
    y_pred = best_y_oof
    
    plt.scatter(y_obs, y_pred, color="#1B5E20", alpha=0.85, edgecolors='black', s=60, label="Out-of-Fold Predictions")
    lims = [min(y_obs.min(), y_pred.min()) - 1, max(y_obs.max(), y_pred.max()) + 1]
    plt.plot(lims, lims, color='red', linestyle='--', linewidth=1.5, label="1:1 Perfect Fit Line")
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Observed Chlorophyll Total (Falker SPAD)")
    plt.ylabel("Predicted Chlorophyll Total (OOF)")
    plt.title(f"Optimized MLP with Discovered Indices & Hue\n"
              f"Block {bs}x{bs} ({stat}) | R² OOF = {best_r2:.4f}", 
              fontsize=11, fontweight='bold')
    plt.text(0.05, 0.95, f"RMSE: {best_rmse:.3f}\nFeatures: {best_combo['Features_Used']}\nMLP: {best_params[0]}, α={best_params[1]}", 
             transform=plt.gca().transAxes, va='top', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.7))
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plot_path = "plots/best_mlp_custom_indices.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Prediction scatter plot saved to {plot_path}")

if __name__ == "__main__":
    main()
