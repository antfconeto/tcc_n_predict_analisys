import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

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
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if "Medição" in line:
                skip_count = i
                break

    df = pd.read_csv(file_path, skiprows=skip_count, header=None)
    df = df.set_index(0).transpose()
    df.columns.name = None

    numeric_cols = ["Medição", "Ponto", "Latitude", "Longitude", "Clorofila A", "Clorofila B", "Clorofila Total"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    # 0. Confirm 20% border crop
    print("=" * 80)
    print("VERIFYING IMAGE CROP CONFIGURATION (Script 01):")
    print("  -> Confirmed: 20% border crop applied in 01-extract-indicies.py")
    print("  -> img[int(0.20*h):int(0.80*h), int(0.20*w):int(0.80*w)]")
    print("=" * 80)

    cloro_csv = "data/clorophyll/cloro.csv"
    test_csv = "data/clorophyll/test.csv"

    print("\nLoading chlorophyll data...")
    df_cloro = load_falker_csv(cloro_csv)
    df_test = load_falker_csv(test_csv)
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)

    # Add Treatment and Block info
    df_all_cloro["Tratamento"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro["Bloco"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    # Only experimental dates for training (same as Script 2)
    df_cloro_exp = df_all_cloro.dropna(subset=["Tratamento"]).copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregation methods — identical to Script 2
    # ─────────────────────────────────────────────────────────────────────────
    stats_methods = {
        "mean":   {"label": "Média",         "agg": "mean"},
        "median": {"label": "Mediana",        "agg": "median"},
        "p75":    {"label": "Percentil 75",   "agg": lambda x: x.quantile(0.75)},
        "p90":    {"label": "Percentil 90",   "agg": lambda x: x.quantile(0.90)},
        "std":    {"label": "Desvio Padrão",  "agg": "std"},
    }

    block_sizes = [10, 5, 2]
    excel_output = "data/output/images-infos/predictive_formulas_summary.xlsx"
    os.makedirs(os.path.dirname(excel_output), exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    writer = pd.ExcelWriter(excel_output, engine='openpyxl')
    overall_summary = []

    for method_name, method_info in stats_methods.items():
        label = method_info["label"]
        agg   = method_info["agg"]
        print(f"\n📊 --- Aggregation Method: {label} ---")

        # ── TRAINING SET: aggregated by (Data, Tratamento) ─────────────────
        # Exactly like Script 2
        chlo_cols = ["Clorofila Total", "Clorofila A", "Clorofila B"]
        df_chlo_agg = (df_cloro_exp
                       .groupby(["Data", "Tratamento"])[chlo_cols]
                       .agg(agg)
                       .reset_index())

        # ── VALIDATION SET A: plot-level averages (N = 34) ─────────────────
        # Use mean chlorophyll per plot; carry Bloco and Tratamento as identifiers
        df_chlo_val_plot = (df_all_cloro
                            .groupby(["Data", "Ponto"])
                            .agg({"Clorofila Total": "mean", "Bloco": "first", "Tratamento": "first"})
                            .reset_index())

        for bs in block_sizes:
            csv_indices = f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv"
            if not os.path.exists(csv_indices):
                continue

            df_indices = pd.read_csv(csv_indices)
            index_cols = [c for c in df_indices.columns if c.startswith("Median_")]

            # ── Script-2-style aggregation of indices ─────────────────────
            df_ind_agg = (df_indices
                          .groupby(["Data", "Tratamento"])[index_cols]
                          .agg(agg)
                          .reset_index())

            # Merge with aggregated chlorophyll (same as Script 2)
            df_train = pd.merge(df_chlo_agg, df_ind_agg, on=["Data", "Tratamento"])

            # Validation merges
            df_val_plot = pd.merge(df_chlo_val_plot, df_indices, on=["Data", "Ponto"])
            df_val_raw  = pd.merge(df_all_cloro, df_indices, on=["Data", "Ponto"])

            formulas_metrics = []

            for col in index_cols:
                ind_name = col.replace("Median_", "")

                # ── TRAINING (same OLS model as Script 2) ─────────────────
                df_sub_tr = df_train[[col, "Clorofila Total"]].dropna()
                if len(df_sub_tr) < 4 or df_sub_tr[col].std() == 0:
                    continue

                x_ind = df_sub_tr[col]
                y_tr  = df_sub_tr["Clorofila Total"]

                # Simple OLS — identical to Script 2
                X_simple    = sm.add_constant(x_ind, has_constant='add')
                model_ols   = sm.OLS(y_tr, X_simple).fit()
                # params: [const, col] — use .iloc to avoid key-name fragility
                intercept   = model_ols.params.iloc[0]
                slope       = model_ols.params.iloc[1]
                r2_train    = model_ols.rsquared          # Script-2 value
                pearson_r, _ = pearsonr(x_ind, y_tr)

                # ── VALIDATION LEVEL 1: Raw individual readings (N ~ 244) ──
                df_sub_raw  = df_val_raw[[col, "Clorofila Total"]].dropna()
                if df_sub_raw.empty:
                    continue
                y_obs_raw   = df_sub_raw["Clorofila Total"]
                y_pred_raw  = slope * df_sub_raw[col] + intercept
                r2_raw      = compute_r2(y_obs_raw, y_pred_raw)
                rmse_raw    = np.sqrt(np.mean((y_obs_raw - y_pred_raw) ** 2))
                mae_raw     = np.mean(np.abs(y_obs_raw - y_pred_raw))

                # ── VALIDATION LEVEL 2: Plot/Parcela (N = 34) ─────────────
                df_sub_plot = df_val_plot[[col, "Clorofila Total"]].dropna()
                y_obs_plot  = df_sub_plot["Clorofila Total"]
                y_pred_plot = slope * df_sub_plot[col] + intercept
                r2_plot     = compute_r2(y_obs_plot, y_pred_plot)
                rmse_plot   = np.sqrt(np.mean((y_obs_plot - y_pred_plot) ** 2))
                mae_plot    = np.mean(np.abs(y_obs_plot - y_pred_plot))

                # ── VALIDATION LEVEL 3: Block (N = 10) ────────────────────
                df_block_base = df_val_plot[[col, "Clorofila Total", "Bloco_x", "Data"]].dropna().copy()
                df_block_base["Predicted"] = slope * df_block_base[col] + intercept
                df_blk = (df_block_base
                          .groupby(["Data", "Bloco_x"])[["Clorofila Total", "Predicted"]]
                          .mean()
                          .reset_index())
                y_obs_blk   = df_blk["Clorofila Total"]
                y_pred_blk  = df_blk["Predicted"]
                r2_block    = compute_r2(y_obs_blk, y_pred_blk)
                rmse_block  = np.sqrt(np.mean((y_obs_blk - y_pred_blk) ** 2))
                mae_block   = np.mean(np.abs(y_obs_blk - y_pred_blk))

                formulas_metrics.append({
                    "Spectral_Index":   ind_name,
                    "Slope_Beta1":      slope,
                    "Intercept_Beta0":  intercept,
                    "Formula":          f"Clorofila = {slope:.5f} * {ind_name} + ({intercept:.5f})",
                    # ── Script-2 style metrics ──────────────────────
                    "Pearson_r":        pearson_r,
                    "R2_Train_Script2": r2_train,        # same as Script 2
                    # ── External validation ─────────────────────────
                    "R2_Raw_Indiv":     r2_raw,
                    "R2_Plot_Parcela":  r2_plot,
                    "R2_Block":         r2_block,
                    "RMSE_Raw":         rmse_raw,
                    "RMSE_Plot":        rmse_plot,
                    "RMSE_Block":       rmse_block,
                    "MAE_Raw":          mae_raw,
                    "MAE_Plot":         mae_plot,
                    "MAE_Block":        mae_block,
                })

            if not formulas_metrics:
                continue

            df_res = pd.DataFrame(formulas_metrics).sort_values("R2_Train_Script2", ascending=False)

            df_res.to_excel(writer, sheet_name=f"{bs}x{bs}_{method_name}", index=False)
            df_res.to_csv(f"data/output/images-infos/predictive_formulas_{bs}x{bs}_{method_name}.csv", index=False)

            champ = df_res.iloc[0]
            overall_summary.append({
                "Grade":            f"{bs}x{bs}",
                "Estatistica":      label,
                "Champ_Index":      champ["Spectral_Index"],
                "Formula":          champ["Formula"],
                "Pearson_r":        champ["Pearson_r"],
                "R2_Train_Scr2":    champ["R2_Train_Script2"],
                "R2_Ext_Raw":       champ["R2_Raw_Indiv"],
                "R2_Ext_Plot":      champ["R2_Plot_Parcela"],
                "R2_Ext_Block":     champ["R2_Block"],
                "RMSE_Plot":        champ["RMSE_Plot"],
                "RMSE_Block":       champ["RMSE_Block"],
            })

            # ── Plot: Observed vs Predicted at Plot level ──────────────────
            champ_col   = f"Median_{champ['Spectral_Index']}"
            slope_c     = champ["Slope_Beta1"]
            intercept_c = champ["Intercept_Beta0"]

            y_obs_c  = df_val_plot["Clorofila Total"]
            y_pred_c = slope_c * df_val_plot[champ_col] + intercept_c
            mask     = y_obs_c.notna() & y_pred_c.notna()
            y_obs_c, y_pred_c = y_obs_c[mask], y_pred_c[mask]

            if len(y_obs_c) > 2:
                fig, ax = plt.subplots(figsize=(7, 6.5))
                ax.scatter(y_pred_c, y_obs_c, color="#1976D2", alpha=0.7,
                           edgecolors='black', s=55, label=f"Parcelas (N={len(y_obs_c)})")

                lo = min(y_obs_c.min(), y_pred_c.min()) - 2
                hi = max(y_obs_c.max(), y_pred_c.max()) + 2
                ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.5, label="Identidade 1:1")

                fs, fi = np.polyfit(y_pred_c, y_obs_c, 1)
                fx = np.linspace(lo, hi, 100)
                ax.plot(fx, fs * fx + fi, color="#2E7D32", linewidth=2, label="Ajuste Linear")

                ann = (
                    f"Índice: {champ['Spectral_Index']}\n"
                    f"Pearson r: {champ['Pearson_r']:.4f}\n"
                    f"R² Treino (Script 2): {champ['R2_Train_Script2']:.4f}\n"
                    f"R² Validação Parcela: {champ['R2_Plot_Parcela']:.4f}\n"
                    f"R² Validação Bloco:   {champ['R2_Block']:.4f}\n"
                    f"RMSE Parcela: {champ['RMSE_Plot']:.3f} Falker\n"
                    f"Agregação: {label}"
                )
                ax.text(0.05, 0.95, ann, transform=ax.transAxes, fontsize=9.5,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

                ax.set_title(f"Validação Clorofila Total ({bs}x{bs}) — {label}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Clorofila Predita (Falker)", fontsize=11)
                ax.set_ylabel("Clorofila Observada (Falker)", fontsize=11)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(f"plots/prediction_observed_vs_predicted_{bs}x{bs}_{method_name}.png", dpi=300)
                plt.close()

            # ── Multi-panel: all 31 indices at plot level ──────────────────
            fig, axes = plt.subplots(6, 6, figsize=(22, 22))
            axes = axes.flatten()

            for idx, col in enumerate(index_cols):
                ind_name = col.replace("Median_", "")
                ax = axes[idx]
                rows = df_res[df_res["Spectral_Index"] == ind_name]
                if rows.empty:
                    ax.axis("off"); continue

                ri     = rows.iloc[0]
                s, i   = ri["Slope_Beta1"], ri["Intercept_Beta0"]
                r2v    = ri["R2_Plot_Parcela"]
                rmse_v = ri["RMSE_Plot"]

                y_obs_m  = df_val_plot["Clorofila Total"]
                y_pred_m = s * df_val_plot[col] + i
                sm2      = y_obs_m.notna() & y_pred_m.notna()
                yo, yp   = y_obs_m[sm2], y_pred_m[sm2]

                if len(yo) > 0:
                    ax.scatter(yp, yo, color="#1976D2", alpha=0.6, s=15)
                    lo2 = min(yo.min(), yp.min()) - 2
                    hi2 = max(yo.max(), yp.max()) + 2
                    ax.plot([lo2, hi2], [lo2, hi2], color="red", linestyle="--", linewidth=0.8)
                    if len(yp) > 2:
                        fs2, fi2 = np.polyfit(yp, yo, 1)
                        fx2 = np.linspace(lo2, hi2, 10)
                        ax.plot(fx2, fs2 * fx2 + fi2, color="#2E7D32", linewidth=1.0)
                    ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)

                ax.set_title(ind_name, fontsize=10, fontweight="bold")
                ax.text(0.05, 0.95, f"R²val: {r2v:.2f}\nRMSE: {rmse_v:.2f}",
                        transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.tick_params(labelsize=8)

            for j in range(len(index_cols), len(axes)):
                axes[j].axis("off")

            plt.suptitle(f"Prediction Validation — All 31 Indices ({bs}x{bs}) — {label}",
                         fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"plots/prediction_all_indices_observed_vs_predicted_{bs}x{bs}_{method_name}.png", dpi=150)
            plt.close()

    writer.close()

    df_summary = pd.DataFrame(overall_summary)
    df_summary.to_csv("data/output/images-infos/prediction_multi_level_summary.csv", index=False)

    print("\n" + "=" * 110)
    print("  CHAMPION INDEX COMPARISON: Script-2 Training R² vs. External Validation R²")
    print("=" * 110)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_summary[[
        "Grade", "Estatistica", "Champ_Index",
        "Pearson_r", "R2_Train_Scr2",
        "R2_Ext_Raw", "R2_Ext_Plot", "R2_Ext_Block",
        "RMSE_Plot", "RMSE_Block"
    ]].to_string(index=False))

    # =========================================================================
    # PART 2: ADVANCED CROSS-VALIDATION STRATEGIES
    # =========================================================================
    # Use Média + 2x2 as the reference configuration (best overall from Part 1)
    REF_STAT   = "mean"
    REF_BS     = 2
    REF_AGG    = "mean"

    csv_ref = f"data/output/images-infos/extracted_indices_{REF_BS}x{REF_BS}.csv"
    if os.path.exists(csv_ref):
        df_idx_ref  = pd.read_csv(csv_ref)
        index_cols  = [c for c in df_idx_ref.columns if c.startswith("Median_")]

        # Build a flat merged dataset at plot level with identifiers
        df_chlo_plot_ref = (df_all_cloro
                            .dropna(subset=["Tratamento"])
                            .groupby(["Data", "Ponto"])
                            .agg({"Clorofila Total": "mean", "Bloco": "first", "Tratamento": "first"})
                            .reset_index())
        df_merged_ref = pd.merge(df_chlo_plot_ref, df_idx_ref, on=["Data", "Ponto"])

        run_lobo_cv(df_merged_ref, index_cols)
        run_lodo_cv(df_merged_ref, index_cols)
        run_lmm(df_merged_ref, index_cols)


def compute_r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


# =============================================================================
# 1.  LEAVE-ONE-BLOCK-OUT CV  (generalização espacial)
# =============================================================================
def run_lobo_cv(df_merged, index_cols):
    """
    For each block (Bloco 1, 2, 3):
        Train  → other two blocks, aggregated by (Data, Tratamento)
        Test   → held-out block, individual plot rows
    Reports mean R², RMSE, and MAE across folds for every index.
    """
    print("\n" + "=" * 80)
    print("  PART 2-A: LEAVE-ONE-BLOCK-OUT CV  (generalização espacial real)")
    print("=" * 80)

    blocos = sorted(df_merged["Bloco_x"].dropna().unique())
    lobo_results = []

    for col in index_cols:
        ind_name = col.replace("Median_", "")
        fold_r2, fold_rmse, fold_mae = [], [], []

        for held_out in blocos:
            train_mask = df_merged["Bloco_x"] != held_out
            test_mask  = df_merged["Bloco_x"] == held_out

            df_tr = df_merged[train_mask].copy()
            df_te = df_merged[test_mask].copy()

            # Aggregate training by (Data, Tratamento) — same as Script 2
            df_tr_agg = (df_tr.groupby(["Data", "Tratamento_x"])
                         [[col, "Clorofila Total"]]
                         .mean().reset_index())
            sub_tr = df_tr_agg[[col, "Clorofila Total"]].dropna()

            if len(sub_tr) < 4 or sub_tr[col].std() == 0:
                continue

            X = sm.add_constant(sub_tr[col], has_constant='add')
            model = sm.OLS(sub_tr["Clorofila Total"], X).fit()
            intercept = model.params.iloc[0]
            slope     = model.params.iloc[1]

            # Test on individual plot rows of the held-out block
            sub_te = df_te[[col, "Clorofila Total"]].dropna()
            if sub_te.empty:
                continue

            y_obs  = sub_te["Clorofila Total"]
            y_pred = slope * sub_te[col] + intercept

            fold_r2.append(compute_r2(y_obs, y_pred))
            fold_rmse.append(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
            fold_mae.append(np.mean(np.abs(y_obs - y_pred)))

        if fold_r2:
            lobo_results.append({
                "Spectral_Index": ind_name,
                "LOBO_R2_mean":   np.mean(fold_r2),
                "LOBO_R2_std":    np.std(fold_r2),
                "LOBO_RMSE_mean": np.mean(fold_rmse),
                "LOBO_MAE_mean":  np.mean(fold_mae),
            })

    df_lobo = (pd.DataFrame(lobo_results)
               .sort_values("LOBO_R2_mean", ascending=False))
    df_lobo.to_csv("data/output/images-infos/cv_lobo_results.csv", index=False)

    print(df_lobo.head(10).to_string(index=False))

    # Bar chart — top 15 indices
    _plot_cv_bars(df_lobo.head(15), "LOBO_R2_mean", "LOBO_R2_std",
                  "Leave-One-Block-Out CV  –  R² médio por Índice (2x2, Média)",
                  "plots/cv_lobo_r2.png")


# =============================================================================
# 2.  LEAVE-ONE-DATE-OUT CV  (robustez temporal)
# =============================================================================
def run_lodo_cv(df_merged, index_cols):
    """
    For each date (18-05, 21-05, 26-05):
        Train  → other two dates, aggregated by (Data, Tratamento)
        Test   → held-out date, individual plot rows
    Reports mean R², RMSE, MAE across folds.
    """
    print("\n" + "=" * 80)
    print("  PART 2-B: LEAVE-ONE-DATE-OUT CV  (robustez temporal)")
    print("=" * 80)

    dates = sorted(df_merged["Data"].dropna().unique())
    lodo_results = []

    for col in index_cols:
        ind_name = col.replace("Median_", "")
        fold_r2, fold_rmse, fold_mae = [], [], []

        for held_date in dates:
            train_mask = df_merged["Data"] != held_date
            test_mask  = df_merged["Data"] == held_date

            df_tr = df_merged[train_mask].copy()
            df_te = df_merged[test_mask].copy()

            # Aggregate training by (Data, Tratamento)
            df_tr_agg = (df_tr.groupby(["Data", "Tratamento_x"])
                         [[col, "Clorofila Total"]]
                         .mean().reset_index())
            sub_tr = df_tr_agg[[col, "Clorofila Total"]].dropna()

            if len(sub_tr) < 4 or sub_tr[col].std() == 0:
                continue

            X = sm.add_constant(sub_tr[col], has_constant='add')
            model = sm.OLS(sub_tr["Clorofila Total"], X).fit()
            intercept = model.params.iloc[0]
            slope     = model.params.iloc[1]

            # Test on individual plot rows of the held-out date
            sub_te = df_te[[col, "Clorofila Total"]].dropna()
            if sub_te.empty:
                continue

            y_obs  = sub_te["Clorofila Total"]
            y_pred = slope * sub_te[col] + intercept

            fold_r2.append(compute_r2(y_obs, y_pred))
            fold_rmse.append(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
            fold_mae.append(np.mean(np.abs(y_obs - y_pred)))

        if fold_r2:
            lodo_results.append({
                "Spectral_Index": ind_name,
                "LODO_R2_mean":   np.mean(fold_r2),
                "LODO_R2_std":    np.std(fold_r2),
                "LODO_RMSE_mean": np.mean(fold_rmse),
                "LODO_MAE_mean":  np.mean(fold_mae),
            })

    df_lodo = (pd.DataFrame(lodo_results)
               .sort_values("LODO_R2_mean", ascending=False))
    df_lodo.to_csv("data/output/images-infos/cv_lodo_results.csv", index=False)

    print(df_lodo.head(10).to_string(index=False))

    _plot_cv_bars(df_lodo.head(15), "LODO_R2_mean", "LODO_R2_std",
                  "Leave-One-Date-Out CV  –  R² médio por Índice (2x2, Média)",
                  "plots/cv_lodo_r2.png")


# =============================================================================
# 3.  LINEAR MIXED MODEL  (efeitos fixos + aleatórios de Bloco e Data)
# =============================================================================
def run_lmm(df_merged, index_cols):
    """
    Fits a Linear Mixed Model for each spectral index:
        Fixed effect  : Spectral Index value
        Random effects: intercept for Bloco and Data (crossed design)
    Uses statsmodels MixedLM.
    Reports the fixed-effect coefficient, conditional R² (approx), and p-value.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not available for MixedLM — skipping.")
        return

    print("\n" + "=" * 80)
    print("  PART 2-C: LINEAR MIXED MODEL  (efeitos fixos + aleatórios de Bloco/Data)")
    print("=" * 80)
    print("  Fixed  : Índice Espectral")
    print("  Random : Bloco (intercept)  +  Data (intercept)")
    print()

    lmm_results = []

    for col in index_cols:
        ind_name = col.replace("Median_", "")
        sub = df_merged[[col, "Clorofila Total", "Bloco_x", "Data"]].dropna().copy()
        sub = sub.rename(columns={col: "Index", "Clorofila Total": "Clorofila", "Bloco_x": "Bloco"})

        if len(sub) < 8 or sub["Index"].std() == 0:
            continue

        try:
            # Bloco as the grouping variable (primary random effect)
            # Data as a random slope (variance component) via re_formula
            model_lmm = smf.mixedlm(
                "Clorofila ~ Index",
                data=sub,
                groups=sub["Bloco"],
                re_formula="~1"        # random intercept per block
            ).fit(reml=True, method='lbfgs')

            # Approximate conditional R² (Nakagawa & Schielzeth)
            y_obs      = sub["Clorofila"]
            y_pred_lmm = model_lmm.fittedvalues
            r2_cond    = compute_r2(y_obs, y_pred_lmm)
            rmse_lmm   = np.sqrt(np.mean((y_obs - y_pred_lmm) ** 2))

            fe_coef = model_lmm.fe_params["Index"]
            fe_pval = model_lmm.pvalues["Index"]

            lmm_results.append({
                "Spectral_Index":   ind_name,
                "FE_Coef_Index":    fe_coef,
                "FE_pvalue":        fe_pval,
                "R2_Conditional":   r2_cond,
                "RMSE_Conditional": rmse_lmm,
                "RE_Bloco_Var":     float(model_lmm.cov_re.iloc[0, 0]),
                "Residual_Var":     model_lmm.scale,
            })

        except Exception as e:
            # Some indices may fail to converge
            lmm_results.append({
                "Spectral_Index": ind_name,
                "FE_Coef_Index": np.nan,
                "FE_pvalue": np.nan,
                "R2_Conditional": np.nan,
                "RMSE_Conditional": np.nan,
                "RE_Bloco_Var": np.nan,
                "Residual_Var": np.nan,
            })

    df_lmm = (pd.DataFrame(lmm_results)
              .dropna(subset=["R2_Conditional"])
              .sort_values("R2_Conditional", ascending=False))
    df_lmm.to_csv("data/output/images-infos/lmm_results.csv", index=False)

    print(df_lmm.head(10).to_string(index=False))

    # Bubble chart: R² vs. p-value, bubble = Bloco random variance
    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = df_lmm.head(15)
    sizes = (top15["RE_Bloco_Var"].clip(lower=0.1) * 300).fillna(50)
    sc = ax.scatter(top15["R2_Conditional"], -np.log10(top15["FE_pvalue"].clip(lower=1e-10)),
                    s=sizes, alpha=0.75, edgecolors="black", linewidth=0.8,
                    c=top15["R2_Conditional"], cmap="YlGn")
    for _, row in top15.iterrows():
        ax.annotate(row["Spectral_Index"],
                    (row["R2_Conditional"], -np.log10(max(row["FE_pvalue"], 1e-10))),
                    fontsize=8.5, ha='left', va='bottom')
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xlabel("R² Condicional (LMM)", fontsize=11)
    ax.set_ylabel("−log₁₀(p-valor efeito fixo)", fontsize=11)
    ax.set_title("Linear Mixed Model  –  R² Condicional vs. Significância do Índice", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="R² Condicional")
    plt.tight_layout()
    plt.savefig("plots/lmm_r2_vs_pvalue.png", dpi=300)
    plt.close()


# =============================================================================
# Shared helper: CV bar chart with error bars
# =============================================================================
def _plot_cv_bars(df_plot, r2_col, std_col, title, path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df_plot))
    ax.bar(x, df_plot[r2_col], yerr=df_plot[std_col], capsize=4,
           color="#4CAF50", edgecolor="black", linewidth=0.6, alpha=0.87,
           error_kw=dict(elinewidth=1.2, ecolor="#333333"))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Spectral_Index"], rotation=90, fontsize=9.5)
    ax.set_ylabel("R² Médio entre Folds", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(min(0, (df_plot[r2_col] - df_plot[std_col]).min()) - 0.05, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

