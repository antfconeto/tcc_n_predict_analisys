import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from gplearn.genetic import SymbolicRegressor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.config import PONTO_TO_DBC as ponto_to_dbc
from tcc_analysis.data import indices_available, load_falker, load_indices
from tcc_analysis.db import plot_path


def run_systematic_search(df_train, df_val, stat):
    # Set of physically-meaningful algebraic templates
    r_tr, g_tr, b_tr = df_train[f"{stat}_r"].values, df_train[f"{stat}_g"].values, df_train[f"{stat}_b"].values
    r_v, g_v, b_v = df_val[f"{stat}_r"].values, df_val[f"{stat}_g"].values, df_val[f"{stat}_b"].values
    y_train = df_train["Clorofila Total"].values
    y_val = df_val["Clorofila Total"].values

    templates = {
        "r/b": lambda r,g,b: r / (b + 1e-8),
        "g/r": lambda r,g,b: g / (r + 1e-8),
        "g/b": lambda r,g,b: g / (b + 1e-8),
        "r/(g*b)": lambda r,g,b: r / (g*b + 1e-8),
        "g/(r*b)": lambda r,g,b: g / (r*b + 1e-8),
        "b/(r*g)": lambda r,g,b: b / (r*g + 1e-8),
        "r/(g*b^2)": lambda r,g,b: r / (g*b**2 + 1e-8),
        "(g-r)/(g+r)": lambda r,g,b: (g - r) / (g + r + 1e-8),
        "(g-b)/(g+b)": lambda r,g,b: (g - b) / (g + b + 1e-8),
        "(r-b)/(r+b)": lambda r,g,b: (r - b) / (r + b + 1e-8),
        "g/(r+b)": lambda r,g,b: g / (r + b + 1e-8),
        "r/(g+b)": lambda r,g,b: r / (g + b + 1e-8),
        "b/(r+g)": lambda r,g,b: b / (r + g + 1e-8),
        "(g-r)/b": lambda r,g,b: (g - r) / (b + 1e-8),
        "(g-b)/r": lambda r,g,b: (g - b) / (r + 1e-8),
        "(r-b)/g": lambda r,g,b: (r - b) / (g + 1e-8),
        "RGBVI": lambda r,g,b: (g**2 - r*b) / (g**2 + r*b + 1e-8),
        "VARI": lambda r,g,b: (g - r) / (g + r - b + 1e-8),
        "GLI": lambda r,g,b: (2*g - r - b) / (2*g + r + b + 1e-8),
        "(2g-r-b)": lambda r,g,b: 2*g - r - b,
        "(r+b-g*r*b)/(r+b+g*r*b)": lambda r,g,b: (r + b - g*r*b) / (r + b + g*r*b + 1e-8),
        "(r+b)/(g*r*b)": lambda r,g,b: (r + b) / (g*r*b + 1e-8),
    }

    results = []
    for name, fn in templates.items():
        try:
            val_tr = fn(r_tr, g_tr, b_tr)
            if np.any(np.isnan(val_tr)) or np.any(np.isinf(val_tr)) or np.std(val_tr) == 0:
                continue
            corr_tr, _ = pearsonr(val_tr, y_train)

            val_v = fn(r_v, g_v, b_v)
            if np.any(np.isnan(val_v)) or np.any(np.isinf(val_v)) or np.std(val_v) == 0:
                corr_val = np.nan
            else:
                corr_val, _ = pearsonr(val_v, y_val)

            results.append({
                "Formula": name,
                "Train_r": corr_tr,
                "Train_R2": corr_tr**2,
                "Val_r": corr_val,
                "Val_R2": corr_val**2 if not np.isnan(corr_val) else np.nan,
                "Type": "Systematic Search"
            })
        except:
            continue
    
    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("SYMBOLIC REGRESSION & PHYSICAL SEARCH FOR SPECTRAL INDICES")
    print("  -> Training Dates: 18-05-2026 and 26-05-2026")
    print("  -> Validation Dates: 21-05-2026 and 01-06-2026")
    print("=" * 80)

    # 1. Load chlorophyll data

    print("\nLoading chlorophyll data...")
    df_cloro = load_falker("cloro")
    df_test = load_falker("test")
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)

    # Add Treatment and Block info
    df_all_cloro["Tratamento"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro["Bloco"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    # Aggregate chlorophyll to plot-level averages
    df_chlo_plot = (df_all_cloro
                    .groupby(["Data", "Ponto"])
                    .agg({"Clorofila Total": "mean", "Bloco": "first", "Tratamento": "first"})
                    .reset_index())

    block_sizes = [10, 5, 2]
    overall_results = []

    # Run for each block size
    for bs in block_sizes:
        if not indices_available(bs):
            print(f"Skipping block size {bs}x{bs} (indices file not found)")
            continue

        print(f"\nProcessing Block Size: {bs}x{bs}...")
        df_indices = load_indices(bs)

        # Merge chlorophyll with image indices
        df_merged = pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])

        # Filter training and validation sets
        train_dates = ["26-05-2026"]
        val_dates = ["18-05-2026", "01-06-2026", "21-05-2026"]

        df_train = df_merged[df_merged["Data"].isin(train_dates)].copy()
        df_val = df_merged[df_merged["Data"].isin(val_dates)].copy()

        print(f"  Training samples (N={len(df_train)}) | Validation samples (N={len(df_val)})")

        if len(df_train) < 5:
            print("  Not enough training samples. Skipping.")
            continue

        # Check Mean and Median configurations
        for stat in ["Mean", "Median"]:
            feat_cols = [f"{stat}_r", f"{stat}_g", f"{stat}_b"]
            
            # Make sure the columns exist
            if not all(col in df_train.columns for col in feat_cols):
                continue

            X_train = df_train[feat_cols].values
            y_train = df_train["Clorofila Total"].values

            X_val = df_val[feat_cols].values
            y_val = df_val["Clorofila Total"].values

            # Initialize and fit ELEGANT, PHYSICALLY CONSTRAINED Symbolic Regressor
            # We restrict operators, remove constants, limit depth, and use high parsimony
            # to prevent weird high-power overfitted expressions.
            est = SymbolicRegressor(
                population_size=3000,
                generations=20,
                tournament_size=20,
                stopping_criteria=0.999,
                const_range=None, # No float constants!
                function_set=['add', 'sub', 'mul', 'div'],
                metric='pearson',
                init_depth=(1, 2),
                parsimony_coefficient=0.08, # High penalty for complex formulas
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )
            
            est.fit(X_train, y_train)
            
            # Evolved formula
            formula = est._program
            pred_train = est.predict(X_train)
            pred_val = est.predict(X_val) if len(X_val) > 0 else np.zeros_like(y_val)

            # Pearson r
            r_train, _ = pearsonr(pred_train, y_train)
            r_val, _ = pearsonr(pred_val, y_val) if len(X_val) > 2 else (np.nan, np.nan)

            formula_str = str(formula)
            formula_friendly = formula_str.replace("X0", "r").replace("X1", "g").replace("X2", "b")

            print(f"  [{stat} GP] Best Evolved Index: {formula_friendly}")
            print(f"    Train Pearson r: {r_train:.4f} (R² = {r_train**2:.4f})")
            if not np.isnan(r_val):
                print(f"    Val Pearson r:   {r_val:.4f} (R² = {r_val**2:.4f})")

            # Run physical systematic search to verify standard models
            df_sys = run_systematic_search(df_train, df_val, stat)
            df_sys_sorted = df_sys.sort_values(by="Train_R2", ascending=False)
            best_sys = df_sys_sorted.iloc[0]
            print(f"  [{stat} Search] Best Template: {best_sys['Formula']}")
            print(f"    Train R² = {best_sys['Train_R2']:.4f} | Val R² = {best_sys['Val_R2']:.4f}")

            # Baseline benchmark
            baseline_col = f"{stat}_r_over_b"
            if baseline_col in df_train.columns:
                r_base_tr, _ = pearsonr(df_train[baseline_col].values, y_train)
                r_base_val, _ = pearsonr(df_val[baseline_col].values, y_val) if len(X_val) > 2 else (np.nan, np.nan)
            else:
                r_base_tr, r_base_val = np.nan, np.nan

            # Save the best of both (GP vs Systematic)
            # If GP has comparable or better validation and training performance, use GP. Otherwise use the best systematic index
            gp_val_r2 = r_val**2 if not np.isnan(r_val) else 0.0
            sys_val_r2 = best_sys["Val_R2"] if not np.isnan(best_sys["Val_R2"]) else 0.0
            
            use_gp = (gp_val_r2 >= sys_val_r2 - 0.02) and (r_train**2 >= best_sys["Train_R2"] - 0.02)
            
            chosen_formula = formula_friendly if use_gp else best_sys["Formula"]
            chosen_train_r = r_train if use_gp else best_sys["Train_r"]
            chosen_val_r = r_val if use_gp else best_sys["Val_r"]

            overall_results.append({
                "Block_Size": bs,
                "Aggregation": stat,
                "GP_Formula": formula_friendly,
                "GP_Train_R2": r_train**2,
                "GP_Val_R2": gp_val_r2,
                "Sys_Formula": best_sys["Formula"],
                "Sys_Train_R2": best_sys["Train_R2"],
                "Sys_Val_R2": sys_val_r2,
                "Chosen_Formula": chosen_formula,
                "Chosen_Train_R2": chosen_train_r**2,
                "Chosen_Val_R2": chosen_val_r**2 if not np.isnan(chosen_val_r) else np.nan,
                "Baseline_Train_R2": r_base_tr**2 if not np.isnan(r_base_tr) else np.nan,
                "Baseline_Val_R2": r_base_val**2 if not np.isnan(r_base_val) else np.nan
            })

            # Plot chosen formula comparison
            plt.figure(figsize=(10, 5))
            
            # Helper to calculate chosen index values
            if use_gp:
                idx_tr = pred_train
                idx_val = pred_val
            else:
                # evaluate systematic formula
                fn_sys = {
                    "r/b": lambda r,g,b: r / (b + 1e-8),
                    "g/r": lambda r,g,b: g / (r + 1e-8),
                    "g/b": lambda r,g,b: g / (b + 1e-8),
                    "r/(g*b)": lambda r,g,b: r / (g*b + 1e-8),
                    "g/(r*b)": lambda r,g,b: g / (r*b + 1e-8),
                    "b/(r*g)": lambda r,g,b: b / (r*g + 1e-8),
                    "r/(g*b^2)": lambda r,g,b: r / (g*b**2 + 1e-8),
                    "(g-r)/(g+r)": lambda r,g,b: (g - r) / (g + r + 1e-8),
                    "(g-b)/(g+b)": lambda r,g,b: (g - b) / (g + b + 1e-8),
                    "(r-b)/(r+b)": lambda r,g,b: (r - b) / (r + b + 1e-8),
                    "g/(r+b)": lambda r,g,b: g / (r + b + 1e-8),
                    "r/(g+b)": lambda r,g,b: r / (g + b + 1e-8),
                    "b/(r+g)": lambda r,g,b: b / (r + g + 1e-8),
                    "(g-r)/b": lambda r,g,b: (g - r) / (b + 1e-8),
                    "(g-b)/r": lambda r,g,b: (g - b) / (r + 1e-8),
                    "(r-b)/g": lambda r,g,b: (r - b) / (g + 1e-8),
                    "RGBVI": lambda r,g,b: (g**2 - r*b) / (g**2 + r*b + 1e-8),
                    "VARI": lambda r,g,b: (g - r) / (g + r - b + 1e-8),
                    "GLI": lambda r,g,b: (2*g - r - b) / (2*g + r + b + 1e-8),
                    "(2g-r-b)": lambda r,g,b: 2*g - r - b,
                    "(r+b-g*r*b)/(r+b+g*r*b)": lambda r,g,b: (r + b - g*r*b) / (r + b + g*r*b + 1e-8),
                    "(r+b)/(g*r*b)": lambda r,g,b: (r + b) / (g*r*b + 1e-8),
                }[chosen_formula]
                r_tr, g_tr, b_tr = df_train[f"{stat}_r"].values, df_train[f"{stat}_g"].values, df_train[f"{stat}_b"].values
                r_v, g_v, b_v = df_val[f"{stat}_r"].values, df_val[f"{stat}_g"].values, df_val[f"{stat}_b"].values
                idx_tr = fn_sys(r_tr, g_tr, b_tr)
                idx_val = fn_sys(r_v, g_v, b_v)

            # Train plot
            plt.subplot(1, 2, 1)
            plt.scatter(idx_tr, y_train, color="#2E7D32", alpha=0.8, edgecolors='black', label="Training Plots")
            m_tr, b_tr = np.polyfit(idx_tr, y_train, 1)
            plt.plot(idx_tr, m_tr * idx_tr + b_tr, color="red", linestyle="--", label="Fit")
            plt.title(f"Train (18-05 & 26-05)\nPearson r: {chosen_train_r:.4f}")
            plt.xlabel("Index Value")
            plt.ylabel("Chlorophyll Total (Falker)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()

            # Val plot
            plt.subplot(1, 2, 2)
            if len(X_val) > 0:
                plt.scatter(idx_val, y_val, color="#1976D2", alpha=0.8, edgecolors='black', label="Validation Plots")
                if len(X_val) > 2:
                    m_v, b_v = np.polyfit(idx_val, y_val, 1)
                    plt.plot(idx_val, m_v * idx_val + b_v, color="red", linestyle="--", label="Fit")
                plt.title(f"Val (21-05 & 01-06)\nPearson r: {chosen_val_r:.4f}")
            else:
                plt.text(0.5, 0.5, "No Validation Data", ha='center', va='center')
            plt.xlabel("Index Value")
            plt.ylabel("Chlorophyll Total (Falker)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()

            plt.suptitle(f"Chosen Index: {chosen_formula} ({bs}x{bs} {stat})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            fig_path = plot_path(f"symbolic_regression_{bs}x{bs}_{stat}.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()

    # Save summary table
    df_results = pd.DataFrame(overall_results)
    db.save_df(df_results, "reg_sym_symbolic_summary",
               description="Regressão simbólica (gplearn) + busca sistemática de índices por grade × estatística")
    print("\nResults successfully saved to table reg_sym_symbolic_summary")

    # Output comparison
    print("\n" + "=" * 100)
    print("SUMMARY OF EVOLVED & DISCOVERED SPECTRAL INDICES")
    print("=" * 100)
    print(df_results[[
        "Block_Size", "Aggregation", "Chosen_Formula", 
        "Chosen_Train_R2", "Chosen_Val_R2", "Baseline_Val_R2"
    ]].to_string(index=False))

if __name__ == "__main__":
    db.run_main(main)
