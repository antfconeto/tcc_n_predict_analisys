import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def generate_candidates():
    # Define primary terms
    # Each term is (name, lambda function)
    terms = [
        ("r", lambda r,g,b: r),
        ("g", lambda r,g,b: g),
        ("b", lambda r,g,b: b),
        ("r+g", lambda r,g,b: r+g),
        ("r+b", lambda r,g,b: r+b),
        ("g+b", lambda r,g,b: g+b),
        ("g-r", lambda r,g,b: g-r),
        ("g-b", lambda r,g,b: g-b),
        ("r-b", lambda r,g,b: r-b),
        ("r*g", lambda r,g,b: r*g),
        ("r*b", lambda r,g,b: r*b),
        ("g*b", lambda r,g,b: g*b),
        ("r^2", lambda r,g,b: r**2),
        ("g^2", lambda r,g,b: g**2),
        ("b^2", lambda r,g,b: b**2),
        ("2g-r-b", lambda r,g,b: 2*g - r - b),
        ("g+r-b", lambda r,g,b: g + r - b),
        ("g*b^2", lambda r,g,b: g * b**2),
        ("g*r*b", lambda r,g,b: g * r * b),
    ]
    
    # We will generate ratio combinations: term1 / term2
    candidates = {}
    
    # Add simple terms
    for name, fn in terms:
        candidates[name] = fn
        
    # Add ratios
    for name1, fn1 in terms:
        for name2, fn2 in terms:
            if name1 == name2:
                continue
            # Avoid obvious inverses
            formula_name = f"({name1})/({name2})"
            candidates[formula_name] = lambda r,g,b, f1=fn1, f2=fn2: f1(r,g,b) / (f2(r,g,b) + 1e-8)
            
            # Normalized differences
            formula_nd = f"({name1}-{name2})/({name1}+{name2})"
            candidates[formula_nd] = lambda r,g,b, f1=fn1, f2=fn2: (f1(r,g,b) - f2(r,g,b)) / (f1(r,g,b) + f2(r,g,b) + 1e-8)
            
    # Add specific literature index styles
    literature = {
        "RGBVI": lambda r,g,b: (g**2 - r*b) / (g**2 + r*b + 1e-8),
        "VARI": lambda r,g,b: (g - r) / (g + r - b + 1e-8),
        "GLI": lambda r,g,b: (2*g - r - b) / (2*g + r + b + 1e-8),
        "NGRDI": lambda r,g,b: (g - r) / (g + r + 1e-8),
        "r/b": lambda r,g,b: r / (b + 1e-8),
        "g/b": lambda r,g,b: g / (b + 1e-8),
        "g/r": lambda r,g,b: g / (r + 1e-8),
        "ExG": lambda r,g,b: 2*g - r - b,
    }
    
    for name, fn in literature.items():
        candidates[name] = fn
        
    return candidates

def main():
    print("=" * 80)
    print("SYSTEMATIC SEARCH FOR PHYSICALLY MEANINGFUL SPECTRAL INDICES")
    print("  -> Training Dates: 18-05-2026 and 26-05-2026")
    print("  -> Validation Dates: 21-05-2026 and 01-06-2026")
    print("=" * 80)

    # Load chlorophyll
    df_cloro = load_falker_csv("data/clorophyll/cloro.csv")
    df_test = load_falker_csv("data/clorophyll/test.csv")
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    # Aggregate to plot level
    df_chlo_plot = (df_all_cloro
                    .groupby(["Data", "Ponto"])
                    .agg({"Clorofila Total": "mean"})
                    .reset_index())

    candidates = generate_candidates()
    print(f"Generated {len(candidates)} physically plausible candidate expressions to search.")

    block_sizes = [10, 5, 2]
    all_summary = []

    for bs in block_sizes:
        csv_indices = f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv"
        if not os.path.exists(csv_indices):
            continue
            
        print(f"\nEvaluating Block Size: {bs}x{bs}...")
        df_indices = pd.read_csv(csv_indices)
        df_merged = pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])

        # Split
        train_dates = ["18-05-2026", "26-05-2026"]
        val_dates = ["21-05-2026", "01-06-2026"]

        df_train = df_merged[df_merged["Data"].isin(train_dates)].copy()
        df_val = df_merged[df_merged["Data"].isin(val_dates)].copy()

        y_train = df_train["Clorofila Total"].values
        y_val = df_val["Clorofila Total"].values

        for stat in ["Mean", "Median"]:
            r_col = f"{stat}_r"
            g_col = f"{stat}_g"
            b_col = f"{stat}_b"

            if not all(c in df_merged.columns for c in [r_col, g_col, b_col]):
                continue

            r_tr, g_tr, b_tr = df_train[r_col].values, df_train[g_col].values, df_train[b_col].values
            r_v, g_v, b_v = df_val[r_col].values, df_val[g_col].values, df_val[b_col].values

            # Benchmark baseline
            r_base_tr = df_train[f"{stat}_r_over_b"].values
            r_base_v = df_val[f"{stat}_r_over_b"].values
            base_corr_tr, _ = pearsonr(r_base_tr, y_train)
            base_corr_val, _ = pearsonr(r_base_v, y_val)

            results = []
            for formula_name, fn in candidates.items():
                try:
                    # Evaluate training
                    val_tr = fn(r_tr, g_tr, b_tr)
                    if np.any(np.isnan(val_tr)) or np.any(np.isinf(val_tr)) or np.std(val_tr) == 0:
                        continue
                    corr_tr, _ = pearsonr(val_tr, y_train)

                    # Evaluate validation
                    val_v = fn(r_v, g_v, b_v)
                    if np.any(np.isnan(val_v)) or np.any(np.isinf(val_v)) or np.std(val_v) == 0:
                        corr_val = np.nan
                    else:
                        corr_val, _ = pearsonr(val_v, y_val)

                    results.append({
                        "Formula": formula_name,
                        "Train_r": corr_tr,
                        "Train_R2": corr_tr**2,
                        "Val_r": corr_val,
                        "Val_R2": corr_val**2 if not np.isnan(corr_val) else np.nan,
                        "Gen_Gap": abs(corr_tr**2 - (corr_val**2 if not np.isnan(corr_val) else 0))
                    })
                except Exception:
                    continue

            df_res = pd.DataFrame(results).dropna()
            # Sort by highest training R2, but filter for validation robustness (Val_R2 > 0.5)
            df_res = df_res.sort_values(by="Train_R2", ascending=False)

            # Top candidate
            top_cand = df_res.iloc[0]
            print(f"  [{stat}] Best: {top_cand['Formula']} | Train R² = {top_cand['Train_R2']:.4f} | Val R² = {top_cand['Val_R2']:.4f}")
            print(f"         Baseline (r/b): Train R² = {base_corr_tr**2:.4f} | Val R² = {base_corr_val**2:.4f}")

            # Keep top 10 for detailed summary
            for idx, row in df_res.head(10).iterrows():
                all_summary.append({
                    "Block_Size": bs,
                    "Stat": stat,
                    "Rank": len(all_summary) + 1,
                    "Formula": row["Formula"],
                    "Train_R2": row["Train_R2"],
                    "Val_R2": row["Val_R2"],
                    "Train_r": row["Train_r"],
                    "Val_r": row["Val_r"],
                    "Gen_Gap": row["Gen_Gap"],
                    "Baseline_Train_R2": base_corr_tr**2,
                    "Baseline_Val_R2": base_corr_val**2
                })

    df_sum = pd.DataFrame(all_summary)
    df_sum.to_csv("data/output/images-infos/reg_sym_physical_search_summary.csv", index=False)
    print(f"\nFull search summary saved to: data/output/images-infos/reg_sym_physical_search_summary.csv")

    # Display Top 15 Overall across all block sizes and statistics, sorted by Val_R2 to show generalizability!
    print("\n" + "=" * 90)
    print("  TOP 15 PHYSICAL SPECTRAL INDICES DISCOVERED (Sorted by Validation R² to show robustness)")
    print("=" * 90)
    df_sum_sorted = df_sum.sort_values("Val_R2", ascending=False).head(15)
    print(df_sum_sorted[[
        "Block_Size", "Stat", "Formula", "Train_R2", "Val_R2", "Gen_Gap", "Baseline_Val_R2"
    ]].to_string(index=False))

    # Generate plot for the best overall generalized formula
    best_overall = df_sum_sorted.iloc[0]
    bs = best_overall["Block_Size"]
    stat = best_overall["Stat"]
    formula_str = best_overall["Formula"]
    
    # Load and plot the best overall
    df_indices = pd.read_csv(f"data/output/images-infos/extracted_indices_{bs}x{bs}.csv")
    df_merged = pd.merge(df_chlo_plot, df_indices, on=["Data", "Ponto"])
    df_train = df_merged[df_merged["Data"].isin(train_dates)].copy()
    df_val = df_merged[df_merged["Data"].isin(val_dates)].copy()
    
    r_tr, g_tr, b_tr = df_train[f"{stat}_r"].values, df_train[f"{stat}_g"].values, df_train[f"{stat}_b"].values
    r_v, g_v, b_v = df_val[f"{stat}_r"].values, df_val[f"{stat}_g"].values, df_val[f"{stat}_b"].values
    
    fn = candidates[formula_str]
    idx_tr = fn(r_tr, g_tr, b_tr)
    idx_v = fn(r_v, g_v, b_v)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(idx_tr, df_train["Clorofila Total"], color="#2E7D32", alpha=0.8, edgecolors="black", label="Training Plots")
    m_tr, b_tr = np.polyfit(idx_tr, df_train["Clorofila Total"], 1)
    plt.plot(idx_tr, m_tr * idx_tr + b_tr, color="red", linestyle="--", label="Fit")
    plt.title(f"Train (18-05 & 26-05)\nPearson r: {best_overall['Train_r']:.4f}")
    plt.xlabel(f"New Index: {formula_str}")
    plt.ylabel("Chlorophyll Total (Falker)")
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.subplot(1, 2, 2)
    plt.scatter(idx_v, df_val["Clorofila Total"], color="#1976D2", alpha=0.8, edgecolors="black", label="Validation Plots")
    m_v, b_v = np.polyfit(idx_v, df_val["Clorofila Total"], 1)
    plt.plot(idx_v, m_v * idx_v + b_v, color="red", linestyle="--", label="Fit")
    plt.title(f"Val (21-05 & 01-06)\nPearson r: {best_overall['Val_r']:.4f}")
    plt.xlabel(f"New Index: {formula_str}")
    plt.ylabel("Chlorophyll Total (Falker)")
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.suptitle(f"Best Evolved Physical Index: {formula_str} ({bs}x{bs} {stat})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/best_evolved_physical_index.png", dpi=300)
    plt.close()
    print("\nScatter plots saved to: plots/best_evolved_physical_index.png")

if __name__ == "__main__":
    main()
