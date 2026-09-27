import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.config import PONTO_TO_DBC as ponto_to_dbc
from tcc_analysis.data import indices_available, load_falker, load_indices
from tcc_analysis.db import plot_path


def _anova_rows(anova_df, date, variable, model):
    out = anova_df.reset_index(names="Fonte")
    out.insert(0, "Modelo", model)
    out.insert(0, "Variavel", variable)
    out.insert(0, "Data", date)
    return out


def run_n_dose_anova(df):
    """
    Runs ANOVA for Nitrogen Dose (Tratamento) and Block (Bloco) on Chlorophyll.
    Since 01-06 has no dose/fertilization, we run One-Way ANOVA of Ponto for 01-06.
    """
    print("\n" + "="*80)
    print("      PART 1: TWO-WAY DBC ANOVA FOR NITROGEN DOSES ON CHLOROPHYLL METRICS")
    print("="*80)
    
    dates = ["18-05-2026", "21-05-2026", "26-05-2026"]
    anova_tables = []
    
    for date in dates:
        df_date = df[df["Data"] == date].copy()
        if df_date.empty:
            print(f"No chlorophyll data found for date {date}!")
            continue
            
        # Map DBC layout
        df_date["Bloco"] = df_date["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
        df_date["Tratamento"] = df_date["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
        df_date["Dose"] = df_date["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Dose"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
        
        # Drop missing values
        df_date_clean = df_date.dropna(subset=["Bloco", "Tratamento", "Clorofila Total"])
        
        print(f"\n📊 --- ANOVA FOR DATE: {date} (DBC Layout, N={len(df_date_clean)}) ---")
        
        # Fit model for Clorofila Total
        model_total = ols('Q("Clorofila Total") ~ C(Bloco) + C(Tratamento)', data=df_date_clean).fit()
        anova_total = sm.stats.anova_lm(model_total, typ=2)
        print("\nDependent Variable: Clorofila Total")
        print(anova_total.round(5))
        anova_tables.append(_anova_rows(anova_total, date, "Clorofila Total", "DBC: Bloco + Tratamento"))
        
        # Fit model for Clorofila A
        model_a = ols('Q("Clorofila A") ~ C(Bloco) + C(Tratamento)', data=df_date_clean).fit()
        anova_a = sm.stats.anova_lm(model_a, typ=2)
        print("\nDependent Variable: Clorofila A")
        print(anova_a.round(5))
        anova_tables.append(_anova_rows(anova_a, date, "Clorofila A", "DBC: Bloco + Tratamento"))
        
        # Conclusion
        p_trat = anova_total.loc["C(Tratamento)", "PR(>F)"]
        if p_trat < 0.05:
            print(f"🟢 Conclusion: Statistically significant effect of N Doses on Total Chlorophyll (p = {p_trat:.5f} < 0.05)")
        else:
            print(f"🔴 Conclusion: No significant effect of N Doses on Total Chlorophyll (p = {p_trat:.5f} >= 0.05)")
            
    # For 01-06-2026 (test.csv) - no doses, run One-Way ANOVA on Point (Ponto)
    df_test = df[df["Data"] == "01-06-2026"].copy()
    if not df_test.empty:
        df_test_clean = df_test.dropna(subset=["Ponto", "Clorofila Total"])
        print(f"\n📊 --- ONE-WAY ANOVA FOR DATE: 01-06-2026 (No N Doses, N={len(df_test_clean)}) ---")
        model_test = ols('Q("Clorofila Total") ~ C(Ponto)', data=df_test_clean).fit()
        anova_test = sm.stats.anova_lm(model_test, typ=1)
        print("Dependent Variable: Clorofila Total vs Points (P1, P2, P3)")
        print(anova_test.round(5))
        anova_tables.append(_anova_rows(anova_test, "01-06-2026", "Clorofila Total", "Um fator: Ponto"))
        p_ponto = anova_test.loc["C(Ponto)", "PR(>F)"]
        if p_ponto < 0.05:
            print(f"🟢 Conclusion: Significant difference in chlorophyll between the points (p = {p_ponto:.5f} < 0.05)")
        else:
            print(f"🔴 Conclusion: No significant difference in chlorophyll between the points (p = {p_ponto:.5f} >= 0.05)")

    if anova_tables:
        db.save_df(pd.concat(anova_tables, ignore_index=True), "anova_n_dose",
                   description="ANOVA do efeito da dose de N (DBC) sobre a clorofila, por data")

def time_to_hours(time_str):
    """Converts HH:MM:SS string to decimal hours."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    try:
        parts = time_str.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return h + m / 60.0 + s / 3600.0
    except Exception:
        return None

def run_indices_regression_anova(df_chlo, block_sizes=[10, 5, 2]):
    """
    Fits linear regressions between chlorophyll (Total, A, B) and every spectral index,
    under 5 different treatment aggregation strategies: Mean, Median, P75, P90, and STD.
    """
    print("\n" + "="*80)
    print("      PART 2: MULTI-STATISTIC REGRESSION ANOVA AND CORRELATION")
    print("="*80)
    
    # Map Tratamento to each reading in df_chlo
    df_chlo["Tratamento"] = df_chlo["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    
    # Drop rows without Tratamento (this automatically filters to the 3 experimental dates)
    df_chlo_clean = df_chlo.dropna(subset=["Tratamento"]).copy()
    
    # Convert Hora to decimal hours
    df_chlo_clean["Chlo_Hour"] = df_chlo_clean["Hora"].apply(time_to_hours)
    
    stats_methods = {
        "mean": {
            "label": "Média",
            "chlo_func": lambda x: x.mean(numeric_only=True),
            "idx_func": lambda x: x.mean(numeric_only=True),
        },
        "median": {
            "label": "Mediana",
            "chlo_func": lambda x: x.median(numeric_only=True),
            "idx_func": lambda x: x.median(numeric_only=True),
        },
        "p75": {
            "label": "Percentil 75",
            "chlo_func": lambda x: x.quantile(0.75, numeric_only=True),
            "idx_func": lambda x: x.quantile(0.75, numeric_only=True),
        },
        "p90": {
            "label": "Percentil 90",
            "chlo_func": lambda x: x.quantile(0.90, numeric_only=True),
            "idx_func": lambda x: x.quantile(0.90, numeric_only=True),
        },
        "std": {
            "label": "Desvio Padrão",
            "chlo_func": lambda x: x.std(numeric_only=True),
            "idx_func": lambda x: x.std(numeric_only=True),
        }
    }
    
    overall_summary = []
    all_regressions = []
    
    for method_name, method_info in stats_methods.items():
        label = method_info["label"]
        print(f"\n📊 --- Aggregation Method: {label} ---")
        
        # Group chlorophyll readings
        df_chlo_grouped = df_chlo_clean.groupby(["Data", "Tratamento"])[["Clorofila A", "Clorofila B", "Clorofila Total"]].apply(method_info["chlo_func"]).reset_index()
        df_chlo_time_grouped = df_chlo_clean.groupby(["Data", "Tratamento"])["Chlo_Hour"].median().reset_index()
        df_chlo_merged = pd.merge(df_chlo_grouped, df_chlo_time_grouped, on=["Data", "Tratamento"])
        
        for bs in block_sizes:
            if not indices_available(bs):
                continue
                
            df_indices = load_indices(bs)
            df_indices["Photo_Hour_Raw"] = df_indices["Photo_Time"].apply(time_to_hours)
            
            index_cols = [c for c in df_indices.columns if c.startswith("Median_")]
            
            # Group indices
            df_ind_grouped = df_indices.groupby(["Data", "Tratamento"])[index_cols].apply(method_info["idx_func"]).reset_index()
            df_time_grouped = df_indices.groupby(["Data", "Tratamento"])["Photo_Hour_Raw"].median().reset_index()
            df_indices_grouped = pd.merge(df_ind_grouped, df_time_grouped, on=["Data", "Tratamento"])
            
            # Merge chlorophyll and image indices
            df_merged = pd.merge(df_chlo_merged, df_indices_grouped, on=["Data", "Tratamento"])
            
            if df_merged.empty:
                continue
                
            valid_diffs = df_merged["Chlo_Hour"] - df_merged["Photo_Hour_Raw"]
            mean_delay = valid_diffs.mean() if not valid_diffs.dropna().empty else 2.64
            df_merged["Photo_Hour"] = df_merged["Photo_Hour_Raw"].fillna(df_merged["Chlo_Hour"] - mean_delay)
            
            chlo_targets = ["Clorofila Total", "Clorofila A", "Clorofila B"]
            reg_results = []
            
            for chlo in chlo_targets:
                for ind in index_cols:
                    df_sub = df_merged[[chlo, ind, "Photo_Hour"]].dropna()
                    if len(df_sub) < 4:
                        continue
                        
                    x_ind = df_sub[ind]
                    x_time = df_sub["Photo_Hour"]
                    y = df_sub[chlo]
                    
                    # Simple linear model
                    X_simple = sm.add_constant(x_ind)
                    model_simple = sm.OLS(y, X_simple).fit()
                    
                    # Multiple linear model
                    X_multi = sm.add_constant(df_sub[[ind, "Photo_Hour"]])
                    model_multi = sm.OLS(y, X_multi).fit()
                    
                    # Pearson correlations (with constant array protection)
                    r_photo, p_photo_corr = pearsonr(x_ind, x_time) if len(x_ind.unique()) > 1 and len(x_time.unique()) > 1 else (0.0, 1.0)
                    r_chlo, p_chlo_corr = pearsonr(x_ind, y) if len(x_ind.unique()) > 1 and len(y.unique()) > 1 else (0.0, 1.0)
                    
                    reg_results.append({
                        "Chlorophyll_Metric": chlo,
                        "Spectral_Index": ind.replace("Median_", ""),
                        "Pearson_r_Chlorophyll": r_chlo,
                        "Simple_R_squared": model_simple.rsquared,
                        "Multiple_R_squared": model_multi.rsquared,
                        "R_squared_Improvement": model_multi.rsquared - model_simple.rsquared,
                        "Index_p_value_in_Multi": model_multi.pvalues[ind],
                        "PhotoHour_p_value_in_Multi": model_multi.pvalues["Photo_Hour"] if "Photo_Hour" in model_multi.pvalues else 1.0,
                        "Pearson_r_PhotoHour": r_photo,
                        "PhotoHour_Correlation_p_value": p_photo_corr,
                        "Is_Time_Sensitive": p_photo_corr < 0.05
                    })
                    
            df_reg = pd.DataFrame(reg_results)
            
            all_regressions.append(df_reg.assign(Grade=f"{bs}x{bs}", Estatistica=method_name))
            
            # Identify champion
            df_total = df_reg[df_reg["Chlorophyll_Metric"] == "Clorofila Total"].copy()
            df_total["abs_r"] = df_total["Pearson_r_Chlorophyll"].abs()
            df_total = df_total.sort_values("abs_r", ascending=False)
            
            if not df_total.empty:
                champ_row = df_total.iloc[0]
                overall_summary.append({
                    "Grade": f"{bs}x{bs}",
                    "Estatistica": label,
                    "Champ_Index": champ_row["Spectral_Index"],
                    "Pearson_r": champ_row["Pearson_r_Chlorophyll"],
                    "R_squared": champ_row["Simple_R_squared"]
                })
                
            # Plot
            plt.figure(figsize=(16, 7.5))
            df_plot = df_total.sort_values("Multiple_R_squared", ascending=False)
            x_arr = np.arange(len(df_plot))
            width = 0.35
            
            plt.bar(x_arr - width/2, df_plot["Simple_R_squared"], width, label="Simple R² (Index Only)", color="#4CAF50", edgecolor='black', linewidth=0.5, alpha=0.85)
            plt.bar(x_arr + width/2, df_plot["Multiple_R_squared"], width, label="Multiple R² (+ Photo Hour)", color="#FF9800", edgecolor='black', linewidth=0.5, alpha=0.85)
            
            plt.axhline(0.0, color='gray', linewidth=0.8)
            plt.title(f"R² predicting Clorofila Total ({bs}x{bs}) - Aggregation: {label}", fontsize=13, fontweight='bold', pad=15)
            plt.xlabel("Spectral Index", fontsize=11, labelpad=8)
            plt.ylabel("Coefficient of Determination (R²)", fontsize=11, labelpad=8)
            plt.xticks(x_arr, df_plot["Spectral_Index"], rotation=90, ha='center', fontsize=9.5)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc='upper right', frameon=True, fontsize=10.5)
            plt.ylim(0, 1.0)
            
            plt.tight_layout()
            fig_path = plot_path(f"anova_chlorophyll_indices_{bs}x{bs}_r2_{method_name}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    db.save_df(pd.concat(all_regressions, ignore_index=True), "anova_indices",
               description="Regressão clorofila × índice (simples e + hora da foto) por grade × estatística")

    df_summary = pd.DataFrame(overall_summary)
    db.save_df(df_summary, "anova_multi_statistic_summary",
               description="Índice campeão (maior |r|) por grade × estatística")
    
    # Print the overall comparison table
    print("\n" + "="*80)
    print("      SUMMARY OF CHAMPION INDICES BY AGGREGATION METHOD")
    print("="*80)
    print(df_summary.to_string(index=False))

def main():
    # 1. Load data
    print("Parsing chlorophyll datasets...")
    df_cloro = load_falker("cloro")
    df_test = load_falker("test")
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)
    print(f"Total Chlorophyll observations loaded: {len(df_all_cloro)}")

    # 2. Run N Dose DBC ANOVA
    run_n_dose_anova(df_all_cloro)

    # 3. Run indices regression ANOVA with Photo Hour
    run_indices_regression_anova(df_all_cloro)


if __name__ == "__main__":
    db.run_main(main)
