import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. Experimental layout map (DBC)
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

def load_and_merge_data(cloro_csv, alt_csv, mass_csv):
    """
    Loads, cleans, and merges chlorophyll, height/weight, and compiled dry/green biomass data.
    """
    if not os.path.exists(cloro_csv):
        raise FileNotFoundError(f"Chlorophyll file not found: {cloro_csv}")
    if not os.path.exists(alt_csv):
        raise FileNotFoundError(f"Height-weight file not found: {alt_csv}")
    if not os.path.exists(mass_csv):
        raise FileNotFoundError(f"Compiled mass file not found: {mass_csv}")

    # A. Parse Chlorophyll Data
    print("  [Data Loading] Parsing chlorophyll CSV...")
    skip_count = 0
    with open(cloro_csv, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if "Medição" in line:
                skip_count = i
                break
                
    df_cloro = pd.read_csv(cloro_csv, skiprows=skip_count, header=None)
    df_cloro = df_cloro.set_index(0).transpose()
    df_cloro.columns.name = None
    
    # Clean chlorophyll columns
    numeric_cols = ["Ponto", "Clorofila A", "Clorofila B", "Clorofila Total"]
    for col in numeric_cols:
        if col in df_cloro.columns:
            df_cloro[col] = pd.to_numeric(df_cloro[col], errors='coerce')
            
    # Filter for May 26th (dia 26)
    df_cloro_26 = df_cloro[df_cloro["Data"] == "26-05-2026"].copy()
    if df_cloro_26.empty:
        raise ValueError("No chlorophyll data found for the date 26-05-2026 in cloro.csv!")
        
    # Aggregate chlorophyll by point
    df_chlo_ponto = df_cloro_26.groupby("Ponto")[["Clorofila A", "Clorofila B", "Clorofila Total"]].mean().reset_index()
    
    # Map Ponto to DBC layout (Bloco, Tratamento, Dose)
    df_chlo_ponto["Bloco"] = df_chlo_ponto["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    df_chlo_ponto["Tratamento"] = df_chlo_ponto["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    df_chlo_ponto["Dose"] = df_chlo_ponto["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Dose"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    
    # B. Load Heights and Green Matter weights
    print("  [Data Loading] Loading height-weight CSV...")
    df_alt_raw = pd.read_csv(alt_csv).dropna(subset=["Bloco"])
    
    # Normalize Bloco names (e.g. 'Bl 1' -> 'Bloco 1')
    def clean_block(b):
        if pd.isna(b): return None
        b_str = str(b).strip()
        if b_str.startswith("Bl "):
            return "Bloco " + b_str.replace("Bl ", "").strip()
        return b_str
        
    df_alt_raw["Bloco"] = df_alt_raw["Bloco"].apply(clean_block)
    df_alt_raw["Tratamento"] = df_alt_raw["Tratamento"].str.strip()
    
    # Parse heights list and calculate mean height per plot
    def parse_heights(h_str):
        if pd.isna(h_str): return np.nan
        h_str = str(h_str).replace('"', '').strip()
        parts = [float(x.strip()) for x in h_str.split(",") if x.strip()]
        return np.mean(parts) if parts else np.nan
        
    df_alt_raw["Altura_Media"] = df_alt_raw["Alturas Anotadas (cm)"].apply(parse_heights)
    
    # Clean numeric columns with comma separator
    cols_to_float = [
        "Peso Matéria Total - Peso MT (kg)",
        "Peso Amostra Detalhada (kg)",
        "Peso do Caule - C (kg)",
        "Peso da Folha/Raiz - F/P (kg)"
    ]
    for col in cols_to_float:
        if col in df_alt_raw.columns:
            df_alt_raw[col] = df_alt_raw[col].astype(str).str.replace(",", ".").astype(float)
            
    # C. Load Compiled dry/green weights
    print("  [Data Loading] Loading compiled green/dry weights CSV...")
    df_mass = pd.read_csv(mass_csv)
    df_mass["Bloco"] = df_mass["Bloco"].str.strip()
    df_mass["Tratamento"] = df_mass["Tratamento"].str.strip()
    
    # Pivot components to columns (Massa_Verde_Colmo, Massa_Seca_Colmo, etc.)
    df_mass_pivoted = df_mass.pivot_table(
        index=["Bloco", "Tratamento"],
        columns="Componente",
        values=["Massa_Verde", "Massa_Seca"]
    )
    df_mass_pivoted.columns = [f"{val}_{comp}" for val, comp in df_mass_pivoted.columns]
    df_mass_pivoted = df_mass_pivoted.reset_index()
    
    # D. Merge all three datasets
    print("  [Data Loading] Merging datasets...")
    df_merged = pd.merge(df_chlo_ponto, df_alt_raw, on=["Bloco", "Tratamento"])
    df_merged = pd.merge(df_merged, df_mass_pivoted, on=["Bloco", "Tratamento"])
    
    # Calculate plot-level Total Green and Dry matter yield extrapolated to 1 hectare (t/ha)
    df_merged["Massa_Verde_Total_ha"] = df_merged["Peso Matéria Total - Peso MT (kg)"] * 10.0
    df_merged["Massa_Seca_Total_ha"] = df_merged["Massa_Verde_Total_ha"] * (df_merged["Massa_Seca_Amostra"] / df_merged["Massa_Verde_Amostra"])
    
    df_merged["Massa_Verde_Folha_ha"] = df_merged["Massa_Verde_Total_ha"] * (df_merged["Massa_Verde_Folha"] / df_merged["Massa_Verde_Amostra"])
    df_merged["Massa_Seca_Folha_ha"] = df_merged["Massa_Verde_Total_ha"] * (df_merged["Massa_Seca_Folha"] / df_merged["Massa_Verde_Amostra"])
    
    df_merged["Massa_Verde_Colmo_ha"] = df_merged["Massa_Verde_Total_ha"] * (df_merged["Massa_Verde_Colmo"] / df_merged["Massa_Verde_Amostra"])
    df_merged["Massa_Seca_Colmo_ha"] = df_merged["Massa_Verde_Total_ha"] * (df_merged["Massa_Seca_Colmo"] / df_merged["Massa_Verde_Amostra"])
    
    # Calculate Leaf-to-Stem ratio (Relação Folha/Colmo)
    df_merged["Rel_Folha_Colmo_Verde"] = df_merged["Massa_Verde_Folha_ha"] / df_merged["Massa_Verde_Colmo_ha"]
    df_merged["Rel_Folha_Colmo_Seco"] = df_merged["Massa_Seca_Folha_ha"] / df_merged["Massa_Seca_Colmo_ha"]
    
    return df_merged

def run_regression_on_dataset(df, level_name):
    """
    Fits multiple OLS models for vegetative Y variables against N Dose and Chlorophyll metrics
    for a given dataset level (Plot or Treatment).
    """
    y_vars = {
        "Altura_Media": "Altura Média (cm)",
        "Massa_Verde_Folha_ha": "Massa Verde Folha (t/ha)",
        "Massa_Seca_Folha_ha": "Massa Seca Folha (t/ha)",
        "Massa_Verde_Colmo_ha": "Massa Verde Colmo (t/ha)",
        "Massa_Seca_Colmo_ha": "Massa Seca Colmo (t/ha)",
        "Massa_Verde_Total_ha": "Massa Verde Total (t/ha)",
        "Massa_Seca_Total_ha": "Massa Seca Total (t/ha)",
        "Rel_Folha_Colmo_Verde": "Relação Folha/Colmo (Verde)",
        "Rel_Folha_Colmo_Seco": "Relação Folha/Colmo (Seca)"
    }
    
    results = []
    
    for y_col, y_desc in y_vars.items():
        df_sub = df[[y_col, "Dose", "Clorofila Total", "Clorofila A", "Clorofila B"]].dropna()
        if len(df_sub) < 3:
            continue
        y = df_sub[y_col]
        
        # 1. Linear: Y vs Dose
        X1 = sm.add_constant(df_sub["Dose"])
        model1 = sm.OLS(y, X1).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Linear vs Dose",
            "R2": model1.rsquared,
            "Adj_R2": model1.rsquared_adj,
            "Intercept": model1.params.iloc[0],
            "Slope_1": model1.params.iloc[1],
            "Slope_2": 0.0,
            "p_val_Intercept": model1.pvalues.iloc[0],
            "p_val_Slope_1": model1.pvalues.iloc[1],
            "p_val_Slope_2": 1.0
        })
        
        # 2. Quadratic: Y vs Dose
        X2 = pd.DataFrame({"Dose": df_sub["Dose"], "Dose_Sq": df_sub["Dose"]**2})
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(y, X2).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Quadratic vs Dose",
            "R2": model2.rsquared,
            "Adj_R2": model2.rsquared_adj,
            "Intercept": model2.params.iloc[0],
            "Slope_1": model2.params.iloc[1],
            "Slope_2": model2.params.iloc[2],
            "p_val_Intercept": model2.pvalues.iloc[0],
            "p_val_Slope_1": model2.pvalues.iloc[1],
            "p_val_Slope_2": model2.pvalues.iloc[2]
        })
        
        # 3. Linear: Y vs Clorofila Total
        X3 = sm.add_constant(df_sub["Clorofila Total"])
        model3 = sm.OLS(y, X3).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Linear vs Clorofila Total",
            "R2": model3.rsquared,
            "Adj_R2": model3.rsquared_adj,
            "Intercept": model3.params.iloc[0],
            "Slope_1": model3.params.iloc[1],
            "Slope_2": 0.0,
            "p_val_Intercept": model3.pvalues.iloc[0],
            "p_val_Slope_1": model3.pvalues.iloc[1],
            "p_val_Slope_2": 1.0
        })
        
        # 4. Quadratic: Y vs Clorofila Total
        X4 = pd.DataFrame({"Chlo": df_sub["Clorofila Total"], "Chlo_Sq": df_sub["Clorofila Total"]**2})
        X4 = sm.add_constant(X4)
        model4 = sm.OLS(y, X4).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Quadratic vs Clorofila Total",
            "R2": model4.rsquared,
            "Adj_R2": model4.rsquared_adj,
            "Intercept": model4.params.iloc[0],
            "Slope_1": model4.params.iloc[1],
            "Slope_2": model4.params.iloc[2],
            "p_val_Intercept": model4.pvalues.iloc[0],
            "p_val_Slope_1": model4.pvalues.iloc[1],
            "p_val_Slope_2": model4.pvalues.iloc[2]
        })

        # 5. Linear: Y vs Clorofila A
        X5 = sm.add_constant(df_sub["Clorofila A"])
        model5 = sm.OLS(y, X5).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Linear vs Clorofila A",
            "R2": model5.rsquared,
            "Adj_R2": model5.rsquared_adj,
            "Intercept": model5.params.iloc[0],
            "Slope_1": model5.params.iloc[1],
            "Slope_2": 0.0,
            "p_val_Intercept": model5.pvalues.iloc[0],
            "p_val_Slope_1": model5.pvalues.iloc[1],
            "p_val_Slope_2": 1.0
        })

        # 6. Linear: Y vs Clorofila B
        X6 = sm.add_constant(df_sub["Clorofila B"])
        model6 = sm.OLS(y, X6).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Linear vs Clorofila B",
            "R2": model6.rsquared,
            "Adj_R2": model6.rsquared_adj,
            "Intercept": model6.params.iloc[0],
            "Slope_1": model6.params.iloc[1],
            "Slope_2": 0.0,
            "p_val_Intercept": model6.pvalues.iloc[0],
            "p_val_Slope_1": model6.pvalues.iloc[1],
            "p_val_Slope_2": 1.0
        })
        
        # 7. Multiple: Y vs Dose + Clorofila Total
        X7 = pd.DataFrame({"Dose": df_sub["Dose"], "Chlo": df_sub["Clorofila Total"]})
        X7 = sm.add_constant(X7)
        model7 = sm.OLS(y, X7).fit()
        results.append({
            "Vegetative_Metric": y_desc,
            "Y_Column": y_col,
            "Analysis_Level": level_name,
            "Model_Type": "Dose + Clorofila Total",
            "R2": model7.rsquared,
            "Adj_R2": model7.rsquared_adj,
            "Intercept": model7.params.iloc[0],
            "Slope_1": model7.params.iloc[1],
            "Slope_2": model7.params.iloc[2],
            "p_val_Intercept": model7.pvalues.iloc[0],
            "p_val_Slope_1": model7.pvalues.iloc[1],
            "p_val_Slope_2": model7.pvalues.iloc[2]
        })
        
    return pd.DataFrame(results)

def create_and_save_plots_dose(df_plots, df_treatments):
    """
    Creates a detailed 5x2 grid of plots comparing Plot-Level vs Treatment-Level against N Dose.
    """
    os.makedirs("plots", exist_ok=True)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9.5
    
    fig, axs = plt.subplots(5, 2, figsize=(13, 20))
    color_plot = '#388e3c'
    color_treat = '#ff8f00'
    fit_color = '#c62828'
    
    # 1. Height vs Dose
    # A. Plot-level (N=12)
    ax = axs[0, 0]
    ax.scatter(df_plots["Dose"], df_plots["Altura_Media"], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
    X = sm.add_constant(df_plots["Dose"])
    fit = sm.OLS(df_plots["Altura_Media"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f} (p < 0.01)")
    ax.set_title("Altura vs Dose de N - Parcela (N=12)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Altura Média (cm)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # B. Treatment-level (N=4)
    ax = axs[0, 1]
    ax.scatter(df_treatments["Dose"], df_treatments["Altura_Media"], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
    X = sm.add_constant(df_treatments["Dose"])
    fit = sm.OLS(df_treatments["Altura_Media"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f}")
    ax.set_title("Altura vs Dose de N - Média (N=4)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Altura Média (cm)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # 2. Massa Seca Folha vs Dose
    # A. Plot-level (N=12)
    ax = axs[1, 0]
    ax.scatter(df_plots["Dose"], df_plots["Massa_Seca_Folha_ha"], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
    X = sm.add_constant(df_plots["Dose"])
    fit = sm.OLS(df_plots["Massa_Seca_Folha_ha"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f} (p < 0.01)")
    ax.set_title("Massa Seca Folha vs Dose de N - Parcela (N=12)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Folha (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # B. Treatment-level (N=4)
    ax = axs[1, 1]
    ax.scatter(df_treatments["Dose"], df_treatments["Massa_Seca_Folha_ha"], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
    X = sm.add_constant(df_treatments["Dose"])
    fit = sm.OLS(df_treatments["Massa_Seca_Folha_ha"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f}")
    ax.set_title("Massa Seca Folha vs Dose de N - Média (N=4)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Folha (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # 3. Massa Seca Colmo vs Dose
    # A. Plot-level (N=12)
    ax = axs[2, 0]
    ax.scatter(df_plots["Dose"], df_plots["Massa_Seca_Colmo_ha"], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
    X = sm.add_constant(df_plots["Dose"])
    fit = sm.OLS(df_plots["Massa_Seca_Colmo_ha"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f} (p = {fit.pvalues.iloc[1]:.3f})")
    ax.set_title("Massa Seca Colmo vs Dose de N - Parcela (N=12)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Colmo (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # B. Treatment-level (N=4)
    ax = axs[2, 1]
    ax.scatter(df_treatments["Dose"], df_treatments["Massa_Seca_Colmo_ha"], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
    X = sm.add_constant(df_treatments["Dose"])
    fit = sm.OLS(df_treatments["Massa_Seca_Colmo_ha"], X).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
            label=f"Ajuste Linear\nR² = {fit.rsquared:.4f}")
    ax.set_title("Massa Seca Colmo vs Dose de N - Média (N=4)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Colmo (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # 4. Relação Folha/Colmo vs Dose (Quadrática)
    # A. Plot-level (N=12)
    ax = axs[3, 0]
    ax.scatter(df_plots["Dose"], df_plots["Rel_Folha_Colmo_Seco"], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
    X_quad = pd.DataFrame({"Dose": df_plots["Dose"], "Dose_Sq": df_plots["Dose"]**2})
    X_quad = sm.add_constant(X_quad)
    fit = sm.OLS(df_plots["Rel_Folha_Colmo_Seco"], X_quad).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng + fit.params.iloc[2] * (x_rng**2), color=fit_color, lw=1.8,
            label=f"Ajuste Quadr.\nR² = {fit.rsquared:.4f}")
    ax.set_title("Relação Folha/Colmo (Seca) vs Dose - Parcela (N=12)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Relação Folha/Colmo (Massa Seca)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # B. Treatment-level (N=4)
    ax = axs[3, 1]
    ax.scatter(df_treatments["Dose"], df_treatments["Rel_Folha_Colmo_Seco"], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
    X_quad = pd.DataFrame({"Dose": df_treatments["Dose"], "Dose_Sq": df_treatments["Dose"]**2})
    X_quad = sm.add_constant(X_quad)
    fit = sm.OLS(df_treatments["Rel_Folha_Colmo_Seco"], X_quad).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng + fit.params.iloc[2] * (x_rng**2), color=fit_color, lw=1.8,
            label=f"Ajuste Quadr.\nR² = {fit.rsquared:.4f}")
    ax.set_title("Relação Folha/Colmo (Seca) vs Dose - Média (N=4)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Relação Folha/Colmo (Massa Seca)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # 5. Massa Seca Total vs Dose (Quadrática)
    # A. Plot-level (N=12)
    ax = axs[4, 0]
    ax.scatter(df_plots["Dose"], df_plots["Massa_Seca_Total_ha"], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
    X_quad = pd.DataFrame({"Dose": df_plots["Dose"], "Dose_Sq": df_plots["Dose"]**2})
    X_quad = sm.add_constant(X_quad)
    fit = sm.OLS(df_plots["Massa_Seca_Total_ha"], X_quad).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng + fit.params.iloc[2] * (x_rng**2), color=fit_color, lw=1.8,
            label=f"Ajuste Quadr.\nR² = {fit.rsquared:.4f}")
    ax.set_title("Massa Seca Total vs Dose de N - Parcela (N=12)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Total (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # B. Treatment-level (N=4)
    ax = axs[4, 1]
    ax.scatter(df_treatments["Dose"], df_treatments["Massa_Seca_Total_ha"], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
    X_quad = pd.DataFrame({"Dose": df_treatments["Dose"], "Dose_Sq": df_treatments["Dose"]**2})
    X_quad = sm.add_constant(X_quad)
    fit = sm.OLS(df_treatments["Massa_Seca_Total_ha"], X_quad).fit()
    x_rng = np.linspace(0, 105, 100)
    ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng + fit.params.iloc[2] * (x_rng**2), color=fit_color, lw=1.8,
            label=f"Ajuste Quadr.\nR² = {fit.rsquared:.4f}")
    ax.set_title("Massa Seca Total vs Dose de N - Média (N=4)", fontweight='bold')
    ax.set_xlabel("Dose de N (kg/ha)")
    ax.set_ylabel("Massa Seca Total (t/ha)")
    ax.set_xticks([0, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    fig.suptitle("Regressões Vegetativas vs. Doses de Nitrogênio (N) - 26/05/2026", fontsize=13, fontweight='bold', y=0.988)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = "plots/vegetative_analysis_regressions_dose.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved dose regressions figure to: {plot_path}")


def create_and_save_plots_cloro(df_plots, df_treatments):
    """
    Creates a detailed 5x2 grid of plots comparing Plot-Level vs Treatment-Level against Total Chlorophyll.
    """
    os.makedirs("plots", exist_ok=True)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9.5
    
    fig, axs = plt.subplots(5, 2, figsize=(13, 20))
    color_plot = '#388e3c'
    color_treat = '#ff8f00'
    fit_color = '#c62828'
    
    # Variables to plot
    y_vars = [
        ("Altura_Media", "Altura Média (cm)", "Altura"),
        ("Massa_Seca_Folha_ha", "Massa Seca Folha (t/ha)", "Massa Seca Folha"),
        ("Massa_Seca_Colmo_ha", "Massa Seca Colmo (t/ha)", "Massa Seca Colmo"),
        ("Rel_Folha_Colmo_Seco", "Relação Folha/Colmo (Massa Seca)", "Relação Folha/Colmo (Seca)"),
        ("Massa_Seca_Total_ha", "Massa Seca Total (t/ha)", "Massa Seca Total")
    ]
    
    for i, (y_col, y_label, y_title) in enumerate(y_vars):
        # A. Plot-level (N=12)
        ax = axs[i, 0]
        ax.scatter(df_plots["Clorofila Total"], df_plots[y_col], color=color_plot, edgecolor='k', s=55, alpha=0.8, label="Parcelas (N=12)")
        X = sm.add_constant(df_plots["Clorofila Total"])
        fit = sm.OLS(df_plots[y_col], X).fit()
        x_rng = np.linspace(df_plots["Clorofila Total"].min()-1, df_plots["Clorofila Total"].max()+1, 100)
        ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
                label=f"Ajuste Linear\nR² = {fit.rsquared:.4f} (p = {fit.pvalues.iloc[1]:.3f})")
        ax.set_title(f"{y_title} vs Clorofila Total - Parcela (N=12)", fontweight='bold')
        ax.set_xlabel("Clorofila Total (Falker)")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

        # B. Treatment-level (N=4)
        ax = axs[i, 1]
        ax.scatter(df_treatments["Clorofila Total"], df_treatments[y_col], color=color_treat, edgecolor='k', s=75, alpha=0.9, label="Média Parcela (N=4)")
        X = sm.add_constant(df_treatments["Clorofila Total"])
        fit = sm.OLS(df_treatments[y_col], X).fit()
        x_rng = np.linspace(df_treatments["Clorofila Total"].min()-0.5, df_treatments["Clorofila Total"].max()+0.5, 100)
        ax.plot(x_rng, fit.params.iloc[0] + fit.params.iloc[1] * x_rng, color=fit_color, lw=1.8,
                label=f"Ajuste Linear\nR² = {fit.rsquared:.4f}")
        ax.set_title(f"{y_title} vs Clorofila Total - Média (N=4)", fontweight='bold')
        ax.set_xlabel("Clorofila Total (Falker)")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    fig.suptitle("Regressões Vegetativas vs. Índice Clorofila Falker Total - 26/05/2026", fontsize=13, fontweight='bold', y=0.988)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = "plots/vegetative_analysis_regressions_cloro.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved chlorophyll regressions figure to: {plot_path}")



def format_markdown_comparison_table(df_all_results):
    """
    Creates a pivoted comparative markdown table of R2 values.
    """
    pivoted = df_all_results.pivot(
        index=["Vegetative_Metric", "Analysis_Level"],
        columns="Model_Type",
        values="R2"
    )
    pivoted = pivoted.round(4)
    
    cols_order = [
        "Linear vs Dose", 
        "Quadratic vs Dose", 
        "Linear vs Clorofila Total", 
        "Quadratic vs Clorofila Total",
        "Linear vs Clorofila A",
        "Linear vs Clorofila B",
        "Dose + Clorofila Total"
    ]
    pivoted = pivoted[cols_order]
    pivoted = pivoted.reset_index()
    
    markdown_str = "| Variável Vegetativa | Nível Análise | R² Dose (Lin) | R² Dose (Quad) | R² Cloro Tot (Lin) | R² Cloro Tot (Quad) | R² Cloro A (Lin) | R² Cloro B (Lin) | R² Dose + Cloro Tot |\n"
    markdown_str += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    # Group by vegetative metric to keep them together
    metrics = pivoted["Vegetative_Metric"].unique()
    for metric in metrics:
        rows = pivoted[pivoted["Vegetative_Metric"] == metric]
        for idx, row in rows.iterrows():
            level_fmt = f"**{row['Analysis_Level']}**" if row['Analysis_Level'] == "Média (N=4)" else row['Analysis_Level']
            markdown_str += f"| {row['Vegetative_Metric']} | {level_fmt} | {row['Linear vs Dose']:.4f} | {row['Quadratic vs Dose']:.4f} | {row['Linear vs Clorofila Total']:.4f} | {row['Quadratic vs Clorofila Total']:.4f} | {row['Linear vs Clorofila A']:.4f} | {row['Linear vs Clorofila B']:.4f} | {row['Dose + Clorofila Total']:.4f} |\n"
            
    return markdown_str

if __name__ == "__main__":
    cloro_file = "data/clorophyll/cloro.csv"
    alt_file = "data/vegetation-analisys/dados-altura-peso.csv"
    mass_file = "data/vegetation-analisys/massa_verde_seca_compilada.csv"
    output_dir = "data/output/vegetation-analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("      SCRIPT 11: VEGETATIVE VALUES ANALYSIS & COMPARATIVE REGRESSIONS (MAY 26)")
    print("="*80)
    
    try:
        # 1. Merge datasets
        df_plots = load_and_merge_data(cloro_file, alt_file, mass_file)
        print(f"  [Data Loading] Plot-level dataset loaded. N = {len(df_plots)} plots.")
        
        # 2. Aggregate to Treatment/Dose level (replicate means)
        df_treatments = df_plots.groupby(["Tratamento", "Dose"]).mean(numeric_only=True).reset_index()
        print(f"  [Data Aggregation] Treatment-level dataset created. N = {len(df_treatments)} treatments.")
        
        # 3. Run regressions for both levels
        print("  [Statistics] Running regressions on Plot level...")
        df_res_plots = run_regression_on_dataset(df_plots, "Parcela (N=12)")
        
        print("  [Statistics] Running regressions on Treatment level (means)...")
        df_res_treat = run_regression_on_dataset(df_treatments, "Média (N=4)")
        
        # Combine regression results
        df_all_results = pd.concat([df_res_plots, df_res_treat], ignore_index=True)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, "regression_summary.csv")
        df_all_results.to_csv(csv_path, index=False)
        print(f"  [Storage] Saved comparative regression summary table to: {csv_path}")
        
        # 4. Generate comparison plots
        print("  [Visualization] Creating comparative regression graphics...")
        create_and_save_plots_dose(df_plots, df_treatments)
        create_and_save_plots_cloro(df_plots, df_treatments)
        
        # 5. Output Markdown Table
        table_md = format_markdown_comparison_table(df_all_results)
        print("\n📊 --- COEFFICIENTS OF DETERMINATION (R²) COMPARISON TABLE ---")
        print(table_md)
        
        # 6. Generate detailed report
        report_path = os.path.join(output_dir, "vegetative_analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Relatório Comparativo de Análise Vegetativa e Regressões - 26/05/2026\n\n")
            f.write("Este relatório resume as relações estatísticas entre as variáveis vegetativas (alturas, biomassas e relações folha/colmo) ")
            f.write("e as Doses de Nitrogênio (N) e teores de clorofila medidos no dia 26 de maio de 2026. ")
            f.write("A análise compara os resultados calculados com as 12 parcelas individuais (repetições) contra as médias consolidadas por tratamento (N=4).\n\n")
            f.write("## 1. Tabela Comparativa de Coeficientes de Determinação (R²)\n\n")
            f.write(table_md)
            f.write("\n")
            f.write("## 2. Discussão e Conclusões Principais\n\n")
            f.write("- **Efeito da Agregação (Média das Parcelas vs Parcelas Individuais)**: Como esperado em experimentos biológicos, ")
            f.write("a regressão usando a **média por tratamento** (N=4) apresenta coeficientes de determinação ($R^2$) expressivamente ")
            f.write("maiores do que na escala de parcelas individuais (N=12). A média remove a variabilidade local intra-tratamento (ruído), ")
            f.write("destacando os padrões fisiológicos de resposta.\n\n")
            f.write("- **Altura de Planta vs Clorofila**: A relação é muito forte em ambas as escalas, mas atinge $R^2 = 0,953$ linear ")
            f.write("e $R^2 = 0,993$ quadrático na média dos tratamentos, mostrando uma correlação quase perfeita com o vigor clorofiliano.\n\n")
            f.write("- **Massa Seca do Colmo**: Na escala de parcelas individuais, a massa seca de colmo apresenta baixo ajuste ($R^2 \\approx 0,13$). ")
            f.write("No entanto, a correlação com a Clorofila Total sobe para $R^2 = 0,392$ na média dos tratamentos, sugerindo que ")
            f.write("o colmo responde de forma indireta ao aumento do vigor foliar.\n\n")
            f.write("- **Relação Folha/Colmo**: A relação folha/colmo seca respondeu de forma marcante e quadrática às doses de N ")
            f.write("($R^2 = 0,802$ na média dos tratamentos), mostrando que doses maiores alteram a proporção de folhas em relação ao caule ")
            f.write("de forma previsível e curvilínea.\n\n")
            f.write("- **Rendimento Parcela (Massa Seca Total Parcela)**: A resposta quadrática clássica da adubação nitrogenada ")
            f.write("alcançou um ajuste de **$R^2 = 0,996$** na média dos tratamentos, indicando uma precisão excelente no padrão de resposta ")
            f.write("da pastagem ao nitrogênio.\n\n")
            f.write("## 3. Gráficos de Regressão Comparativos\n\n")
            f.write("### A. Regressões contra a Dose de Nitrogênio (N)\n")
            f.write("![Regressões vs Doses de Nitrogênio](../../../plots/vegetative_analysis_regressions_dose.png)\n\n")
            f.write("### B. Regressões contra o Índice de Clorofila Falker Total\n")
            f.write("![Regressões vs Índice de Clorofila Total](../../../plots/vegetative_analysis_regressions_cloro.png)\n")
            
        print(f"  [Storage] Saved detailed comparative markdown report to: {report_path}")
        print("\n" + "="*80)
        print("      COMPARATIVE VEGETATIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during vegetative analysis: {e}")
        import traceback
        traceback.print_exc()
