import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df = df.set_index(0)
    if 1 in df.columns:
        df = df.drop(columns=[1])
    df = df.transpose()
    df.index.name = "ID"
    df.columns.name = None
    
    numeric_cols = ["Medição", "Ponto", "Latitude", "Longitude", "Clorofila A", "Clorofila B", "Clorofila Total"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def main():
    cloro_csv = "data/clorophyll/cloro.csv"
    test_csv = "data/clorophyll/test.csv"
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Load and combine data
    print("Loading chlorophyll data...")
    df_cloro = load_falker_csv(cloro_csv)
    df_test = load_falker_csv(test_csv)
    df_all = pd.concat([df_cloro, df_test], ignore_index=True)
    print(f"Loaded {len(df_all)} total measurements.")
    
    # Add Dose, Treatment, and Block for experimental dates
    df_all["Bloco"] = df_all["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    df_all["Tratamento"] = df_all["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    df_all["Dose"] = df_all["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Dose"] if pd.notna(x) and int(x) in ponto_to_dbc else None)
    
    # Filter dates
    dates_exp = ["18-05-2026", "21-05-2026", "26-05-2026"]
    df_exp = df_all[df_all["Data"].isin(dates_exp)].copy()
    
    # ----------------------------------------------------
    # PLOT 1: Chlorophyll Dose-Response by Date
    # ----------------------------------------------------
    print("Plotting Plot 1: Dose-Response...")
    plt.figure(figsize=(10, 6))
    
    # Colors for each dose
    dose_colors = {0: "#E57373", 50: "#FFB74D", 75: "#81C784", 100: "#4DB6AC"}
    doses = [0, 50, 75, 100]
    
    # Prepare data for plotting
    data_to_plot = []
    positions = []
    colors_list = []
    labels = []
    
    for i, date in enumerate(dates_exp):
        df_date = df_exp[df_exp["Data"] == date]
        for j, dose in enumerate(doses):
            sub_df = df_date[df_date["Dose"] == dose].dropna(subset=["Clorofila Total"])
            data_to_plot.append(sub_df["Clorofila Total"].values)
            positions.append(i * 5 + j)
            colors_list.append(dose_colors[dose])
            if i == 0:
                labels.append(f"{dose} kg/ha")
            else:
                labels.append("")
                
    bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    
    # Apply colors
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
        
    # Legend handling
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=dose_colors[d], label=f"{d} kg N/ha", alpha=0.8) for d in doses]
    plt.legend(handles=legend_patches, loc="upper left", title="Nitrogen Dose")
    
    plt.xticks([1.5, 6.5, 11.5], dates_exp, fontsize=11)
    plt.title("Dose-Response of Nitrogen Fertilization on Total Chlorophyll", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Evaluation Date", fontsize=12, labelpad=8)
    plt.ylabel("Total Chlorophyll Index (Falker)", fontsize=12, labelpad=8)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.ylim(20, 55)
    plt.tight_layout()
    plt.savefig("plots/chlorophyll_dose_response.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # PLOT 2: Temporal Evolution of Chlorophyll Components
    # ----------------------------------------------------
    print("Plotting Plot 2: Temporal Evolution...")
    plt.figure(figsize=(9, 5))
    
    all_dates = ["18-05-2026", "21-05-2026", "26-05-2026", "01-06-2026"]
    
    chlo_a_means = []
    chlo_b_means = []
    chlo_tot_means = []
    chlo_a_se = []
    chlo_b_se = []
    chlo_tot_se = []
    
    for date in all_dates:
        df_date = df_all[df_all["Data"] == date]
        chlo_a_means.append(df_date["Clorofila A"].mean())
        chlo_b_means.append(df_date["Clorofila B"].mean())
        chlo_tot_means.append(df_date["Clorofila Total"].mean())
        
        chlo_a_se.append(df_date["Clorofila A"].sem())
        chlo_b_se.append(df_date["Clorofila B"].sem())
        chlo_tot_se.append(df_date["Clorofila Total"].sem())
        
    x_indices = np.arange(len(all_dates))
    
    # Plot curves with shaded error bands (standard error)
    plt.errorbar(x_indices, chlo_tot_means, yerr=chlo_tot_se, fmt='-o', color='#2E7D32', label='Total Chlorophyll', linewidth=2, capsize=4)
    plt.errorbar(x_indices, chlo_a_means, yerr=chlo_a_se, fmt='-s', color='#4CAF50', label='Chlorophyll A', linewidth=1.8, capsize=4)
    plt.errorbar(x_indices, chlo_b_means, yerr=chlo_b_se, fmt='-d', color='#81C784', label='Chlorophyll B', linewidth=1.5, capsize=4)
    
    plt.xticks(x_indices, all_dates, fontsize=11)
    plt.title("Temporal Evolution of Chlorophyll Levels (A, B and Total)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Sampling Date", fontsize=11, labelpad=8)
    plt.ylabel("Chlorophyll Index (Falker)", fontsize=11, labelpad=8)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(5, 50)
    plt.tight_layout()
    plt.savefig("plots/chlorophyll_temporal_evolution.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # PLOT 3: Chlorophyll A vs B Correlation
    # ----------------------------------------------------
    print("Plotting Plot 3: Chlorophyll A vs B Correlation...")
    plt.figure(figsize=(7, 6))
    
    df_clean = df_all.dropna(subset=["Clorofila A", "Clorofila B"])
    
    x = df_clean["Clorofila A"]
    y = df_clean["Clorofila B"]
    
    plt.scatter(x, y, color="#2E7D32", alpha=0.5, edgecolors='none', s=25, label='Measurements')
    
    # Calculate regression line
    slope, intercept = np.polyfit(x, y, 1)
    r_val = np.corrcoef(x, y)[0, 1]
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    plt.plot(x_line, y_line, color="#FF9800", linestyle="--", linewidth=2, 
             label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\nr = {r_val:.3f}")
             
    plt.title("Physiological Correlation between Chlorophyll A & B", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Chlorophyll A Index", fontsize=11, labelpad=8)
    plt.ylabel("Chlorophyll B Index", fontsize=11, labelpad=8)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("plots/chlorophyll_a_vs_b_correlation.png", dpi=300)
    plt.close()
    
    print("All plots generated successfully in the 'plots/' directory!")

if __name__ == "__main__":
    main()
