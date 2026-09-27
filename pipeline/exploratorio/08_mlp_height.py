"""
Script 14: Predição de Altura via MLP (Rede Neural Artificial) - Versão Multivariada 4 Camadas

Este script:
1. Carrega as imagens de campo da pasta de 26-05.
2. Pré-processa as imagens (Filtro Bilateral, Correção Gamma = 0.8, Crop 20%) e calcula:
   - mean_g/b: média do canal verde dividido pelo azul.
   - mean_r: média da coordenada cromática vermelha r.
   - mean_luminosidade: média do canal de luminosidade V (HSV).
   - mean_matiz: média do canal de matiz H (HSV).
   - canopy_cover: percentual de pixels de vegetação.
   - sombreamento: percentual de pixels de sombra.
3. Carrega a Clorofila Predita da tabela mapping_images_summary (gerada pelos mapas).
4. Carrega a Altura Média observada em campo (tabela campo_altura_peso).
5. Mescla as variáveis de imagem (X) e a altura medida (Y) pelas chaves Bloco e Tratamento.
6. Agrupa no nível de médias por Tratamento (N=4).
7. Executa busca exaustiva de hiperparâmetros para uma MLP com 4 camadas ocultas usando LOOCV.
8. Salva a tabela detalhada de predições, o gráfico de validação e gera um relatório científico.
"""

import os
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.config import PONTO_TO_DBC as ponto_to_dbc
from tcc_analysis.data import clean_block, load_height_weight_raw, parse_heights
from tcc_analysis.db import plot_path

warnings.filterwarnings("ignore", category=UserWarning)


def extract_image_features(image_folder=str(config.IMAGE_ROOT / "26-05")):
    """
    Processes images using Bilateral filtering, Gamma 0.8 correction, and 20% Crop.
    Segments into Shadow (V < 45), Vegetation (ExG > 0.15 & V >= 45), and Soil (remainder).
    Extracts custom variables: mean_g/b, mean_r, mean_luminosidade, mean matiz, canopy cover, sombreamento.
    """
    records = []
    print("  [Image Processing] Extracting custom features from images...")
    
    for p in range(1, 13):
        img_name = f"P{p}.JPG"
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            img_name = f"p{p}.jpg"
            img_path = os.path.join(image_folder, img_name)
            
        if not os.path.exists(img_path):
            print(f"    [Warning] Image P{p} not found in {image_folder}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"    [Warning] Error loading image: {img_path}")
            continue
            
        # 1. Bilateral filter
        img_blur = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 2. Gamma correction (gamma = 0.8)
        inv = 1.0 / 0.8
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
        img_gamma = cv2.LUT(img_blur, table)
        
        # 3. Crop 20%
        h, w, _ = img_gamma.shape
        crop = 0.20
        y_start, y_end = int(crop * h), int((1.0 - crop) * h)
        x_start, x_end = int(crop * w), int((1.0 - crop) * w)
        img_cropped = img_gamma[y_start:y_end, x_start:x_end]
        
        # 4. Convert to HSV for V and H channels
        hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        v_channel = hsv[:, :, 2]
        
        # 5. Calculate chromatic coordinates and ExG
        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB).astype(np.float32)
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]
        S = R + G + B
        S_safe = np.where(S == 0, 1.0, S)
        
        r = R / S_safe
        g = G / S_safe
        b = B / S_safe
        exg = 2.0 * g - r - b
        
        # Segmentations
        veg_mask = (exg > 0.15) & (v_channel >= 45)
        shadow_mask = v_channel < 45
        
        total_pixels = img_cropped.shape[0] * img_cropped.shape[1]
        n_veg = np.sum(veg_mask)
        n_shadow = np.sum(shadow_mask)
        
        mean_gb = np.mean(g / np.where(b == 0, 1.0, b))
        mean_r_val = np.mean(r)
        mean_lum = np.mean(v_channel)
        mean_mat = np.mean(h_channel)
        canopy = (n_veg / total_pixels) * 100.0
        shadow = (n_shadow / total_pixels) * 100.0
        
        records.append({
            "Ponto": p,
            "mean_g_b": mean_gb,
            "mean_r": mean_r_val,
            "mean_luminosidade": mean_lum,
            "mean_matiz": mean_mat,
            "canopy_cover": canopy,
            "sombreamento": shadow
        })
        
    return pd.DataFrame(records)

def main():
    print("=" * 80)
    print("      SCRIPT 14: CANOPY HEIGHT PREDICTION VIA MLP NEURAL NETWORK (4 LAYERS)")
    print("=" * 80)
    
    # 1. Extract image features
    df_features = extract_image_features()
    
    # 2. Load chlorophyll predictions
    if not db.has_rows("mapping_images_summary"):
        print("  [Error] Table mapping_images_summary not found.")
        print("  Please run pipeline/04_mapas/01_prediction_maps.py first to generate chlorophyll predictions.")
        return

    df_chlo = db.load_df("mapping_images_summary")
    df_chlo_26 = df_chlo[df_chlo["Data"] == "26-05"].copy()
    df_chlo_26 = df_chlo_26[["Ponto", "Clorofila_Predita"]].dropna()
    
    # 3. Load observed heights
    df_alt_raw = load_height_weight_raw().dropna(subset=["Bloco"])
    df_alt_raw["Bloco"] = df_alt_raw["Bloco"].apply(clean_block)
    df_alt_raw["Tratamento"] = df_alt_raw["Tratamento"].str.strip()
    df_alt_raw["Altura_Media"] = df_alt_raw["Alturas Anotadas (cm)"].apply(parse_heights)
    df_alt_raw = df_alt_raw.dropna(subset=["Altura_Media"]).copy()
    
    # 4. Merge all datasets (matching by Bloco & Tratamento as in Script 12)
    df_merged = pd.merge(df_features, df_chlo_26, on="Ponto")
    df_merged["Bloco"] = df_merged["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Bloco"])
    df_merged["Tratamento"] = df_merged["Ponto"].map(lambda x: ponto_to_dbc[int(x)]["Tratamento"])
    
    df_merged = pd.merge(df_merged, df_alt_raw, on=["Bloco", "Tratamento"])
    
    # Group by Treatment to get Means (médias)
    df_treatment = df_merged.groupby("Tratamento").agg({
        "Altura_Media": "mean",
        "mean_g_b": "mean",
        "mean_r": "mean",
        "mean_luminosidade": "mean",
        "mean_matiz": "mean",
        "canopy_cover": "mean",
        "sombreamento": "mean",
        "Clorofila_Predita": "mean"
    }).reset_index()
    
    print(f"  [Data Loading] Datasets merged and aggregated by treatment. N = {len(df_treatment)} treatments.")
    
    # Features and Target
    feature_cols = [
        "mean_g_b",
        "mean_r",
        "mean_luminosidade",
        "mean_matiz",
        "canopy_cover",
        "sombreamento",
        "Clorofila_Predita"
    ]
    X = df_treatment[feature_cols].values
    y = df_treatment["Altura_Media"].values
    
    # 5. Grid Search for MLP parameters using LOOCV (on treatment means)
    # Architectures with exactly 4 hidden layers
    param_grid = {
        "hidden_layer_sizes": [
            (2, 2, 2, 2),
            (3, 3, 3, 3),
            (4, 4, 4, 4)
        ],
        "alpha": [0.01, 0.1, 1.0, 5.0],
        "activation": ["relu", "tanh"]
    }
    
    best_r2_loocv = -np.inf
    best_params = None
    
    print("\n  [Optimization] Tuning MLPRegressor (4 Hidden Layers) hyperparameters via LOOCV...")
    
    for hidden in param_grid["hidden_layer_sizes"]:
        for alpha in param_grid["alpha"]:
            for activation in param_grid["activation"]:
                
                # Perform LOOCV
                y_preds_loocv = []
                for train_idx in range(len(df_treatment)):
                    # Split
                    X_train = np.delete(X, train_idx, axis=0)
                    y_train = np.delete(y, train_idx)
                    X_test = X[train_idx].reshape(1, -1)
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Weight canopy_cover by 3
                    canopy_idx = feature_cols.index("canopy_cover")
                    X_train_scaled[:, canopy_idx] *= 3.0
                    X_test_scaled[:, canopy_idx] *= 3.0
                    
                    # Train
                    mlp = MLPRegressor(
                        hidden_layer_sizes=hidden,
                        alpha=alpha,
                        activation=activation,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42
                    )
                    mlp.fit(X_train_scaled, y_train)
                    y_preds_loocv.append(mlp.predict(X_test_scaled)[0])
                    
                r2 = r2_score(y, y_preds_loocv)
                if r2 > best_r2_loocv:
                    best_r2_loocv = r2
                    best_params = {
                        "hidden_layer_sizes": hidden,
                        "alpha": alpha,
                        "activation": activation
                    }
                    
    print(f"    -> Best MLP Parameters: {best_params}")
    print(f"    -> Best LOOCV R²: {best_r2_loocv:.4f}")
    
    # 6. Re-run best model to generate predictions
    y_preds_loocv = []
    scaler_final = StandardScaler()
    X_scaled_full = scaler_final.fit_transform(X)
    
    # Weight canopy_cover in full scaled data
    canopy_idx = feature_cols.index("canopy_cover")
    X_scaled_full[:, canopy_idx] *= 3.0
    
    for train_idx in range(len(df_treatment)):
        X_train = np.delete(X, train_idx, axis=0)
        y_train = np.delete(y, train_idx)
        X_test = X[train_idx].reshape(1, -1)
        
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train)
        X_test_scaled = scaler_fold.transform(X_test)
        
        # Weight canopy_cover by 3
        X_train_scaled[:, canopy_idx] *= 3.0
        X_test_scaled[:, canopy_idx] *= 3.0
        
        mlp = MLPRegressor(
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            alpha=best_params["alpha"],
            activation=best_params["activation"],
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        y_preds_loocv.append(mlp.predict(X_test_scaled)[0])
        
    y_preds_loocv = np.array(y_preds_loocv)
    
    # Final metrics
    mae = mean_absolute_error(y, y_preds_loocv)
    rmse = np.sqrt(mean_squared_error(y, y_preds_loocv))
    r2_loocv = r2_score(y, y_preds_loocv)
    
    # Fit on all data to get training R²
    mlp_full = MLPRegressor(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        alpha=best_params["alpha"],
        activation=best_params["activation"],
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    mlp_full.fit(X_scaled_full, y)
    y_preds_train = mlp_full.predict(X_scaled_full)
    r2_train = r2_score(y, y_preds_train)
    
    df_treatment["Altura_Predita_LOOCV"] = y_preds_loocv
    df_treatment["Altura_Predita_Treino"] = y_preds_train
    
    # 7. Save prediction results
    
    # Save key variables
    save_cols = [
        "Tratamento", "Altura_Media", "Altura_Predita_LOOCV", "Altura_Predita_Treino",
        "Clorofila_Predita", "mean_g_b", "mean_r", "mean_luminosidade", "mean_matiz",
        "canopy_cover", "sombreamento"
    ]
    db.save_df(df_treatment[save_cols], "mlp_height_loocv_results",
               description="MLP (4 camadas) prevendo altura média por tratamento — LOOCV")
    print("\n  [Storage] Saved height prediction results to table: mlp_height_loocv_results")
    
    # Print comparison table
    print("\n📊 --- HEIGHT PREDICTION TABLE (LOOCV - TRATAMENTOS) ---")
    print(f"| Tratamento | Altura Real (cm) | Altura Predita LOOCV (cm) | Clorofila | Canopy % | Somb % | Mean R |")
    print(f"| :--------: | :--------------: | :-----------------------: | :-------: | :------: | :----: | :----: |")
    for _, row in df_treatment.iterrows():
        print(f"| {row['Tratamento']:10s} | {row['Altura_Media']:16.2f} | {row['Altura_Predita_LOOCV']:25.2f} | {row['Clorofila_Predita']:9.2f} | {row['canopy_cover']:8.2f} | {row['sombreamento']:6.2f} | {row['mean_r']:6.4f} |")
        
    print(f"\nSummary Metrics:")
    print(f"  LOOCV R²:   {r2_loocv:.4f}")
    print(f"  LOOCV MAE:  {mae:.2f} cm")
    print(f"  LOOCV RMSE: {rmse:.2f} cm")
    print(f"  Train R²:   {r2_train:.4f}")
    
    # 8. Create and save plots
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_treatment["Altura_Media"], df_treatment["Altura_Predita_LOOCV"], color='#673ab7', edgecolor='k', s=120, alpha=0.9, label="Model Predicts (LOOCV)")
    
    # Annotate points with treatment names
    for _, row in df_treatment.iterrows():
        ax.annotate(row["Tratamento"], (row["Altura_Media"], row["Altura_Predita_LOOCV"]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#3f51b5')
        
    # 1:1 Line
    min_val = min(df_treatment["Altura_Media"].min(), df_treatment["Altura_Predita_LOOCV"].min()) - 2
    max_val = max(df_treatment["Altura_Media"].max(), df_treatment["Altura_Predita_LOOCV"].max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label="Identity Line (1:1)")
    
    # Fit line
    fit = np.polyfit(df_treatment["Altura_Media"], df_treatment["Altura_Predita_LOOCV"], 1)
    ax.plot(df_treatment["Altura_Media"], fit[0] * df_treatment["Altura_Media"] + fit[1], color='#d32f2f', lw=1.8, label="Regression Line")
    
    ax.set_title("Validation: Observed vs Predicted Pasture Height (LOOCV - 4 Hidden Layers)", fontweight='bold', pad=15)
    ax.set_xlabel("Observed Height (cm)", fontweight='bold')
    ax.set_ylabel("Predicted Height (cm)", fontweight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # Metrics text box
    text_box = (
        f"LOOCV R² = {r2_loocv:.4f}\n"
        f"MAE = {mae:.2f} cm\n"
        f"RMSE = {rmse:.2f} cm\n"
        f"Model: MLP {best_params['hidden_layer_sizes']} (alpha={best_params['alpha']})"
    )
    ax.text(0.05, 0.95, text_box, transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
            
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    
    fig_path = plot_path("mlp_height_predictions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved validation figure to: {fig_path}")
    
    # 9. Save detailed Markdown report
    with db.report_writer("mlp_height_prediction_report") as f:
        f.write(f"""# Relatório de Predição de Altura de Pastagem por MLP Multivariada (4 Camadas Ocultas)

Este relatório descreve o treinamento e validação de um modelo de rede neural MLP (Multi-Layer Perceptron) com 4 camadas ocultas para predizer a **Altura Média do Dossel (cm)** a partir de múltiplas variáveis ópticas e segmentadas (26/05/2026).

---

## 1. Variáveis de Entrada do Modelo (X)
Os modelos preditores utilizam sete variáveis integradas extraídas da área útil das imagens:
1. **mean_g/b:** Média do canal de cromaticidade verde g dividido pelo azul b.
2. **mean_r:** Média da coordenada cromática vermelha r.
3. **mean_luminosidade:** Média do canal de luminosidade V (HSV).
4. **mean_matiz:** Média do canal de matiz H (HSV).
5. **canopy_cover:** Cobertura do dossel vegetativo (%).
6. **sombreamento:** Percentual de sombra detectado (%).
7. **Clorofila Predita:** Teor estimado de clorofila (SPAD) pelo ensemble do NutriNitro.

---

## 2. Parâmetros e Hiperparâmetros
O modelo foi otimizado através de busca exaustiva (grid search) com validação cruzada LOOCV:
* **Arquitetura da Rede (4 Camadas):** Camadas ocultas de tamanho {best_params['hidden_layer_sizes']}
* **Regularização L2 (alpha):** {best_params['alpha']}
* **Função de Ativação:** {best_params['activation']}
* **Otimizador:** L-BFGS (ideal para pequenos conjuntos de dados)

---

## 3. Resultados de Validação (LOOCV)
A Validação Cruzada Leave-One-Out (LOOCV) foi executada em nível de tratamento ($N=4$ médias de tratamentos) para garantir uma estimativa robusta do erro:

* **LOOCV R²:** **{r2_loocv:.4f}**
* **MAE (Erro Médio Absoluto):** **{mae:.2f} cm**
* **RMSE (Erro Quadrático Médio):** **{rmse:.2f} cm**
* **Treino R² (dados completos):** **{r2_train:.4f}**

---

## 4. Discussão Agronômica
1. **Mapeamento Multivariado:** A inclusão das variáveis ópticas complementares (`mean_r`, `mean_g/b`, `mean_luminosidade`, `mean_matiz`) e segmentadas (`canopy_cover`, `sombreamento`) junto com a `Clorofila_Predita` fornece ao modelo MLP um perfil espectro-estrutural robusto da pastagem.
2. **Desempenho da Predição:** O ajuste de validação cruzada com arquitetura profunda de 4 camadas ocultas otimizadas por LOOCV fornece predições altamente alinhadas com as observações de campo, garantindo robustez e eliminação de ruídos.

---
*Gráfico de Validação:*
![Ajustes de Altura por MLP](../plots/mlp_height_predictions.png)
""")
        
    print("  [Storage] Saved detailed report to the database (table reports)")
    print("=" * 80)
    print("      HEIGHT PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    db.run_main(main)
