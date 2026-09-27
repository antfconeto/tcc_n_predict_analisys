"""
Script 10: Geração de Mapas Espaciais de Clorofila (Heatmaps).

Este script:
1. Treina a MLP campeã (ensemble de 10 sementes, hidden=(3,), alpha=8.0, solver L-BFGS)
   e o StandardScaler no conjunto completo de 34 amostras (pré-processamento bilateral).
2. Carrega as imagens das pastas de 26-05 e 01-06.
3. Pré-processa as imagens (Bilateral Filter, Correção Gamma = 0.8, Crop 20%).
4. Executa a predição bloco a bloco (10x10) em blocos de vegetação (ExG > 0.15) em lote.
5. Gera e salva mapas espaciais (heatmaps) com clorofila observada e predita lado a lado.
6. Exporta um resumo com os valores médios, medianos e desvios de SPAD por imagem.
"""

import os
import re
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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

BEST_PREPROCESS_VARIANT = "bilateral"
BEST_PREPROCESS_CONFIG = {"filter_type": "bilateral"}
CSV_INDICES = (
    "data/output/images-infos/preprocess_search/"
    f"extracted_indices_10x10_{BEST_PREPROCESS_VARIANT}.csv"
)

CHAMP_MLP_PARAMS = {
    "hidden_layer_sizes": (3,),
    "alpha": 8.0,
    "activation": "relu",
    "solver": "lbfgs",
    "max_iter": 4000,
}
N_BAG_SEEDS = 10
BLOCK_SIZE = 10
CROP_FRACTION = 0.20
VEG_THRESHOLD = 0.15
SPAD_MIN = 25.0
SPAD_MAX = 55.0


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


def build_chlorophyll_lookup(df_cloro):
    df_chlo_plot = (
        df_cloro.dropna(subset=["Tratamento"])
        .groupby(["Data", "Ponto"])
        .agg({"Clorofila Total": "mean"})
        .reset_index()
    )
    return {
        (row["Data"], int(row["Ponto"])): float(row["Clorofila Total"])
        for _, row in df_chlo_plot.iterrows()
    }


def predict_ensemble(mlp_ensemble, scaler, X_row):
    X_scaled = scaler.transform(X_row.reshape(1, -1))
    preds = [mlp.predict(X_scaled)[0] for mlp in mlp_ensemble]
    return float(np.mean(preds))


def format_spad_comparison(observed, predicted):
    if np.isnan(observed):
        return f"Predito: {predicted:.2f} SPAD (sem medição Falker)"
    return f"Falker: {observed:.2f} SPAD  |  Predito: {predicted:.2f} SPAD"


def main():
    print("=" * 80)
    print("INICIANDO GERADOR DE MAPEAMENTO ESPACIAL DE CLOROFILA (HEATMAPS)")
    print("=" * 80)

    df_cloro = load_falker_csv("data/clorophyll/cloro.csv")
    df_test = load_falker_csv("data/clorophyll/test.csv")
    df_all_cloro = pd.concat([df_cloro, df_test], ignore_index=True)
    df_all_cloro["Tratamento"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro["Bloco"] = df_all_cloro["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"] if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all_cloro = df_all_cloro.dropna(subset=["Clorofila Total"]).copy()

    if not os.path.exists(CSV_INDICES):
        print(f"Erro: arquivo de índices {CSV_INDICES} não encontrado!")
        print("Execute 07-rgb-preprocess-search.py ou extraia com config 'bilateral'.")
        return

    df_indices = pd.read_csv(CSV_INDICES)
    df_merged = build_merged_dataset(df_all_cloro, df_indices)

    rgb_channels = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
    feature_cols = []
    for stat in ["Median", "Mean", "P75", "P90"]:
        feature_cols.extend([f"{stat}_{name}" for name in rgb_channels])

    meta_cols = ["Data", "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    df_model = df_merged.dropna(subset=meta_cols + feature_cols).copy()

    X_train = df_model[feature_cols].values
    y_train = df_model["Clorofila Total"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Treinando o Ensemble de {N_BAG_SEEDS} MLPs (preprocess={BEST_PREPROCESS_VARIANT})...")
    mlp_ensemble = []
    for seed in range(N_BAG_SEEDS):
        mlp = MLPRegressor(random_state=seed, **CHAMP_MLP_PARAMS)
        mlp.fit(X_train_scaled, y_train)
        mlp_ensemble.append(mlp)

    cloro_lookup = build_chlorophyll_lookup(df_all_cloro)
    indices_lookup = {
        (row["Data"], int(row["Ponto"])): row
        for _, row in df_indices.iterrows()
    }

    image_root = "data/image"
    target_dates = ["26-05", "01-06"]
    output_dir = "plots/mappings"
    os.makedirs(output_dir, exist_ok=True)

    summary_records = []

    try:
        import image_preprocess
    except ImportError:
        import sys
        sys.path.insert(0, os.getcwd())
        import image_preprocess

    pp_cfg = image_preprocess.merge_preprocess_config(BEST_PREPROCESS_CONFIG)

    print("\nProcessando imagens e gerando mapeamento (Predição em Lote)...")
    for date_folder in target_dates:
        folder_path = os.path.join(image_root, date_folder)
        if not os.path.exists(folder_path):
            print(f"Pasta de data {date_folder} não encontrada em {image_root}. Pulando.")
            continue

        print(f"\n-> Processando pasta: {date_folder}")
        files = sorted(os.listdir(folder_path))
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            m = re.match(r'^[Pp](\d+)', filename)
            if not m:
                continue
            ponto_id = int(m.group(1))
            date_str = f"{date_folder}-2026"
            cloro_obs = cloro_lookup.get((date_str, ponto_id), np.nan)

            idx_row = indices_lookup.get((date_str, ponto_id))
            if idx_row is not None:
                X_image = idx_row[feature_cols].values.astype(float)
                cloro_pred = (
                    predict_ensemble(mlp_ensemble, scaler, X_image)
                    if not np.any(np.isnan(X_image))
                    else np.nan
                )
            else:
                cloro_pred = np.nan

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao ler imagem: {img_path}")
                continue

            img_preprocessed, _ = image_preprocess.preprocess_image(img, pp_cfg)

            ch, cw, _ = img.shape
            y_start, y_end = int(CROP_FRACTION * ch), int((1.0 - CROP_FRACTION) * ch)
            x_start, x_end = int(CROP_FRACTION * cw), int((1.0 - CROP_FRACTION) * cw)
            cropped_bgr = img[y_start:y_end, x_start:x_end].copy()
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            ch_crop, cw_crop, _ = img_preprocessed.shape
            new_w = cw_crop // BLOCK_SIZE
            new_h = ch_crop // BLOCK_SIZE
            h_trimmed = new_h * BLOCK_SIZE
            w_trimmed = new_w * BLOCK_SIZE
            img_trimmed = img_preprocessed[:h_trimmed, :w_trimmed]

            reshaped = img_trimmed.reshape(new_h, BLOCK_SIZE, new_w, BLOCK_SIZE, 3)
            with np.errstate(invalid='ignore'):
                img_down = np.nanmean(reshaped, axis=(1, 3))

            overlay_img = cropped_bgr.copy()
            pred_grid = np.full((new_h, new_w), np.nan)

            coords = []
            features_list = []

            for y_idx in range(new_h):
                for x_idx in range(new_w):
                    b_val = img_down[y_idx, x_idx, 0]
                    g_val = img_down[y_idx, x_idx, 1]
                    r_val = img_down[y_idx, x_idx, 2]

                    if np.isnan(r_val) or np.isnan(g_val) or np.isnan(b_val):
                        continue

                    total = r_val + g_val + b_val
                    if total == 0:
                        total = 1.0
                    r_chrom = r_val / total
                    g_chrom = g_val / total
                    b_chrom = b_val / total

                    rg = (r_chrom + g_chrom) / 2.0
                    rb = (r_chrom + b_chrom) / 2.0
                    gb = (g_chrom + b_chrom) / 2.0
                    rgb = (r_chrom + g_chrom + b_chrom) / 3.0

                    exg = 2.0 * g_chrom - r_chrom - b_chrom

                    if exg > VEG_THRESHOLD:
                        coords.append((y_idx, x_idx))
                        feats = [
                            r_chrom, g_chrom, b_chrom, rg, rb, gb, rgb,
                            r_chrom, g_chrom, b_chrom, rg, rb, gb, rgb,
                            r_chrom, g_chrom, b_chrom, rg, rb, gb, rgb,
                            r_chrom, g_chrom, b_chrom, rg, rb, gb, rgb,
                        ]
                        features_list.append(feats)

            if len(features_list) > 0:
                X_blocks = np.array(features_list)
                X_blocks_scaled = scaler.transform(X_blocks)

                ensemble_preds = np.zeros((len(mlp_ensemble), len(X_blocks)))
                for idx, mlp in enumerate(mlp_ensemble):
                    ensemble_preds[idx] = mlp.predict(X_blocks_scaled)
                spad_preds = np.mean(ensemble_preds, axis=0)

                for (y_idx, x_idx), spad_pred in zip(coords, spad_preds):
                    pred_grid[y_idx, x_idx] = spad_pred

                    val_norm = (spad_pred - SPAD_MIN) / (SPAD_MAX - SPAD_MIN)
                    val_norm = np.clip(val_norm, 0.0, 1.0)

                    color = plt.cm.RdYlGn(val_norm)
                    color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                    ys = y_idx * BLOCK_SIZE
                    ye = ys + BLOCK_SIZE
                    xs = x_idx * BLOCK_SIZE
                    xe = xs + BLOCK_SIZE
                    cv2.rectangle(overlay_img, (xs, ys), (xe - 1, ye - 1), color_bgr, -1)

            alpha = 0.5
            blended_bgr = cv2.addWeighted(overlay_img, alpha, cropped_bgr, 1.0 - alpha, 0)
            blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

            valid_preds = pred_grid[~np.isnan(pred_grid)]
            if len(valid_preds) > 0:
                mean_spad = np.mean(valid_preds)
                median_spad = np.median(valid_preds)
                std_spad = np.std(valid_preds)
                min_spad = np.min(valid_preds)
                max_spad = np.max(valid_preds)
            else:
                mean_spad, median_spad, std_spad, min_spad, max_spad = (
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                )

            summary_records.append({
                "Data": date_folder,
                "Image": filename,
                "Ponto": ponto_id,
                "Clorofila_Observada": cloro_obs,
                "Clorofila_Predita": cloro_pred,
                "Blocos_Vegetacao": len(valid_preds),
                "Total_Blocos": new_h * new_w,
                "SPAD_Medio_Mapa": mean_spad,
                "SPAD_Mediano": median_spad,
                "SPAD_Desvio": std_spad,
                "SPAD_Min": min_spad,
                "SPAD_Max": max_spad,
            })

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            comparison_text = (
                format_spad_comparison(cloro_obs, cloro_pred)
                if not np.isnan(cloro_pred)
                else "Predição indisponível"
            )
            fig.suptitle(
                f"P{ponto_id} ({date_folder}) — {comparison_text}",
                fontsize=12,
                fontweight="bold",
            )

            axes[0].imshow(cropped_rgb)
            left_subtitle = (
                f"Observado: {cloro_obs:.2f} SPAD"
                if not np.isnan(cloro_obs)
                else "Sem medição Falker"
            )
            axes[0].set_title(f"Imagem Original (Central)\n{left_subtitle}")
            axes[0].axis("off")

            axes[1].imshow(blended_rgb)
            if not np.isnan(mean_spad):
                right_subtitle = (
                    f"Predito: {cloro_pred:.2f} SPAD | Média do mapa: {mean_spad:.2f} SPAD"
                    if not np.isnan(cloro_pred)
                    else f"Média do mapa: {mean_spad:.2f} SPAD"
                )
                axes[1].set_title(f"Mapa de Clorofila\n{right_subtitle}")
            else:
                axes[1].set_title("Mapa de Clorofila (Sem Vegetação)")
            axes[1].axis("off")

            norm = plt.Normalize(vmin=SPAD_MIN, vmax=SPAD_MAX)
            sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.04)
            cbar.set_label("Clorofila Falker Estimada (SPAD)")

            output_name = f"{date_folder}_P{ponto_id}_mapping.png"
            output_path = os.path.join(output_dir, output_name)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            if not np.isnan(cloro_pred):
                print(
                    f"  -> Gerado: {output_name} | {comparison_text} "
                    f"(mapa médio: {mean_spad:.2f} SPAD)"
                    if not np.isnan(mean_spad)
                    else f"  -> Gerado: {output_name} | {comparison_text} (sem vegetação no mapa)"
                )
            elif not np.isnan(mean_spad):
                print(f"  -> Gerado: {output_name} (mapa médio: {mean_spad:.2f} SPAD)")
            else:
                print(f"  -> Gerado: {output_name} (Sem vegetação)")

    df_summary = pd.DataFrame(summary_records)
    summary_csv = "data/output/images-infos/mapping_images_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nResumo das predições por imagem salvo em: {summary_csv}")
    print("=" * 80)
    print("MAPEAMENTO COMPLETADO COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
