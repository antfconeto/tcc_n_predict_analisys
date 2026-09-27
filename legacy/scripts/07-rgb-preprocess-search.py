"""
Busca variantes de pré-processamento de imagem para maximizar R² OOF (MLP RGB).

Para cada configuração:
  1. Extrai índices RGB (grade 10×10, modo rgb28)
  2. Avalia MLP com GroupKFold por Data (leave-one-date-out)
  3. Ranqueia por R² OOF

Requer imagens em data/image/ e clorofila em data/clorophyll/.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from image_preprocess import DEFAULT_PREPROCESS_CONFIG, merge_preprocess_config

warnings.filterwarnings("ignore")

# Importa extração e utilitários dos scripts existentes
import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_extract = _load_module("01-extract-indicies.py", "extract_indices")
_nn_rgb = _load_module("06-neural-network-rgb.py", "nn_rgb")

extract_features_from_images = _extract.extract_features_from_images
load_falker_csv = _nn_rgb.load_falker_csv
build_merged_dataset = _nn_rgb.build_merged_dataset
compute_r2 = _nn_rgb.compute_r2
compute_rmse = _nn_rgb.compute_rmse
ponto_to_dbc = _nn_rgb.ponto_to_dbc
GROUP_COL = _nn_rgb.GROUP_COL
RGB_CHANNEL_NAMES = _nn_rgb.RGB_CHANNEL_NAMES

BLOCK_SIZE = 10
N_BAG = 10
MLP_PARAMS = {"hidden_layer_sizes": (3,), "alpha": 8.0}

PREPROCESS_VARIANTS = {
    "baseline": {},
    "gray_world": {
        "white_balance": "gray_world",
    },
    "clahe_v": {
        "clahe": True,
        "clahe_clip": 2.0,
    },
    "illum_norm": {
        "illum_norm": "luminance",
        "target_luminance": 128.0,
    },
    "gw_clahe": {
        "white_balance": "gray_world",
        "clahe": True,
        "clahe_clip": 2.0,
    },
    "gw_illum": {
        "white_balance": "gray_world",
        "illum_norm": "luminance",
    },
    "full_norm": {
        "white_balance": "gray_world",
        "clahe": True,
        "clahe_clip": 2.0,
        "illum_norm": "luminance",
    },
    "bilateral": {
        "filter_type": "bilateral",
    },
    "blur_light": {
        "blur_kernel": (3, 3),
    },
    "blur_strong": {
        "blur_kernel": (9, 9),
    },
    "no_blur": {
        "filter_type": "none",
    },
    "crop_10": {
        "crop_fraction": 0.10,
    },
    "crop_30": {
        "crop_fraction": 0.30,
    },
    "hsv_tight": {
        "hsv_v_min": 60,
        "hsv_v_max": 220,
        "hsv_s_min": 35,
    },
    "hsv_wide": {
        "hsv_v_min": 30,
        "hsv_v_max": 250,
        "hsv_s_min": 15,
    },
    "no_hsv": {
        "hsv_enabled": False,
    },
    "gamma_08": {
        "gamma": 0.8,
    },
    "gamma_12": {
        "gamma": 1.2,
    },
    "gw_hsv_tight": {
        "white_balance": "gray_world",
        "hsv_v_min": 60,
        "hsv_v_max": 220,
        "hsv_s_min": 35,
    },
    "gw_exg": {
        "white_balance": "gray_world",
    },
    "gw_clahe_exg": {
        "white_balance": "gray_world",
        "clahe": True,
    },
    "best_combo_a": {
        "white_balance": "gray_world",
        "clahe": True,
        "clahe_clip": 3.0,
        "illum_norm": "luminance",
        "hsv_v_min": 50,
        "hsv_v_max": 230,
        "hsv_s_min": 30,
    },
    "best_combo_b": {
        "white_balance": "gray_world",
        "filter_type": "bilateral",
        "clahe": True,
        "crop_fraction": 0.15,
    },
    # --- NOVOS PROCESSAMENTOS E COMBINAÇÕES ---
    "median_filter": {
        "filter_type": "median",
        "blur_kernel": (5, 5),
    },
    "nlmeans_filter": {
        "filter_type": "nlmeans",
        "nlmeans_h": 3,
    },
    "clahe_lab": {
        "clahe": True,
        "clahe_mode": "lab",
        "clahe_clip": 2.0,
    },
    "specular_removal": {
        "remove_specular": True,
        "specular_v_min": 240,
        "specular_s_max": 20,
    },
    "hsv_morph": {
        "hsv_morphology": True,
        "hsv_morphology_kernel": (3, 3),
    },
    "veg_morph": {
        "veg_morphology": True,
        "veg_morphology_kernel": (3, 3),
    },
    "combo_new_denoise_lab": {
        "filter_type": "median",
        "blur_kernel": (5, 5),
        "clahe": True,
        "clahe_mode": "lab",
        "clahe_clip": 2.0,
    },
    "combo_new_morph_specular": {
        "remove_specular": True,
        "hsv_morphology": True,
        "veg_morphology": True,
    },
    "best_combo_a_optimized": {
        "white_balance": "gray_world",
        "clahe": True,
        "clahe_mode": "lab",
        "clahe_clip": 3.0,
        "filter_type": "median",
        "blur_kernel": (5, 5),
        "illum_norm": "luminance",
        "hsv_v_min": 50,
        "hsv_v_max": 230,
        "hsv_s_min": 30,
        "remove_specular": True,
        "hsv_morphology": True,
    },
    "best_combo_b_optimized": {
        "white_balance": "gray_world",
        "filter_type": "nlmeans",
        "nlmeans_h": 2,
        "clahe": True,
        "clahe_mode": "lab",
        "crop_fraction": 0.15,
        "remove_specular": True,
        "veg_morphology": True,
    },
}


def get_rgb28_columns():
    cols = []
    for prefix in ["Median", "Mean", "P75", "P90"]:
        cols.extend([f"{prefix}_{name}" for name in RGB_CHANNEL_NAMES])
    return cols


def make_mlp_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=MLP_PARAMS["hidden_layer_sizes"],
            alpha=MLP_PARAMS["alpha"],
            solver="lbfgs",
            max_iter=4000,
            random_state=42,
        )),
    ])


def bagged_oof(X, y, groups, n_bag=N_BAG):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    y_oof = np.zeros(len(y))
    pipe_template = make_mlp_pipeline()

    for train_idx, test_idx in gkf.split(X, y, groups):
        preds = []
        for seed in range(n_bag):
            pipe = clone(pipe_template)
            pipe.set_params(mlp__random_state=seed)
            pipe.fit(X[train_idx], y[train_idx])
            preds.append(pipe.predict(X[test_idx]))
        y_oof[test_idx] = np.mean(preds, axis=0)

    return y_oof


def load_chlorophyll_data():
    df_cloro = load_falker_csv("data/clorophyll/cloro.csv")
    df_test = load_falker_csv("data/clorophyll/test.csv")
    df_all = pd.concat([df_cloro, df_test], ignore_index=True)
    df_all["Tratamento"] = df_all["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Tratamento"]
        if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    df_all["Bloco"] = df_all["Ponto"].map(
        lambda x: ponto_to_dbc[int(x)]["Bloco"]
        if pd.notna(x) and int(x) in ponto_to_dbc else None
    )
    return df_all.dropna(subset=["Clorofila Total"]).copy()


def evaluate_variant(variant_name, preprocess_overrides, df_cloro, save_csv=True):
    pp_cfg = merge_preprocess_config(preprocess_overrides)
    pp_cfg["veg_mask"] = "exg"
    df_indices = extract_features_from_images(
        block_size=BLOCK_SIZE,
        preprocess_config=pp_cfg,
        verbose=False,
    )
    if df_indices.empty:
        return None

    rgb_cols = get_rgb28_columns()
    missing = [c for c in rgb_cols if c not in df_indices.columns]
    if missing:
        print(f"  Colunas ausentes: {missing[:3]}...")
        return None

    df_merged = build_merged_dataset(df_cloro, df_indices)
    df_model = df_merged[[GROUP_COL, "Ponto", "Clorofila Total"] + rgb_cols].dropna()
    if len(df_model) < 10:
        return None

    X = df_model[rgb_cols].values
    y = df_model["Clorofila Total"].values
    groups = df_model[GROUP_COL].values
    y_oof = bagged_oof(X, y, groups)

    r2 = compute_r2(y, y_oof)
    rmse = compute_rmse(y, y_oof)

    if save_csv:
        out_dir = "data/output/images-infos/preprocess_search"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/extracted_indices_10x10_{variant_name}.csv"
        df_indices.to_csv(csv_path, index=False)

    # R² por fold
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_r2 = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_r2.append(compute_r2(y[test_idx], y_oof[test_idx]))

    return {
        "Variant": variant_name,
        "R2_OOF": r2,
        "RMSE_OOF": rmse,
        "N_Samples": len(y),
        "N_Groups": len(np.unique(groups)),
        "R2_Fold_min": min(fold_r2),
        "R2_Fold_max": max(fold_r2),
        "R2_Fold_std": float(np.std(fold_r2)),
        "Canopy_Cover_mean": df_indices["Canopy_Cover"].mean(),
        "Veg_Blocks_mean": df_indices["Vegetation_Blocks"].mean(),
        **{f"cfg_{k}": v for k, v in pp_cfg.items()
           if k in preprocess_overrides or variant_name == "baseline"},
    }


def main():
    print("=" * 80)
    print("BUSCA DE PRÉ-PROCESSAMENTO — MLP RGB (rgb28, 10×10, GroupKFold por Data)")
    print(f"  Variantes: {len(PREPROCESS_VARIANTS)}")
    print(f"  Baseline OOF anterior: ~0.618 (sem otimização de preprocessamento)")
    print("=" * 80)

    if not os.path.isdir("data/image"):
        print("\nErro: pasta data/image/ não encontrada.")
        return

    df_cloro = load_chlorophyll_data()
    results = []
    baseline_r2 = 0.6177

    for name, overrides in PREPROCESS_VARIANTS.items():
        print(f"\n[{name}] extraindo + avaliando...", flush=True)
        try:
            row = evaluate_variant(name, overrides, df_cloro)
        except Exception as exc:
            print(f"  ERRO: {exc}")
            continue
        if row is None:
            print("  Sem dados suficientes.")
            continue
        row["Delta_vs_baseline"] = row["R2_OOF"] - baseline_r2
        results.append(row)
        print(
            f"  R² OOF = {row['R2_OOF']:.4f}  "
            f"(Δ {row['Delta_vs_baseline']:+.4f})  "
            f"RMSE = {row['RMSE_OOF']:.3f}  "
            f"folds [{row['R2_Fold_min']:.2f}, {row['R2_Fold_max']:.2f}]",
            flush=True,
        )

    if not results:
        print("\nNenhuma variante avaliada.")
        return

    df_res = pd.DataFrame(results).sort_values("R2_OOF", ascending=False)
    out_csv = "data/output/images-infos/preprocess_search/rgb_preprocess_search_summary.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)

    best = df_res.iloc[0]
    print("\n" + "=" * 80)
    print("RANKING — top 10 pré-processamentos")
    print("=" * 80)
    show_cols = ["Variant", "R2_OOF", "Delta_vs_baseline", "RMSE_OOF", "R2_Fold_min", "R2_Fold_max"]
    print(df_res[show_cols].head(10).to_string(index=False))

    print(f"\nMelhor: '{best['Variant']}'  |  R² OOF = {best['R2_OOF']:.4f}  "
          f"(Δ {best['Delta_vs_baseline']:+.4f} vs baseline 0.618)")
    print(f"\nResumo salvo: {out_csv}")

    if best["R2_OOF"] > baseline_r2:
        print(f"\nPara usar o melhor preprocessamento como padrão, rode 01-extract-indicies.py")
        print(f"com a config '{best['Variant']}' ou copie extracted_indices de:")
        print(f"  data/output/images-infos/preprocess_search/extracted_indices_10x10_{best['Variant']}.csv")


if __name__ == "__main__":
    main()
