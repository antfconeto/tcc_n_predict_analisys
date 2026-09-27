#!/usr/bin/env python3
"""
Validação do pipeline alinhado nas imagens dos dias 18-05, 21-05, 26-05 e 01-06.

Validação oficial (TCC): bilateral + crop 20% + ExG + MLP — igual ao treino.
Referência app (informativo): bilateral sem crop — como o Nutrinitro em produção.

Métricas: R² OOF (script 09), R² in-sample, RMSE, MAE, correlação.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from image_preprocess import merge_preprocess_config, preprocess_image, build_vegetation_mask
from mlp_aligned_weights import FEATURE_COLS, predict_mlp_aligned

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s09", ROOT / "09-definitive-nn-model.py")
s09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s09)

DATES = ["18-05", "21-05", "26-05", "01-06"]
IMAGE_ROOT = ROOT / "data/image"
REF_CSV = ROOT / "data/output/images-infos/definitive_model_predictions.csv"
BLOCK_SIZE = 10
EXG_THRESHOLD = 0.15
PP_CFG_TCC = merge_preprocess_config({"filter_type": "bilateral", "crop_fraction": 0.20})
PP_CFG_APP = merge_preprocess_config({"filter_type": "bilateral", "crop_fraction": 0.0})

INDEX_STATS = {
    "Median": lambda a: float(np.median(a)),
    "Mean": lambda a: float(np.mean(a)),
    "P75": lambda a: float(np.percentile(a, 75)),
    "P90": lambda a: float(np.percentile(a, 90)),
}
CHANNEL_KEYS = ["r", "g", "b", "rg", "rb", "gb", "rgb"]


def compute_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def build_rgb28_from_lists(channel_lists: list[list[float]]) -> list[float]:
    feats = []
    for ch in channel_lists:
        feats.append(INDEX_STATS["Median"](ch) if ch else 0.0)
    for ch in channel_lists:
        feats.append(INDEX_STATS["Mean"](ch) if ch else 0.0)
    for ch in channel_lists:
        feats.append(INDEX_STATS["P75"](ch) if ch else 0.0)
    for ch in channel_lists:
        feats.append(INDEX_STATS["P90"](ch) if ch else 0.0)
    return feats


def predict_from_image(img_path: str, pp_cfg: dict | None = None) -> tuple[float, int]:
    img = __import__("cv2").imread(img_path)
    if img is None:
        raise ValueError(f"Não leu {img_path}")

    img_float, _ = preprocess_image(img, pp_cfg or PP_CFG_TCC)
    ch, cw, _ = img_float.shape
    nh, nw = ch // BLOCK_SIZE, cw // BLOCK_SIZE
    trimmed = img_float[: nh * BLOCK_SIZE, : nw * BLOCK_SIZE]
    reshaped = trimmed.reshape(nh, BLOCK_SIZE, nw, BLOCK_SIZE, 3)
    with np.errstate(invalid="ignore"):
        block = np.nanmean(reshaped, axis=(1, 3))

    lists = {k: [] for k in CHANNEL_KEYS}
    veg = 0
    for y in range(nh):
        for x in range(nw):
            b, g, r = block[y, x]
            if np.isnan(r):
                continue
            total = r + g + b
            if total <= 0:
                continue
            rn, gn, bn = r / total, g / total, b / total
            exg = 2.0 * gn - rn - bn
            if exg <= EXG_THRESHOLD:
                continue
            rg = (rn + gn) / 2.0
            rb = (rn + bn) / 2.0
            gb = (gn + bn) / 2.0
            rgb = (rn + gn + bn) / 3.0
            lists["r"].append(rn)
            lists["g"].append(gn)
            lists["b"].append(bn)
            lists["rg"].append(rg)
            lists["rb"].append(rb)
            lists["gb"].append(gb)
            lists["rgb"].append(rgb)
            veg += 1

    if veg == 0:
        return 0.0, 0
    feats = build_rgb28_from_lists([lists[k] for k in CHANNEL_KEYS])
    return predict_mlp_aligned(feats), veg


def load_reference() -> dict[tuple[str, int], dict]:
    df = pd.read_csv(REF_CSV)
    out = {}
    for _, row in df.iterrows():
        key = (str(row["Data"]), int(float(row["Ponto"])))
        out[key] = {
            "falker": float(row["Clorofila Total"]),
            "py_oof": float(row["Predicao_MLP"]),
            "py_si": float(row["Predicao_Melhor_Indice"]),
        }
    return out


def load_feature_dataset():
    df_cloro = s09.load_falker_csv("data/clorophyll/cloro.csv")
    df_test = s09.load_falker_csv("data/clorophyll/test.csv")
    df_all = pd.concat([df_cloro, df_test], ignore_index=True)
    df_all["Tratamento"] = df_all["Ponto"].map(
        lambda x: s09.ponto_to_dbc[int(x)]["Tratamento"]
        if pd.notna(x) and int(x) in s09.ponto_to_dbc
        else None
    )
    df_all["Bloco"] = df_all["Ponto"].map(
        lambda x: s09.ponto_to_dbc[int(x)]["Bloco"]
        if pd.notna(x) and int(x) in s09.ponto_to_dbc
        else None
    )
    df_all = df_all.dropna(subset=["Clorofila Total"]).copy()
    df_idx = pd.read_csv(s09.CSV_INDICES)
    df = s09.build_merged_dataset(df_all, df_idx)
    meta = [s09.GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    return df.dropna(subset=meta + FEATURE_COLS).copy()


def print_metrics(title: str, y_true, y_pred) -> dict:
    m = {
        "R2": compute_r2(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
    }
    print(f"  {title}")
    print(f"    R²   = {m['R2']:.4f}")
    print(f"    RMSE = {m['RMSE']:.4f} SPAD")
    print(f"    MAE  = {m['MAE']:.4f} SPAD")
    print(f"    r    = {m['r']:.4f}")
    return m


def main():
    ref = load_reference()
    rows_tcc = []
    rows_app = []

    for folder in DATES:
        folder_path = IMAGE_ROOT / folder
        date_key = f"{folder}-2026"
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue
            m = re.match(r"^[Pp](\d+)", filename)
            if not m:
                continue
            ponto = int(m.group(1))
            img_path = str(folder_path / filename)
            r = ref.get((date_key, ponto))
            if r is None:
                continue

            pred_tcc, veg_tcc = predict_from_image(img_path, PP_CFG_TCC)
            pred_app, veg_app = predict_from_image(img_path, PP_CFG_APP)
            base = {
                "Data": date_key,
                "Ponto": ponto,
                "Arquivo": filename,
                "Falker": r["falker"],
                "Py_OOF_s09": r["py_oof"],
                "Py_SI_s09": r["py_si"],
            }
            rows_tcc.append({
                **base,
                "Aligned_MLP": pred_tcc,
                "Veg_Blocks": veg_tcc,
                "Erro_vs_Falker": abs(pred_tcc - r["falker"]),
                "Erro_vs_OOF": abs(pred_tcc - r["py_oof"]),
            })
            rows_app.append({
                **base,
                "Aligned_MLP": pred_app,
                "Veg_Blocks": veg_app,
                "Erro_vs_Falker": abs(pred_app - r["falker"]),
                "Erro_vs_OOF": abs(pred_app - r["py_oof"]),
            })

    df_img = pd.DataFrame(rows_tcc)
    df_app = pd.DataFrame(rows_app)
    out_img = ROOT / "data/output/images-infos/aligned_image_validation.csv"
    df_img.to_csv(out_img, index=False)

    # Features CSV bilateral — MLP alinhada
    df_feat = load_feature_dataset()
    X = df_feat[FEATURE_COLS].values
    y = df_feat["Clorofila Total"].values
    groups = df_feat[s09.GROUP_COL].values
    y_aligned = np.array([predict_mlp_aligned(x) for x in X])
    pipe = s09.make_mlp_pipeline(**s09.CHAMP_MLP_PARAMS)
    y_oof = s09.group_kfold_oof_mlp(X, y, groups, pipe, n_bag=s09.N_BAG_SEEDS)

    print("=" * 80)
    print("VALIDAÇÃO PIPELINE — dias 18, 21, 26-05 e 01-06")
    print("  Oficial (TCC): crop 20%  |  App (Nutrinitro): sem crop")
    print("=" * 80)

    print("\n[A] MLP alinhada (fit completo sklearn) — features CSV bilateral, n=34")
    m_a = print_metrics("vs Falker", y, y_aligned)
    print_metrics("vs Script09 OOF", y_oof, y_aligned)

    print("\n[B] Referência Script 09 — OOF bagged (baseline documentado)")
    m_oof = print_metrics("OOF vs Falker", y, y_oof)

    print("\n[C] Validação oficial por imagem (bilateral + crop 20% + ExG>0.15), n={}".format(len(df_img)))
    y_fk = df_img["Falker"].values
    y_al = df_img["Aligned_MLP"].values
    y_oof_img = df_img["Py_OOF_s09"].values
    m_c = print_metrics("TCC crop: vs Falker", y_fk, y_al)
    print_metrics("TCC crop: vs Script09 OOF", y_oof_img, y_al)
    print_metrics("Script09 OOF vs Falker", y_fk, y_oof_img)

    print("\n[G] App Nutrinitro (sem crop — informativo), n={}".format(len(df_app)))
    y_app = df_app["Aligned_MLP"].values
    m_app = print_metrics("App sem crop: vs Falker", y_fk, y_app)
    print_metrics("App sem crop: vs Script09 OOF", y_oof_img, y_app)

    print("\n[D] Igualdade numérica MLP alinhada vs sklearn")
    pipe.fit(X, y)
    y_sk = pipe.predict(X)
    max_diff = float(np.max(np.abs(y_aligned - y_sk)))
    print(f"  max|diff| features CSV = {max_diff:.2e}  ({'OK' if max_diff < 1e-9 else 'FALHA'})")

    print("\n[E] Métricas por data — validação oficial (crop)")
    for date in sorted(df_img["Data"].unique()):
        sub = df_img[df_img["Data"] == date]
        print_metrics(f"  {date} (n={len(sub)})", sub["Falker"], sub["Aligned_MLP"])

    print("\n[F] Amostras — validação oficial (crop)")
    print(df_img.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_app = ROOT / "data/output/images-infos/aligned_image_validation_app_nocrop.csv"
    df_app.to_csv(out_app, index=False)

    summary = pd.DataFrame([
        {"Cenario": "MLP alinhada (features CSV) vs Falker", **m_a},
        {"Cenario": "Script09 OOF vs Falker", **m_oof},
        {"Cenario": "TCC por imagem (crop) vs Falker", **m_c},
        {"Cenario": "App sem crop vs Falker", **m_app},
    ])
    out_sum = ROOT / "data/output/images-infos/aligned_validation_summary.csv"
    summary.to_csv(out_sum, index=False)

    print(f"\nCSV imagens (TCC crop): {out_img}")
    print(f"CSV imagens (app):      {out_app}")
    print(f"CSV resumo:  {out_sum}")
    print("=" * 80)


if __name__ == "__main__":
    main()
