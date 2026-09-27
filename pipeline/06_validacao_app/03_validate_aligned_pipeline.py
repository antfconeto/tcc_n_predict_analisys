#!/usr/bin/env python3
"""
Validação do pipeline alinhado nas imagens dos dias 18-05, 21-05, 26-05 e 01-06.

Validação oficial (TCC): bilateral + crop 20% + ExG + MLP — igual ao treino.
Referência app (informativo): bilateral sem crop — como o Nutrinitro em produção.

Métricas: R² OOF (modelo definitivo), R² in-sample, RMSE, MAE, correlação.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis import modeling as s09
from tcc_analysis.app_model import FEATURE_COLS, predict_mlp_aligned
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse
from tcc_analysis.preprocess import merge_preprocess_config, preprocess_image

DATES = config.DATE_FOLDERS
IMAGE_ROOT = config.IMAGE_ROOT
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
    df = db.load_df("definitive_model_predictions")
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
    return s09.load_rgb28_dataset()


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
    db.save_df(df_img, "aligned_image_validation",
               description="MLP alinhada por imagem: pipeline TCC (bilateral + crop 20%)")

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

    print("\n[B] Referência modelo definitivo — OOF bagged (baseline documentado)")
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

    db.save_df(df_app, "aligned_image_validation_app_nocrop",
               description="MLP alinhada por imagem: pipeline do app (sem crop)")

    summary = pd.DataFrame([
        {"Cenario": "MLP alinhada (features CSV) vs Falker", **m_a},
        {"Cenario": "Script09 OOF vs Falker", **m_oof},
        {"Cenario": "TCC por imagem (crop) vs Falker", **m_c},
        {"Cenario": "App sem crop vs Falker", **m_app},
    ])
    db.save_df(summary, "aligned_validation_summary", description="Métricas por cenário de validação do pipeline")

    print("\nTabelas: aligned_image_validation, aligned_image_validation_app_nocrop, aligned_validation_summary")
    print("=" * 80)


if __name__ == "__main__":
    db.run_main(main)
