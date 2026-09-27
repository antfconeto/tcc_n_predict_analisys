#!/usr/bin/env python3
"""
Validação etapa a etapa do pipeline de clorofila (34 imagens).

Etapas:
  1. Pré-processamento (bilateral + gamma) — geometria e checksum
  2. Grade 10×10 — dimensões e média do bloco central
  3. Máscara ExG > 0.15 — blocos vegetados
  4. Features rgb28 — 28 estatísticas
  5. MLP — predição SPAD

Validação oficial (TCC): usa crop 20% — igual ao treino e ao CSV de referência.
App Nutrinitro: sem crop — apenas comparação informativa (não altera o app).

Pipelines:
  - csv_ref / tcc_recrop: referência com crop 20%
  - tcc_nocrop / dart_replica: réplica do app (sem crop)
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_preprocess import merge_preprocess_config, preprocess_image
from mlp_aligned_weights import FEATURE_COLS, predict_mlp_aligned

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extract", ROOT / "01-extract-indicies.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

spec09 = importlib.util.spec_from_file_location("s09", ROOT / "09-definitive-nn-model.py")
s09 = importlib.util.module_from_spec(spec09)
spec09.loader.exec_module(s09)

DATES = ["18-05", "21-05", "26-05", "01-06"]
IMAGE_ROOT = ROOT / "data/image"
CSV_REF = ROOT / "data/output/images-infos/preprocess_search/extracted_indices_10x10_bilateral.csv"
REF_PRED = ROOT / "data/output/images-infos/definitive_model_predictions.csv"
BLOCK_SIZE = 10
EXG_THRESHOLD = 0.15
CHANNEL_KEYS = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
RGB28_KEYS = FEATURE_COLS

PP_CROP = merge_preprocess_config({"filter_type": "bilateral", "crop_fraction": 0.20, "veg_mask": "exg"})
PP_NOCROP = merge_preprocess_config({"filter_type": "bilateral", "crop_fraction": 0.0, "veg_mask": "exg"})


@dataclass
class StageSnapshot:
    pipeline: str
    raw_h: int = 0
    raw_w: int = 0
    proc_h: int = 0
    proc_w: int = 0
    proc_mean_r: float = 0.0
    proc_mean_g: float = 0.0
    proc_mean_b: float = 0.0
    grid_rows: int = 0
    grid_cols: int = 0
    total_blocks: int = 0
    veg_blocks: int = 0
    canopy_pct: float = 0.0
    center_block_r: float = 0.0
    center_block_g: float = 0.0
    center_block_b: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    prediction: float = 0.0


def dart_gamma_lut() -> np.ndarray:
    gamma = 0.8
    return np.array(
        [int(round(255.0 * (i / 255.0) ** (1.0 / gamma))) for i in range(256)],
        dtype=np.uint8,
    )


def tcc_gamma_lut() -> np.ndarray:
    inv = 1.0 / 0.8
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)


def dart_percentile(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = round((len(sorted_vals) - 1) * frac)
    return float(sorted_vals[idx])


def build_rgb28_dart(channel_lists: list[list[float]]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for ch_name, values in zip(CHANNEL_KEYS, channel_lists):
        feats[f"Median_{ch_name}"] = dart_percentile(values, 0.50)
    for ch_name, values in zip(CHANNEL_KEYS, channel_lists):
        feats[f"Mean_{ch_name}"] = float(np.mean(values)) if values else 0.0
    for ch_name, values in zip(CHANNEL_KEYS, channel_lists):
        feats[f"P75_{ch_name}"] = dart_percentile(values, 0.75)
    for ch_name, values in zip(CHANNEL_KEYS, channel_lists):
        feats[f"P90_{ch_name}"] = dart_percentile(values, 0.90)
    return feats


def build_rgb28_tcc(indices: dict, veg_mask: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    for key in CHANNEL_KEYS:
        arr = indices[key][veg_mask] if np.any(veg_mask) else np.array([0.0])
        for stat_name, agg_fn in ext.INDEX_STATISTICS.items():
            feats[f"{stat_name}_{key}"] = float(agg_fn(arr))
    return feats


def collect_veg_lists_from_blocks(blocks: np.ndarray, exg_mask: np.ndarray) -> list[list[float]]:
    """blocks: (rows, cols, 3) com R,G,B em 0-255."""
    lists = {k: [] for k in CHANNEL_KEYS}
    rows, cols, _ = blocks.shape
    for y in range(rows):
        for x in range(cols):
            if not exg_mask[y, x]:
                continue
            r, g, b = blocks[y, x]
            total = r + g + b
            if total <= 0:
                continue
            rn, gn, bn = r / total, g / total, b / total
            exg = 2.0 * gn - rn - bn
            if exg <= EXG_THRESHOLD:
                continue
            lists["r"].append(rn)
            lists["g"].append(gn)
            lists["b"].append(bn)
            lists["rg"].append((rn + gn) / 2.0)
            lists["rb"].append((rn + bn) / 2.0)
            lists["gb"].append((gn + bn) / 2.0)
            lists["rgb"].append((rn + gn + bn) / 3.0)
    return [lists[k] for k in CHANNEL_KEYS]


def block_grid_from_bgr_uint8(img_bgr: np.ndarray, block_size: int) -> tuple[np.ndarray, int, int]:
    h, w = img_bgr.shape[:2]
    cols = w // block_size
    rows = h // block_size
    grid = np.zeros((rows, cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * block_size, c * block_size
            patch = img_bgr[y0 : y0 + block_size, x0 : x0 + block_size]
            grid[r, c, 0] = patch[:, :, 2].mean()
            grid[r, c, 1] = patch[:, :, 1].mean()
            grid[r, c, 2] = patch[:, :, 0].mean()
    return grid, rows, cols


def exg_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros(grid.shape[:2], dtype=bool)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            r, g, b = grid[y, x]
            total = r + g + b
            if total <= 0:
                continue
            rn, gn, bn = r / total, g / total, b / total
            mask[y, x] = (2.0 * gn - rn - bn) > EXG_THRESHOLD
    return mask


def run_dart_replica(img_bgr: np.ndarray) -> StageSnapshot:
    snap = StageSnapshot(pipeline="dart_replica")
    snap.raw_h, snap.raw_w = img_bgr.shape[:2]

    filtered = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    processed = cv2.LUT(filtered, dart_gamma_lut())
    snap.proc_h, snap.proc_w = processed.shape[:2]
    snap.proc_mean_r = float(processed[:, :, 2].mean())
    snap.proc_mean_g = float(processed[:, :, 1].mean())
    snap.proc_mean_b = float(processed[:, :, 0].mean())

    grid, rows, cols = block_grid_from_bgr_uint8(processed, BLOCK_SIZE)
    snap.grid_rows, snap.grid_cols = rows, cols
    snap.total_blocks = rows * cols
    if rows > 0 and cols > 0:
        cr, cc = rows // 2, cols // 2
        snap.center_block_r, snap.center_block_g, snap.center_block_b = grid[cr, cc]

    veg_mask = exg_mask_from_grid(grid)
    snap.veg_blocks = int(veg_mask.sum())
    snap.canopy_pct = 100.0 * snap.veg_blocks / snap.total_blocks if snap.total_blocks else 0.0

    ch_lists = collect_veg_lists_from_blocks(grid, veg_mask)
    snap.features = build_rgb28_dart(ch_lists)
    if snap.veg_blocks > 0:
        snap.prediction = predict_mlp_aligned([snap.features[k] for k in RGB28_KEYS])
    return snap


def run_tcc_pipeline(img_bgr: np.ndarray, pp_cfg: dict, label: str) -> StageSnapshot:
    snap = StageSnapshot(pipeline=label)
    snap.raw_h, snap.raw_w = img_bgr.shape[:2]

    img_float, _ = preprocess_image(img_bgr, pp_cfg)
    valid = ~np.isnan(img_float).any(axis=2)

    snap.proc_h, snap.proc_w = img_float.shape[:2]
    if valid.any():
        snap.proc_mean_r = float(np.nanmean(img_float[:, :, 2]))
        snap.proc_mean_g = float(np.nanmean(img_float[:, :, 1]))
        snap.proc_mean_b = float(np.nanmean(img_float[:, :, 0]))

    ch, cw, _ = img_float.shape
    new_h, new_w = ch // BLOCK_SIZE, cw // BLOCK_SIZE
    snap.grid_rows, snap.grid_cols = new_h, new_w
    snap.total_blocks = new_h * new_w

    trimmed = img_float[: new_h * BLOCK_SIZE, : new_w * BLOCK_SIZE]
    reshaped = trimmed.reshape(new_h, BLOCK_SIZE, new_w, BLOCK_SIZE, 3)
    with np.errstate(invalid="ignore"):
        block = np.nanmean(reshaped, axis=(1, 3))

    if new_h > 0 and new_w > 0:
        cr, cc = new_h // 2, new_w // 2
        snap.center_block_r = float(block[cr, cc, 2]) if not np.isnan(block[cr, cc, 2]) else 0.0
        snap.center_block_g = float(block[cr, cc, 1]) if not np.isnan(block[cr, cc, 1]) else 0.0
        snap.center_block_b = float(block[cr, cc, 0]) if not np.isnan(block[cr, cc, 0]) else 0.0

    R_raw, G_raw, B_raw = block[:, :, 2], block[:, :, 1], block[:, :, 0]
    indices = ext.calculate_all_indices(R_raw, G_raw, B_raw)
    veg_mask = ext.build_vegetation_mask(indices, pp_cfg)
    snap.veg_blocks = int(np.sum(veg_mask))
    snap.canopy_pct = 100.0 * snap.veg_blocks / snap.total_blocks if snap.total_blocks else 0.0

    snap.features = build_rgb28_tcc(indices, veg_mask)
    if snap.veg_blocks > 0:
        snap.prediction = predict_mlp_aligned([snap.features[k] for k in RGB28_KEYS])
    return snap


def snapshot_from_csv_row(row: pd.Series) -> StageSnapshot:
    snap = StageSnapshot(pipeline="csv_ref")
    snap.veg_blocks = int(row.get("Vegetation_Blocks", 0))
    snap.total_blocks = int(row.get("Total_Blocks", 0))
    snap.canopy_pct = float(row.get("Canopy_Cover", 0.0))
    if snap.total_blocks > 0:
        side = int(round(math.sqrt(snap.total_blocks)))
        snap.grid_rows, snap.grid_cols = side, snap.total_blocks // side
    snap.features = {k: float(row[k]) for k in RGB28_KEYS if k in row.index}
    if snap.veg_blocks > 0 and len(snap.features) == 28:
        snap.prediction = predict_mlp_aligned([snap.features[k] for k in RGB28_KEYS])
    return snap


def load_csv_ref() -> dict[tuple[str, int], pd.Series]:
    df = pd.read_csv(CSV_REF)
    out = {}
    for _, row in df.iterrows():
        out[(str(row["Data"]), int(row["Ponto"]))] = row
    return out


def load_falker_ref() -> dict[tuple[str, int], float]:
    df = pd.read_csv(REF_PRED)
    return {(str(r["Data"]), int(float(r["Ponto"]))): float(r["Clorofila Total"]) for _, r in df.iterrows()}


def feat_vector(snap: StageSnapshot) -> np.ndarray:
    return np.array([snap.features.get(k, np.nan) for k in RGB28_KEYS])


def compute_r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def print_stage_summary(name: str, diffs: dict[str, list[float]]) -> None:
    print(f"\n  [{name}]")
    for stage, vals in diffs.items():
        if not vals:
            continue
        arr = np.array(vals)
        print(f"    {stage:<22} MAE={arr.mean():.4f}  max={arr.max():.4f}")


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    falker = df["Falker"].values
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    pairs = [
        ("dart_vs_csv_feat", "Etapa 4: Dart vs CSV"),
        ("nocrop_vs_csv_feat", "Etapa 4: TCC nocrop vs CSV"),
        ("dart_vs_nocrop_feat", "Etapa 4: Dart vs TCC nocrop"),
    ]
    for ax, (col, title) in zip(axes[:3], pairs):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=15, color="#1E88E5", edgecolor="k", alpha=0.85)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.set_xlabel("max|Δ| rgb28")
    ax = axes[3]
    for col, color, label in [
        ("pred_csv", "#43A047", "CSV ref"),
        ("pred_dart", "#1E88E5", "Dart replica"),
        ("pred_nocrop", "#FB8C00", "TCC sem crop"),
    ]:
        if col in df.columns:
            y = df[col].values
            ax.scatter(falker, y, alpha=0.75, s=40, c=color, label=label, edgecolors="k", linewidths=0.3)
    lims = [min(falker.min(), df[["pred_csv", "pred_dart", "pred_nocrop"]].min().min()) - 2,
            max(falker.max(), df[["pred_csv", "pred_dart", "pred_nocrop"]].max().max()) + 2]
    ax.plot(lims, lims, "r--", lw=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Falker (SPAD)")
    ax.set_ylabel("Predito")
    ax.set_title("Etapa 5: MLP vs Falker", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Validação por etapa — pipeline completo (n=34)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_stages_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Painel 2x2: métricas finais por pipeline
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    pipelines = [
        ("pred_csv", "CSV ref (crop, treino)"),
        ("pred_recrop", "TCC re-extract (crop)"),
        ("pred_nocrop", "TCC sem crop (app)"),
        ("pred_dart", "Dart replica (sem crop)"),
    ]
    for ax, (col, title) in zip(axes.flat, pipelines):
        if col not in df.columns:
            continue
        y = df[col].values
        r2 = compute_r2(falker, y)
        rmse = float(np.sqrt(np.mean((falker - y) ** 2)))
        mae = float(np.mean(np.abs(falker - y)))
        ax.scatter(falker, y, color="#1E88E5", alpha=0.85, s=50, edgecolors="k", linewidths=0.4)
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Falker (SPAD)")
        ax.set_ylabel("Predito (SPAD)")
        ax.set_title(f"{title}\nR²={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_stages_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Veg blocks comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["veg_csv"], df["veg_dart"], c="#8E24AA", alpha=0.85, s=50, edgecolors="k", linewidths=0.4)
    mx = max(df["veg_csv"].max(), df["veg_dart"].max())
    ax.plot([0, mx], [0, mx], "r--")
    ax.set_xlabel("Blocos vegetados — CSV ref (crop)")
    ax.set_ylabel("Blocos vegetados — Dart replica (sem crop)")
    ax.set_title("Etapa 3: ExG > 0.15", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_stages_veg_blocks.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    csv_ref = load_csv_ref()
    falker_ref = load_falker_ref()
    rows = []

    lut_diff = float(np.max(np.abs(dart_gamma_lut().astype(int) - tcc_gamma_lut().astype(int))))
    print("=" * 80)
    print("VALIDAÇÃO COMPLETA POR ETAPA — pipeline de clorofila")
    print("  Oficial: TCC com crop 20%  |  App: sem crop (informativo)")
    print("=" * 80)
    print(f"LUT gamma Dart vs TCC: max|Δ| = {lut_diff} níveis (0=igual)")

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
            key = (date_key, ponto)
            if key not in csv_ref:
                continue

            img = cv2.imread(str(folder_path / filename))
            if img is None:
                continue

            s_csv = snapshot_from_csv_row(csv_ref[key])
            s_recrop = run_tcc_pipeline(img, PP_CROP, "tcc_recrop")
            s_nocrop = run_tcc_pipeline(img, PP_NOCROP, "tcc_nocrop")
            s_dart = run_dart_replica(img)

            v_csv = feat_vector(s_csv)
            v_recrop = feat_vector(s_recrop)
            v_nocrop = feat_vector(s_nocrop)
            v_dart = feat_vector(s_dart)

            rows.append({
                "Data": date_key,
                "Ponto": ponto,
                "Arquivo": filename,
                "Falker": falker_ref.get(key, np.nan),
                "raw_h": s_dart.raw_h,
                "raw_w": s_dart.raw_w,
                # Etapa 1
                "proc_h_crop": s_recrop.proc_h,
                "proc_h_nocrop": s_nocrop.proc_h,
                "proc_h_dart": s_dart.proc_h,
                "proc_mean_r_diff_dart_nocrop": abs(s_dart.proc_mean_r - s_nocrop.proc_mean_r),
                # Etapa 2
                "grid_rows_csv": s_csv.grid_rows,
                "grid_rows_dart": s_dart.grid_rows,
                "total_blocks_csv": s_csv.total_blocks,
                "total_blocks_dart": s_dart.total_blocks,
                # Etapa 3
                "veg_csv": s_csv.veg_blocks,
                "veg_recrop": s_recrop.veg_blocks,
                "veg_nocrop": s_nocrop.veg_blocks,
                "veg_dart": s_dart.veg_blocks,
                # Etapa 4
                "recrop_vs_csv_feat": float(np.nanmax(np.abs(v_recrop - v_csv))),
                "nocrop_vs_csv_feat": float(np.nanmax(np.abs(v_nocrop - v_csv))),
                "dart_vs_csv_feat": float(np.nanmax(np.abs(v_dart - v_csv))),
                "dart_vs_nocrop_feat": float(np.nanmax(np.abs(v_dart - v_nocrop))),
                "dart_vs_recrop_feat": float(np.nanmax(np.abs(v_dart - v_recrop))),
                # Etapa 5
                "pred_csv": s_csv.prediction,
                "pred_recrop": s_recrop.prediction,
                "pred_nocrop": s_nocrop.prediction,
                "pred_dart": s_dart.prediction,
            })

    df = pd.DataFrame(rows)
    out_csv = ROOT / "data/output/images-infos/full_pipeline_stages.csv"
    df.to_csv(out_csv, index=False)

    falker = df["Falker"].values
    print(f"\nAmostras validadas: n={len(df)}")

    print("\n=== VALIDAÇÃO OFICIAL (TCC, crop 20%) ===")

    print("\n--- Etapa 1–3: pré-proc., grade, ExG ---")
    print(f"  Geometria após crop:  h médio = {df['proc_h_crop'].mean():.0f}")
    print(f"  Total blocos (crop):   média = {df['total_blocks_csv'].mean():.0f}")
    print(f"  Blocos veg. CSV:       média = {df['veg_csv'].mean():.0f}")
    print(f"  Blocos veg. re-crop:   média = {df['veg_recrop'].mean():.0f}  MAE vs CSV = {(df['veg_recrop']-df['veg_csv']).abs().mean():.1f}")

    print("\n--- Etapa 4: features rgb28 vs CSV ref ---")
    print_stage_summary("oficial", {"TCC re-crop": df["recrop_vs_csv_feat"].tolist()})

    print("\n--- Etapa 5: MLP vs Falker (oficial) ---")
    for col, label in [
        ("pred_csv", "CSV ref"),
        ("pred_recrop", "TCC re-extract (crop)"),
    ]:
        y = df[col].values
        r2 = compute_r2(falker, y)
        rmse = float(np.sqrt(np.mean((falker - y) ** 2)))
        mae = float(np.mean(np.abs(falker - y)))
        print(f"  {label:<28} R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    print("\n=== APP NUTRINITRO (sem crop — informativo) ===")

    print("\n--- Etapas 1–3 ---")
    print(f"  Geometria sem crop:    h médio = {df['proc_h_nocrop'].mean():.0f}")
    print(f"  Total blocos (app):    média = {df['total_blocks_dart'].mean():.0f}")
    print(f"  Blocos veg. Dart:      média = {df['veg_dart'].mean():.0f}")
    print(f"  Dart ≈ TCC nocrop:     mean R MAE = {df['proc_mean_r_diff_dart_nocrop'].mean():.2f}")

    print("\n--- Etapa 4: Dart vs TCC sem crop ---")
    print_stage_summary("app", {"Dart vs TCC nocrop": df["dart_vs_nocrop_feat"].tolist()})

    print("\n--- Etapa 5: MLP vs Falker (app) ---")
    for col, label in [
        ("pred_nocrop", "TCC sem crop"),
        ("pred_dart", "Dart replica (app)"),
    ]:
        y = df[col].values
        r2 = compute_r2(falker, y)
        rmse = float(np.sqrt(np.mean((falker - y) ** 2)))
        mae = float(np.mean(np.abs(falker - y)))
        print(f"  {label:<28} R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    plot_dir = ROOT / "plots"
    save_plots(df, plot_dir)
    print(f"\nCSV: {out_csv}")
    print(f"Plots: {plot_dir}/pipeline_stages_*.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
