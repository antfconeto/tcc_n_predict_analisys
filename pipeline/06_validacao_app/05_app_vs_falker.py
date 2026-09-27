#!/usr/bin/env python3
"""Gráfico App (Nutrinitro) vs Falker a partir do JSON de debug exportado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.app_model import load_manifest_images
from tcc_analysis.db import plot_path


def load_manifest() -> list[dict]:
    rows = []
    for m in load_manifest_images():
        cs = m["cropped_size"]
        rows.append(
            {
                "id": m["id"],
                "date_folder": m["date_folder"],
                "ponto": m["ponto"],
                "w": cs["width"],
                "h": cs["height"],
                "falker": m["falker_spad_reference"],
            }
        )
    return rows


def match_by_dimensions(
    width: int, height: int, manifest: list[dict]
) -> tuple[dict, float]:
    best, score = None, float("inf")
    for fp in manifest:
        s = abs(width - fp["w"]) / fp["w"] + abs(height - fp["h"]) / fp["h"]
        if s < score:
            score, best = s, fp
    conf = max(0.0, 1.0 - min(score, 1.0))
    return best, conf


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": len(y_true),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def build_dataframe(json_path: Path) -> pd.DataFrame:
    export = json.loads(json_path.read_text())
    manifest = load_manifest()
    rows = []
    for img in export["images"]:
        w = img.get("processed_width")
        h = img.get("processed_height")
        if w is None or h is None:
            continue
        match, dim_conf = match_by_dimensions(w, h, manifest)
        spad = float(img["chlorophyll_spad_numeric"])
        falker = float(match["falker"])
        rows.append(
            {
                "display_order": img["display_order"],
                "filename": img.get("filename"),
                "source_name_hint": img.get("source_name_hint"),
                "parcel_id": match["id"],
                "date_folder": match["date_folder"],
                "ponto": match["ponto"],
                "processed_width": w,
                "processed_height": h,
                "vegetation_blocks": img.get("vegetation_blocks"),
                "spad_predicted": spad,
                "falker_reference": falker,
                "error": spad - falker,
                "abs_error": abs(spad - falker),
                "dim_match_confidence": dim_conf,
            }
        )
    return pd.DataFrame(rows).sort_values(["date_folder", "ponto"]).reset_index(drop=True)


def save_plot(df: pd.DataFrame, out_path: str, title: str, m: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    dates = sorted(df["date_folder"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(dates), 1)))
    color_map = {d: palette[i] for i, d in enumerate(dates)}

    for date in dates:
        sub = df[df["date_folder"] == date]
        ax.scatter(
            sub["falker_reference"],
            sub["spad_predicted"],
            s=90,
            alpha=0.85,
            color=color_map[date],
            edgecolors="white",
            linewidths=0.8,
            label=f"{date} (n={len(sub)})",
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"P{int(row['ponto']):02d}",
                (row["falker_reference"], row["spad_predicted"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="#333333",
            )

    lo = min(df["falker_reference"].min(), df["spad_predicted"].min()) - 2
    hi = max(df["falker_reference"].max(), df["spad_predicted"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="1:1", zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Falker observado (SPAD)", fontsize=12)
    ax.set_ylabel("App Nutrinitro previsto (SPAD)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", framealpha=0.9)

    text = (
        f"n = {m['n']}\n"
        f"R² = {m['r2']:.4f}\n"
        f"MAE = {m['mae']:.2f} SPAD\n"
        f"RMSE = {m['rmse']:.2f} SPAD\n"
        f"Viés = {m['bias']:+.2f} SPAD"
    )
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(config.APP_DEBUG_EXPORT_JSON),
        help="JSON exportado pelo app (debug)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Filtrar uma data (ex: 26-05). Padrão: todas as imagens.",
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=None,
        help="Excluir pontos com |erro| acima deste limite (SPAD).",
    )
    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Atalho para --max-abs-error 2.5 (outliers da validação).",
    )
    args = parser.parse_args()

    max_abs_error = 2.5 if args.no_outliers else args.max_abs_error

    json_path = Path(args.json_file)
    df_all = build_dataframe(json_path)
    if args.date:
        df_all = df_all[df_all["date_folder"] == args.date].copy()

    if df_all.empty:
        raise SystemExit("Nenhuma imagem encontrada para plotar.")

    outliers = pd.DataFrame()
    df = df_all
    if max_abs_error is not None:
        outliers = df_all[df_all["abs_error"] > max_abs_error].copy()
        df = df_all[df_all["abs_error"] <= max_abs_error].copy()
        if df.empty:
            raise SystemExit("Todos os pontos foram removidos pelo filtro de outliers.")

    m = metrics(
        df["falker_reference"].to_numpy(),
        df["spad_predicted"].to_numpy(),
    )

    table = "app_validation_vs_falker_no_outliers" if max_abs_error else "app_validation_vs_falker"
    db.save_df(df.assign(Filtro_Data=args.date, Max_Abs_Error=max_abs_error), table,
               description="Predição do app Nutrinitro vs Falker (JSON de debug exportado)")

    date_suffix = f"_{args.date.replace('-', '')}" if args.date else "_all34"
    out_suffix = f"{date_suffix}_no_outliers" if max_abs_error else date_suffix
    fig_path = plot_path(f"app_vs_falker{out_suffix}.png")

    if args.date:
        title = f"Validação App vs Falker — {args.date}"
    elif max_abs_error:
        title = (
            f"Validação App vs Falker — {len(df)} imagens "
            f"(sem outliers, |erro| ≤ {max_abs_error:.1f} SPAD)"
        )
    else:
        title = "Validação App vs Falker — 34 imagens (dataset croppado)"
    save_plot(df, fig_path, title, m)

    print("=" * 72)
    print("APP NUTRINITRO vs FALKER")
    print("=" * 72)
    print(f"JSON: {json_path}")
    if max_abs_error:
        print(f"Filtro outliers: |erro| ≤ {max_abs_error:.1f} SPAD")
        print(f"Removidos: {len(outliers)} de {len(df_all)}")
        for _, r in outliers.sort_values("abs_error", ascending=False).iterrows():
            print(
                f"  - {r['date_folder']} P{int(r['ponto']):02d}: "
                f"pred={r['spad_predicted']:.1f} falker={r['falker_reference']:.2f} "
                f"erro={r['error']:+.2f}"
            )
    print(f"Imagens plotadas: {m['n']}")
    print(f"R² final: {m['r2']:.4f}")
    print(f"MAE: {m['mae']:.2f} SPAD")
    print(f"RMSE: {m['rmse']:.2f} SPAD")
    print(f"Viés: {m['bias']:+.2f} SPAD")
    print(f"\nTabela: {table}")
    print(f"Gráfico: {fig_path}")
    print("=" * 72)


if __name__ == "__main__":
    db.run_main(main)
