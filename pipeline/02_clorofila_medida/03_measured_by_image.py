#!/usr/bin/env python3
"""
Compara as médias de clorofila MEDIDA (Falker) associadas a cada imagem de parcela.

Para cada foto em <IMAGE_ROOT>/{18-05,21-05,26-05,01-06}, calcula a média das leituras
Falker do mesmo Ponto+Data e gera gráficos comparativos.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.config import PONTO_TO_DBC
from tcc_analysis.data import load_falker_all
from tcc_analysis.db import plot_path

IMAGE_ROOT = config.IMAGE_ROOT
DATES = config.DATE_FOLDERS

DOSE_COLORS = {0: "#E57373", 50: "#FFB74D", 75: "#81C784", 100: "#4DB6AC"}
TRAT_COLORS = {"T1": "#E57373", "T2": "#FFB74D", "T3": "#81C784", "T4": "#4DB6AC"}


def load_all_measurements() -> pd.DataFrame:
    df = load_falker_all()
    df["Ponto"] = pd.to_numeric(df["Ponto"], errors="coerce")
    df = df.dropna(subset=["Ponto", "Data", "Clorofila Total"]).copy()
    df["Ponto"] = df["Ponto"].astype(int)
    return df


def stats_for_image(df_meas: pd.DataFrame, date_key: str, ponto: int) -> dict | None:
    sub = df_meas[(df_meas["Data"] == date_key) & (df_meas["Ponto"] == ponto)]
    if sub.empty:
        return None
    vals = sub["Clorofila Total"].values
    return {
        "n_medicoes": len(vals),
        "media": float(np.mean(vals)),
        "desvio": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "sem": float(sub["Clorofila Total"].sem()),
        "minimo": float(np.min(vals)),
        "maximo": float(np.max(vals)),
        "mediana": float(np.median(vals)),
    }


def build_image_dataset(df_meas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder in DATES:
        folder_path = IMAGE_ROOT / folder
        if not folder_path.is_dir():
            continue
        date_key = f"{folder}-2026"
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue
            m = re.match(r"^[Pp](\d+)", filename)
            if not m:
                continue
            ponto = int(m.group(1))
            stats = stats_for_image(df_meas, date_key, ponto)
            if stats is None:
                continue
            dbc = PONTO_TO_DBC.get(ponto, {})
            rows.append({
                "Data": date_key,
                "Data_curta": folder,
                "Ponto": ponto,
                "Arquivo": filename,
                "Bloco": dbc.get("Bloco"),
                "Tratamento": dbc.get("Tratamento"),
                "Dose": dbc.get("Dose"),
                **stats,
            })
    return pd.DataFrame(rows)


def plot_bars_by_date(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    for ax, date in zip(axes, DATES):
        date_key = f"{date}-2026"
        sub = df[df["Data"] == date_key].sort_values("Ponto")
        if sub.empty:
            ax.set_title(f"{date} — sem dados")
            continue
        colors = [TRAT_COLORS.get(t, "#90A4AE") for t in sub["Tratamento"]]
        x = np.arange(len(sub))
        ax.bar(
            x,
            sub["media"],
            yerr=sub["sem"],
            capsize=4,
            color=colors,
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{int(p)}" for p in sub["Ponto"]], rotation=45)
        ax.set_title(f"{date} (n={len(sub)} imagens)", fontweight="bold")
        ax.set_ylabel("Clorofila Total (SPAD)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, row["media"] + row["sem"] + 0.3, f"{row['media']:.1f}", ha="center", fontsize=8)

    patches = [Patch(facecolor=TRAT_COLORS[t], label=t) for t in ["T1", "T2", "T3", "T4"]]
    fig.legend(handles=patches, loc="upper center", ncol=4, title="Tratamento", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Média de clorofila medida (Falker) por imagem/parcela", fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_bars_by_date.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="Ponto", columns="Data_curta", values="media", aggfunc="first")
    pivot = pivot.reindex(columns=DATES)
    data = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(DATES)))
    ax.set_xticklabels(DATES)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"P{int(p)}" for p in pivot.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Clorofila Total (SPAD)")
    ax.set_title("Heatmap — média medida por Ponto × Data", fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Ponto (parcela)")
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_by_ponto(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    pontos = sorted(df["Ponto"].unique())
    x = np.arange(len(DATES))
    for ponto in pontos:
        sub = df[df["Ponto"] == ponto].set_index("Data_curta").reindex(DATES)
        if sub["media"].isna().all():
            continue
        ax.plot(
            x,
            sub["media"],
            marker="o",
            linewidth=1.8,
            label=f"P{ponto}",
            alpha=0.85,
        )
        ax.errorbar(x, sub["media"], yerr=sub["sem"], fmt="none", capsize=3, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DATES)
    ax.set_xlabel("Data")
    ax.set_ylabel("Clorofila Total medida (SPAD)")
    ax.set_title("Evolução temporal da clorofila medida por parcela", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_temporal_by_ponto.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_grouped_all_images(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values(["Data_curta", "Ponto"]).copy()
    df_sorted["label"] = df_sorted.apply(
        lambda r: f"{r['Data_curta']}\nP{int(r['Ponto'])}", axis=1
    )
    colors = [TRAT_COLORS.get(t, "#90A4AE") for t in df_sorted["Tratamento"]]
    fig, ax = plt.subplots(figsize=(max(12, len(df_sorted) * 0.35), 6))
    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted["media"], yerr=df_sorted["sem"], capsize=3, color=colors, edgecolor="k", linewidth=0.4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["label"], rotation=90, fontsize=7)
    ax.set_ylabel("Clorofila Total (SPAD)")
    ax.set_title("Comparação de todas as imagens — média ± EP das medições Falker", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    patches = [Patch(facecolor=TRAT_COLORS[t], label=t) for t in ["T1", "T2", "T3", "T4"]]
    ax.legend(handles=patches, loc="upper right", title="Tratamento")
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_all_images.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_boxplot_replicates(df_meas: pd.DataFrame, df_img: pd.DataFrame) -> None:
    """Boxplot das leituras individuais Falker, agrupadas por imagem."""
    keys = set(zip(df_img["Data"], df_img["Ponto"]))
    sub = df_meas[df_meas.apply(lambda r: (r["Data"], int(r["Ponto"])) in keys, axis=1)].copy()
    order = (
        df_img.sort_values(["Data_curta", "Ponto"])
        .apply(lambda r: f"{r['Data_curta']}\nP{int(r['Ponto'])}", axis=1)
        .tolist()
    )
    groups = []
    for label in order:
        parts = label.split("\n")
        date_short, p_str = parts[0], parts[1]
        ponto = int(p_str.replace("P", ""))
        date_key = f"{date_short}-2026"
        vals = sub[(sub["Data"] == date_key) & (sub["Ponto"] == ponto)]["Clorofila Total"].values
        groups.append(vals)

    fig, ax = plt.subplots(figsize=(max(14, len(order) * 0.4), 6))
    bp = ax.boxplot(groups, positions=range(len(order)), widths=0.6, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#A5D6A7")
        patch.set_alpha(0.85)
    means = df_img.sort_values(["Data_curta", "Ponto"])
    for i, (_, row) in enumerate(means.iterrows()):
        ax.scatter(i, row["media"], color="#D32F2F", s=60, zorder=5, marker="D")
    ax.scatter([], [], color="#D32F2F", s=60, marker="D", label="Média da imagem")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("Clorofila Total (SPAD)")
    ax.set_title("Distribuição das medições Falker por imagem (losango = média)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_boxplot_replicates.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_date_comparison_lines(df: pd.DataFrame) -> None:
    """Compara a mesma parcela entre datas (apenas pontos com imagem em ≥2 datas)."""
    counts = df.groupby("Ponto")["Data"].nunique()
    pontos_multi = counts[counts >= 2].index.tolist()
    if not pontos_multi:
        return

    ncols = 4
    nrows = int(np.ceil(len(pontos_multi) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, ponto in enumerate(sorted(pontos_multi)):
        ax = axes[idx // ncols, idx % ncols]
        sub = df[df["Ponto"] == ponto].set_index("Data_curta").reindex(DATES)
        trat = sub["Tratamento"].dropna().iloc[0] if sub["Tratamento"].notna().any() else "?"
        ax.errorbar(
            range(len(DATES)),
            sub["media"],
            yerr=sub["sem"],
            fmt="-o",
            capsize=4,
            color=TRAT_COLORS.get(trat, "#455A64"),
            linewidth=2,
        )
        ax.set_xticks(range(len(DATES)))
        ax.set_xticklabels(DATES, rotation=45, fontsize=8)
        ax.set_title(f"P{ponto} ({trat})", fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    for idx in range(len(pontos_multi), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fig.suptitle("Comparação temporal — média medida por parcela", fontweight="bold")
    fig.supylabel("Clorofila Total (SPAD)")
    plt.tight_layout()
    plt.savefig(plot_path("measured_chlorophyll_ponto_panels.png"), dpi=150, bbox_inches="tight")
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("COMPARAÇÃO — clorofila MEDIDA (Falker) por imagem")
    print("=" * 80)
    print(f"Imagens com referência Falker: n={len(df)}")
    print(f"\nMédia geral: {df['media'].mean():.2f} SPAD  |  Desvio entre imagens: {df['media'].std():.2f}")
    print("\nPor data:")
    for date in DATES:
        sub = df[df["Data_curta"] == date]
        if sub.empty:
            continue
        print(
            f"  {date}: média={sub['media'].mean():.2f}  "
            f"min={sub['media'].min():.2f}  max={sub['media'].max():.2f}  n={len(sub)}"
        )
    print("\nTabela:")
    cols = ["Data_curta", "Ponto", "Arquivo", "Tratamento", "media", "sem", "n_medicoes"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("=" * 80)


def main():
    df_meas = load_all_measurements()
    df_img = build_image_dataset(df_meas)
    if df_img.empty:
        raise RuntimeError("Nenhuma imagem encontrada com medições Falker correspondentes.")

    db.save_df(df_img, "measured_chlorophyll_by_image",
               description="Estatísticas das leituras Falker associadas a cada imagem (Data + Ponto)")
    print_summary(df_img)

    plot_bars_by_date(df_img)
    plot_heatmap(df_img)
    plot_temporal_by_ponto(df_img)
    plot_grouped_all_images(df_img)
    plot_boxplot_replicates(df_meas, df_img)
    plot_date_comparison_lines(df_img)

    print("\nTabela: measured_chlorophyll_by_image")
    print(f"Gráficos em: {config.PLOTS_DIR}")


if __name__ == "__main__":
    db.run_main(main)
