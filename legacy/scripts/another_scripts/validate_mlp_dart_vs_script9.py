#!/usr/bin/env python3
"""
Compara a MLP do Dart (pesos fixos) com o script 09-definitive-nn-model.py.

Testes:
  A) Dart MLP vs sklearn MLP treinada no conjunto completo (mesmas features rgb28)
  B) Dart MLP vs Predicao_MLP do script 09 (GroupKFold OOF + bagging 10 seeds)
  C) Correlações e igualdade numérica (tolerância float)
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Importa funções do script 09
spec = importlib.util.spec_from_file_location("script09", ROOT / "09-definitive-nn-model.py")
s09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s09)

CSV_INDICES = s09.CSV_INDICES
REF_CSV = ROOT / "data/output/images-infos/definitive_model_predictions.csv"
RGB_CHANNELS = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
FEATURE_COLS = [
    f"{stat}_{ch}"
    for stat in ["Median", "Mean", "P75", "P90"]
    for ch in RGB_CHANNELS
]
TOL = 1e-4

from mlp_aligned_weights import (
    B1,
    B2,
    FEATURE_COLS as ALIGNED_FEATURE_COLS,
    MLP28_MEANS,
    MLP28_SCALES,
    W1,
    W2,
    predict_mlp_aligned,
)


def predict_mlp_dart(features: list[float]) -> float:
    return predict_mlp_aligned(features)


def load_dataset():
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

    df_idx = pd.read_csv(CSV_INDICES)
    df = s09.build_merged_dataset(df_all, df_idx)
    meta = [s09.GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    df = df.dropna(subset=meta + FEATURE_COLS).copy()
    return df


def sklearn_full_fit_predict(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, Pipeline]:
    pipe = s09.make_mlp_pipeline(**s09.CHAMP_MLP_PARAMS)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    return pred, pipe


def dart_predict_batch(X: np.ndarray) -> np.ndarray:
    return np.array([predict_mlp_dart(row.tolist()) for row in X])


def extract_sklearn_weights(pipe: Pipeline) -> dict:
    scaler: StandardScaler = pipe.named_steps["scaler"]
    mlp: MLPRegressor = pipe.named_steps["mlp"]
    return {
        "means": scaler.mean_.copy(),
        "scales": scaler.scale_.copy(),
        "W1": mlp.coefs_[0].copy(),  # (28, 3)
        "b1": mlp.intercepts_[0].copy(),
        "W2": mlp.coefs_[1].copy().reshape(-1),  # (3,)
        "b2": float(mlp.intercepts_[1][0]),
    }


def compare_weights(sk: dict) -> None:
    print("\n--- Comparação de pesos: Dart alinhado vs sklearn (fit completo, seed=42) ---")
    for name, dart, skw in [
        ("means", MLP28_MEANS, sk["means"]),
        ("scales", MLP28_SCALES, sk["scales"]),
        ("W1", W1, sk["W1"]),
        ("b1", B1, sk["b1"]),
        ("W2", W2, sk["W2"]),
        ("b2", np.array([B2]), np.array([sk["b2"]])),
    ]:
        diff = np.max(np.abs(dart - skw))
        ok = diff < TOL
        print(f"  {name}: max|diff| = {diff:.2e}  {'OK' if ok else 'DIFERE'}")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    lims = [
        min(x.min(), y.min()) - 2,
        max(x.max(), y.max()) + 2,
    ]
    ax.scatter(x, y, color=color, alpha=0.85, edgecolors="k", linewidths=0.5, s=50)
    ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)


def save_comparison_plot(
    y_falker: np.ndarray,
    y_dart: np.ndarray,
    y_oof: np.ndarray,
    y_full: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r2_dart = compute_r2(y_falker, y_dart)
    r2_oof = compute_r2(y_falker, y_oof)
    r2_full = compute_r2(y_falker, y_full)
    mae_dart_oof = compute_mae(y_dart, y_oof)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    _scatter_panel(
        axes[0, 0],
        y_falker,
        y_dart,
        xlabel="Clorofila Falker (SPAD)",
        ylabel="Dart MLP (fit completo)",
        title=(
            f"Dart MLP vs Falker\n"
            f"R² = {r2_dart:.4f} | RMSE = {compute_rmse(y_falker, y_dart):.2f} | "
            f"MAE = {compute_mae(y_falker, y_dart):.2f}"
        ),
        color="#1E88E5",
    )
    _scatter_panel(
        axes[0, 1],
        y_falker,
        y_oof,
        xlabel="Clorofila Falker (SPAD)",
        ylabel="Script 09 OOF bagged",
        title=(
            f"OOF Script 09 vs Falker\n"
            f"R² = {r2_oof:.4f} | RMSE = {compute_rmse(y_falker, y_oof):.2f} | "
            f"MAE = {compute_mae(y_falker, y_oof):.2f}"
        ),
        color="#43A047",
    )
    _scatter_panel(
        axes[1, 0],
        y_oof,
        y_dart,
        xlabel="Script 09 OOF bagged",
        ylabel="Dart MLP (fit completo)",
        title=(
            f"Dart MLP vs OOF (mesmas features rgb28)\n"
            f"r = {corr(y_dart, y_oof):.4f} | MAE = {mae_dart_oof:.2f} | "
            f"max|Δ| = {np.max(np.abs(y_dart - y_oof)):.2f}"
        ),
        color="#8E24AA",
    )
    _scatter_panel(
        axes[1, 1],
        y_falker,
        y_full,
        xlabel="Clorofila Falker (SPAD)",
        ylabel="sklearn fit completo",
        title=(
            f"sklearn full-fit vs Falker\n"
            f"R² = {r2_full:.4f} | RMSE = {compute_rmse(y_falker, y_full):.2f} | "
            f"max|Dart−sklearn| = {np.max(np.abs(y_dart - y_full)):.2e}"
        ),
        color="#FB8C00",
    )

    fig.suptitle(
        "Validação MLP rgb28 — Dart alinhado vs Script 09 (features CSV bilateral, n=34)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df = load_dataset()
    X = df[FEATURE_COLS].values
    y = df["Clorofila Total"].values
    groups = df[s09.GROUP_COL].values

    # Script 09: OOF bagged (o que está em definitive_model_predictions.csv)
    pipe_template = s09.make_mlp_pipeline(**s09.CHAMP_MLP_PARAMS)
    y_oof = s09.group_kfold_oof_mlp(X, y, groups, pipe_template, n_bag=s09.N_BAG_SEEDS)

    ref = pd.read_csv(REF_CSV)
    merged = df[[s09.GROUP_COL, "Ponto"]].copy()
    merged["y_oof_recalc"] = y_oof
    merged = merged.merge(
        ref[[s09.GROUP_COL, "Ponto", "Predicao_MLP"]],
        left_on=[s09.GROUP_COL, "Ponto"],
        right_on=["Data", "Ponto"],
        how="inner",
    )
    oof_match = np.max(np.abs(merged["y_oof_recalc"] - merged["Predicao_MLP"]))
    print("=" * 80)
    print("VALIDAÇÃO MLP: Dart vs Script 09")
    print("=" * 80)
    print(f"Reprodução OOF script 09 vs CSV salvo: max|diff| = {oof_match:.2e}")

    # Dart MLP (pesos fixos, com clamp 12-70)
    y_dart = dart_predict_batch(X)

    # sklearn fit completo (seed 42, sem bagging, sem clamp)
    y_full, pipe_full = sklearn_full_fit_predict(X, y)
    compare_weights(extract_sklearn_weights(pipe_full))

    # sklearn fit completo com clamp como Dart
    y_full_clamped = np.clip(y_full, 12.0, 70.0)

    # Bagging médio no conjunto completo (10 seeds, todas as amostras)
    bag_preds = []
    for seed in range(s09.N_BAG_SEEDS):
        p = clone(pipe_template)
        p.set_params(mlp__random_state=seed)
        p.fit(X, y)
        bag_preds.append(p.predict(X))
    y_bag_full = np.mean(bag_preds, axis=0)
    y_bag_clamped = np.clip(y_bag_full, 12.0, 70.0)

    err_dart_full = np.abs(y_dart - y_full_clamped)
    err_dart_oof = np.abs(y_dart - y_oof)
    err_dart_bag = np.abs(y_dart - y_bag_clamped)

    print("\n--- Predições (mesmas 28 features do CSV bilateral) ---")
    print(f"{'Comparação':<35} {'max|diff|':>12} {'MAE':>10} {'RMSE':>10} {'r':>8} {'iguais?':>8}")
    print("-" * 90)

    rows = [
        ("Dart vs sklearn fit completo", y_dart, y_full_clamped),
        ("Dart vs script09 OOF bagged", y_dart, y_oof),
        ("Dart vs bagging 10 seeds (full)", y_dart, y_bag_clamped),
        ("sklearn full vs script09 OOF", y_full_clamped, y_oof),
        ("bagging full vs script09 OOF", y_bag_clamped, y_oof),
    ]
    for label, a, b in rows:
        diff = np.abs(a - b)
        exact = np.all(diff < TOL)
        print(
            f"{label:<35} {diff.max():>12.6f} {diff.mean():>10.6f} "
            f"{math.sqrt((diff**2).mean()):>10.6f} {corr(a, b):>8.4f} "
            f"{'SIM' if exact else 'NÃO':>8}"
        )

    print("\n--- Amostra (primeiras 10 linhas) ---")
    out = df[[s09.GROUP_COL, "Ponto", "Clorofila Total"]].copy()
    out["Dart_MLP"] = y_dart
    out["Py_OOF_s09"] = y_oof
    out["Py_FullFit"] = y_full_clamped
    out["Py_Bag10_Full"] = y_bag_clamped
    out["Diff_Dart_OOF"] = np.abs(y_dart - y_oof)
    print(out.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = ROOT / "data/output/images-infos/mlp_dart_vs_script9.csv"
    out.to_csv(out_path, index=False)
    print(f"\nCSV salvo: {out_path}")

    plot_path = ROOT / "plots/mlp_dart_vs_script09.png"
    save_comparison_plot(y, y_dart, y_oof, y_full, plot_path)
    print(f"Gráfico salvo: {plot_path}")
    print("=" * 80)

    if np.all(err_dart_full < TOL):
        print("RESULTADO: Dart é EXATAMENTE igual ao sklearn fit completo (mesmas features).")
    else:
        print(
            "RESULTADO: Dart NÃO é igual ao OOF do script 09 (esperado: script 09 usa CV+bagging).\n"
            f"  Dart vs OOF: MAE={err_dart_oof.mean():.4f}, r={corr(y_dart, y_oof):.4f}\n"
            f"  Dart vs full-fit: MAE={err_dart_full.mean():.4f}, max|diff|={err_dart_full.max():.6f}"
        )


if __name__ == "__main__":
    main()
