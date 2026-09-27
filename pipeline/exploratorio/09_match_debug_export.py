#!/usr/bin/env python3
"""Reidentifica parcelas em um JSON exportado do app Nutrinitro (v1 ou v2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from tcc_analysis import config, db

# Impressões digitais 26-05 (vegetation_blocks após pipeline croppado)
FINGERPRINTS_26_05 = [
    {"ponto": 1, "blocks": 918, "w": 288, "h": 339, "falker": 36.94},
    {"ponto": 2, "blocks": 23417, "w": 1425, "h": 1698, "falker": 37.29},
    {"ponto": 3, "blocks": 21593, "w": 1354, "h": 1685, "falker": 41.17},
    {"ponto": 4, "blocks": 20701, "w": 1351, "h": 1695, "falker": 33.19},
    {"ponto": 5, "blocks": 23291, "w": 1395, "h": 1824, "falker": 40.46},
    {"ponto": 6, "blocks": 20630, "w": 1379, "h": 1593, "falker": 41.07},
    {"ponto": 7, "blocks": 17152, "w": 1382, "h": 1635, "falker": 30.34},
    {"ponto": 8, "blocks": 17470, "w": 1197, "h": 1577, "falker": 30.89},
    {"ponto": 9, "blocks": 23431, "w": 1401, "h": 1765, "falker": 36.46},
    {"ponto": 10, "blocks": 20218, "w": 1299, "h": 1632, "falker": 36.19},
    {"ponto": 11, "blocks": 18070, "w": 1230, "h": 1586, "falker": 36.99},
    {"ponto": 12, "blocks": 17626, "w": 1299, "h": 1620, "falker": 30.79},
]


def parse_ponto_from_name(name: str | None) -> int | None:
    if not name:
        return None
    import re
    m = re.search(r"(?:\d{2}-\d{2}[_-])P(\d{1,2})", name, re.I) or re.search(
        r"(?:^|/)P(\d{1,2})(?:[^0-9]|$)", name, re.I
    )
    if m:
        p = int(m.group(1))
        if 1 <= p <= 12:
            return p
    return None


def match_fingerprint(blocks: int, w: int | None, h: int | None) -> tuple[int, float]:
    best_p, best_score = 1, float("inf")
    for fp in FINGERPRINTS_26_05:
        score = abs(blocks - fp["blocks"]) / max(fp["blocks"], 1)
        if w and h:
            score += abs(w - fp["w"]) / fp["w"] * 0.35
            score += abs(h - fp["h"]) / fp["h"] * 0.35
        if score < best_score:
            best_score, best_p = score, fp["ponto"]
    conf = max(0.0, 1.0 - min(best_score, 1.0))
    return best_p, conf


def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    err = y_pred - y_true
    ss_res = (err**2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", nargs="?", default=str(config.APP_DEBUG_EXPORT_JSON),
                        help="Arquivo JSON exportado pelo app ('-' lê de stdin)")
    parser.add_argument("--date", default="26-05")
    args = parser.parse_args()

    if args.json_file != "-":
        data = json.loads(Path(args.json_file).read_text())
    else:
        data = json.loads(sys.stdin.read())

    rows = []
    used_pontos: set[int] = set()

    for img in data.get("images", []):
        spad = img.get("chlorophyll_spad_numeric")
        if spad is None and img.get("chlorophyll_spad"):
            spad = float(str(img["chlorophyll_spad"]).replace("SPAD", "").strip())
        blocks = img.get("vegetation_blocks") or (img.get("result") or {}).get("vegetation_blocks")
        w = img.get("processed_width") or (img.get("result") or {}).get("processed_width")
        h = img.get("processed_height") or (img.get("result") or {}).get("processed_height")
        name = img.get("filename") or img.get("source_name_hint")

        ponto = parse_ponto_from_name(name)
        method = "filename"
        conf = 1.0
        if ponto is None and blocks:
            ponto, conf = match_fingerprint(int(blocks), w, h)
            method = "fingerprint"

        falker = next((f["falker"] for f in FINGERPRINTS_26_05 if f["ponto"] == ponto), None)
        rows.append({
            "display_order": img.get("display_order"),
            "filename": name,
            "matched_ponto": ponto,
            "match_method": method,
            "confidence": round(conf, 3),
            "vegetation_blocks": blocks,
            "spad_app": spad,
            "falker": falker,
            "error": round(spad - falker, 2) if spad and falker else None,
        })
        if ponto:
            used_pontos.add(ponto)

    # Resolver colisões: mesma parcela atribuída 2x — manter maior confiança
    by_ponto: dict[int, dict] = {}
    for r in sorted(rows, key=lambda x: -(x["confidence"] or 0)):
        p = r["matched_ponto"]
        if p and p not in by_ponto:
            by_ponto[p] = r

    matched = sorted(by_ponto.values(), key=lambda x: x["matched_ponto"])
    falker_vals = [r["falker"] for r in matched if r["falker"]]
    app_vals = [r["spad_app"] for r in matched if r["spad_app"] and r["falker"]]

    print("=" * 72)
    print("REIDENTIFICAÇÃO DE PARCELAS — JSON DO APP")
    print("=" * 72)
    print(f"{'Ord':>4} {'Arquivo':>38} {'→':^3} {'P':>3} {'Blocks':>7} {'App':>6} {'Falker':>7} {'Erro':>6} {'Conf':>5}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['display_order']!s:>4} {(r['filename'] or '')[-38:]:>38} {'→':^3} "
            f"P{r['matched_ponto'] or 0:02d} {str(r['vegetation_blocks'] or '-'):>7} "
            f"{r['spad_app'] or '-':>6} {r['falker'] or '-':>7} {r['error'] or '-':>6} "
            f"{r['confidence'] or 0:>5.0%}"
        )

    if len(app_vals) >= 3:
        m = metrics(falker_vals[: len(app_vals)], app_vals)
        print()
        print(f"Métricas (parcelas únicas identificadas, n={len(app_vals)}):")
        print(f"  R²={m['r2']:.3f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  Viés={m['bias']:+.2f}")

    db.save_df(pd.DataFrame(rows), "app_debug_parcel_matching",
               description="Reidentificação das parcelas no JSON do app (nome do arquivo ou fingerprint)")

    dupes = len(rows) - len(by_ponto)
    if dupes:
        print(f"\n⚠ {dupes} imagem(ns) com parcela duplicada — revise fingerprint ou use nomes 26-05_P01.jpg")


if __name__ == "__main__":
    db.run_main(main)
