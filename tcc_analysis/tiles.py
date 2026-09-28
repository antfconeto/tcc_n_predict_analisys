"""Índices por foto inteira e por recortes (grade N×N), e a variação entre blocos de cada foto.

Mesmo pré-processamento da extração definitiva (config.BEST_PREPROCESS_CONFIG, blocos 10×10): a linha com
Recorte = -1 é a foto inteira e reproduz exatamente a tabela image_indices (variante definitiva). Os recortes
servem como repetições dentro da parcela (como as folhas do Falker) e a dispersão entre blocos mede a
heterogeneidade do dossel.
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import pandas as pd

from . import config
from .indices import INDEX_STATISTICS, calculate_all_indices
from .preprocess import build_vegetation_mask, merge_preprocess_config, preprocess_image

HETERO_INDICES = None   # None = todos os índices de calculate_all_indices


def _stats_row(indices, mask):
    row = {}
    for key, arr in indices.items():
        vals = arr[mask] if mask.any() else np.array([0.0])
        for stat_name, fn in INDEX_STATISTICS.items():
            row[f"{stat_name}_{key}"] = fn(vals)
    return row


def _hetero_row(indices, mask):
    row = {}
    for key in (HETERO_INDICES or indices.keys()):
        if key == "rgb":   # (r + g + b)/3 é constante: só ruído numérico
            continue
        vals = indices[key][mask]
        if vals.size < 2:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        mean = float(np.mean(vals))
        cv = 100 * np.std(vals, ddof=1) / abs(mean) if abs(mean) > 1e-6 else np.nan
        row.update({f"DP_{key}": float(np.std(vals, ddof=1)), f"IQR_{key}": float(q3 - q1), f"CV_{key}": float(cv)})
    return row


def extract_photo_units(tiles: int = 3, block_size: int = 10, min_veg_blocks: int = 30, image_transform=None) -> pd.DataFrame:
    """Uma linha por foto (Recorte = -1, com a heterogeneidade entre blocos) e uma por recorte (0..tiles²-1).

    image_transform: função opcional aplicada à imagem BGR (uint8) antes do pré-processamento (ex.: Retinex).
    tiles = 0: só a foto inteira.
    """
    pp = merge_preprocess_config(config.BEST_PREPROCESS_CONFIG)
    root = str(config.IMAGE_ROOT)
    rows = []
    for folder in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        date = f"{folder}-{config.YEAR}"
        for fn in sorted(os.listdir(os.path.join(root, folder))):
            m = re.match(r"^[Pp](\d+)", fn)
            if not m or not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(root, folder, fn))
            if img is None:
                continue
            if image_transform is not None:
                img = image_transform(img)
            f, _ = preprocess_image(img, pp)
            h, w, _ = f.shape
            nh, nw = h // block_size, w // block_size
            with np.errstate(invalid="ignore"):
                down = np.nanmean(f[:nh * block_size, :nw * block_size].reshape(nh, block_size, nw, block_size, 3), axis=(1, 3))
            ind = calculate_all_indices(down[:, :, 2], down[:, :, 1], down[:, :, 0])
            mask = build_vegetation_mask(ind, pp)
            base = {"Data": date, "Ponto": int(m.group(1)), "File_Name": fn}
            rows.append({**base, "Recorte": -1, "Veg_blocks": int(mask.sum()), "Cobertura": 100 * mask.mean(),
                         **_hetero_row(ind, mask), **_stats_row(ind, mask)})
            ys = np.linspace(0, nh, tiles + 1).astype(int)
            xs = np.linspace(0, nw, tiles + 1).astype(int)
            for i in range(tiles):
                for j in range(tiles):
                    sl = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
                    tm = mask[sl]
                    if tm.sum() < min_veg_blocks:
                        continue
                    rows.append({**base, "Recorte": i * tiles + j, "Veg_blocks": int(tm.sum()), "Cobertura": 100 * tm.mean(),
                                 **_stats_row({k: v[sl] for k, v in ind.items()}, tm)})
    return pd.DataFrame(rows)
