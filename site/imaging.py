"""Imagens servidas pelo site: miniaturas das fotos e as etapas do processamento.

Tudo é gerado sob demanda a partir das fotos originais (config.IMAGE_ROOT) e guardado
em outputs/cache/ para as próximas requisições.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tcc_analysis import config
from tcc_analysis.indices import calculate_all_indices
from tcc_analysis.preprocess import merge_preprocess_config, preprocess_image

CACHE_DIR = config.OUTPUT_DIR / "cache"
STEPS = ("crop", "pre", "mask", "blocks")
BLOCK = 10
EXG_THRESHOLD = 0.15


def resolve_photo(rel: str) -> Path:
    root = config.IMAGE_ROOT.resolve()
    path = (root / rel).resolve()
    path.relative_to(root)  # ValueError se tentar sair da pasta
    if not path.is_file():
        raise FileNotFoundError(rel)
    return path


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _encode(img: np.ndarray, out: Path) -> bytes:
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 84])
    data = buf.tobytes()
    out.write_bytes(data)
    return data


def thumbnail(rel: str, width: int = 640) -> bytes:
    width = int(min(max(width, 120), 1600))
    out = CACHE_DIR / "photos" / str(width) / (rel + ".jpg")
    if out.exists():
        return out.read_bytes()
    img = cv2.imread(str(resolve_photo(rel)))
    return _encode(_resize(img, width), out)


def step_image(rel: str, step: str, width: int = 640) -> bytes:
    """Etapas do pipeline do modelo definitivo (filtro bilateral, gamma 0,8, recorte 20%, ExG > 0,15)."""
    if step not in STEPS:
        raise ValueError(f"Etapa desconhecida: {step}")
    width = int(min(max(width, 120), 1600))
    out = CACHE_DIR / "steps" / step / str(width) / (rel + ".jpg")
    if out.exists():
        return out.read_bytes()

    img = cv2.imread(str(resolve_photo(rel)))
    cfg = merge_preprocess_config(config.BEST_PREPROCESS_CONFIG)
    h, w = img.shape[:2]
    crop = cfg["crop_fraction"]
    y0, y1, x0, x1 = int(crop * h), int((1 - crop) * h), int(crop * w), int((1 - crop) * w)

    if step == "crop":
        shade = img.copy()
        overlay = (img * 0.35).astype(np.uint8)
        shade[:] = overlay
        shade[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        thick = max(2, w // 300)
        cv2.rectangle(shade, (x0, y0), (x1, y1), (255, 255, 255), thick)
        return _encode(_resize(shade, width), out)

    pre, _ = preprocess_image(img, cfg)                    # float, já recortada
    pre_u8 = np.nan_to_num(pre, nan=0).clip(0, 255).astype(np.uint8)
    if step == "pre":
        return _encode(_resize(pre_u8, width), out)

    ch, cw = pre.shape[:2]
    nh, nw = ch // BLOCK, cw // BLOCK
    trimmed = pre[: nh * BLOCK, : nw * BLOCK]
    with np.errstate(invalid="ignore"):
        blocks = np.nanmean(trimmed.reshape(nh, BLOCK, nw, BLOCK, 3), axis=(1, 3))
    idx = calculate_all_indices(blocks[:, :, 2], blocks[:, :, 1], blocks[:, :, 0])
    veg = idx["ExG"] > EXG_THRESHOLD

    if step == "mask":
        base = pre_u8[: nh * BLOCK, : nw * BLOCK].copy()
        mask_px = np.repeat(np.repeat(veg, BLOCK, axis=0), BLOCK, axis=1)
        gray = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        faded = (gray * 0.45 + 140).clip(0, 255).astype(np.uint8)
        base[~mask_px] = faded[~mask_px]
        return _encode(_resize(base, width), out)

    # blocks: cada bloco pintado com a sua cor média (o que as estatísticas "enxergam")
    mosaic = np.repeat(np.repeat(blocks, BLOCK, axis=0), BLOCK, axis=1)
    mosaic = np.nan_to_num(mosaic, nan=255).clip(0, 255).astype(np.uint8)
    mask_px = np.repeat(np.repeat(veg, BLOCK, axis=0), BLOCK, axis=1)
    mosaic[~mask_px] = 245
    return _encode(_resize(mosaic, width), out)
