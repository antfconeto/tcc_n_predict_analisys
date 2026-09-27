"""Pré-processamento configurável de imagens RGB para extração de índices."""

import cv2
import numpy as np

DEFAULT_PREPROCESS_CONFIG = {
    "blur_kernel": (5, 5),
    "blur_sigma": 0,
    "filter_type": "gaussian",       # gaussian | bilateral | median | nlmeans | none
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "nlmeans_h": 3,
    "nlmeans_h_color": 3,
    "nlmeans_template_w": 7,
    "nlmeans_search_w": 21,
    "crop_fraction": 0.20,
    "white_balance": None,           # None | gray_world | white_patch
    "clahe": False,
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),
    "clahe_mode": "v",               # v | lab
    "illum_norm": None,              # None | luminance
    "target_luminance": 128.0,
    "gamma": 0.8,
    "hsv_enabled": False,
    "hsv_v_min": 45,
    "hsv_v_max": 240,
    "hsv_s_min": 25,
    "remove_specular": False,
    "specular_v_min": 240,
    "specular_s_max": 20,
    "hsv_morphology": False,
    "hsv_morphology_kernel": (3, 3),
    "veg_mask": "exg",               # vari | exg | gli | exg_vari
    "veg_threshold": 0.15,
    "veg_morphology": False,
    "veg_morphology_kernel": (3, 3),
}


def _gray_world_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    mean_gray = (mb + mg + mr) / 3.0
    scale = mean_gray / (np.array([mb, mg, mr]) + 1e-6)
    balanced = cv2.merge([
        np.clip(b * scale[0], 0, 255),
        np.clip(g * scale[1], 0, 255),
        np.clip(r * scale[2], 0, 255),
    ])
    return balanced.astype(np.uint8)


def _apply_clahe_v(img_bgr, clip_limit, grid_size):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def _apply_clahe_lab(img_bgr, clip_limit, grid_size):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _normalize_luminance(img_bgr, target=128.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    current = gray.mean()
    if current < 1e-3:
        return img_bgr
    scale = target / current
    adjusted = np.clip(img_bgr.astype(np.float32) * scale, 0, 255)
    return adjusted.astype(np.uint8)


def _apply_gamma(img_bgr, gamma):
    if abs(gamma - 1.0) < 1e-6:
        return img_bgr
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255 for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def _apply_blur(img_bgr, config):
    ftype = config.get("filter_type", "gaussian")
    if ftype == "none":
        return img_bgr
    if ftype == "bilateral":
        return cv2.bilateralFilter(
            img_bgr,
            config.get("bilateral_d", 9),
            config.get("bilateral_sigma_color", 75),
            config.get("bilateral_sigma_space", 75),
        )
    if ftype == "median":
        k = config.get("blur_kernel", (5, 5))
        ksize = k[0] if isinstance(k, tuple) else k
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(img_bgr, ksize)
    if ftype == "nlmeans":
        h = config.get("nlmeans_h", 3)
        h_color = config.get("nlmeans_h_color", 3)
        template_w = config.get("nlmeans_template_w", 7)
        search_w = config.get("nlmeans_search_w", 21)
        return cv2.fastNlMeansDenoisingColored(
            img_bgr, None, h, h_color, template_w, search_w
        )
    k = config.get("blur_kernel", (5, 5))
    return cv2.GaussianBlur(img_bgr, k, config.get("blur_sigma", 0))


def _build_pixel_mask(hsv, config):
    if not config.get("hsv_enabled", True):
        h, w = hsv.shape[:2]
        return np.ones((h, w), dtype=bool)

    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    
    if config.get("remove_specular", False):
        spec_v = config.get("specular_v_min", 240)
        spec_s = config.get("specular_s_max", 20)
        specular = (v_ch > spec_v) & (s_ch < spec_s)
    else:
        specular = np.zeros_like(s_ch, dtype=bool)

    mask = (
        (v_ch >= config.get("hsv_v_min", 45))
        & (v_ch <= config.get("hsv_v_max", 240))
        & (s_ch >= config.get("hsv_s_min", 25))
        & (~specular)
    )

    if config.get("hsv_morphology", False):
        ksize = config.get("hsv_morphology_kernel", (3, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask = mask_uint8.astype(bool)

    return mask


def _build_veg_mask(indices, config):
    threshold = 0.15
    mask_type = config.get("veg_mask", "exg")

    if mask_type == "exg":
        return indices["ExG"] > threshold
    if mask_type == "gli":
        return indices["GLI"] > threshold
    if mask_type == "exg_vari":
        return (indices["ExG"] > threshold) & (indices["VARI"] > threshold)
    return indices["ExG"] > threshold


def merge_preprocess_config(overrides=None):
    cfg = dict(DEFAULT_PREPROCESS_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def preprocess_image(img_bgr, config=None):
    """
    Aplica pipeline de pré-processamento e retorna imagem float32 com NaN em pixels inválidos.
    """
    cfg = merge_preprocess_config(config)
    img = img_bgr.copy()

    img = _apply_blur(img, cfg)

    wb = cfg.get("white_balance")
    if wb == "gray_world":
        img = _gray_world_balance(img)

    gamma = cfg.get("gamma", 1.0)
    img = _apply_gamma(img, gamma)

    if cfg.get("clahe", False):
        mode = cfg.get("clahe_mode", "v")
        if mode == "lab":
            img = _apply_clahe_lab(
                img, cfg.get("clahe_clip", 2.0), cfg.get("clahe_grid", (8, 8)),
            )
        else:
            img = _apply_clahe_v(
                img, cfg.get("clahe_clip", 2.0), cfg.get("clahe_grid", (8, 8)),
            )

    if cfg.get("illum_norm") == "luminance":
        img = _normalize_luminance(img, cfg.get("target_luminance", 128.0))

    h, w, _ = img.shape
    crop = cfg.get("crop_fraction", 0.20)
    y_start, y_end = int(crop * h), int((1.0 - crop) * h)
    x_start, x_end = int(crop * w), int((1.0 - crop) * w)
    img_cropped = img[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    keep_mask = _build_pixel_mask(hsv, cfg)

    img_float = img_cropped.astype(np.float32)
    img_float[~keep_mask] = np.nan

    return img_float, cfg


def build_vegetation_mask(indices, config=None):
    cfg = merge_preprocess_config(config)
    mask = _build_veg_mask(indices, cfg)
    if cfg.get("veg_morphology", False):
        ksize = cfg.get("veg_morphology_kernel", (3, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask = mask_uint8.astype(bool)
    return mask
