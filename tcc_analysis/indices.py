"""Cálculo dos índices espectrais RGB e extração de estatísticas por imagem.

Lógica idêntica ao antigo 01-extract-indicies.py; a gravação em banco fica a cargo
do script de pipeline que chama `extract_features_from_images`.
"""

import os
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

from . import config
from .config import PONTO_TO_DBC as ponto_to_dbc
from .preprocess import build_vegetation_mask, merge_preprocess_config, preprocess_image


def get_exif_time(img_path):
    try:
        with Image.open(img_path) as img:
            exif = img._getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        # Exif is formatted as 'YYYY:MM:DD HH:MM:SS'
                        if ' ' in val:
                            return val.split(' ')[1]
                        return val
    except Exception:
        pass
    return None


# Agregação dos valores por bloco vegetado (dentro de cada imagem)
INDEX_STATISTICS = {
    "Median": lambda arr: float(np.median(arr)),
    "Mean":   lambda arr: float(np.nanmean(arr)),
    "P75":    lambda arr: float(np.nanpercentile(arr, 75)),
    "P90":    lambda arr: float(np.nanpercentile(arr, 90)),
}

def calculate_all_indices(R_raw, G_raw, B_raw):
    """
    Calculates the 27 requested spectral indices + previously used indices (IPCA, HSL parameters).
    Input raw values should be float32 arrays (0.0 to 255.0).
    """
    # 1. Total sum and normalized coordinates r, g, b
    total = R_raw + G_raw + B_raw
    total_safe = np.where(total == 0, 1.0, total)
    
    r = R_raw / total_safe
    g = G_raw / total_safe
    b = B_raw / total_safe
    
    # 4. rg (Azevedo, 2024)
    rg = (r + g) / 2.0
    
    # 5. rb (Azevedo, 2024)
    rb = (r + b) / 2.0
    
    # 6. gb (Azevedo, 2024)
    gb = (g + b) / 2.0
    
    # 7. rgb (Olivoto, 2022)
    rgb = (r + g + b) / 3.0
    
    # Ratios (with division-by-zero protection)
    # 8. r/g (Gamon e Surfus, 1999)
    r_over_g = np.where(g == 0, 0, r / g)
    
    # 9. r/b (Zakaluk e Ranjan, 2008)
    r_over_b = np.where(b == 0, 0, r / b)
    
    # 10. g/r (Zakaluk e Ranjan, 2008)
    g_over_r = np.where(r == 0, 0, g / r)
    
    # 11. g/b (Zakaluk e Ranjan, 2008)
    g_over_b = np.where(b == 0, 0, g / b)
    
    # 12. b/r (Zarco-Tejada et al., 2005)
    b_over_r = np.where(r == 0, 0, b / r)
    
    # 13. b/g (Zarco-Tejada et al., 2005)
    b_over_g = np.where(g == 0, 0, b / g)
    
    # 14. S (Zakaluk e Ranjan, 2008)
    # Saturation formula: ((r+g+b)-3b)/(r+g+b)
    # Let's use standard chromatic: ((r+g+b) - 3*b) / (r+g+b)
    S_chromatic = np.where((r+g+b) == 0, 0, ((r+g+b) - 3.0*b) / (r+g+b))
    
    # 15. BI (Richardson e Wiegand, 1977)
    BI = np.sqrt((r**2 + g**2 + b**2) / 3.0)
    
    # 16. BIM (Olivoto, 2022)
    BIM = np.sqrt((2.0 * r + 2.0 * g + 2.0 * b) / 3.0)
    
    # 17. SCI (Mathieu et al., 1998)
    denom_sci = r + g
    denom_sci_safe = np.where(denom_sci == 0, 1.0, denom_sci)
    SCI = (r - g) / denom_sci_safe
    
    # 18. GLI (Louhaichi et al., 2001)
    denom_gli = 2.0*g + r + b
    denom_gli_safe = np.where(denom_gli == 0, 1.0, denom_gli)
    GLI = (2.0*g - r - b) / denom_gli_safe
    
    # 19. HI (Escadafal et al., 1994)
    denom_hi = g - b
    denom_hi_safe = np.where(denom_hi == 0, 1.0, denom_hi)
    HI = (2.0*r - g - b) / denom_hi_safe
    
    # 20. NGRDI (Tucker, 1979)
    denom_ngrdi = g + r
    denom_ngrdi_safe = np.where(denom_ngrdi == 0, 1.0, denom_ngrdi)
    NGRDI = (g - r) / denom_ngrdi_safe
    
    # 21. SI (Escadafal et al., 1994)
    denom_si = r + b
    denom_si_safe = np.where(denom_si == 0, 1.0, denom_si)
    SI = (r - b) / denom_si_safe
    
    # 22. VARI (Gitelson et al., 2002)
    denom_vari = g + r - b
    denom_vari_safe = np.where(denom_vari == 0, 1.0, denom_vari)
    VARI = (g - r) / denom_vari_safe
    
    # 23. HUE (Escadafal et al., 1994)
    denom_hue = 30.5 * (g - r)
    denom_hue_safe = np.where(denom_hue == 0, 1.0, denom_hue)
    HUE = np.arctan(2.0 * (b - g - r) / denom_hue_safe)
    
    # 24. MGRVI (Bendig et al., 2015)
    denom_mgrvi = g**2 + r**2
    denom_mgrvi_safe = np.where(denom_mgrvi == 0, 1.0, denom_mgrvi)
    MGRVI = (g**2 - r**2) / denom_mgrvi_safe
    
    # 25. MPRI (Yang et al., 2008)
    denom_mpri = g + r
    denom_mpri_safe = np.where(denom_mpri == 0, 1.0, denom_mpri)
    MPRI = (g - r) / denom_mpri_safe
    
    # 26. ExG (Woebbecke et al., 1995)
    ExG = 2.0 * g - r - b
    
    # 27. RGBVI (Bendig et al., 2015)
    denom_rgbvi = g**2 + b*r
    denom_rgbvi_safe = np.where(denom_rgbvi == 0, 1.0, denom_rgbvi)
    RGBVI = (g**2 - b*r) / denom_rgbvi_safe
    
    # =========================================================================
    # PREVIOUSLY USED ABSTRACTIONS (IPCA and HSL)
    # =========================================================================
    # 28. IPCA (Índice de Principais Componentes Adaptado)
    IPCA = 0.7582 * np.abs(r - b) - 0.1168 * np.abs(r - g) + 0.6414 * np.abs(g - b)
    
    # 29. HSL conversion using 0-1 values from R_raw, G_raw, B_raw
    R_n, G_n, B_n = R_raw / 255.0, G_raw / 255.0, B_raw / 255.0
    max_c = np.maximum(np.maximum(R_n, G_n), B_n)
    min_c = np.minimum(np.minimum(R_n, G_n), B_n)
    delta = max_c - min_c
    L_HSL = (max_c + min_c) / 2.0
    
    # Saturation HSL
    S_HSL = np.zeros_like(L_HSL)
    non_zero = delta != 0
    L_lt_05 = (L_HSL < 0.5) & non_zero
    L_ge_05 = (L_HSL >= 0.5) & non_zero
    S_HSL[L_lt_05] = delta[L_lt_05] / (max_c[L_lt_05] + min_c[L_lt_05])
    S_HSL[L_ge_05] = delta[L_ge_05] / (2.0 - max_c[L_ge_05] - min_c[L_ge_05])
    
    # Hue HSL
    H_HSL = np.zeros_like(L_HSL)
    if np.any(non_zero):
        mask_r = (max_c == R_n) & non_zero
        mask_g = (max_c == G_n) & non_zero
        mask_b = (max_c == B_n) & non_zero
        delta_safe = np.where(delta == 0, 1.0, delta)
        
        H_HSL[mask_r] = ((G_n[mask_r] - B_n[mask_r]) / delta_safe[mask_r]) % 6
        H_HSL[mask_g] = (B_n[mask_g] - R_n[mask_g]) / delta_safe[mask_g] + 2
        H_HSL[mask_b] = (R_n[mask_b] - G_n[mask_b]) / delta_safe[mask_b] + 4
        H_HSL = H_HSL * 60.0
        H_HSL[H_HSL < 0] += 360.0
        
    return {
        "r": r, "g": g, "b": b, "rg": rg, "rb": rb, "gb": gb, "rgb": rgb,
        "r_over_g": r_over_g, "r_over_b": r_over_b, "g_over_r": g_over_r,
        "g_over_b": g_over_b, "b_over_r": b_over_r, "b_over_g": b_over_g,
        "S": S_chromatic, "BI": BI, "BIM": BIM, "SCI": SCI, "GLI": GLI, "HI": HI,
        "NGRDI": NGRDI, "SI": SI, "VARI": VARI, "HUE": HUE, "MGRVI": MGRVI,
        "MPRI": MPRI, "ExG": ExG, "RGBVI": RGBVI,
        
        # Original features
        "IPCA": IPCA,
        "H": H_HSL,
        "S_HSL": S_HSL,
        "L": L_HSL
    }

def extract_features_from_images(
    image_root=None,
    block_size=10,
    threshold=0.0,
    preprocess_config=None,
    verbose=True,
):
    image_root = str(image_root or config.IMAGE_ROOT)
    all_rows = []
    pp_cfg = merge_preprocess_config(preprocess_config)
    if threshold != 0.0:
        pp_cfg = merge_preprocess_config({**pp_cfg, "veg_threshold": threshold})
    
    if not os.path.exists(image_root):
        if verbose:
            print(f"Error: image root directory {image_root} does not exist!")
        return pd.DataFrame()
        
    # List subdirectories representing dates (e.g., 18-05, 21-05, etc.)
    date_folders = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
    
    for folder in date_folders:
        folder_path = os.path.join(image_root, folder)
        # Format date as DD-MM-YYYY (assume year 2026 based on previous scripts)
        date_str = f"{folder}-{config.YEAR}"
        if verbose:
            print(f"\nProcessing folder: {folder} (Date: {date_str})")
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            m = re.match(r'^[Pp](\d+)', filename)
            if not m:
                if verbose:
                    print(f"  Skipping non-point file: {filename}")
                continue

            ponto_id = int(m.group(1))
            if folder == "01-06":
                dbc_info = {"Bloco": None, "Tratamento": None, "Dose": None}
            else:
                if ponto_id not in ponto_to_dbc:
                    if verbose:
                        print(f"  Skipping point P{ponto_id} (not in layout map)")
                    continue
                dbc_info = ponto_to_dbc[ponto_id]

            img_path = os.path.join(folder_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                if verbose:
                    print(f"  Error: Could not read image {img_path}")
                continue
                
            img_float, _ = preprocess_image(img, pp_cfg)

            # Downsample block by block ignoring NaNs
            ch, cw, _ = img_float.shape
            new_w = cw // block_size
            new_h = ch // block_size
            h_trimmed = new_h * block_size
            w_trimmed = new_w * block_size
            img_trimmed = img_float[:h_trimmed, :w_trimmed]
            
            reshaped = img_trimmed.reshape(new_h, block_size, new_w, block_size, 3)
            with np.errstate(invalid='ignore'):
                img_down = np.nanmean(reshaped, axis=(1, 3))
            
            # Extract BGR channels
            B_raw = img_down[:, :, 0]
            G_raw = img_down[:, :, 1]
            R_raw = img_down[:, :, 2]
            
            # Compute indices
            indices = calculate_all_indices(R_raw, G_raw, B_raw)
            
            veg_mask = build_vegetation_mask(indices, pp_cfg)
            
            num_veg_blocks = int(np.sum(veg_mask))
            total_blocks = new_h * new_w
            canopy_cover = 100.0 * num_veg_blocks / total_blocks if total_blocks > 0 else 0.0
            
            photo_time = get_exif_time(img_path)
            row = {
                "Ponto": ponto_id,
                "Bloco": dbc_info["Bloco"],
                "Tratamento": dbc_info["Tratamento"],
                "Dose": dbc_info["Dose"],
                "Data": date_str,
                "File_Name": filename,
                "Photo_Time": photo_time,
                "Canopy_Cover": canopy_cover,
                "Vegetation_Blocks": num_veg_blocks,
                "Total_Blocks": total_blocks
            }
            
            # Mediana, média e percentis de cada índice na máscara de vegetação
            for key, val_arr in indices.items():
                if num_veg_blocks > 0:
                    veg_vals = val_arr[veg_mask]
                else:
                    veg_vals = np.array([0.0])
                for stat_name, agg_fn in INDEX_STATISTICS.items():
                    row[f"{stat_name}_{key}"] = agg_fn(veg_vals)
                
            all_rows.append(row)
            if verbose:
                print(f"  P{ponto_id}: Loaded {filename} | Cropped {ch}x{cw} -> Down {new_h}x{new_w} | Veg Blocks: {num_veg_blocks}/{total_blocks} ({canopy_cover:.2f}%)")
            
    df = pd.DataFrame(all_rows)
    return df
