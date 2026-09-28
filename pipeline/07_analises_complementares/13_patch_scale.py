"""
Escala das manchas: a desuniformidade de cor é feita de manchas pequenas espalhadas ou de áreas grandes?

Para cada foto (pré-processamento definitivo, blocos 10×10 px), o mapa de b/r nos blocos de vegetação:
  - correlograma: correlação entre blocos separados por 1, 2, 4, 8, 16 e 32 blocos (na horizontal e na vertical);
    comprimento de correlação = distância em que a correlação cai abaixo de 0,5 (interpolada), em blocos e em % da
    largura da imagem;
  - manchas amareladas: blocos no quinto mais baixo de b/r da própria foto (os menos verdes); tamanho médio das
    manchas (componentes conectados, em blocos) e fração da área amarelada que está em manchas de ≥ 10 blocos.
Relação com a dose (ANOVA em blocos, datas com todas as parcelas fotografadas) e com o SPAD.

Tabelas: ext_patch_photos, ext_patch_tests, ext_patch_correlogram.
"""

import os
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.indices import calculate_all_indices
from tcc_analysis.preprocess import build_vegetation_mask, merge_preprocess_config, preprocess_image
from tcc_analysis.science import dbc_anova, plot_table

warnings.filterwarnings("ignore")
LAGS = [1, 2, 4, 8, 16, 32]
BLOCK = 10


def correlogram(z, mask):
    out = {}
    for lag in LAGS:
        num, den1, den2, n = 0.0, 0.0, 0.0, 0
        rs = []
        for a, b, ma, mb in [(z[:, :-lag], z[:, lag:], mask[:, :-lag], mask[:, lag:]), (z[:-lag, :], z[lag:, :], mask[:-lag, :], mask[lag:, :])]:
            m = ma & mb
            if m.sum() > 50:
                rs.append(np.corrcoef(a[m], b[m])[0, 1])
        out[lag] = float(np.mean(rs)) if rs else np.nan
    return out


def corr_length(cg):
    lags = np.array(LAGS, float)
    r = np.array([cg[k] for k in LAGS])
    below = np.flatnonzero(r < 0.5)
    if not below.size:
        return float(lags[-1])
    i = below[0]
    if i == 0:
        return float(lags[0])
    # interpolação em log(lag)
    x0, x1 = np.log(lags[i - 1]), np.log(lags[i])
    return float(np.exp(x0 + (0.5 - r[i - 1]) * (x1 - x0) / (r[i] - r[i - 1])))


def main():
    pp = merge_preprocess_config(config.BEST_PREPROCESS_CONFIG)
    root = str(config.IMAGE_ROOT)
    rows, cgrows = [], []
    for folder in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        date = f"{folder}-{config.YEAR}"
        if date not in config.EXPERIMENTAL_DATES:
            continue
        for fn in sorted(os.listdir(os.path.join(root, folder))):
            m = re.match(r"^[Pp](\d+)", fn)
            if not m:
                continue
            img = cv2.imread(os.path.join(root, folder, fn))
            if img is None:
                continue
            f, _ = preprocess_image(img, pp)
            h, w, _ = f.shape
            nh, nw = h // BLOCK, w // BLOCK
            with np.errstate(invalid="ignore"):
                down = np.nanmean(f[:nh * BLOCK, :nw * BLOCK].reshape(nh, BLOCK, nw, BLOCK, 3), axis=(1, 3))
            ind = calculate_all_indices(down[:, :, 2], down[:, :, 1], down[:, :, 0])
            mask = build_vegetation_mask(ind, pp) & np.isfinite(ind["b_over_r"])
            br = ind["b_over_r"]
            z = np.where(mask, (br - np.nanmean(br[mask])) / np.nanstd(br[mask]), 0.0)
            cg = correlogram(z, mask)
            thr = np.percentile(br[mask], 20)
            yellow = mask & (br <= thr)
            lab, n = ndimage.label(yellow)
            sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
            rows.append({"Data": date, "Ponto": int(m.group(1)), "Blocos_vegetacao": int(mask.sum()), "Largura_blocos": nw,
                         "Comprimento_correlacao_blocos": corr_length(cg),
                         "Comprimento_correlacao_pct_largura": 100 * corr_length(cg) / nw,
                         "Manchas_n": int(n), "Mancha_tamanho_medio": float(sizes.mean()),
                         "Mancha_tamanho_ponderado": float((sizes ** 2).sum() / sizes.sum()) if sizes.sum() else 0.0,
                         "Fracao_em_manchas_grandes": float(sizes[sizes >= 10].sum() / sizes.sum()) if sizes.sum() else 0.0,
                         **{f"r_lag_{k}": v for k, v in cg.items()}})
    ph = pd.DataFrame(rows)
    pt = plot_table()[["Data", "Ponto", "Bloco", "Dose", "SPAD", "CV_b_over_r", "Cobertura"]]
    ph = ph.merge(pt, on=["Data", "Ponto"], how="left")

    tests = []
    for col, lab in [("Comprimento_correlacao_blocos", "Comprimento de correlação (blocos)"),
                     ("Comprimento_correlacao_pct_largura", "Comprimento de correlação (% da largura)"),
                     ("Mancha_tamanho_ponderado", "Tamanho típico das manchas amareladas (blocos)"),
                     ("Fracao_em_manchas_grandes", "Fração do amarelado em manchas ≥ 10 blocos"),
                     ("r_lag_1", "Correlação entre blocos vizinhos")]:
        rec = {"Medida": col, "Rotulo": lab}
        for d in ["21-05-2026", "26-05-2026"]:
            an = dbc_anova(ph[ph.Data == d], col)
            rec[f"F_dose_{d[:5]}"], rec[f"p_dose_{d[:5]}"] = an.get("F_dose"), an.get("p_dose")
        r = stats.spearmanr(ph.Dose, ph[col])
        rs = stats.spearmanr(ph.SPAD, ph[col])
        rc = stats.spearmanr(ph.CV_b_over_r, ph[col])
        rec.update({"rho_dose": r.correlation, "p_rho_dose": r.pvalue, "rho_SPAD": rs.correlation, "p_rho_SPAD": rs.pvalue,
                    "rho_CV_cor": rc.correlation, **{f"Media_{int(d)}kg": x for d, x in ph.groupby("Dose")[col].mean().items()}})
        tests.append(rec)
    cgl = ph.melt(id_vars=["Data", "Ponto", "Dose"], value_vars=[f"r_lag_{k}" for k in LAGS], var_name="lag", value_name="r")
    cgl["Lag_blocos"] = cgl.lag.str.replace("r_lag_", "").astype(int)
    cgm = cgl.groupby(["Dose", "Lag_blocos"]).r.agg(["mean", "std", "count"]).reset_index()
    db.save_df(ph, "ext_patch_photos", description="Escala espacial da cor em cada foto: correlograma e manchas amareladas")
    db.save_df(pd.DataFrame(tests), "ext_patch_tests", description="Escala das manchas × dose, SPAD e variação de cor")
    db.save_df(cgm, "ext_patch_correlogram", description="Correlograma médio de b/r entre blocos por dose")
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    print(cgm.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
