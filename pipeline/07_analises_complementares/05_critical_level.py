"""
Nível crítico de clorofila para a produção (26/05, a única data com corte; 12 parcelas).

Produção relativa (PR) = massa da parcela ÷ maior massa do experimento, para massa seca total e massa seca de folha.
  - Cate e Nelson (1971), versão estatística: os pontos são divididos em dois grupos por um valor x; o nível crítico
    é o x que maximiza o R² da PR explicada pelos dois grupos (média de cada grupo como previsão);
  - linear-platô: PR = a + b·min(x, x0); x0 é o ponto a partir do qual a produção não aumenta mais.
x = SPAD do Falker e as medidas da foto (orientadas para crescer com a clorofila). Para comparar, o nível crítico
da foto é convertido em SPAD pela reta índice → SPAD da própria data. IC 95 % por bootstrap das parcelas.
Com 12 parcelas o resultado é indicativo.

Tabelas: sci_critical_level, sci_critical_points.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.data import load_vegetation_plots
from tcc_analysis.science import MEASURE_LABELS, PHOTO_MEASURES, orientation, plot_table

warnings.filterwarnings("ignore")
DATE = "26-05-2026"
N_BOOT = 2000
YIELDS = {"Massa_Seca_Total_ha": "Massa seca total", "Massa_Seca_Folha_ha": "Massa seca de folha"}


def cate_nelson(x, y):
    """Nível crítico (ponto médio entre dois valores consecutivos de x) que maximiza o R² de dois grupos."""
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    tss = ((ys - ys.mean()) ** 2).sum()
    best = (np.nan, -np.inf, 0)
    for i in range(2, len(xs) - 1):   # pelo menos 2 pontos em cada grupo
        if xs[i] == xs[i - 1]:
            continue
        lo, hi = ys[:i], ys[i:]
        rss = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        r2 = 1 - rss / tss
        if r2 > best[1]:
            best = ((xs[i - 1] + xs[i]) / 2, r2, i)
    return best


def plateau(x, y):
    grid = np.linspace(np.percentile(x, 15), np.percentile(x, 85), 200)
    best = (np.nan, np.inf, np.nan, np.nan)
    for x0 in grid:
        z = np.minimum(x, x0)
        if np.std(z) == 0:
            continue
        b, a = np.polyfit(z, y, 1)
        rss = ((y - (a + b * z)) ** 2).sum()
        if rss < best[1] and b > 0:
            best = (x0, rss, a, b)
    tss = ((y - y.mean()) ** 2).sum()
    return best[0], 1 - best[1] / tss if np.isfinite(best[1]) else np.nan


def main():
    rng = np.random.default_rng(42)
    df = plot_table()
    df = df[df.Data == DATE].copy()
    veg = load_vegetation_plots(verbose=False)
    veg["Ponto"] = veg.Ponto.astype(int)
    df = df.merge(veg[["Ponto"] + list(YIELDS)], on="Ponto")
    measures = ["SPAD"] + PHOTO_MEASURES + ["SPAD_rede"]
    sign = {m: (1.0 if m in ("SPAD", "SPAD_rede") else orientation(plot_table(), m)) for m in measures}

    rows, pts = [], []
    for ycol, ylabel in YIELDS.items():
        ry = (df[ycol] / df[ycol].max()).to_numpy()
        for m in measures:
            x_raw = df[m].to_numpy(float)
            x = sign[m] * x_raw
            # conversão para SPAD pela reta da própria data (só para comparar com o Falker)
            b_cal, a_cal = np.polyfit(x, df.SPAD.to_numpy(), 1)
            to_spad = (lambda v: v) if m == "SPAD" else (lambda v, a=a_cal, b=b_cal: a + b * v)
            cn_x, cn_r2, cn_i = cate_nelson(x, ry)
            lp_x, lp_r2 = plateau(x, ry)
            boot_cn, boot_lp = [], []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(x), len(x))
                if np.unique(x[idx]).size < 5:
                    continue
                boot_cn.append(cate_nelson(x[idx], ry[idx])[0])
                boot_lp.append(plateau(x[idx], ry[idx])[0])
            below = x < cn_x
            for method, xc, r2, boot in [("Cate-Nelson", cn_x, cn_r2, boot_cn), ("Linear-platô", lp_x, lp_r2, boot_lp)]:
                boot = np.array(boot, float)
                rows.append({"Producao": ylabel, "Medida": m, "Rotulo": MEASURE_LABELS[m], "Metodo": method, "n": len(x),
                             "Nivel_critico": sign[m] * xc, "Nivel_critico_SPAD_equiv": to_spad(xc), "R2": r2,
                             "IC_min_SPAD_equiv": to_spad(np.nanpercentile(boot, 2.5)), "IC_max_SPAD_equiv": to_spad(np.nanpercentile(boot, 97.5)),
                             "n_abaixo": int(below.sum()), "n_acima": int((~below).sum()),
                             "PR_media_abaixo": float(ry[below].mean()), "PR_media_acima": float(ry[~below].mean())})
            for i, r in df.reset_index(drop=True).iterrows():
                pts.append({"Producao": ylabel, "Medida": m, "Ponto": int(r.Ponto), "Dose": r.Dose, "x": x_raw[i],
                            "x_SPAD_equiv": to_spad(x[i]), "Producao_relativa": ry[i], "Massa": r[ycol]})
    out = pd.DataFrame(rows)
    db.save_df(out, "sci_critical_level", description="Nível crítico de clorofila para a produção (26/05): Cate-Nelson e linear-platô")
    db.save_df(pd.DataFrame(pts), "sci_critical_points", description="Pontos: produção relativa × SPAD e medidas da foto (26/05)")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
