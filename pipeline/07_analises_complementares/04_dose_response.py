"""
Resposta à dose de N: a foto aponta a mesma dose ótima que o Falker?

Em cada data, para o SPAD (Falker, 12 parcelas) e para as medidas da foto (parcelas fotografadas), ajustam-se:
  - linear:            y = a + b·dose
  - quadrático:        y = a + b·dose + c·dose²   → dose de máxima (vértice), se c < 0 e dentro de 0–100 kg
  - linear-platô:      y = a + b·min(dose, x0)     → x0 = dose a partir da qual não há mais ganho
Escolha pelo AICc. As medidas da foto são orientadas para crescer com a clorofila (IPCA e SI decrescem) e
padronizadas (z) dentro da data; a dose ótima não depende dessa transformação linear. IC 95 % da dose ótima por
bootstrap (2000 reamostragens de parcelas dentro de cada dose). Com 4 doses e 3 blocos, os intervalos são largos.

Tabelas: sci_dose_response_fits, sci_dose_response_best, sci_dose_response_means.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import MEASURE_LABELS, PHOTO_MEASURES, orientation, plot_table

warnings.filterwarnings("ignore")
N_BOOT = 2000
X0_GRID = np.arange(10, 100.5, 1.0)


def aicc(rss, n, k):
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)


def fit_linear(x, y):
    b, a = np.polyfit(x, y, 1)
    return {"a": a, "b": b}, float(((y - (a + b * x)) ** 2).sum()), 2


def fit_quadratic(x, y):
    c, b, a = np.polyfit(x, y, 2)
    return {"a": a, "b": b, "c": c}, float(((y - (a + b * x + c * x ** 2)) ** 2).sum()), 3


def fit_plateau(x, y):
    """Mínimos quadrados em y = a + b·min(x, x0) para todos os x0 da grade de uma vez."""
    z = np.minimum(x[:, None], X0_GRID[None, :])
    zc = z - z.mean(0)
    szz = (zc ** 2).sum(0)
    ok = szz > 1e-12
    b = np.where(ok, (zc * (y - y.mean())[:, None]).sum(0) / np.where(ok, szz, 1), 0.0)
    a = y.mean() - b * z.mean(0)
    rss = ((y[:, None] - (a + b * z)) ** 2).sum(0)
    rss[~ok] = np.inf
    j = int(np.argmin(rss))
    return {"a": float(a[j]), "b": float(b[j]), "x0": float(X0_GRID[j])}, float(rss[j]), 3


def optimum(model, p):
    if model == "quadratico":
        if p["c"] >= 0:
            return np.nan
        v = -p["b"] / (2 * p["c"])
        return v if 0 < v <= 100 else np.nan
    if model == "linear_plato":
        return p["x0"] if p["b"] > 0 and p["x0"] < 100 else np.nan
    return np.nan


def fit_all(x, y):
    out = {}
    for name, fn in [("linear", fit_linear), ("quadratico", fit_quadratic), ("linear_plato", fit_plateau)]:
        p, rss, k = fn(x, y)
        tss = float(((y - y.mean()) ** 2).sum())
        out[name] = {"params": p, "rss": rss, "R2": 1 - rss / tss if tss else np.nan, "AICc": aicc(rss, len(y), k)}
    return out


def main():
    rng = np.random.default_rng(42)
    df = plot_table()
    measures = ["SPAD"] + PHOTO_MEASURES + ["SPAD_rede"]
    sign = {m: (1.0 if m in ("SPAD", "SPAD_rede") else orientation(df, m)) for m in measures}

    fits, best_rows, means = [], [], []
    for d in config.EXPERIMENTAL_DATES:
        for m in measures:
            sub = df[(df.Data == d) & df[m].notna()]
            if sub.Dose.nunique() < 3:
                continue
            x = sub.Dose.to_numpy(float)
            y_raw = sign[m] * sub[m].to_numpy(float)
            y = (y_raw - y_raw.mean()) / y_raw.std(ddof=1)
            res = fit_all(x, y)
            best = min(res, key=lambda k: res[k]["AICc"])
            for name, r in res.items():
                fits.append({"Data": d, "Medida": m, "Rotulo": MEASURE_LABELS[m], "n": len(y), "Modelo": name, "R2": r["R2"],
                             "AICc": r["AICc"], "Delta_AICc": r["AICc"] - res[best]["AICc"],
                             **{f"p_{k}": v for k, v in r["params"].items()}, "Dose_otima": optimum(name, r["params"])})
            # bootstrap: reamostra parcelas dentro de cada dose e refaz a escolha de modelo
            boot = []
            groups = [np.flatnonzero(x == v) for v in np.unique(x)]
            for _ in range(N_BOOT):
                idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                rb = fit_all(x[idx], y[idx])
                bb = min(rb, key=lambda k: rb[k]["AICc"])
                boot.append(optimum(bb, rb[bb]["params"]))
            boot = np.array(boot, float)
            opt = optimum(best, res[best]["params"])
            best_rows.append({"Data": d, "Medida": m, "Rotulo": MEASURE_LABELS[m], "n": len(y), "Melhor_modelo": best,
                              "R2": res[best]["R2"], "Dose_otima": opt,
                              "Dose_otima_IC_min": float(np.nanpercentile(boot, 2.5)) if np.isfinite(boot).mean() > 0.5 else np.nan,
                              "Dose_otima_IC_max": float(np.nanpercentile(boot, 97.5)) if np.isfinite(boot).mean() > 0.5 else np.nan,
                              "Prop_boot_com_otimo": float(np.isfinite(boot).mean()),
                              "Resposta": "cresce até 100 kg (sem máximo)" if not np.isfinite(opt) else f"máximo/platô em {opt:.0f} kg"})
            for dose in sorted(sub.Dose.unique()):
                v = y[x == dose]
                means.append({"Data": d, "Medida": m, "Rotulo": MEASURE_LABELS[m], "Dose": dose, "Media_z": v.mean(),
                              "EP_z": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, "n": len(v),
                              "Media_original": sub[m][sub.Dose == dose].mean()})

    db.save_df(pd.DataFrame(fits), "sci_dose_response_fits", description="Resposta à dose: modelos linear, quadrático e linear-platô por data e medida")
    db.save_df(pd.DataFrame(best_rows), "sci_dose_response_best", description="Dose ótima (vértice ou platô) por data e medida, com IC bootstrap")
    db.save_df(pd.DataFrame(means), "sci_dose_response_means", description="Médias padronizadas por dose, data e medida")
    print(pd.DataFrame(best_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
