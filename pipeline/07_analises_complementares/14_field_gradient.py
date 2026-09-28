"""
O terreno tem um lado mais fértil? Gradiente espacial do SPAD a partir do GPS das leituras do Falker.

Posição de cada parcela = mediana das coordenadas das suas leituras (convertidas em metros a partir do canto
sudoeste). Precisão típica do GPS: alguns metros, parecida com o tamanho de uma parcela.
  - Resíduo de cada parcela na ANOVA em blocos (SPAD − efeito da dose − média), por data e na média das datas;
  - regressão do resíduo em x (leste) e y (norte): existe tendência que o bloco não absorveu?
  - o mesmo com o SPAD sem descontar a dose e com o bloco como fator, para ver se o efeito de bloco é espacial;
  - I de Moran dos resíduos (vizinhos = 3 parcelas mais próximas), com teste de permutação.

Tabelas: ext_field_positions, ext_field_gradient.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all
from tcc_analysis.science import plot_table

warnings.filterwarnings("ignore")
N_PERM = 5000


def moran(values, xy, k=3, rng=None, n_perm=N_PERM):
    n = len(values)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(d[i])[:k]] = 1
    W = (W + W.T) / 2
    def I(v):
        z = v - v.mean()
        return n / W.sum() * (z @ W @ z) / (z @ z)
    obs = I(values)
    perm = np.array([I(rng.permutation(values)) for _ in range(n_perm)])
    return float(obs), float((np.sum(perm >= obs) + 1) / (n_perm + 1))


def main():
    rng = np.random.default_rng(42)
    f = load_falker_all()
    f = f[f.Data.isin(config.EXPERIMENTAL_DATES)].copy()
    f["Ponto"] = f.Ponto.astype(int)
    lat0, lon0 = f.Latitude.min(), f.Longitude.min()
    f["x_m"] = (f.Longitude - lon0) * 111320 * np.cos(np.radians(f.Latitude.mean()))
    f["y_m"] = (f.Latitude - lat0) * 110540
    pos = f.groupby("Ponto")[["x_m", "y_m"]].median().reset_index()
    spread = f.groupby("Ponto")[["x_m", "y_m"]].std().mean().mean()

    df = plot_table()
    df = df.merge(pos, on="Ponto")
    df["Resid"] = smf.ols("SPAD ~ C(Dose) + C(Data)", df).fit().resid
    mean_res = df.groupby("Ponto").agg(Resid=("Resid", "mean"), x_m=("x_m", "first"), y_m=("y_m", "first"),
                                        Bloco=("Bloco", "first"), Dose=("Dose", "first")).reset_index()

    rows = []
    for scope, sub in [("Média das datas", mean_res)] + [(d, df[df.Data == d]) for d in config.EXPERIMENTAL_DATES]:
        s = sub.copy()
        if scope != "Média das datas":
            s["Resid"] = smf.ols("SPAD ~ C(Dose)", s).fit().resid
        fit = smf.ols("Resid ~ x_m + y_m", s).fit()
        fit_b = smf.ols("Resid ~ C(Bloco)", s).fit()
        fit_bxy = smf.ols("Resid ~ C(Bloco) + x_m + y_m", s).fit()
        I, p = moran(s.Resid.to_numpy(), s[["x_m", "y_m"]].to_numpy(), rng=rng)
        rows.append({"Escopo": scope, "n": len(s), "Inclinacao_leste_SPAD_por_10m": 10 * fit.params["x_m"], "p_leste": fit.pvalues["x_m"],
                     "Inclinacao_norte_SPAD_por_10m": 10 * fit.params["y_m"], "p_norte": fit.pvalues["y_m"],
                     "R2_gradiente": fit.rsquared, "p_gradiente": fit.f_pvalue, "R2_bloco": fit_b.rsquared, "p_bloco": fit_b.f_pvalue,
                     "R2_bloco_mais_gradiente": fit_bxy.rsquared, "Moran_I": I, "p_Moran": p})
    grad = pd.DataFrame(rows)
    pos = pos.merge(mean_res[["Ponto", "Resid", "Bloco", "Dose"]], on="Ponto")
    pos["Dispersao_GPS_m"] = spread
    db.save_df(pos, "ext_field_positions", description="Posição de cada parcela (GPS das leituras, em metros) e resíduo médio do SPAD")
    db.save_df(grad, "ext_field_gradient", description="Gradiente espacial do SPAD (resíduo da dose) e I de Moran")
    print(f"dispersão média do GPS dentro da parcela: {spread:.1f} m")
    print(pos.round(2).to_string(index=False))
    print(grad.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
