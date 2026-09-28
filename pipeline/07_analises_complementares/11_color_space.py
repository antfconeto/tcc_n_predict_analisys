"""
Cor da folha × cor da luz: no espaço de cor, a clorofila e a troca de data empurram as fotos na mesma direção?

Cor de cada foto = coordenadas cromáticas medianas dos blocos de vegetação (r, g, b somam 1: dois graus de liberdade;
usa-se x = r − b e y = 2g − r − b, eixos ortogonais do triângulo cromático).
  1. Modelo multivariado por canal: cor = a + β·SPAD + efeito da data. β é a "direção da clorofila" (dentro de cada
     data, onde a luz é a mesma); o efeito da data é a mudança de cor que o SPAD real não explica ("direção da luz").
  2. Ângulo entre as duas direções e o quanto a mudança de luz de cada data se parece com clorofila: projeção do
     efeito da data sobre β, em SPAD equivalente — o viés que um modelo de cor leria como clorofila.
     Comparado com o viés por data do Bland-Altman (sci_bland_altman).
  3. Correção: se a luz empurra numa direção diferente da clorofila, a componente da cor ORTOGONAL à direção da luz
     deveria ser mais estável entre datas. A direção da luz é estimada só com as datas de treino (diferença entre as
     duas), e a reta SPAD ~ componente é testada na data que ficou fora (R² e acerto das 3 classes), comparada com a
     reta nas coordenadas brutas.

Tabelas: ext_color_directions, ext_color_points, ext_color_correction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.science import plot_table


def classify(v, limits):
    v = np.asarray(v, float)
    return np.where(v < limits[0], 0, np.where(v <= limits[1], 1, 2))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_color_model(C, spad, dates):
    """Mínimos quadrados de cada canal em [1, SPAD, dummies de data]; devolve β (2,) e efeitos de data (dict)."""
    ud = sorted(set(dates))
    X = np.column_stack([np.ones(len(spad)), spad] + [(dates == d).astype(float) for d in ud[1:]])
    B = np.linalg.lstsq(X, C, rcond=None)[0]
    beta = B[1]
    eff = {ud[0]: np.zeros(2)}
    for i, d in enumerate(ud[1:]):
        eff[d] = B[2 + i]
    return beta, eff


def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    df = plot_table()
    df = df[df.Tem_foto].reset_index(drop=True)
    limits = db.load_df("chl_class_scheme").sort_values("Ordem").SPAD_max.dropna().tolist()
    r, g, b = df.Median_r.to_numpy(), df.Median_g.to_numpy(), df.Median_b.to_numpy()
    C = np.column_stack([r - b, 2 * g - r - b])
    spad, dates = df.SPAD.to_numpy(float), df.Data.to_numpy()

    # 1–2. direções com todas as datas
    beta, eff = fit_color_model(C, spad, dates)
    ref = sorted(set(dates))[0]
    mean_spad = df.groupby("Data").SPAD.mean()
    ba = db.load_df("sci_bland_altman")
    ba = ba[ba.Metodo == "Reta no melhor índice"].iloc[0] if (ba.Metodo == "Reta no melhor índice").any() else None
    rows = [{"Item": "Direção da clorofila (por 1 SPAD)", "dx": beta[0], "dy": beta[1], "Norma": float(np.linalg.norm(beta))}]
    centered = {d: eff[d] - np.mean([eff[k] for k in eff], axis=0) for d in eff}
    for d in sorted(eff):
        e = centered[d]
        proj_spad = float(np.dot(e, beta) / np.dot(beta, beta))
        rows.append({"Item": f"Efeito da luz em {d[:5]} (relativo à média das datas)", "Data": d, "dx": e[0], "dy": e[1],
                     "Norma": float(np.linalg.norm(e)), "Angulo_com_clorofila_graus": angle(e, beta) if np.linalg.norm(e) > 0 else np.nan,
                     "Parece_clorofila_SPAD_equiv": proj_spad,
                     "Vies_BlandAltman_reta_indice": float(ba[f"Vies_{d[:5]}"]) if ba is not None else np.nan,
                     "SPAD_medio_real": float(mean_spad[d])})
    # direção principal da luz (1ª componente dos efeitos de data centrados)
    E = np.array([centered[d] for d in sorted(eff)])
    light = np.linalg.svd(E, full_matrices=False)[2][0]
    rows.append({"Item": "Direção principal da luz", "dx": light[0], "dy": light[1], "Norma": 1.0,
                 "Angulo_com_clorofila_graus": min(angle(light, beta), angle(-light, beta))})
    # parcela de variância da cor explicada por clorofila × data
    tot = ((C - C.mean(0)) ** 2).sum()
    X_s = np.column_stack([np.ones(len(spad)), spad])
    res_s = C - X_s @ np.linalg.lstsq(X_s, C, rcond=None)[0]
    Xd = np.column_stack([np.ones(len(spad))] + [(dates == d).astype(float) for d in sorted(set(dates))[1:]])
    res_d = C - Xd @ np.linalg.lstsq(Xd, C, rcond=None)[0]
    Xsd = np.column_stack([Xd, spad])
    res_sd = C - Xsd @ np.linalg.lstsq(Xsd, C, rcond=None)[0]
    rows.append({"Item": "Variância da cor explicada só pelo SPAD", "Norma": 1 - (res_s ** 2).sum() / tot})
    rows.append({"Item": "Variância da cor explicada só pela data", "Norma": 1 - (res_d ** 2).sum() / tot})
    rows.append({"Item": "Variância da cor explicada por SPAD + data", "Norma": 1 - (res_sd ** 2).sum() / tot})

    # 3. correção: componente ortogonal à direção da luz, estimada só com as datas de treino
    ud = sorted(set(dates))
    preds = {k: np.empty(len(spad)) for k in ("bruta_2d", "ortogonal_luz", "projecao_clorofila")}
    detail = []
    for d in ud:
        te = dates == d
        tr = ~te
        beta_tr, eff_tr = fit_color_model(C[tr], spad[tr], dates[tr])
        ds = sorted(eff_tr)
        lvec = eff_tr[ds[1]] - eff_tr[ds[0]]
        lvec = lvec / np.linalg.norm(lvec)
        perp = np.array([-lvec[1], lvec[0]])
        # (a) reta em 2D bruto
        A = np.column_stack([np.ones(tr.sum()), C[tr]])
        coef = np.linalg.lstsq(A, spad[tr], rcond=None)[0]
        preds["bruta_2d"][te] = np.column_stack([np.ones(te.sum()), C[te]]) @ coef
        # (b) só a componente ortogonal à luz
        z = C @ perp
        bb, aa = np.polyfit(z[tr], spad[tr], 1)
        preds["ortogonal_luz"][te] = aa + bb * z[te]
        # (c) projeção na direção da clorofila (sem correção)
        zc = C @ (beta_tr / np.linalg.norm(beta_tr))
        bb, aa = np.polyfit(zc[tr], spad[tr], 1)
        preds["projecao_clorofila"][te] = aa + bb * zc[te]
        detail.append({"Data_testada": d, "Angulo_luz_clorofila_no_treino": min(angle(lvec, beta_tr), angle(-lvec, beta_tr)),
                       **{f"R2_{k}": r2(spad[te], p[te]) for k, p in preds.items()},
                       **{f"Vies_{k}": float(np.mean(p[te] - spad[te])) for k, p in preds.items()}})
    corr = []
    labels = {"bruta_2d": "Cor bruta (r − b e 2g − r − b)", "projecao_clorofila": "Projeção na direção da clorofila",
              "ortogonal_luz": "Componente ortogonal à direção da luz (corrigida)"}
    yc = classify(spad, limits)
    for k, p in preds.items():
        corr.append({"Modelo": labels[k], "R2_data_nova": r2(spad, p), "MAE": float(np.mean(np.abs(spad - p))),
                     "Classe_exata": float(np.mean(classify(p, limits) == yc)),
                     **{f"Vies_{d[:5]}": float(np.mean(p[dates == d] - spad[dates == d])) for d in ud}})
    corr = pd.concat([pd.DataFrame(corr), pd.DataFrame(detail)], ignore_index=True)

    pts = df[["Data", "Ponto", "Dose", "SPAD"]].copy()
    pts["x_r_menos_b"], pts["y_2g_menos_r_b"] = C[:, 0], C[:, 1]
    db.save_df(pd.DataFrame(rows), "ext_color_directions", description="Direção da clorofila × direção da luz no espaço de cor")
    db.save_df(pts, "ext_color_points", description="Cor de cada foto no plano cromático (r − b, 2g − r − b)")
    db.save_df(corr, "ext_color_correction", description="Correção da luz: componente da cor ortogonal à direção da luz, testada numa data nova")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
