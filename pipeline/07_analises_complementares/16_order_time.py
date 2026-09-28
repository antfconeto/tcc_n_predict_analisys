"""
Ordem e horário: o valor "anda" ao longo da sessão de coleta?

Falker (horário de cada leitura):
  - dentro da parcela: posição da leitura na sequência (1ª a 7ª) × desvio em relação à média da parcela (a primeira
    folha medida é diferente das outras?);
  - ao longo da sessão: resíduo de cada parcela (SPAD − efeito da dose e do bloco, por data) × horário médio das suas
    leituras.
Foto (horário do EXIF):
  - resíduo de cada foto na reta índice → SPAD da própria data × horário da foto: a luz mudou durante a sessão?
    Medidas: b/r e IPCA (medianas) e a luminosidade média (L), que depende diretamente da luz.

Tabelas: ext_order_time, ext_session_points.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all, load_indices
from tcc_analysis.science import plot_table

warnings.filterwarnings("ignore")


def minutes(t):
    try:
        h, m, s = (int(x) for x in str(t).split(":"))
        return h * 60 + m + s / 60
    except Exception:
        return np.nan


def main():
    f = load_falker_all()
    f = f[f.Data.isin(config.EXPERIMENTAL_DATES)].copy()
    f["Ponto"] = f.Ponto.astype(int)
    f["Min"] = f.Hora.map(minutes)
    f = f.sort_values(["Data", "Ponto", "Min"])
    f["Ordem"] = f.groupby(["Data", "Ponto"]).cumcount() + 1
    f["Desvio"] = f["Clorofila Total"] - f.groupby(["Data", "Ponto"])["Clorofila Total"].transform("mean")
    rows, pts = [], []

    # dentro da parcela
    r = stats.spearmanr(f.Ordem, f.Desvio)
    fit = smf.ols("Desvio ~ Ordem", f).fit()
    first = f[f.Ordem == 1].Desvio
    rows.append({"Analise": "Falker: posição da leitura na parcela (1ª a 7ª) × desvio da média", "n": len(f),
                 "Efeito": 10 * fit.params["Ordem"] / 10, "Unidade": "SPAD por posição", "rho": r.correlation, "p": r.pvalue})
    rows.append({"Analise": "Falker: desvio médio da 1ª leitura de cada parcela", "n": len(first), "Efeito": float(first.mean()),
                 "Unidade": "SPAD", "p": float(stats.ttest_1samp(first, 0).pvalue)})
    for k, s in f.groupby("Ordem").Desvio:
        pts.append({"Tipo": "Falker: ordem na parcela", "x": k, "y": float(s.mean()), "ep": float(s.std(ddof=1) / np.sqrt(len(s)))})

    # ao longo da sessão
    df = plot_table()
    tmean = f.groupby(["Data", "Ponto"]).Min.mean().rename("Min_Falker").reset_index()
    df = df.merge(tmean, on=["Data", "Ponto"])
    res = []
    for d in config.EXPERIMENTAL_DATES:
        s = df[df.Data == d].copy()
        s["Resid"] = smf.ols("SPAD ~ C(Dose) + C(Bloco)", s).fit().resid
        s["Min_rel"] = s.Min_Falker - s.Min_Falker.min()
        res.append(s)
        r = stats.spearmanr(s.Min_rel, s.Resid)
        rows.append({"Analise": f"Falker {d[:5]}: resíduo da parcela × horário na sessão", "n": len(s),
                     "Efeito": 60 * np.polyfit(s.Min_rel, s.Resid, 1)[0], "Unidade": "SPAD por hora",
                     "rho": r.correlation, "p": r.pvalue, "Duracao_sessao_min": float(s.Min_rel.max())})
        for _, rr in s.iterrows():
            pts.append({"Tipo": f"Falker {d[:5]}", "Data": d, "Ponto": int(rr.Ponto), "x": rr.Min_rel, "y": rr.Resid})
    allr = pd.concat(res)
    r = stats.spearmanr(allr.Min_rel, allr.Resid)
    rows.append({"Analise": "Falker (3 datas): resíduo × horário na sessão", "n": len(allr), "Efeito": 60 * np.polyfit(allr.Min_rel, allr.Resid, 1)[0],
                 "Unidade": "SPAD por hora", "rho": r.correlation, "p": r.pvalue})

    # fotos
    idx = load_indices(10, config.BEST_PREPROCESS_VARIANT)[["Data", "Ponto", "Photo_Time"]]
    idx["Ponto"] = idx.Ponto.astype(int)
    ph = df[df.Tem_foto].merge(idx, on=["Data", "Ponto"], how="left")
    ph["Min_foto"] = ph.Photo_Time.map(minutes)
    for m, lab in [("Median_b_over_r", "b/r"), ("Median_IPCA", "IPCA"), ("Median_L", "luminosidade L")]:
        allp = []
        for d in config.EXPERIMENTAL_DATES:
            s = ph[(ph.Data == d) & ph.Min_foto.notna()].copy()
            if len(s) < 5:
                continue
            s["Min_rel"] = s.Min_foto - s.Min_foto.min()
            # resíduo da medida descontando o SPAD real da parcela (o que sobra deveria ser luz/ruído)
            s["Resid"] = s[m] - np.polyval(np.polyfit(s.SPAD, s[m], 1), s.SPAD)
            s["Resid_z"] = s.Resid / s[m].std(ddof=1)
            allp.append(s)
            r = stats.spearmanr(s.Min_rel, s.Resid)
            rows.append({"Analise": f"Foto {d[:5]}: resíduo de {lab} (descontado o SPAD) × horário", "n": len(s),
                         "Efeito": float(np.polyfit(s.Min_rel, s.Resid_z, 1)[0] * 10), "Unidade": "desvios padrão por 10 min",
                         "rho": r.correlation, "p": r.pvalue, "Duracao_sessao_min": float(s.Min_rel.max())})
            if m in ("Median_b_over_r", "Median_L"):
                for _, rr in s.iterrows():
                    pts.append({"Tipo": f"Foto: {lab}", "Data": d, "Ponto": int(rr.Ponto), "x": rr.Min_rel, "y": rr.Resid_z})
        a = pd.concat(allp)
        r = stats.spearmanr(a.Min_rel, a.Resid_z)
        rows.append({"Analise": f"Foto (3 datas): resíduo de {lab} × horário", "n": len(a), "Efeito": float(np.polyfit(a.Min_rel, a.Resid_z, 1)[0] * 10),
                     "Unidade": "desvios padrão por 10 min", "rho": r.correlation, "p": r.pvalue})

    out = pd.DataFrame(rows)
    db.save_df(out, "ext_order_time", description="Efeito da ordem das leituras e do horário na sessão (Falker e fotos)")
    db.save_df(pd.DataFrame(pts), "ext_session_points", description="Pontos: resíduos × ordem/horário na sessão")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
