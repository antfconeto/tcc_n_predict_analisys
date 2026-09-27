"""
Classes de clorofila (baixa, média, alta) criadas a partir das próprias medições de SPAD.

Critério: faixas de Beaufils (1973), usadas no diagnóstico nutricional para criar classes a partir da média (m) e
do desvio padrão (s) de uma população de referência (Urano et al., 2007; Serra et al., 2012; Guimarães, 2014).
As cinco faixas originais (±2/3 s e ±4/3 s) foram agrupadas em três:

    SPAD < m − 2/3·s                 baixa   (deficiência + tendência à deficiência)
    m − 2/3·s ≤ SPAD ≤ m + 2/3·s     média   (suficiente)
    SPAD > m + 2/3·s                 alta    (tendência ao excesso + excesso)

Com cinco classes, as faixas têm só 2,7 SPAD de largura, menos que a incerteza da média do Falker (±2 SPAD com 7
leituras): a própria referência mudaria de classe numa nova medição. A comparação 5 × 4 × 3 classes fica na
tabela chl_class_scheme_comparison.

População: SPAD médio (Falker, 7 leituras) de cada parcela nas três coletas do experimento (12 × 3 = 36).

1. Classes medidas e verificação: normalidade (Shapiro-Wilk), relação com a dose e com altura e biomassa (26/05).
2. Incerteza do Falker: intervalo de confiança de 95 % da média de cada parcela (t de Student, n − 1 graus de
   liberdade); a classe é "segura" quando o intervalo inteiro cai numa só classe. Teto: concordância esperada
   entre duas medições da mesma parcela (simulação com o erro padrão de cada parcela).
3. Classes pela foto, com o SPAD absoluto (sem área de referência):
     - cada índice × estatística: reta índice → SPAD ajustada em duas datas e testada na terceira;
     - escolha aninhada do índice: escolhido só com as duas datas de treino;
     - MLP pequena da busca aninhada (pipeline/exploratorio/11_mlp_class_search.py), se já tiver sido executada;
     - rede neural definitiva (predições fora da amostra, data deixada de fora).
4. Concordância: acerto exato, kappa ponderado (Landis; Koch, 1977), acerto só nas parcelas com classe segura e
   acerto dentro da incerteza do Falker (a classe prevista está entre as classes possíveis pelo intervalo).

Tabelas: chl_class_scheme, chl_class_references, chl_class_distribution, chl_class_plots, chl_class_validation,
chl_class_index_ranking, chl_class_agreement, chl_class_confusion, chl_class_scheme_comparison.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, skew, spearmanr, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import load_falker_all, load_indices, load_vegetation_plots

warnings.filterwarnings("ignore")

CLASSES = ["Baixa", "Média", "Alta"]
BEAUFILS_NAMES = ["Deficiência e tendência à deficiência", "Suficiente", "Tendência ao excesso e excesso"]
Z_LIMITS = [-2 / 3, 2 / 3]
STAT_PREFIXES = ("Median", "Mean", "P75", "P90")
N_SIM = 5000
SCHEMES = [  # comparação de esquemas (limites em unidades de desvio padrão)
    ("5 classes (Beaufils completo)", [-4 / 3, -2 / 3, 2 / 3, 4 / 3]),
    ("4 classes: separa a deficiência", [-4 / 3, -2 / 3, 2 / 3]),
    ("4 classes: separa o excesso", [-2 / 3, 2 / 3, 4 / 3]),
    ("4 classes: parte a média na média", [-2 / 3, 0.0, 2 / 3]),
    ("3 classes (adotado)", [-2 / 3, 2 / 3]),
]

REFERENCES = [
    {"Chave": "Beaufils (1973)",
     "Referencia": "BEAUFILS, E. R. Diagnosis and recommendation integrated system (DRIS): a general scheme for "
                   "experimentation and calibration based on principles developed from research in plant nutrition. "
                   "Pietermaritzburg: University of Natal, 1973. 132 p. (Soil Science Bulletin, 1).",
     "URL": "",
     "Uso": "Origem das faixas pela média e pelo desvio padrão (±2/3 s e ±4/3 s)."},
    {"Chave": "Guimarães (2014)",
     "Referencia": "GUIMARÃES, F. C. N. Normas DRIS e diagnóstico nutricional de cana-soca na região sul de Goiás. "
                   "2014. Dissertação (Mestrado em Agronomia) – Universidade Federal da Grande Dourados, Dourados, 2014.",
     "URL": "https://files.ufgd.edu.br/arquivos/arquivos/78/MESTRADO-DOUTORADO-AGRONOMIA/Disserta%C3%A7%C3%A3o%20Franklyn%20Clawdy%20Nunes%20Guimar%C3%A3es.pdf",
     "Uso": "Descreve o critério: deficiência < −4/3 s; tendência à deficiência entre −4/3 e −2/3 s; suficiente entre "
            "−2/3 e 2/3 s; tendência ao excesso entre 2/3 e 4/3 s; excesso > 4/3 s (Beaufils, 1973)."},
    {"Chave": "Serra et al. (2012)",
     "Referencia": "SERRA, A. P.; MARCHETTI, M. E.; ROJAS, E. P.; VITORINO, A. C. T. Beaufils ranges to assess the cotton "
                   "nutrient status in the southern region of Mato Grosso. Revista Brasileira de Ciência do Solo, v. 36, "
                   "p. 171-182, 2012.",
     "URL": "",
     "Uso": "Exemplo de uso das faixas de Beaufils no Brasil (algodoeiro)."},
    {"Chave": "Urano et al. (2007)",
     "Referencia": "URANO, E. O. M.; KURIHARA, C. H.; MAEDA, S.; VITORINO, A. C. T.; GONÇALVES, M. C.; MARCHETTI, M. E. "
                   "Determinação de teores ótimos de nutrientes em soja pelos métodos chance matemática, sistema integrado "
                   "de diagnose e recomendação e diagnose da composição nutricional. Revista Brasileira de Ciência do "
                   "Solo, v. 31, p. 63-72, 2007.",
     "URL": "",
     "Uso": "Exemplo de uso das faixas de Beaufils no Brasil (soja)."},
    {"Chave": "Bacelar et al. (2015)",
     "Referencia": "BACELAR, B. M. F. S.; SALMAN, A. K. D.; ALVES, E. A.; CRUZ, P. G.; MENDES, A. M. Uso do medidor de "
                   "clorofila portátil (clorofilômetro) na adubação nitrogenada de pastagens. Porto Velho: Embrapa Rondônia, "
                   "2015. 21 p. (Documentos, 161).",
     "URL": "https://www.infoteca.cnptia.embrapa.br/infoteca/bitstream/doc/1125217/1/cpafro-18443-doc-161.pdf",
     "Uso": "Não existe nível crítico fixo e universal de índice de clorofila para pastagens; as classes precisam ser "
            "calibradas para as condições do local, o que justifica criá-las a partir das próprias medições."},
    {"Chave": "Batista (2002)",
     "Referencia": "BATISTA, K. Respostas do capim-Marandu a combinações de doses de nitrogênio e enxofre. 2002. "
                   "Dissertação (Mestrado em Agronomia) – Escola Superior de Agricultura Luiz de Queiroz, Universidade de "
                   "São Paulo, Piracicaba, 2002.",
     "URL": "https://www.teses.usp.br/teses/disponiveis/11/11140/tde-18022003-155114/publico/karina.pdf",
     "Uso": "Leituras SPAD no capim-Marandu feitas nas lâminas de folhas recém-expandidas: padronizar a folha medida "
            "reduz a variação entre leituras."},
    {"Chave": "Landis; Koch (1977)",
     "Referencia": "LANDIS, J. R.; KOCH, G. G. The measurement of observer agreement for categorical data. Biometrics, "
                   "v. 33, n. 1, p. 159-174, 1977. DOI: 10.2307/2529310.",
     "URL": "https://doi.org/10.2307/2529310",
     "Uso": "Interpretação do kappa: 0,41–0,60 moderada; 0,61–0,80 substancial; > 0,80 quase perfeita."},
]


def classify(v, limits):
    """Índice da classe: abaixo do 1º limite → 0; limites internos fechados na classe central, como no critério."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape, dtype=int)
    for i, lim in enumerate(limits):
        out = np.where(v > lim if i >= len(limits) / 2 else v >= lim, i + 1, out)
    return out


def weighted_kappa(a, b, k):
    """Kappa de Cohen com pesos lineares para classes ordenadas (0..k-1)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denom = (w * exp).sum()
    return 1 - (w * obs).sum() / denom if denom > 0 else np.nan


def class_metrics(true_idx, pred_idx, lo_idx=None, hi_idx=None, k=len(CLASSES)):
    true_idx, pred_idx = np.asarray(true_idx, dtype=int), np.asarray(pred_idx, dtype=int)
    out = {"n": len(true_idx), "Acerto_exato": float(np.mean(true_idx == pred_idx)),
           "Acerto_1_classe": float(np.mean(np.abs(true_idx - pred_idx) <= 1)),
           "Kappa_ponderado": weighted_kappa(true_idx, pred_idx, k),
           "Previsto_acima": int((pred_idx > true_idx).sum()), "Previsto_abaixo": int((pred_idx < true_idx).sum())}
    if lo_idx is not None:
        lo_idx, hi_idx = np.asarray(lo_idx, dtype=int), np.asarray(hi_idx, dtype=int)
        sure = lo_idx == hi_idx
        out.update({"n_seguras": int(sure.sum()),
                    "Acerto_seguras": float(np.mean(true_idx[sure] == pred_idx[sure])) if sure.any() else np.nan,
                    "Acerto_dentro_incerteza": float(np.mean((pred_idx >= lo_idx) & (pred_idx <= hi_idx)))})
    return out


def falker_ceiling(means, se, limits, rng, n_sim=N_SIM):
    """Concordância esperada entre duas medições independentes da mesma parcela (mesmo nº de leituras)."""
    k = len(limits) + 1
    ex, ks = [], []
    for _ in range(n_sim):
        a = classify(means + rng.normal(0, se), limits)
        b = classify(means + rng.normal(0, se), limits)
        ex.append(np.mean(a == b))
        ks.append(weighted_kappa(a, b, k))
    return float(np.mean(ex)), float(np.nanmean(ks))


def lodo_line(x, y, dates):
    """SPAD previsto por uma reta índice → SPAD ajustada nas outras datas."""
    pred = np.empty(len(y))
    for d in np.unique(dates):
        te = dates == d
        b, a = np.polyfit(x[~te], y[~te], 1)
        pred[te] = a + b * x[te]
    return pred


def main():
    rng = np.random.default_rng(42)

    # 1. população: SPAD médio de cada parcela nas três coletas ───────────────────
    f = load_falker_all()
    f = f[f.Data.isin(config.EXPERIMENTAL_DATES)]
    plots = f.groupby(["Data", "Ponto"])["Clorofila Total"].agg(["mean", "std", "size"]).reset_index()
    plots.columns = ["Data", "Ponto", "SPAD", "SPAD_dp", "n_leituras"]
    plots["Ponto"] = plots.Ponto.astype(int)
    for field in ("Bloco", "Tratamento", "Dose"):
        plots[field] = plots.Ponto.map(lambda p, fl=field: config.PONTO_TO_DBC[p][fl])

    m, s = plots.SPAD.mean(), plots.SPAD.std(ddof=1)
    limits = [m + z * s for z in Z_LIMITS]
    plots["z"] = (plots.SPAD - m) / s
    plots["Classe_idx"] = classify(plots.SPAD, limits)
    plots["Classe"] = [CLASSES[i] for i in plots.Classe_idx]

    # 2. incerteza do Falker ─────────────────────────────────────────────────
    plots["SPAD_ep"] = plots.SPAD_dp / np.sqrt(plots.n_leituras)
    tcrit = student_t.ppf(0.975, plots.n_leituras - 1)
    plots["IC95_min"] = plots.SPAD - tcrit * plots.SPAD_ep
    plots["IC95_max"] = plots.SPAD + tcrit * plots.SPAD_ep
    plots["Classe_IC_min"] = [CLASSES[i] for i in classify(plots.IC95_min, limits)]
    plots["Classe_IC_max"] = [CLASSES[i] for i in classify(plots.IC95_max, limits)]
    plots["Classe_segura"] = plots.Classe_IC_min == plots.Classe_IC_max
    teto_ex, teto_k = falker_ceiling(plots.SPAD.to_numpy(), plots.SPAD_ep.to_numpy(), limits, rng)
    sd_leitura = float(np.sqrt((plots.SPAD_dp ** 2).mean()))   # desvio típico entre leituras da mesma parcela
    n_para_1 = int(np.ceil((1.96 * sd_leitura / 1.0) ** 2))
    se_28 = plots.SPAD_dp.to_numpy() / np.sqrt(28)
    teto28_ex, teto28_k = falker_ceiling(plots.SPAD.to_numpy(), se_28, limits, rng)

    W, p_sw = shapiro(plots.SPAD)
    cuts = [-np.inf] + Z_LIMITS + [np.inf]
    expected = [norm.cdf(cuts[i + 1]) - norm.cdf(cuts[i]) for i in range(len(CLASSES))]
    dist = {"n": len(plots), "Media": m, "Desvio_padrao": s, "CV_pct": 100 * s / m,
            "Minimo": plots.SPAD.min(), "Maximo": plots.SPAD.max(), "Assimetria": skew(plots.SPAD),
            "Shapiro_W": W, "Shapiro_p": p_sw,
            "Leituras_por_parcela": int(plots.n_leituras.median()), "DP_entre_leituras": sd_leitura,
            "EP_medio": float(plots.SPAD_ep.mean()), "IC95_meia_largura_media": float((tcrit * plots.SPAD_ep).mean()),
            "Largura_classe_media": limits[1] - limits[0],
            "n_seguras": int(plots.Classe_segura.sum()), "Teto_exato": teto_ex, "Teto_kappa": teto_k,
            "Leituras_para_IC_1_SPAD": n_para_1, "Teto_exato_28_leituras": teto28_ex, "Teto_kappa_28_leituras": teto28_k}

    scheme = []
    bounds = [None] + limits + [None]
    for i, c in enumerate(CLASSES):
        n_i = int((plots.Classe_idx == i).sum())
        scheme.append({"Ordem": i + 1, "Classe": c, "Classe_Beaufils": BEAUFILS_NAMES[i],
                       "z_min": None if i == 0 else Z_LIMITS[i - 1], "z_max": None if i == len(CLASSES) - 1 else Z_LIMITS[i],
                       "SPAD_min": bounds[i], "SPAD_max": bounds[i + 1],
                       "n_parcelas": n_i, "Proporcao_observada": n_i / len(plots), "Proporcao_esperada_normal": expected[i]})

    # 3. verificação: dose e (26/05) altura e biomassa por classe ────────────────
    veg = load_vegetation_plots(verbose=False)[["Ponto", "Altura_Media", "Massa_Verde_Folha_ha", "Massa_Seca_Total_ha"]]
    veg["Ponto"] = veg.Ponto.astype(int)
    d26 = plots[plots.Data == "26-05-2026"].merge(veg, on="Ponto")
    valid = []
    for i, c in enumerate(CLASSES):
        sub, s26 = plots[plots.Classe_idx == i], d26[d26.Classe_idx == i]
        valid.append({"Classe": c, "n": len(sub), "SPAD_medio": sub.SPAD.mean(), "Dose_media": sub.Dose.mean(),
                      "n_0kg": int((sub.Dose == 0).sum()), "n_50kg": int((sub.Dose == 50).sum()),
                      "n_75kg": int((sub.Dose == 75).sum()), "n_100kg": int((sub.Dose == 100).sum()),
                      "n_18_05": int((sub.Data == "18-05-2026").sum()), "n_21_05": int((sub.Data == "21-05-2026").sum()),
                      "n_26_05": int((sub.Data == "26-05-2026").sum()),
                      "Altura_media_26_05": s26.Altura_Media.mean() if len(s26) else None,
                      "Massa_verde_folha_26_05": s26.Massa_Verde_Folha_ha.mean() if len(s26) else None,
                      "Massa_seca_total_26_05": s26.Massa_Seca_Total_ha.mean() if len(s26) else None})
    valid = pd.DataFrame(valid)
    r = spearmanr(plots.Classe_idx, plots.Dose)
    dist.update({"rho_classe_dose": r.correlation, "p_classe_dose": r.pvalue})
    for k, label in [("Altura_Media", "altura"), ("Massa_Verde_Folha_ha", "massa_verde_folha"), ("Massa_Seca_Total_ha", "massa_seca_total")]:
        r = spearmanr(d26.Classe_idx, d26[k])
        dist.update({f"rho_classe_{label}": r.correlation, f"p_classe_{label}": r.pvalue})

    # 4. classes pela foto ────────────────────────────────────────────────────
    pred = db.load_df("definitive_model_predictions")
    pred = pred[pred.Data.isin(config.EXPERIMENTAL_DATES)][["Data", "Ponto", "Predicao_MLP"]].copy()
    pred["Ponto"] = pred.Ponto.astype(int)
    plots = plots.merge(pred.rename(columns={"Predicao_MLP": "SPAD_rede"}), on=["Data", "Ponto"], how="left")
    if db.has_rows("mlp_class_predictions"):
        mp = db.load_df("mlp_class_predictions")[["Data", "Ponto", "SPAD_aninhada"]].copy()
        mp["Ponto"] = mp.Ponto.astype(int)
        plots = plots.merge(mp.rename(columns={"SPAD_aninhada": "SPAD_mlp"}), on=["Data", "Ponto"], how="left")
    else:
        plots["SPAD_mlp"] = np.nan

    idx = load_indices(10, config.BEST_PREPROCESS_VARIANT)
    idx = idx[idx.Data.isin(config.EXPERIMENTAL_DATES)].copy()
    idx["Ponto"] = idx.Ponto.astype(int)
    feats = [c for c in idx.columns if c.split("_")[0] in STAT_PREFIXES and pd.api.types.is_numeric_dtype(idx[c])
             and not c.endswith("_rgb") and idx[c].std() > 1e-9]
    ph = plots.merge(idx[["Data", "Ponto"] + feats], on=["Data", "Ponto"]).reset_index(drop=True)
    y, yc, dates = ph.SPAD.to_numpy(), ph.Classe_idx.to_numpy(), ph.Data.to_numpy()
    lo_c, hi_c = classify(ph.IC95_min, limits), classify(ph.IC95_max, limits)

    rank_rows, preds = [], {}
    for f_ in feats:
        x = ph[f_].to_numpy(float)
        p_ = lodo_line(x, y, dates)
        rank_rows.append({"Variavel": f_, "Indice": f_.partition("_")[2], "Estatistica": f_.partition("_")[0],
                          "rho_SPAD": spearmanr(x, y).correlation, "rho_classe": spearmanr(x, yc).correlation,
                          "R2_SPAD_LODO": 1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2),
                          "MAE_SPAD_LODO": float(np.mean(np.abs(y - p_))),
                          **class_metrics(yc, classify(p_, limits), lo_c, hi_c)})
        preds[f_] = p_
    rank = pd.DataFrame(rank_rows).sort_values(["Kappa_ponderado", "Acerto_exato", "R2_SPAD_LODO"], ascending=False).reset_index(drop=True)
    rank["Rank"] = np.arange(1, len(rank) + 1)
    best = rank.Variavel.iloc[0]

    # escolha aninhada: o índice é escolhido só com as duas datas de treino
    nested = np.empty(len(y))
    chosen = {}
    for d in config.EXPERIMENTAL_DATES:
        te = dates == d
        tr = [x for x in config.EXPERIMENTAL_DATES if x != d]
        best_f, best_k = None, -np.inf
        for f_ in feats:
            x = ph[f_].to_numpy(float)
            ks = []
            for d_fit, d_eval in [(tr[0], tr[1]), (tr[1], tr[0])]:
                a_m, e_m = dates == d_fit, dates == d_eval
                b, a = np.polyfit(x[a_m], y[a_m], 1)
                ks.append(weighted_kappa(yc[e_m], classify(a + b * x[e_m], limits), len(CLASSES)))
            k = np.nanmean(ks)
            if k > best_k:
                best_f, best_k = f_, k
        x = ph[best_f].to_numpy(float)
        b, a = np.polyfit(x[~te], y[~te], 1)
        nested[te] = a + b * x[te]
        chosen[d] = best_f

    ph["SPAD_melhor_indice"] = preds[best]
    ph["SPAD_indice_aninhado"] = nested
    ph["Indice_aninhado"] = ph.Data.map(chosen)
    plots = plots.merge(ph[["Data", "Ponto", "SPAD_melhor_indice", "SPAD_indice_aninhado", "Indice_aninhado"]],
                        on=["Data", "Ponto"], how="left")
    method_cols = [(f"Melhor índice ({best})", "SPAD_melhor_indice", "Classe_melhor_indice"),
                   ("Índice escolhido sem a data testada (aninhado)", "SPAD_indice_aninhado", "Classe_indice_aninhado"),
                   ("MLP pequena, escolhida sem a data testada (aninhada)", "SPAD_mlp", "Classe_mlp"),
                   ("Rede neural definitiva (24 entradas)", "SPAD_rede", "Classe_rede")]
    for _, col, out in method_cols:
        has = plots[col].notna()
        plots[out] = None
        plots.loc[has, out] = [CLASSES[i] for i in classify(plots.loc[has, col], limits)]
    plots["Tem_foto"] = plots.SPAD_melhor_indice.notna()
    plots = plots.assign(_d=pd.to_datetime(plots.Data, format="%d-%m-%Y")).sort_values(["_d", "Ponto"]).drop(columns="_d")

    order = {c: i for i, c in enumerate(CLASSES)}
    agree, conf = [], []
    pf = plots[plots.Tem_foto]
    agree.append({"Metodo": "Falker × Falker (teto: medir a mesma parcela de novo)", "Escopo": "Todas as datas", "Indice": "",
                  "n": len(plots), "Acerto_exato": teto_ex, "Kappa_ponderado": teto_k})
    for name, col, out in method_cols:
        if pf[col].isna().all():
            continue
        for scope, sub in [("Todas as datas", pf)] + [(d, pf[pf.Data == d]) for d in config.EXPERIMENTAL_DATES]:
            agree.append({"Metodo": name, "Escopo": scope,
                          "Indice": chosen.get(scope, "") if "Índice escolhido" in name else "",
                          **class_metrics(sub.Classe.map(order), sub[out].map(order),
                                          sub.Classe_IC_min.map(order), sub.Classe_IC_max.map(order))})
        for (cm, cp), n in pf.groupby(["Classe", out]).size().items():
            conf.append({"Metodo": name, "Classe_medida": cm, "Classe_prevista": cp, "n": int(n)})

    # 5. comparação de esquemas de classes ─────────────────────────────────────
    comp = []
    for name, zl in SCHEMES:
        lim = [m + z * s for z in zl]
        k = len(lim) + 1
        te_ex, te_k = falker_ceiling(plots.SPAD.to_numpy(), plots.SPAD_ep.to_numpy(), lim, rng, n_sim=2000)
        yy, lo, hi = classify(pf.SPAD, lim), classify(pf.IC95_min, lim), classify(pf.IC95_max, lim)
        row = {"Esquema": name, "n_classes": k, "Limites_SPAD": " | ".join(f"{v:.1f}".replace(".", ",") for v in lim),
               "Parcelas_por_classe": "/".join(map(str, np.bincount(classify(plots.SPAD, lim), minlength=k))),
               "Teto_Falker_exato": te_ex, "Teto_Falker_kappa": te_k, "n_seguras": int((lo == hi).sum())}
        for label, col in [("Indice_aninhado", "SPAD_indice_aninhado"), ("MLP", "SPAD_mlp"), ("Rede", "SPAD_rede")]:
            if pf[col].isna().all():
                continue
            q = classify(pf[col], lim)
            row[f"{label}_exato"] = float(np.mean(yy == q))
            row[f"{label}_kappa"] = weighted_kappa(yy, q, k)
            row[f"{label}_dentro_incerteza"] = float(np.mean((q >= lo) & (q <= hi)))
        comp.append(row)

    db.save_df(pd.DataFrame(scheme), "chl_class_scheme", description="Classes de clorofila pelas faixas de Beaufils (3 classes): limites em SPAD e proporções")
    db.save_df(pd.DataFrame(REFERENCES), "chl_class_references", description="Referências bibliográficas da classificação da clorofila")
    db.save_df(pd.DataFrame([dist]), "chl_class_distribution", description="SPAD por parcela: distribuição, incerteza do Falker, teto e relação das classes com dose, altura e massa")
    db.save_df(plots, "chl_class_plots", description="Classe de cada parcela e data: medida (Falker, com intervalo de confiança) e prevista pela foto")
    db.save_df(valid, "chl_class_validation", description="Por classe: SPAD, dose, datas e (26/05) altura e biomassa médias")
    db.save_df(rank, "chl_class_index_ranking", description="Índices da foto como classificadores (reta índice→SPAD, data deixada de fora)")
    db.save_df(pd.DataFrame(agree), "chl_class_agreement", description="Concordância entre a classe medida e a prevista, com o teto do Falker e a incerteza")
    db.save_df(pd.DataFrame(conf), "chl_class_confusion", description="Matriz de confusão: classe medida × classe prevista")
    db.save_df(pd.DataFrame(comp), "chl_class_scheme_comparison", description="Comparação entre 5, 4 e 3 classes: teto do Falker e acerto da foto")

    print(f"m = {m:.2f}  s = {s:.2f}  limites = {[round(v, 2) for v in limits]}  Shapiro p = {p_sw:.3f}")
    print(pd.DataFrame(scheme)[["Classe", "SPAD_min", "SPAD_max", "n_parcelas", "Proporcao_observada", "Proporcao_esperada_normal"]].round(3).to_string(index=False))
    print(pd.Series(dist).round(3).to_string())
    print(pd.DataFrame(agree)[lambda d: d.Escopo == "Todas as datas"].round(3).to_string(index=False))
    print(pd.DataFrame(comp).round(3).to_string(index=False))


if __name__ == "__main__":
    db.run_main(main)
