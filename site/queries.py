"""Consultas ao banco usadas pela API do site (site/server.py).

Tudo é calculado sob demanda a partir do SQLite; nada é pré-gerado em arquivo.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE_DIR.parent))

from tcc_analysis import config, db
from tcc_analysis.data import load_vegetation_plots
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse


# ─────────────────────────────────────────────────────────────────────────────
# Serialização
# ─────────────────────────────────────────────────────────────────────────────
def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{float(v):.6g}")
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def table_payload(df: pd.DataFrame) -> dict:
    return {
        "columns": [str(c) for c in df.columns],
        "rows": [[_clean(v) for v in row] for row in df.itertuples(index=False, name=None)],
    }


def latest(table: str, **filters) -> pd.DataFrame:
    return db.load_df(table, filters or None)


def try_latest(table: str) -> pd.DataFrame | None:
    return db.load_df(table) if db.has_rows(table) else None


# ─────────────────────────────────────────────────────────────────────────────
# Blocos de conteúdo pré-calculados
# ─────────────────────────────────────────────────────────────────────────────
def dose_response() -> list[dict]:
    """Média ± erro padrão da Clorofila Total por data × dose (leituras individuais)."""
    df = db.query(
        """SELECT f.data AS Data, l.dose AS Dose, f.clorofila_total AS v
           FROM falker_readings f JOIN dbc_layout l ON l.ponto = CAST(f.ponto AS INTEGER)
           WHERE f.source = 'cloro' AND f.clorofila_total IS NOT NULL"""
    )
    out = []
    for (data, dose), g in df.groupby(["Data", "Dose"]):
        out.append({
            "Data": data, "Dose": dose, "n": len(g),
            "Media": g.v.mean(), "EP": g.v.sem(), "DP": g.v.std(),
        })
    return out


def falker_by_date() -> list[dict]:
    df = db.query("SELECT data, source, clorofila_total, clorofila_a, clorofila_b FROM falker_readings")
    rows = []
    for data, g in df.groupby("data"):
        rows.append({
            "Data": data, "Fonte": g.source.iloc[0], "Leituras": len(g),
            "Clorofila_Total_media": g.clorofila_total.mean(), "Clorofila_Total_EP": g.clorofila_total.sem(),
            "Clorofila_A_media": g.clorofila_a.mean(), "Clorofila_B_media": g.clorofila_b.mean(),
        })
    order = {d: i for i, d in enumerate(config.EXPERIMENTAL_DATES + [config.TEST_DATE])}
    return sorted(rows, key=lambda r: order.get(r["Data"], 99))


def definitive_block() -> dict:
    pred = latest("definitive_model_predictions")
    summ = latest("definitive_model_summary")
    per_date = []
    for data, g in pred.groupby("Data"):
        per_date.append({
            "Data": data, "n": len(g),
            "R2_MLP": compute_r2(g["Clorofila Total"], g.Predicao_MLP),
            "MAE_MLP": compute_mae(g["Clorofila Total"], g.Predicao_MLP),
            "MAE_Indice": compute_mae(g["Clorofila Total"], g.Predicao_Melhor_Indice),
        })
    return {"summary": summ.to_dict("records"), "per_date": per_date}


def treatment_yields() -> list[dict]:
    df = load_vegetation_plots(verbose=False)
    cols = ["Clorofila Total", "Altura_Media", "Massa_Verde_Total_ha", "Massa_Seca_Total_ha",
            "Massa_Seca_Folha_ha", "Massa_Seca_Colmo_ha", "Rel_Folha_Colmo_Seco"]
    g = df.groupby(["Tratamento", "Dose"])[cols].mean().reset_index()
    return g.to_dict("records")


def approaches() -> list[dict]:
    """R² fora da amostra de cada abordagem para clorofila (validação por data quando disponível)."""
    out = []
    summ = latest("definitive_model_summary")
    for rec in summ.to_dict("records"):
        is_mlp = str(rec["Modelo"]).startswith("MLP")
        label = f"{rec['Modelo']} (modelo definitivo)" if is_mlp else f"Índice único {rec['Modelo']}"
        out.append({"Abordagem": label, "R2": rec["R2_OOF"], "Validacao": "GroupKFold por data",
                    "Fonte": "definitive_model_summary", "Principal": True})

    def add(label, table, col, validation, note=None):
        df = try_latest(table)
        if df is None or col not in df.columns:
            return
        out.append({"Abordagem": label, "R2": df[col].max(), "Validacao": validation, "Fonte": table,
                    "Principal": False, "Observacao": note})

    hist = try_latest("model_version_history")
    if hist is not None:
        for rec in hist.to_dict("records"):
            if "v1 sem" in rec["Modelo"]:
                out.append({"Abordagem": "MLP v1 sem o artefato (sem máscara HSV)", "R2": rec["R2_OOF"],
                            "Validacao": "GroupKFold por data", "Fonte": "model_version_history", "Principal": False})
    # MLP só RGB, MLP avançada e ensemble da análise anterior ficam de fora: usavam as colunas *_rgb
    # e podem carregar o mesmo artefato numérico de Mean_rgb (ver README).
    add("MLP com índices customizados + HUE", "legacy__mlp_custom_indices_summary", "R2_OOF", "GroupKFold por data")
    add("Fórmula física (melhor na validação)", "legacy__reg_sym_physical_search_summary", "Val_R2", "Treino 18+26/05, validação 21/05+01/06")
    add("Regressão simbólica (gplearn)", "legacy__reg_sym_symbolic_summary", "Val_R2", "Treino 26/05, validação demais datas")
    lodo = try_latest("cv_lodo_results")
    if lodo is not None:
        out.append({"Abordagem": "Melhor índice linear (treino em médias, LODO)", "R2": lodo.LODO_R2_mean.max(),
                    "Validacao": "Deixa uma data de fora", "Fonte": "cv_lodo_results", "Principal": False})
    return sorted(out, key=lambda r: -(r["R2"] if r["R2"] is not None else -9))


def mlp_height_r2() -> float | None:
    df = try_latest("mlp_height_loocv_results")
    if df is None:
        return None
    return compute_r2(df.Altura_Media, df.Altura_Predita_LOOCV)


def key_numbers(defin: dict) -> dict:
    k: dict = {}
    s = {r["Modelo"]: r for r in defin["summary"]}
    mlp_name = next(m for m in s if str(m).startswith("MLP"))
    mlp = s[mlp_name]
    idx = next(r for m, r in s.items() if m != mlp_name)
    best = mlp if mlp["R2_OOF"] >= idx["R2_OOF"] else idx
    k.update({
        "mlp_name": mlp_name, "best_r2": best["R2_OOF"], "best_rmse": best["RMSE_OOF"], "best_mae": best["MAE_OOF"],
        "best_name": "a rede neural" if best is mlp else f"o índice {idx['Modelo']}",
        "mlp_r2": mlp["R2_OOF"], "mlp_rmse": mlp["RMSE_OOF"], "mlp_mae": mlp["MAE_OOF"],
        "idx_name": idx["Modelo"], "idx_r2": idx["R2_OOF"], "idx_rmse": idx["RMSE_OOF"],
        "idx_mae": idx["MAE_OOF"], "idx_formula": idx["Formula"],
    })
    k["n_falker"] = int(db.query("SELECT COUNT(*) n FROM falker_readings").n[0])
    k["n_images"] = int(db.query("SELECT COUNT(*) n FROM images WHERE ponto IS NOT NULL").n[0])
    k["n_samples"] = len(latest("definitive_model_predictions"))
    k["n_dates"] = len(config.DATE_FOLDERS)

    anova = latest("anova_n_dose")
    trat = anova[(anova.Fonte == "C(Tratamento)") & (anova.Variavel == "Clorofila Total")]
    k["anova_p_max"] = trat["PR(>F)"].max()
    k["anova_F_min"] = trat["F"].min()
    ponto = anova[anova.Fonte == "C(Ponto)"]
    if not ponto.empty:
        k["anova_test_p"] = ponto["PR(>F)"].iloc[0]

    champ = latest("anova_multi_statistic_summary")
    top = champ.iloc[champ.R_squared.idxmax()]
    k.update({"champ_index": top.Champ_Index, "champ_r": top.Pearson_r, "champ_r2": top.R_squared,
              "champ_grade": top.Grade, "champ_stat": top.Estatistica})

    lodo = try_latest("cv_lodo_results")
    if lodo is not None:
        k["lodo_best"] = lodo.LODO_R2_mean.max()
        k["lodo_best_index"] = lodo.sort_values("LODO_R2_mean").Spectral_Index.iloc[-1]
    lobo = try_latest("cv_lobo_results")
    if lobo is not None:
        k["lobo_best"] = lobo.LOBO_R2_mean.max()
        k["lobo_best_index"] = lobo.sort_values("LOBO_R2_mean").Spectral_Index.iloc[-1]

    al = try_latest("aligned_validation_summary")
    if al is not None:
        al = al.set_index("Cenario")
        k["aligned_crop_r2"] = al.loc["TCC por imagem (crop) vs Falker", "R2"]
        k["aligned_nocrop_r2"] = al.loc["App sem crop vs Falker", "R2"]
        k["aligned_full_r2"] = al.loc["MLP alinhada (features CSV) vs Falker", "R2"]
    app = try_latest("app_validation_vs_falker")
    if app is not None:
        k["app_r2"] = compute_r2(app.falker_reference, app.spad_predicted)
        k["app_mae"] = compute_mae(app.falker_reference, app.spad_predicted)
        k["app_rmse"] = compute_rmse(app.falker_reference, app.spad_predicted)
        k["app_n"] = len(app)
        inl = app[app.abs_error <= 2.5]
        k["app_r2_clean"] = compute_r2(inl.falker_reference, inl.spad_predicted)
        k["app_n_clean"] = len(inl)

    reg = latest("regression_summary")
    def r2(metric, level, model):
        sub = reg[(reg.Y_Column == metric) & (reg.Analysis_Level == level) & (reg.Model_Type == model)]
        return float(sub.R2.iloc[0]) if len(sub) else None
    k["alt_clo_plot"] = r2("Altura_Media", "Parcela (N=12)", "Linear vs Clorofila Total")
    k["alt_clo_mean"] = r2("Altura_Media", "Média (N=4)", "Linear vs Clorofila Total")
    k["msf_clo_plot"] = r2("Massa_Seca_Folha_ha", "Parcela (N=12)", "Linear vs Clorofila Total")
    k["mst_dose_mean_q"] = r2("Massa_Seca_Total_ha", "Média (N=4)", "Quadratic vs Dose")
    k["mst_dose_plot_q"] = r2("Massa_Seca_Total_ha", "Parcela (N=12)", "Quadratic vs Dose")
    k["mlp_height_r2"] = mlp_height_r2()

    hist = try_latest("model_version_history")
    if hist is not None:
        for rec in hist.to_dict("records"):
            if "artefato" in rec["Modelo"]:
                k["v1_artifact_r2"] = rec["R2_OOF"]
            elif "v1 sem" in rec["Modelo"]:
                k["v1_clean_r2"] = rec["R2_OOF"]
    pds = try_latest("per_date_summary")
    if pds is not None:
        rel = pds[pds.Confiavel.astype(bool)]
        k["per_date_best_loo_max"] = rel.Melhor_R2_LOO.max()
        k["per_date_best_loo_min"] = rel.Melhor_R2_LOO.min()
        k["per_date_rb_r_min"] = rel.rb_r.abs().min() if "rb_r" in rel else None
        k["per_date_rb_r_max"] = rel.rb_r.abs().max() if "rb_r" in rel else None
        for rec in pds.to_dict("records"):
            key = rec["Data"][:5].replace("-", "_")
            k[f"pd_{key}_loo"] = rec["Melhor_R2_LOO"]
            k[f"pd_{key}_var"] = rec["Melhor_variavel"]
            k[f"pd_{key}_n"] = rec["n_fotos"]

    ps = try_latest("rgb_preprocess_search_summary")
    leg = try_latest("legacy__preprocess_search__rgb_preprocess_search_summary")
    if leg is not None:
        k["preprocess_n"] = len(leg) + (0 if ps is None else len(set(ps.Variant) - set(leg.Variant)))
        k["preprocess_baseline"] = leg.set_index("Variant").R2_OOF.get("baseline")
    if ps is not None and "bilateral" in set(ps.Variant):
        k["preprocess_bilateral"] = ps.set_index("Variant").R2_OOF["bilateral"]
    return k


def preprocess_ranking() -> list[dict]:
    rows = {}
    leg = try_latest("legacy__preprocess_search__rgb_preprocess_search_summary")
    if leg is not None:
        for r in leg[["Variant", "R2_OOF", "RMSE_OOF"]].to_dict("records"):
            rows[r["Variant"]] = {**r, "Fonte": "análise anterior"}
    new = try_latest("rgb_preprocess_search_summary")
    if new is not None:
        for r in new[["Variant", "R2_OOF", "RMSE_OOF"]].to_dict("records"):
            rows[r["Variant"]] = {**r, "Fonte": "pipeline atual"}
    return sorted(rows.values(), key=lambda r: -r["R2_OOF"])


def map_gallery() -> list[dict]:
    """Resumo por imagem + caminho do mapa em outputs/plots/mappings (servido em /plots/...)."""
    summ = try_latest("mapping_images_summary")
    if summ is None:
        return []
    items = []
    for rec in summ.to_dict("records"):
        name = f"mappings/{rec['Data']}_P{int(rec['Ponto'])}_mapping.png"
        exists = (config.PLOTS_DIR / name).exists()
        items.append({**{k: _clean(v) for k, v in rec.items()}, "image": f"/plots/{name}" if exists else None})
    items.sort(key=lambda r: (r["Data"] != "26-05", r["Ponto"]))
    return items


RAW_TABLES = {"dbc_layout", "falker_readings", "images", "campo_altura_peso", "campo_massa_verde_seca"}
INTERNAL_TABLES = {"runs", "catalog", "reports", "artifacts", "sqlite_sequence"}


def table_group(name: str) -> str:
    if name in RAW_TABLES:
        return "Dados brutos"
    return "Análise anterior (legado)" if name.startswith("legacy__") else "Resultados do pipeline"


def table_catalog() -> list[dict]:
    catalog = db.query("SELECT table_name, description, script, last_run_id, updated_at FROM catalog")
    cat = {r["table_name"]: r for r in catalog.to_dict("records")}
    meta = []
    with db.connect() as conn:
        for name in db.list_tables():
            if name in INTERNAL_TABLES:
                continue
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")') if r[1] != "run_id"]
            has_run = name not in RAW_TABLES and "run_id" in [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')]
            if has_run:
                n = conn.execute(f'SELECT COUNT(*) FROM "{name}" WHERE run_id = (SELECT MAX(run_id) FROM "{name}")').fetchone()[0]
                n_runs = conn.execute(f'SELECT COUNT(DISTINCT run_id) FROM "{name}"').fetchone()[0]
            else:
                n, n_runs = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0], 1
            c = cat.get(name, {})
            meta.append({
                "name": name, "group": table_group(name), "rows": n, "cols": len(cols), "n_runs": n_runs,
                "description": c.get("description"), "script": c.get("script"),
                "run_id": _clean(c.get("last_run_id")), "updated_at": c.get("updated_at"),
            })
    order = {"Dados brutos": 0, "Resultados do pipeline": 1, "Análise anterior (legado)": 2}
    meta.sort(key=lambda m: (order[m["group"]], m["name"]))
    return meta


def report_list() -> list[dict]:
    return db.query(
        "SELECT r.name, r.format, r.created_at, r.run_id FROM reports r "
        "JOIN (SELECT name, MAX(report_id) AS mx FROM reports GROUP BY name) m ON m.mx = r.report_id "
        "ORDER BY r.name LIKE 'legacy/%', r.name"
    ).to_dict("records")


def runs() -> list[dict]:
    return db.query("SELECT run_id, script, status, started_at, finished_at FROM runs ORDER BY run_id DESC").to_dict("records")


def overview() -> dict:
    """Blocos calculados que alimentam as seções narrativas do site."""
    defin = definitive_block()
    return {
        "key": {k: _clean(v) for k, v in key_numbers(defin).items()},
        "dose_response": [{k: _clean(v) for k, v in r.items()} for r in dose_response()],
        "falker_by_date": [{k: _clean(v) for k, v in r.items()} for r in falker_by_date()],
        "definitive": json.loads(json.dumps(defin, default=_clean)),
        "yields": [{k: _clean(v) for k, v in r.items()} for r in treatment_yields()],
        "approaches": [{k: _clean(v) for k, v in r.items()} for r in approaches()],
        "preprocess_ranking": [{k: _clean(v) for k, v in r.items()} for r in preprocess_ranking()],
        "config": {
            "dbc": {str(k): v for k, v in config.PONTO_TO_DBC.items()},
            "best_variant": config.BEST_PREPROCESS_VARIANT,
            "champ_params": {k: list(v) if isinstance(v, tuple) else v for k, v in config.CHAMP_MLP_PARAMS.items()},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Campo: parcelas, fotos, detalhe por ponto
# ─────────────────────────────────────────────────────────────────────────────
TEST_DATE = config.TEST_DATE
RES_FACTOR = 1.5          # |erro do modelo| > 1,5 × RMSE fora da amostra
CV_LIMIT = 0.12           # coeficiente de variação das leituras de uma parcela
MODZ_LIMIT = 3.5          # z robusto (mediana/MAD) de uma leitura isolada
CANOPY_LIMIT = 60.0       # % de blocos com vegetação
APP_LIMIT = 2.5           # |erro do app| (mesmo corte da validação original)


def plot_id(data: str, ponto) -> str:
    return f"teste-{int(ponto)}" if data == TEST_DATE else str(int(ponto))


def _short(data: str) -> str:
    return data[:5].replace("-", "/")


def _photo_sizes() -> dict:
    from PIL import Image
    out = {}
    for r in db.query("SELECT rel_path FROM images").itertuples():
        try:
            with Image.open(config.IMAGE_ROOT / r.rel_path) as im:
                out[r.rel_path] = im.size
        except OSError:
            out[r.rel_path] = (None, None)
    return out


_SIZES: dict | None = None


def photo_sizes() -> dict:
    global _SIZES
    if _SIZES is None:
        _SIZES = _photo_sizes()
    return _SIZES


def image_table() -> pd.DataFrame:
    """Uma linha por foto de parcela com referência, diagnósticos e predições."""
    imgs = db.query("SELECT data, date_folder, file_name, rel_path, ponto, photo_time FROM images WHERE ponto IS NOT NULL")
    imgs = imgs.rename(columns={"data": "Data", "ponto": "Ponto"})
    fr = db.query("SELECT data AS Data, ponto AS Ponto, clorofila_total AS v FROM falker_readings")
    stats = fr.groupby(["Data", "Ponto"]).v.agg(obs="mean", sd="std", n="count", vmin="min", vmax="max").reset_index()
    stats["cv"] = stats.sd / stats.obs
    df = imgs.merge(stats, on=["Data", "Ponto"], how="left")
    if db.has_rows("image_indices", {"Variant": config.BEST_PREPROCESS_VARIANT, "Block_Size": 10}):
        idx = latest("image_indices", Variant=config.BEST_PREPROCESS_VARIANT, Block_Size=10)
        df = df.merge(idx[["Data", "Ponto", "Canopy_Cover", "Vegetation_Blocks", "Total_Blocks", "Median_r_over_b"]],
                      on=["Data", "Ponto"], how="left")
    pred = try_latest("definitive_model_predictions")
    if pred is not None:
        df = df.merge(pred[["Data", "Ponto", "Predicao_MLP", "Predicao_Melhor_Indice"]], on=["Data", "Ponto"], how="left")
        df["res"] = df.Predicao_MLP - df.obs
    app = try_latest("app_validation_vs_falker")
    if app is not None:
        a = app.assign(Data=app.date_folder + f"-{config.YEAR}", Ponto=app.ponto.astype(float))
        df = df.merge(a[["Data", "Ponto", "spad_predicted"]].rename(columns={"spad_predicted": "app_pred"}), on=["Data", "Ponto"], how="left")
        df["app_res"] = df.app_pred - df.obs
    sizes = photo_sizes()
    df["width"] = df.rel_path.map(lambda r: sizes.get(r, (None, None))[0])
    df["height"] = df.rel_path.map(lambda r: sizes.get(r, (None, None))[1])
    df["id"] = [plot_id(d, p) for d, p in zip(df.Data, df.Ponto)]
    for field in ("Bloco", "Tratamento", "Dose"):
        df[field] = [None if d == TEST_DATE else config.PONTO_TO_DBC.get(int(p), {}).get(field) for d, p in zip(df.Data, df.Ponto)]
    order = {d: i for i, d in enumerate(config.EXPERIMENTAL_DATES + [TEST_DATE])}
    return df.sort_values(["Data", "Ponto"], key=lambda s: s.map(order) if s.name == "Data" else s).reset_index(drop=True)


def photos() -> list[dict]:
    df = image_table()
    cols = ["id", "Data", "date_folder", "Ponto", "Bloco", "Tratamento", "Dose", "rel_path", "photo_time", "obs",
            "Predicao_MLP", "res", "Canopy_Cover", "Vegetation_Blocks", "width", "height"]
    out = [{k: _clean(v) for k, v in r.items()} for r in df[[c for c in cols if c in df]].to_dict("records")]
    extra = db.query("SELECT data, date_folder, rel_path, photo_time FROM images WHERE ponto IS NULL")
    for r in extra.to_dict("records"):
        out.append({"id": None, "Data": r["data"], "date_folder": r["date_folder"], "rel_path": r["rel_path"],
                    "photo_time": r["photo_time"], "extra": True})
    return out


def field_map() -> dict:
    df = image_table()
    fr = db.query("SELECT data AS Data, ponto AS Ponto, clorofila_total AS v FROM falker_readings")
    means = fr.groupby(["Data", "Ponto"]).v.mean()
    veg = load_vegetation_plots(verbose=False).set_index("Ponto")
    flags = outlier_counts()
    plots = []
    for ponto, lay in config.PONTO_TO_DBC.items():
        rec = {"id": str(ponto), "Ponto": ponto, **lay, "obs": {}, "res": {}, "photos": {}, "flags": flags.get(str(ponto), 0)}
        for d in config.EXPERIMENTAL_DATES:
            if (d, float(ponto)) in means.index:
                rec["obs"][d] = _clean(means[(d, float(ponto))])
            sub = df[(df.Data == d) & (df.Ponto == ponto)]
            if len(sub):
                rec["photos"][d] = sub.rel_path.iloc[0]
                rec["res"][d] = _clean(sub.res.iloc[0]) if "res" in sub else None
        if float(ponto) in veg.index:
            v = veg.loc[float(ponto)]
            rec["altura"] = _clean(v.Altura_Media)
            rec["massa_seca"] = _clean(v.Massa_Seca_Total_ha)
        plots.append(rec)
    tests = []
    for r in df[df.Data == TEST_DATE].to_dict("records"):
        tests.append({"id": r["id"], "Ponto": int(r["Ponto"]), "obs": {TEST_DATE: _clean(r["obs"])},
                      "res": {TEST_DATE: _clean(r.get("res"))}, "photos": {TEST_DATE: r["rel_path"]}, "flags": flags.get(r["id"], 0)})
    overview_photo = db.query("SELECT rel_path FROM images WHERE ponto IS NULL LIMIT 1")
    return {"plots": plots, "tests": tests, "dates": config.EXPERIMENTAL_DATES, "test_date": TEST_DATE,
            "overview_photo": overview_photo.rel_path.iloc[0] if len(overview_photo) else None}


def plot_detail(pid: str) -> dict:
    is_test = pid.startswith("teste-")
    ponto = int(pid.split("-")[-1])
    if not is_test and ponto not in config.PONTO_TO_DBC:
        raise LookupError(f"Parcela desconhecida: {pid}")
    df = image_table()
    dates = [TEST_DATE] if is_test else config.EXPERIMENTAL_DATES
    readings = db.query(
        "SELECT data, hora, medicao, clorofila_a, clorofila_b, clorofila_total FROM falker_readings "
        "WHERE CAST(ponto AS INTEGER) = ? ORDER BY reading_id", (ponto,))
    readings = readings[readings.data.isin(dates)]
    per_date = []
    for d in dates:
        sub = df[(df.Data == d) & (df.Ponto == ponto)]
        r = readings[readings.data == d]
        row = {"Data": d, "n_leituras": len(r)}
        if len(r):
            row.update({"obs": r.clorofila_total.mean(), "sd": r.clorofila_total.std(),
                        "clorofila_a": r.clorofila_a.mean(), "clorofila_b": r.clorofila_b.mean()})
        if len(sub):
            s = sub.iloc[0]
            for k in ("rel_path", "photo_time", "Predicao_MLP", "Predicao_Melhor_Indice", "res", "app_pred",
                      "Canopy_Cover", "Vegetation_Blocks", "Total_Blocks", "width", "height", "cv"):
                if k in s:
                    row[k] = s[k]
        per_date.append({k: _clean(v) for k, v in row.items()})
    out = {
        "id": pid, "Ponto": ponto, "teste": is_test,
        "layout": None if is_test else config.PONTO_TO_DBC[ponto],
        "dates": per_date,
        "readings": [{k: _clean(v) for k, v in r.items()} for r in readings.to_dict("records")],
        "outliers": [o for o in outliers() if o["id"] == pid],
    }
    if not is_test:
        veg = load_vegetation_plots(verbose=False)
        v = veg[veg.Ponto == ponto]
        if len(v):
            v = v.iloc[0]
            out["vegetacao"] = {k: _clean(v[k]) for k in ("Altura_Media", "Alturas Anotadas (cm)", "Massa_Verde_Total_ha",
                                "Massa_Seca_Total_ha", "Massa_Seca_Folha_ha", "Massa_Seca_Colmo_ha", "Rel_Folha_Colmo_Seco")}
            trat = veg.groupby("Tratamento")[["Altura_Media", "Massa_Seca_Total_ha"]].mean()
            out["vegetacao"]["media_tratamento"] = {k: _clean(v) for k, v in trat.loc[config.PONTO_TO_DBC[ponto]["Tratamento"]].items()}
        maps = try_latest("mapping_images_summary")
        if maps is not None:
            mm = maps[(maps.Data == "26-05") & (maps.Ponto == ponto)]
            if len(mm):
                name = f"mappings/26-05_P{ponto}_mapping.png"
                out["mapa"] = {"image": f"/plots/{name}" if (config.PLOTS_DIR / name).exists() else None,
                               **{k: _clean(mm.iloc[0][k]) for k in ("SPAD_Medio_Mapa", "SPAD_Desvio", "Clorofila_Predita")}}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Outliers e suas justificativas
# ─────────────────────────────────────────────────────────────────────────────
def br(v, d: int = 1, sign: bool = False) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    txt = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return txt.replace(".", ",")


def outliers() -> list[dict]:
    df = image_table()
    out: list[dict] = []
    summ = latest("definitive_model_summary").set_index("Modelo")
    rmse = float(summ.loc[[m for m in summ.index if str(m).startswith("MLP")][0], "RMSE_OOF"])
    bias = df.groupby("Data").res.mean()
    app_bias = df.groupby("Data").app_res.mean() if "app_res" in df else pd.Series(dtype=float)
    obs_p95 = df.obs.quantile(0.95)
    med_blocks = df.groupby("Data").Vegetation_Blocks.median()

    def diagnostics(r, resid, bias_s, who):
        motivos = []
        if r["Data"] == TEST_DATE:
            motivos.append("Ponto da data de teste (01/06): fora do delineamento, sem dose controlada e com bastante palha seca na foto.")
        b = bias_s.get(r["Data"])
        if b is not None and abs(b) >= 1 and np.sign(b) == np.sign(resid):
            motivos.append(f"Toda a data {_short(r['Data'])} foi {'superestimada' if b > 0 else 'subestimada'} "
                           f"pelo {who} (viés médio {br(b, 1, True)} SPAD): a condição da coleta (luz, umidade, estágio) "
                           "mudou a cor das folhas de um jeito que o treino com as outras datas não cobre.")
        if r["obs"] >= obs_p95 and resid < 0:
            motivos.append(f"O SPAD medido ({br(r['obs'], 1)}) está entre os 5% mais altos do conjunto. Há poucos exemplos "
                           "nessa faixa, e a cor RGB muda pouco em dosséis já muito verdes (saturação), então a predição "
                           "tende a ficar abaixo.")
        if r.get("cv", 0) and r["cv"] > CV_LIMIT:
            motivos.append(f"As leituras Falker dessa parcela variam muito entre si (CV {br(r['cv'] * 100, 0)}%, "
                           f"de {br(r['vmin'], 1)} a {br(r['vmax'], 1)}): a referência é pouco estável.")
        if r.get("Canopy_Cover") is not None and r["Canopy_Cover"] < CANOPY_LIMIT:
            motivos.append(f"Só {br(r['Canopy_Cover'], 0)}% dos blocos são vegetação; solo e palha ocupam boa parte da foto.")
        mb = med_blocks.get(r["Data"])
        if mb and r.get("Vegetation_Blocks") and r["Vegetation_Blocks"] < 0.25 * mb:
            motivos.append(f"A foto tem poucos blocos de vegetação ({br(r['Vegetation_Blocks'], 0)}, contra mediana de "
                           f"{br(mb, 0)} na data) — resolução baixa ({r['width']}×{r['height']} px) ou enquadramento diferente.")
        if not motivos:
            motivos.append("Nenhum diagnóstico explica o desvio: pode ser variação natural entre a folha medida e o dossel fotografado.")
        return motivos

    # 1. erro do modelo
    for r in df.dropna(subset=["res"]).to_dict("records"):
        if abs(r["res"]) > RES_FACTOR * rmse:
            out.append({
                "categoria": "Erro do modelo", "id": r["id"], "Data": r["Data"], "Ponto": int(r["Ponto"]),
                "foto": r["rel_path"], "valor": r["Predicao_MLP"], "referencia": r["obs"], "desvio": r["res"],
                "titulo": f"MLP previu {br(r['Predicao_MLP'], 1)}, Falker mediu {br(r['obs'], 1)}",
                "criterio": f"|erro| > {br(RES_FACTOR, 1)} × RMSE ({br(RES_FACTOR * rmse, 1)} SPAD)",
                "motivos": diagnostics(r, r["res"], bias, "modelo"),
            })

    # 2. erro do app
    if "app_res" in df:
        for r in df.dropna(subset=["app_res"]).to_dict("records"):
            if abs(r["app_res"]) > APP_LIMIT:
                out.append({
                    "categoria": "Erro do app", "id": r["id"], "Data": r["Data"], "Ponto": int(r["Ponto"]),
                    "foto": r["rel_path"], "valor": r["app_pred"], "referencia": r["obs"], "desvio": r["app_res"],
                    "titulo": f"App previu {br(r['app_pred'], 1)}, Falker mediu {br(r['obs'], 1)}",
                    "criterio": f"|erro| > {br(APP_LIMIT, 1)} SPAD",
                    "motivos": diagnostics(r, r["app_res"], app_bias, "app"),
                })

    # 3. referência instável
    for r in df.to_dict("records"):
        if r.get("cv") and r["cv"] > CV_LIMIT:
            motivos = [f"Leituras de {br(r['vmin'], 1)} a {br(r['vmax'], 1)} SPAD na mesma parcela (desvio padrão {br(r['sd'], 1)})."]
            motivos.append("Folhas em estágios diferentes ou partes senescentes aumentam a dispersão; a média de "
                           f"{br(r['n'], 0)} leituras representa mal o dossel inteiro.")
            if r["Data"] == TEST_DATE:
                motivos.append("Área de teste sem adubação controlada, com mistura de folhas verdes e palha seca.")
            out.append({
                "categoria": "Referência instável", "id": r["id"], "Data": r["Data"], "Ponto": int(r["Ponto"]),
                "foto": r["rel_path"], "valor": r["cv"] * 100, "referencia": None, "desvio": None,
                "titulo": f"Coeficiente de variação de {br(r['cv'] * 100, 0)}% entre as leituras",
                "criterio": f"CV > {br(CV_LIMIT * 100, 0)}%", "motivos": motivos,
            })

    # 4. leitura Falker isolada (z robusto)
    fr = db.query("SELECT data AS Data, ponto AS Ponto, hora, clorofila_total AS v FROM falker_readings")
    for (d, p), g in fr.groupby(["Data", "Ponto"]):
        med = g.v.median()
        mad = (g.v - med).abs().median()
        if not mad:
            continue
        z = 0.6745 * (g.v - med) / mad
        for (_, row), zz in zip(g.iterrows(), z):
            if abs(zz) > MODZ_LIMIT:
                sem = g.v[g.index != row.name].mean()
                img = df[(df.Data == d) & (df.Ponto == p)]
                out.append({
                    "categoria": "Leitura isolada", "id": plot_id(d, p), "Data": d, "Ponto": int(p),
                    "foto": img.rel_path.iloc[0] if len(img) else None,
                    "valor": row.v, "referencia": med, "desvio": row.v - med,
                    "titulo": f"Leitura de {br(row.v, 1)} SPAD às {row.hora} (mediana da parcela: {br(med, 1)})",
                    "criterio": f"z robusto {br(zz, 1, True)} (|z| > {br(MODZ_LIMIT, 1)})",
                    "motivos": [
                        "Uma leitura muito distante das demais da mesma parcela: provável folha atípica "
                        "(lesionada, sombreada, mais velha) ou pinça mal posicionada.",
                        f"Sem ela, a média da parcela passaria de {br(g.v.mean(), 2)} para {br(sem, 2)} SPAD.",
                    ],
                })

    # 5. qualidade da foto
    for r in df.to_dict("records"):
        mb = med_blocks.get(r["Data"])
        if mb and r.get("Vegetation_Blocks") and r["Vegetation_Blocks"] < 0.25 * mb:
            out.append({
                "categoria": "Qualidade da foto", "id": r["id"], "Data": r["Data"], "Ponto": int(r["Ponto"]),
                "foto": r["rel_path"], "valor": r["Vegetation_Blocks"], "referencia": mb, "desvio": None,
                "titulo": f"Foto de {r['width']}×{r['height']} px com {br(r['Vegetation_Blocks'], 0)} blocos de vegetação",
                "criterio": "menos de 25% da mediana de blocos da data",
                "motivos": [
                    "A imagem tem resolução muito menor que as demais e foi enquadrada mais de perto.",
                    "As estatísticas de cor saem de poucos blocos, e cada bloco de 10×10 px cobre uma área de folha "
                    "diferente das outras fotos. Mesmo assim, o erro do modelo nessa foto ficou "
                    f"{'dentro' if abs(r.get('res') or 0) <= RES_FACTOR * rmse else 'fora'} do limite de outlier.",
                ],
            })

    # 6. altura × clorofila (26/05)
    veg = load_vegetation_plots(verbose=False)
    b = np.polyfit(veg["Clorofila Total"], veg.Altura_Media, 1)
    resid = veg.Altura_Media - np.polyval(b, veg["Clorofila Total"])
    zres = resid / resid.std()
    for (_, r), zz, rr in zip(veg.iterrows(), zres, resid):
        if abs(zz) > 1.8:
            img = df[(df.Data == "26-05-2026") & (df.Ponto == r.Ponto)]
            out.append({
                "categoria": "Altura × clorofila", "id": str(int(r.Ponto)), "Data": "26-05-2026", "Ponto": int(r.Ponto),
                "foto": img.rel_path.iloc[0] if len(img) else None,
                "valor": r.Altura_Media, "referencia": r.Altura_Media - rr, "desvio": rr,
                "titulo": f"Altura de {br(r.Altura_Media, 1)} cm; a clorofila indicaria {br(r.Altura_Media - rr, 1)} cm",
                "criterio": f"resíduo padronizado {br(zz, 1, True)} (|z| > 1,8)",
                "motivos": [
                    f"A parcela ({r.Tratamento}) está {'mais alta' if rr > 0 else 'mais baixa'} do que a sua clorofila sugere.",
                    "A altura também depende de fatores que não mudam a cor da folha, como falhas no estande, "
                    "compactação ou a posição das 5 medidas de altura na parcela "
                    f"({r['Alturas Anotadas (cm)']} cm).",
                ],
            })
    order = {"Erro do modelo": 0, "Erro do app": 1, "Referência instável": 2, "Leitura isolada": 3, "Qualidade da foto": 4, "Altura × clorofila": 5}
    out.sort(key=lambda o: (order[o["categoria"]], -abs(o["desvio"] or o["valor"] or 0)))
    return [{k: (_clean(v) if not isinstance(v, list) else v) for k, v in o.items()} for o in out]


def outlier_counts() -> dict:
    counts: dict = {}
    for o in outliers():
        if o["categoria"] == "Leitura isolada":   # folha isolada não marca a parcela inteira
            continue
        counts[o["id"]] = counts.get(o["id"], 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Dados dos gráficos da página de análise
# ─────────────────────────────────────────────────────────────────────────────
def figures() -> dict:
    df = image_table()
    fr = db.query(
        """SELECT f.data AS Data, CAST(f.ponto AS INTEGER) AS Ponto, l.tratamento AS Tratamento, l.dose AS Dose,
                  f.clorofila_total AS v
           FROM falker_readings f LEFT JOIN dbc_layout l ON l.ponto = CAST(f.ponto AS INTEGER)
           WHERE f.clorofila_total IS NOT NULL""")
    fr.loc[fr.Data == TEST_DATE, ["Tratamento", "Dose"]] = None
    cols = ["id", "Data", "Ponto", "Tratamento", "Dose", "obs", "cv", "Predicao_MLP", "Predicao_Melhor_Indice", "res",
            "app_pred", "app_res", "Canopy_Cover", "Vegetation_Blocks", "Median_r_over_b", "photo_time", "rel_path"]
    images = [{k: _clean(v) for k, v in r.items()} for r in df[[c for c in cols if c in df]].to_dict("records")]
    date_stats = []
    for d, g in df.groupby("Data"):
        date_stats.append({"Data": d, "n": len(g), "obs_media": g.obs.mean(), "vies_modelo": g.res.mean(),
                           "mae_modelo": g.res.abs().mean(), "vies_app": g.app_res.mean() if "app_res" in g else None,
                           "cobertura_media": g.Canopy_Cover.mean(), "hora": g.photo_time.min()})
    veg = load_vegetation_plots(verbose=False)
    veg_rows = veg[["Ponto", "Tratamento", "Dose", "Clorofila Total", "Altura_Media", "Massa_Seca_Total_ha",
                    "Massa_Seca_Folha_ha", "Massa_Verde_Total_ha"]]
    return {
        "readings": [{k: _clean(v) for k, v in r.items()} for r in fr.to_dict("records")],
        "images": images,
        "date_stats": [{k: _clean(v) for k, v in r.items()} for r in date_stats],
        "vegetation": [{k: _clean(v) for k, v in r.items()} for r in veg_rows.to_dict("records")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Análise por coleta
# ─────────────────────────────────────────────────────────────────────────────
def per_date() -> dict:
    def rows(table, **f):
        df = try_latest(table)
        if df is None:
            return []
        for key, val in f.items():
            df = df[df[key] == val]
        return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

    return {
        "summary": rows("per_date_summary"),
        "anova": rows("per_date_anova"),
        "points": rows("per_date_points"),
        "consistency": rows("per_date_consistency")[:20],
        "top": [{k: _clean(v) for k, v in r.items()} for r in
                (try_latest("per_date_correlations").sort_values(["Data", "Rank"]).groupby("Data").head(15).to_dict("records")
                 if try_latest("per_date_correlations") is not None else [])],
    }


def vegetative() -> dict:
    """Índices da foto × altura e biomassa (26/05) e a busca de MLP."""
    def rows(df):
        return [] if df is None else [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

    corr = try_latest("veg_index_correlations")
    top = None
    if corr is not None:
        img = corr[~corr.Referencia.astype(bool)].sort_values(["Variavel_campo", "Rank"])
        top = img.groupby("Variavel_campo").head(15)
    search = try_latest("veg_mlp_search")
    return {
        "summary": rows(try_latest("veg_index_summary")),
        "top": rows(top),
        "points": rows(try_latest("veg_points")),
        "models": rows(try_latest("veg_model_summary")),
        "predictions": rows(try_latest("veg_model_predictions")),
        "search_top": rows(None if search is None else search.sort_values(["Variavel_campo", "Rank"]).groupby("Variavel_campo").head(10)),
        "search_n": 0 if search is None else int(search.groupby("Variavel_campo").size().max()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Downloads: tabelas agrupadas por tema + planilhas consolidadas montadas na hora
# ─────────────────────────────────────────────────────────────────────────────
def _falker_by_plot() -> pd.DataFrame:
    from tcc_analysis.data import load_falker_all
    f = load_falker_all()
    g = (f.groupby(["Data", "Ponto"])
         .agg(n_leituras=("Clorofila Total", "size"),
              SPAD_total_media=("Clorofila Total", "mean"), SPAD_total_dp=("Clorofila Total", "std"),
              Clorofila_A_media=("Clorofila A", "mean"), Clorofila_B_media=("Clorofila B", "mean"))
         .reset_index())
    g["Ponto"] = g.Ponto.astype(int)
    test = g.Data == config.TEST_DATE
    for field in ("Bloco", "Tratamento", "Dose"):
        g[field] = [None if t else config.PONTO_TO_DBC.get(int(p), {}).get(field) for p, t in zip(g.Ponto, test)]
    g["Dose"] = pd.array(g.Dose, dtype="Int64")
    g["Tipo"] = np.where(test, "ponto de teste", "parcela do experimento")
    order = pd.to_datetime(g.Data, format="%d-%m-%Y")
    g = g.assign(_o=order).sort_values(["_o", "Ponto"]).drop(columns="_o")
    return g[["Data", "Tipo", "Ponto", "Bloco", "Tratamento", "Dose", "n_leituras", "SPAD_total_media", "SPAD_total_dp",
              "Clorofila_A_media", "Clorofila_B_media"]]


def _plots_field() -> pd.DataFrame:
    v = load_vegetation_plots(verbose=False)
    cols = ["Ponto", "Bloco", "Tratamento", "Dose", "Clorofila Total", "Clorofila A", "Clorofila B", "Alturas Anotadas (cm)",
            "Altura_Media", "Massa_Verde_Total_ha", "Massa_Seca_Total_ha", "Massa_Verde_Folha_ha", "Massa_Seca_Folha_ha",
            "Massa_Verde_Colmo_ha", "Massa_Seca_Colmo_ha", "Rel_Folha_Colmo_Verde", "Rel_Folha_Colmo_Seco"]
    out = v[cols].rename(columns={"Clorofila Total": "SPAD_total_media_26_05", "Clorofila A": "Clorofila_A_media_26_05",
                                  "Clorofila B": "Clorofila_B_media_26_05"})
    out["Ponto"] = out.Ponto.astype(int)
    return out.sort_values("Ponto")


VIRTUAL_TABLES = {
    "consolidado_clorofila_por_parcela": ("Média, desvio e número de leituras Falker por parcela e data, com bloco, tratamento e dose", _falker_by_plot),
    "consolidado_parcelas_26_05": ("Por parcela em 26/05: SPAD médio, altura (5 medidas e média) e massas verde e seca em t/ha", _plots_field),
}

DOWNLOAD_CATEGORIES = [
    ("Planilhas consolidadas", "Montadas na hora a partir dos dados brutos: o ponto de partida mais prático."),
    ("Medições de campo", "Leituras Falker, altura, pesagens, layout do experimento e fotos, como foram coletados."),
    ("ANOVA", "Efeito da dose e do bloco no SPAD e nos índices."),
    ("Índices da imagem", "Os índices extraídos de cada foto e dos mapas."),
    ("Clorofila: modelos e validação", "Fórmulas, validação cruzada, modelo definitivo e estudo de iluminação."),
    ("Análise por coleta", "Correlações e ANOVA dentro de cada data."),
    ("Classes de clorofila", "Classes baixa, média e alta pelas faixas de Beaufils sobre o SPAD, medidas e pela foto."),
    ("Análises complementares", "Repetibilidade, concordância, dose ótima, nível crítico, clorofila a × b, heterogeneidade, dinâmica e planejamento."),
    ("Altura e biomassa", "Regressões e busca de modelos para altura e massa."),
    ("App", "Validação do app, pesos do modelo e comparação com o Falker."),
    ("Buscas exploratórias", "Varreduras de pré-processamento e de fórmulas."),
    ("Análise anterior (legado)", "Resultados da análise original, sem alteração."),
]


def download_category(name: str) -> str:
    if name in VIRTUAL_TABLES:
        return "Planilhas consolidadas"
    if name.startswith("legacy__"):
        return "Análise anterior (legado)"
    if name in RAW_TABLES or name == "measured_chlorophyll_by_image":
        return "Medições de campo"
    if name.startswith("per_date_"):
        return "Análise por coleta"
    if name.startswith("chl_class_"):
        return "Classes de clorofila"
    if name.startswith(("sci_", "ext_")):
        return "Análises complementares"
    if name.startswith("anova"):
        return "ANOVA"
    if name in {"image_indices", "mapping_images_summary"}:
        return "Índices da imagem"
    if name.startswith(("veg_", "height_", "mlp_height")) or name in {"regression_summary", "predicted_regression_summary"}:
        return "Altura e biomassa"
    if name.startswith(("app_", "aligned_", "full_pipeline", "mlp_dart")):
        return "App"
    if name.startswith(("rgb_preprocess", "reg_sym")):
        return "Buscas exploratórias"
    return "Clorofila: modelos e validação"


def downloads() -> list[dict]:
    cats = {c: {"category": c, "note": note, "tables": []} for c, note in DOWNLOAD_CATEGORIES}
    for name, (desc, _) in VIRTUAL_TABLES.items():
        cats["Planilhas consolidadas"]["tables"].append({"name": name, "description": desc, "rows": None, "cols": None,
                                                         "n_runs": 1, "virtual": True})
    raw_desc = {
        "falker_readings": "Cada leitura do clorofilômetro Falker: data, ponto, hora, coordenadas, clorofila A, B e total",
        "campo_altura_peso": "Alturas anotadas (5 por parcela) e pesos de matéria total, amostra, caule e folha (26/05)",
        "campo_massa_verde_seca": "Massa verde e seca por componente (folha, colmo, amostra) de cada parcela (26/05)",
        "dbc_layout": "Delineamento: ponto, bloco, tratamento e dose de N",
        "images": "Fotos: arquivo, data, ponto e horário",
    }
    for m in table_catalog():
        m = {**m, "description": m.get("description") or raw_desc.get(m["name"])}
        cats[download_category(m["name"])]["tables"].append({**m, "virtual": False})
    return [c for c in cats.values() if c["tables"]]


def virtual_table(name: str) -> pd.DataFrame:
    return VIRTUAL_TABLES[name][1]()


def chl_classes() -> dict:
    """Classes de clorofila (faixas de Beaufils sobre o SPAD): esquema, distribuição, parcelas, concordância."""
    def rows(table):
        df = try_latest(table)
        return [] if df is None else [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    return {name: rows(f"chl_class_{name}") for name in ("scheme", "references", "distribution", "plots", "validation",
                                                         "index_ranking", "agreement", "confusion", "scheme_comparison")}


SCIENCE_TABLES = ["repeatability", "bland_altman", "bland_altman_points", "dose_response_best", "dose_response_means",
                  "critical_level", "critical_points", "ab_anova", "ab_means", "ab_summary", "heterogeneity_plots",
                  "heterogeneity_tests", "temporal_means", "temporal_tests", "temporal_slopes", "plan_readings",
                  "plan_ceiling", "plan_blocks", "plan_sensitivity", "plan_dates"]
SCIENCE_REFERENCES = [
    "SHROUT, P. E.; FLEISS, J. L. Intraclass correlations: uses in assessing rater reliability. Psychological Bulletin, "
    "v. 86, n. 2, p. 420-428, 1979. DOI: 10.1037/0033-2909.86.2.420.",
    "BLAND, J. M.; ALTMAN, D. G. Statistical methods for assessing agreement between two methods of clinical measurement. "
    "The Lancet, v. 327, n. 8476, p. 307-310, 1986.",
    "CATE, R. B.; NELSON, L. A. A simple statistical procedure for partitioning soil test correlation data into two classes. "
    "Soil Science Society of America Proceedings, v. 35, n. 4, p. 658-660, 1971. DOI: 10.2136/sssaj1971.03615995003500040048x.",
    "BEAUFILS, E. R. Diagnosis and recommendation integrated system (DRIS). Pietermaritzburg: University of Natal, 1973. "
    "(Soil Science Bulletin, 1).",
]


def science() -> dict:
    """Análises complementares (pipeline/07_analises_complementares)."""
    out = {}
    for name in SCIENCE_TABLES:
        df = try_latest(f"sci_{name}")
        out[name] = [] if df is None else [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    lim = try_latest("chl_class_scheme")
    out["class_limits"] = [] if lim is None else [_clean(v) for v in lim.sort_values("Ordem").SPAD_max.dropna().tolist()]
    out["references"] = SCIENCE_REFERENCES
    return out


EXPLORATION_TABLES = ["partition", "saturation", "nue", "height_uniformity", "biomass_plots", "color_directions", "color_points",
                      "color_correction", "canopy_leaf", "patch_tests", "patch_correlogram", "field_positions", "field_gradient",
                      "reading_shape", "reading_mean_median", "order_time", "session_points"]


def explorations() -> dict:
    """Explorações (scripts 10–16 de pipeline/07_analises_complementares)."""
    out = {}
    for name in EXPLORATION_TABLES:
        df = try_latest(f"ext_{name}")
        out[name] = [] if df is None else [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    return out
