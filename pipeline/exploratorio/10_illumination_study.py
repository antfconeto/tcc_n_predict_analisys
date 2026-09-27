"""
Estudo de correções de iluminação/cor antes dos índices e da rede neural.

Cada foto é lida uma única vez; guardamos a cor média de cada bloco de 10×10 px (em três espaços
de cor) e as máscaras. As variantes aplicam correções sobre esses blocos e recalculam as entradas
da MLP campeã — 24, sem as colunas *_rgb: (r+g+b)/3 vale sempre 1/3, e o resíduo de arredondamento
de Mean_rgb era o que levava o R² de ~0,59 para 0,753 (ver seção "Achado" no README), sempre com a mesma máscara de vegetação (ExG > 0,15 no pipeline atual), a mesma
arquitetura e a mesma validação por data (GroupKFold, 10 sementes). Assim só o tratamento de cor muda.

Variantes:
  espaço de cor   atual (gamma 0,8) · sem gamma (sRGB da câmera) · linear (sRGB → luz linear)
  máscara HSV     remove pixels de sombra (V < 45), estourados (V > 240) e pouco saturados (S < 25)
  solo            von Kries: ganhos por canal para levar a cor do solo da foto/da data à cor média
                  do solo em todo o conjunto (blocos com ExG < 0)
  log-cromaticidade  + estatísticas de log(R/G) e log(B/G)
  EXIF            + log2(tempo de exposição × ISO/100) como entrada
  centralizado    subtrai a mediana de cada entrada por data (sem usar o SPAD)
  aumento         treino com cópias de cada foto com ganhos de cor aleatórios (±5%)
  âncoras         depois da predição, corrige o deslocamento de cada data com k parcelas medidas

Saídas: illumination_study_summary, illumination_study_predictions, illumination_study_anchor.
"""

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.metrics import compute_mae, compute_r2, compute_rmse
from tcc_analysis.modeling import CHAMP_MLP_PARAMS, group_kfold_oof_lr, load_rgb28_dataset, make_mlp_pipeline

BLOCK = 10
EXG_VEG = 0.15
EXG_SOIL = 0.0
MIN_SOIL_BLOCKS = 30
N_BAG = 10
AUG_COPIES = 8
AUG_SIGMA = 0.05
ANCHOR_REPS = 300
STATS = ["Median", "Mean", "P75", "P90"]
CHANNELS = ["r", "g", "b", "rg", "rb", "gb"]
FEATURES24 = [f"{st}_{ch}" for st in STATS for ch in CHANNELS]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Blocos por foto
# ─────────────────────────────────────────────────────────────────────────────
def srgb_to_linear(x):
    """x em [0, 1] → luz linear em [0, 1] (curva sRGB IEC 61966-2-1)."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def block_mean(img, valid=None):
    h, w = img.shape[:2]
    nh, nw = h // BLOCK, w // BLOCK
    arr = img[: nh * BLOCK, : nw * BLOCK].astype(np.float64)
    if valid is not None:
        arr = arr.copy()
        arr[~valid[: nh * BLOCK, : nw * BLOCK]] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr.reshape(nh, BLOCK, nw, BLOCK, 3), axis=(1, 3))


def chroma(bgr):
    tot = bgr.sum(axis=-1, keepdims=True)
    tot = np.where(tot == 0, 1.0, tot)
    return bgr / tot


def exif_ev(path):
    try:
        with Image.open(path) as im:
            ex = {TAGS.get(k, k): v for k, v in (im._getexif() or {}).items()}
        return float(np.log2(float(ex["ExposureTime"]) * float(ex["ISOSpeedRatings"]) / 100.0))
    except Exception:
        return np.nan


def load_blocks(rel_path):
    """Blocos (BGR) em três espaços de cor, com e sem máscara HSV, mais máscaras de vegetação e solo."""
    img = cv2.imread(str(config.IMAGE_ROOT / rel_path))
    filt = cv2.bilateralFilter(img, 9, 75, 75)
    h, w = filt.shape[:2]
    c = 0.20
    crop = filt[int(c * h): int((1 - c) * h), int(c * w): int((1 - c) * w)]
    lut = np.array([((i / 255.0) ** (1 / 0.8)) * 255 for i in range(256)]).astype(np.uint8)
    g08 = cv2.LUT(crop, lut)

    hsv = cv2.cvtColor(g08, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    specular = (v_ch > 240) & (s_ch < 20)
    hsv_valid = (v_ch >= 45) & (v_ch <= 240) & (s_ch >= 25) & ~specular

    spaces = {
        "atual": g08.astype(np.float64),
        "sem_gamma": crop.astype(np.float64),
        "linear": srgb_to_linear(crop.astype(np.float64) / 255.0) * 255.0,
    }
    ref = block_mean(spaces["atual"])
    exg = (lambda ch: 2 * ch[..., 1] - ch[..., 2] - ch[..., 0])(chroma(ref))   # BGR → g, r, b
    out = {"veg": exg > EXG_VEG, "soil": exg < EXG_SOIL, "ev": exif_ev(config.IMAGE_ROOT / rel_path)}
    for name, arr in spaces.items():
        out[name] = block_mean(arr)
        out[f"{name}_hsv"] = block_mean(arr, hsv_valid)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Entradas da rede a partir dos blocos
# ─────────────────────────────────────────────────────────────────────────────
def stats(values):
    return [float(np.median(values)), float(np.nanmean(values)), float(np.nanpercentile(values, 75)), float(np.nanpercentile(values, 90))]


def features(blocks_bgr, veg, gains=None, logchroma=False):
    px = blocks_bgr[veg]
    px = px[~np.isnan(px).any(axis=1)]
    if gains is not None:
        px = px * gains
    b, g, r = chroma(px).T
    chans = {"r": r, "g": g, "b": b, "rg": (r + g) / 2, "rb": (r + b) / 2, "gb": (g + b) / 2}
    per = {k: stats(v) for k, v in chans.items()}
    x = [per[ch][i] for i in range(4) for ch in CHANNELS]   # mesma ordem de RGB28_FEATURES, sem *_rgb
    rb_median = float(np.median(np.where(b == 0, 0, r / np.where(b == 0, 1, b))))
    if logchroma:
        eps = 1e-6
        x += stats(np.log((px[:, 2] + eps) / (px[:, 1] + eps))) + stats(np.log((px[:, 0] + eps) / (px[:, 1] + eps)))
    return np.array(x), rb_median


def soil_color(blocks_bgr, soil):
    px = blocks_bgr[soil]
    px = px[~np.isnan(px).any(axis=1)]
    return px.mean(axis=0) if len(px) >= MIN_SOIL_BLOCKS else None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Avaliação (mesma rede campeã, validação por data)
# ─────────────────────────────────────────────────────────────────────────────
def oof_mlp(X, y, groups, seeds, train_aug=None):
    """train_aug(train_idx, rng) → (X_extra, y_extra) para o aumento de dados."""
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    pred = np.full(len(y), np.nan)
    template = make_mlp_pipeline(**CHAMP_MLP_PARAMS)
    for tr, te in gkf.split(X, y, groups):
        outs = []
        for seed in seeds:
            Xtr, ytr = X[tr], y[tr]
            if train_aug is not None:
                Xa, ya = train_aug(tr, np.random.default_rng(seed))
                Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
            model = clone(template).set_params(mlp__random_state=seed)
            model.fit(Xtr, ytr)
            outs.append(model.predict(X[te]))
        pred[te] = np.mean(outs, axis=0)
    return pred


def fold_r2(y, pred, groups):
    return {g: compute_r2(y[groups == g], pred[groups == g]) for g in np.unique(groups)}


def anchor_eval(y, pred, groups, k, rng):
    """Corrige o deslocamento de cada data com k parcelas medidas (sorteadas); mede nas demais."""
    r2s, maes, rmses, r2_raw, mae_raw = [], [], [], [], []
    dates = np.unique(groups)
    for _ in range(ANCHOR_REPS):
        yo, pc, pr = [], [], []
        for d in dates:
            idx = np.where(groups == d)[0]
            kk = min(k, len(idx) - 1)
            anchors = rng.choice(idx, kk, replace=False)
            rest = np.setdiff1d(idx, anchors)
            offset = np.mean(y[anchors] - pred[anchors])
            yo.append(y[rest]); pc.append(pred[rest] + offset); pr.append(pred[rest])
        yo, pc, pr = map(np.concatenate, (yo, pc, pr))
        r2s.append(compute_r2(yo, pc)); maes.append(compute_mae(yo, pc)); rmses.append(compute_rmse(yo, pc))
        r2_raw.append(compute_r2(yo, pr)); mae_raw.append(compute_mae(yo, pr))
    return {"R2": np.mean(r2s), "MAE": np.mean(maes), "RMSE": np.mean(rmses), "R2_sem_ancora": np.mean(r2_raw), "MAE_sem_ancora": np.mean(mae_raw)}


def main():
    df = load_rgb28_dataset()
    imgs = db.query("SELECT data, ponto, rel_path FROM images WHERE ponto IS NOT NULL")
    df = df.merge(imgs.rename(columns={"data": "Data", "ponto": "Ponto"}), on=["Data", "Ponto"])
    y = df["Clorofila Total"].to_numpy()
    groups = df["Data"].to_numpy()
    print(f"{len(df)} fotos · datas: {', '.join(sorted(set(groups)))}")

    print("Lendo fotos e calculando blocos...", flush=True)
    cache = [load_blocks(r) for r in df.rel_path]

    # sanidade: a base "atual" tem de reproduzir as entradas gravadas no banco
    X_db = df[FEATURES24].to_numpy()
    X_chk = np.array([features(c["atual"], c["veg"])[0] for c in cache])
    print(f"Conferência com image_indices (bilateral): max|Δ| = {np.nanmax(np.abs(X_chk - X_db)):.2e}")

    # referência de solo: por foto, por data e global
    def soil_refs(space):
        per = [soil_color(c[space], c["soil"]) for c in cache]
        date_ref = {d: np.median([p for p, g in zip(per, groups) if g == d and p is not None], axis=0) for d in np.unique(groups)}
        glob = np.median([p for p in per if p is not None], axis=0)
        return per, date_ref, glob

    def build(space, corr="none", logchroma=False, exif=False, gains_extra=None):
        per, date_ref, glob = soil_refs(space) if corr.startswith("solo") else (None, None, None)
        X, rb = [], []
        for i, c in enumerate(cache):
            gains = None
            if corr == "solo_foto":
                ref = per[i] if per[i] is not None else date_ref[groups[i]]
                gains = glob / ref
            elif corr == "solo_data":
                gains = glob / date_ref[groups[i]]
            if gains_extra is not None:
                gains = (gains if gains is not None else 1.0) * gains_extra[i]
            x, r = features(c[space], c["veg"], gains, logchroma)
            if exif:
                x = np.append(x, c["ev"])
            X.append(x); rb.append(r)
        X = np.array(X)
        if corr == "centralizado":
            for d in np.unique(groups):
                m = groups == d
                X[m] -= np.median(X[m], axis=0)
        return X, np.array(rb)

    variants = [
        # nome, espaço, correção, logchroma, exif, aumento, sementes
        ("Referência (pipeline atual, 24 entradas)", "atual", "none", False, False, False, range(10)),
        ("Referência, outras sementes", "atual", "none", False, False, False, range(10, 20)),
        ("Sem gamma (sRGB da câmera)", "sem_gamma", "none", False, False, False, range(10)),
        ("RGB linear", "linear", "none", False, False, False, range(10)),
        ("Máscara HSV (sombra/reflexo)", "atual_hsv", "none", False, False, False, range(10)),
        ("RGB linear + máscara HSV", "linear_hsv", "none", False, False, False, range(10)),
        ("Solo por foto", "atual", "solo_foto", False, False, False, range(10)),
        ("Solo por data", "atual", "solo_data", False, False, False, range(10)),
        ("RGB linear + solo por data", "linear", "solo_data", False, False, False, range(10)),
        ("+ log-cromaticidade", "atual", "none", True, False, False, range(10)),
        ("RGB linear + solo por data + log-cromaticidade", "linear", "solo_data", True, False, False, range(10)),
        ("+ exposição (EXIF)", "atual", "none", False, True, False, range(10)),
        ("Centralizado por data", "atual", "centralizado", False, False, False, range(10)),
        ("Aumento de cor no treino", "atual", "none", False, False, True, range(10)),
        ("RGB linear + solo por data + aumento", "linear", "solo_data", False, False, True, range(10)),
    ]

    rows, preds, anchors = [], [], []
    for name, space, corr, logc, exif, aug, seeds in variants:
        X, rb = build(space, corr, logc, exif)
        train_aug = None
        if aug:
            def train_aug(tr, rng, space=space, corr=corr, logc=logc, exif=exif):
                Xa = []
                for _ in range(AUG_COPIES):
                    g = np.exp(rng.normal(0, AUG_SIGMA, size=(len(cache), 3)))
                    Xa.append(build(space, corr, logc, exif, gains_extra=g)[0][tr])
                return np.vstack(Xa), np.tile(y[tr], AUG_COPIES)
        pred = oof_mlp(X, y, groups, list(seeds), train_aug)
        rb_pred = group_kfold_oof_lr(rb, y, groups)
        f = fold_r2(y, pred, groups)
        res = pred - y
        row = {
            "Variante": name, "Espaco": space, "Correcao": corr, "Log_cromaticidade": logc, "EXIF": exif, "Aumento": aug,
            "N_entradas": X.shape[1], "R2": compute_r2(y, pred), "RMSE": compute_rmse(y, pred), "MAE": compute_mae(y, pred),
            "Pior_fold_R2": min(f.values()), "R2_indice_rb": compute_r2(y, rb_pred),
            **{f"Vies_{d[:5]}": res[groups == d].mean() for d in np.unique(groups)},
            "Vies_abs_medio": np.mean([abs(res[groups == d].mean()) for d in np.unique(groups)]),
        }
        rows.append(row)
        print(f"  {name:<48} R² {row['R2']:.3f}  RMSE {row['RMSE']:.2f}  pior fold {row['Pior_fold_R2']:+.2f}  "
              f"|viés| {row['Vies_abs_medio']:.2f}  r/b {row['R2_indice_rb']:.3f}", flush=True)
        preds.append(pd.DataFrame({"Variante": name, "Data": groups, "Ponto": df.Ponto, "Observado": y, "Previsto": pred, "Previsto_rb": rb_pred}))
        rng = np.random.default_rng(0)
        for k in (1, 2, 3):
            anchors.append({"Variante": name, "Ancoras_por_data": k, **anchor_eval(y, pred, groups, k, rng)})

    summary = pd.DataFrame(rows)
    db.save_df(summary, "illumination_study_summary", description="Correções de iluminação/cor: MLP rgb28 com validação por data")
    db.save_df(pd.concat(preds, ignore_index=True), "illumination_study_predictions", description="Predições fora da amostra por variante")
    db.save_df(pd.DataFrame(anchors), "illumination_study_anchor", description="Calibração por data com k parcelas medidas no Falker")

    print("\nCalibração com âncoras (média de", ANCHOR_REPS, "sorteios; métricas nas parcelas não usadas como âncora):")
    print(pd.DataFrame(anchors).pivot(index="Variante", columns="Ancoras_por_data", values="R2").round(3).to_string())


if __name__ == "__main__":
    db.run_main(main)
