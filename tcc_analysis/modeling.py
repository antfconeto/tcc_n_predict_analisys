"""MLP campeã (24 entradas) e regressão linear com validação GroupKFold por data.

Funções extraídas do antigo 09-definitive-nn-model.py, compartilhadas pelo modelo
definitivo e pelos scripts de validação do app.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import APP_V1_VARIANT, BEST_PREPROCESS_VARIANT, CHAMP_MLP_PARAMS, MODEL_FEATURES, RGB28_FEATURES
from .data import build_merged_dataset, load_chlorophyll_with_layout, load_indices

RANDOM_STATE = 42
GROUP_COL = "Data"
N_BAG_SEEDS = 10
MAX_ITER = 4000
BLOCK_SIZE = 10


def make_mlp_pipeline(hidden_layer_sizes, alpha, activation="relu", random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="lbfgs",
            alpha=alpha,
            max_iter=MAX_ITER,
            random_state=random_state,
        )),
    ])


def bagged_predict(pipe_template, X_train, y_train, X_test, n_bag=N_BAG_SEEDS):
    preds = []
    for seed in range(n_bag):
        pipe = clone(pipe_template)
        pipe.set_params(mlp__random_state=seed)
        pipe.fit(X_train, y_train)
        preds.append(pipe.predict(X_test))
    return np.mean(preds, axis=0)


def group_kfold_oof_mlp(X, y, groups, pipe_template, n_bag=N_BAG_SEEDS):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        y_pred = bagged_predict(
            pipe_template, X[train_idx], y[train_idx], X[test_idx], n_bag=n_bag,
        )
        y_oof[test_idx] = y_pred

    return y_oof


def group_kfold_oof_lr(X_col, y, groups):
    n_splits = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)
    y_oof = np.full(len(y), np.nan)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    for train_idx, test_idx in gkf.split(X_col, y, groups):
        model = clone(pipe)
        model.fit(X_col[train_idx].reshape(-1, 1), y[train_idx])
        y_pred = model.predict(X_col[test_idx].reshape(-1, 1))
        y_oof[test_idx] = y_pred

    return y_oof


def load_model_dataset(variant: str = BEST_PREPROCESS_VARIANT, features=None) -> pd.DataFrame:
    """Fotos com clorofila média da parcela e as entradas do modelo na variante escolhida."""
    features = features or MODEL_FEATURES
    df = build_merged_dataset(load_chlorophyll_with_layout(), load_indices(BLOCK_SIZE, variant))
    meta = [GROUP_COL, "Ponto", "Bloco_x", "Tratamento_x", "Clorofila Total"]
    return df.dropna(subset=meta + list(features)).copy()


def load_rgb28_dataset(variant: str = APP_V1_VARIANT) -> pd.DataFrame:
    """Entradas da versão 1 do app (28, com *_rgb) — só para validar os pesos antigos."""
    return load_model_dataset(variant, RGB28_FEATURES)


__all__ = [
    "CHAMP_MLP_PARAMS", "GROUP_COL", "N_BAG_SEEDS", "RANDOM_STATE",
    "make_mlp_pipeline", "bagged_predict", "group_kfold_oof_mlp", "group_kfold_oof_lr",
    "load_model_dataset", "load_rgb28_dataset",
]
