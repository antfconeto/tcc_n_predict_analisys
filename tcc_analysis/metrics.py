"""Métricas de regressão compartilhadas pelos scripts de modelagem."""

import numpy as np


def compute_r2(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def compute_rmse(y_obs, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_obs) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_obs, y_pred):
    return float(np.mean(np.abs(np.asarray(y_obs) - np.asarray(y_pred))))
