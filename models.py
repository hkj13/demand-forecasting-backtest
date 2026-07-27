"""Forecasting models: two baselines every real model must beat, and two models.

Each forecaster takes a training series (daily, one store) and a horizon, and
returns point forecasts. The gradient-boosting model also produces quantile
forecasts so the backtest can score interval calibration, not just accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed week. The baseline to beat."""
    last_week = train.to_numpy()[-7:]
    return np.tile(last_week, horizon // 7 + 1)[:horizon]


def moving_average(train: pd.Series, horizon: int, window: int = 28) -> np.ndarray:
    return np.full(horizon, train.to_numpy()[-window:].mean())


def holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(
        train.to_numpy(), trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.maximum(model.forecast(horizon), 0)


def _features(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(values, index=dates)
    return pd.DataFrame({
        "lag7": s.shift(7),
        "lag14": s.shift(14),
        "lag28": s.shift(28),
        "roll7": s.shift(1).rolling(7).mean(),
        "roll28": s.shift(1).rolling(28).mean(),
        "dow": dates.dayofweek,
        "month": dates.month,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    }, index=dates)


def gbm_with_quantiles(train: pd.Series, horizon: int) -> dict:
    """Gradient boosting on lag/calendar features; recursive multi-step forecast.

    Returns point (median) plus 10th/90th percentile forecasts for an 80%
    prediction interval.
    """
    history = train.copy()
    fit_X = _features(history.to_numpy(), history.index).dropna()
    fit_y = history.loc[fit_X.index]

    models = {}
    for name, kwargs in {
        "p50": {"loss": "quantile", "quantile": 0.5},
        "p10": {"loss": "quantile", "quantile": 0.1},
        "p90": {"loss": "quantile", "quantile": 0.9},
    }.items():
        m = HistGradientBoostingRegressor(max_iter=120, max_depth=6, random_state=0, **kwargs)
        m.fit(fit_X, fit_y)
        models[name] = m

    # Recursive forecast: feed the median prediction back as history so lag
    # features exist for later steps of the horizon.
    out = {"p50": [], "p10": [], "p90": []}
    extended = history.copy()
    future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    for d in future_dates:
        tmp = pd.concat([extended, pd.Series([np.nan], index=[d])])
        X = _features(tmp.to_numpy(), tmp.index)[-1:]
        for name in out:
            out[name].append(max(models[name].predict(X)[0], 0))
        extended = pd.concat([extended, pd.Series([out["p50"][-1]], index=[d])])
    return {k: np.array(v) for k, v in out.items()}


POINT_MODELS = {
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "holt_winters": holt_winters,
}
