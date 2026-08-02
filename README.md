# demand-forecasting-backtest

Honest evaluation for demand forecasting: rolling-origin backtesting, baselines that must be
beaten, and quantile forecasts scored on calibration, not just point accuracy.

## Why this repo

Most forecasting write-ups show one train/test split and one accuracy number. That flatters
whichever model got lucky on that window, and it says nothing about whether the model's
uncertainty estimates can be trusted. In operational forecasting (staffing, ordering,
capacity), the questions that actually matter are:

1. **Does the model beat a trivially cheap baseline** across many time windows, not one?
2. **Are the prediction intervals calibrated?** An "80% interval" that only covers 60% of
   actuals will systematically under-plan, and no point-accuracy metric will reveal it.

This repo is a compact, fully reproducible harness for answering both.

## What's inside

| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic daily demand for 12 stores over 2 years: trend, weekly and annual seasonality, promo spikes, noise. Seeded, deterministic. |
| `models.py` | Two baselines (seasonal naive, moving average) and two models (Holt-Winters, gradient boosting on lag/calendar features with p10/p50/p90 quantile heads). |
| `backtest.py` | Rolling-origin evaluation: 4 cutoffs per store, 14-day horizon, aggregated WAPE / sMAPE / MASE plus interval coverage and pinball loss. |

## Run it

```bash
pip install -r requirements.txt
python generate_data.py
python backtest.py        # ~2-3 minutes: 12 stores x 4 folds, GBM refit per fold
```

## Results

Rolling-origin backtest, 12 stores x 4 folds, 14-day horizon:

| model | WAPE | sMAPE | MASE |
|---|---|---|---|
| gbm_quantile | 0.098 | 0.100 | 0.903 |
| holt_winters | 0.104 | 0.104 | 0.951 |
| seasonal_naive | 0.120 | 0.120 | 1.104 |
| moving_average | 0.162 | 0.164 | 1.468 |

```
80% interval coverage (gbm_quantile): 58.5%  (target 80%)
Mean pinball loss (p10/p90): 7.50
```

## Reading the results, including the failure

- **Point accuracy looks healthy.** The GBM beats Holt-Winters, and both beat the baselines
  with MASE below 1.0. If this were a one-split evaluation with only WAPE reported, the model
  would ship.
- **The calibration check caught a real problem.** The nominal 80% interval only covers 58.5%
  of actuals: the quantile model is overconfident. Anyone planning safety stock or staffing
  buffers against these intervals would be systematically under-covered, and no point-accuracy
  metric would ever reveal it. This is precisely why the harness scores calibration separately.
- **Why it happens here:** the multi-step forecast is recursive, feeding the median prediction
  back as lag features. The p10/p90 heads were trained on one-step-ahead residuals, so they
  never see the error that accumulates over a 14-day recursive horizon; intervals that are
  honest at step 1 are too narrow by step 14.
- **What to do about it in production:** conformalize the intervals or train direct
  per-horizon quantile models instead of recursive ones. The first option is now implemented
  below. Either way, the fix is only findable because the evaluation measures coverage at the
  deployed horizon.
- **The baselines are the point.** Seasonal naive is embarrassingly strong on weekly-seasonal
  demand; any pipeline that never compares against it can ship a model that is worse than
  copying last week.

## Follow-up: the conformal fix (`conformal.py`)

Sequential conformalized quantile regression: for each day the conformity score is
`max(p10 - y, y - p90)`, and the correction Q applied to fold k is the finite-sample 80%
quantile of scores pooled from chronologically earlier folds only, so the fix never looks
forward. Fold 1 runs raw because a deployed system spends its first cycle collecting scores.

| fold | coverage raw | coverage conformal | mean width raw | mean width conformal |
|---|---|---|---|---|
| 1 (no calibration yet) | 54.8% | - | 50.4 | - |
| 2 (Q=+21.2) | 62.5% | 88.1% | 41.7 | 84.1 |
| 3 (Q=+16.0) | 58.9% | 84.5% | 55.3 | 87.3 |
| 4 (Q=+14.9) | 57.7% | 78.0% | 49.2 | 78.9 |

```
folds 2-4 overall: raw 59.7% -> conformal 83.5%  (target 80%)
```

Three things worth noticing:

- **Coverage lands where it should.** 59.7% becomes 83.5% against an 80% target, using only
  past information at every step.
- **The correction shrinks as evidence accumulates** (+21.2 with one calibration fold, +14.9
  with three): early corrections are conservative because the finite-sample quantile is, and
  they converge toward the target as the score pool grows. The slight overshoot on fold 2 is
  that conservatism, not a bug.
- **Honest uncertainty costs width.** Intervals roughly widen by two thirds. That is the true
  price of 80% coverage under recursive multi-step error, and it was being hidden by the raw
  intervals. If the width is operationally unacceptable, the model needs to improve (direct
  per-horizon heads), not the intervals to lie.

## Design notes

- Rolling-origin (a.k.a. time-series cross-validation) steps the cutoff back 28 days at a time,
  so every metric is an average over genuinely out-of-sample windows in different seasonal
  positions.
- MASE uses the in-sample seasonal-naive MAE as its scale, which makes it comparable across
  stores with very different volumes, unlike raw MAE.
- Quantile forecasts come from three gradient-boosting heads (p10/p50/p90) rather than a
  distributional assumption, so the interval width adapts to local volatility such as promo
  weeks.

All data is synthetic and generated by `generate_data.py`; no external datasets are required.
