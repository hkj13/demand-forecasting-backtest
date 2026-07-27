#!/usr/bin/env python3
"""Rolling-origin backtest with accuracy AND calibration metrics.

The point of this repo: a forecast model is only as good as its out-of-sample
evaluation. A single train/test split flatters whichever model got lucky on
that window; rolling-origin evaluation across multiple cutoffs is the honest
version. And for anything used to make decisions (staffing, ordering), the
interval calibration matters as much as the point accuracy: an 80% interval
that only covers 60% of actuals will systematically under-plan.

Metrics:
  WAPE     sum(|err|) / sum(actual)          scale-free accuracy, robust to zeros
  sMAPE    symmetric MAPE                    common in forecasting papers
  MASE     MAE / in-sample seasonal-naive MAE  <1 means "beats naive on train scale"
  coverage share of actuals inside the 80% prediction interval (target: 0.80)
  pinball  mean quantile loss at p10/p90     rewards sharp AND calibrated intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path

from models import POINT_MODELS, gbm_with_quantiles

DATA = Path(__file__).parent / "data" / "demand.csv"
HORIZON = 14          # forecast 14 days ahead
N_FOLDS = 4           # rolling cutoffs, 28 days apart
STEP = 28


def wape(actual, pred):
    return np.abs(actual - pred).sum() / actual.sum()


def smape(actual, pred):
    return np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-9))


def mase(actual, pred, train):
    scale = np.mean(np.abs(train[7:] - train[:-7]))  # in-sample seasonal-naive MAE
    return np.mean(np.abs(actual - pred)) / scale


def pinball(actual, pred, q):
    diff = actual - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    stores = sorted(df["store_id"].unique())
    last_date = df["date"].max()
    cutoffs = [last_date - pd.Timedelta(days=HORIZON + i * STEP) for i in range(N_FOLDS)]

    rows, cov_rows = [], []
    for i, store in enumerate(stores, 1):
        print(f"backtesting store {i}/{len(stores)}...", flush=True)
        s = df[df.store_id == store].set_index("date")["units"].asfreq("D")
        for cutoff in cutoffs:
            train = s[s.index <= cutoff]
            actual = s[(s.index > cutoff)][:HORIZON].to_numpy()
            tr = train.to_numpy()

            for name, fn in POINT_MODELS.items():
                pred = fn(train, HORIZON)
                rows.append((name, wape(actual, pred), smape(actual, pred), mase(actual, pred, tr)))

            q = gbm_with_quantiles(train, HORIZON)
            rows.append(("gbm_quantile", wape(actual, q["p50"]), smape(actual, q["p50"]),
                         mase(actual, q["p50"], tr)))
            covered = np.mean((actual >= q["p10"]) & (actual <= q["p90"]))
            pb = (pinball(actual, q["p10"], 0.1) + pinball(actual, q["p90"], 0.9)) / 2
            cov_rows.append((covered, pb))

    res = pd.DataFrame(rows, columns=["model", "wape", "smape", "mase"])
    summary = res.groupby("model").mean().sort_values("wape")

    print(f"\nRolling-origin backtest: {len(stores)} stores x {N_FOLDS} folds, "
          f"horizon {HORIZON}d\n")
    print("| model | WAPE | sMAPE | MASE |")
    print("|---|---|---|---|")
    for name, r in summary.iterrows():
        print(f"| {name} | {r.wape:.3f} | {r.smape:.3f} | {r.mase:.3f} |")

    cov = np.mean([c for c, _ in cov_rows])
    pb = np.mean([p for _, p in cov_rows])
    print(f"\n80% interval coverage (gbm_quantile): {cov:.1%}  (target 80%)")
    print(f"Mean pinball loss (p10/p90): {pb:.2f}")


if __name__ == "__main__":
    main()
