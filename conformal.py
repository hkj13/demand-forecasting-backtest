#!/usr/bin/env python3
"""Conformalize the GBM prediction intervals (the fix the backtest called for).

The backtest showed the raw p10/p90 intervals covering ~58% instead of 80%:
the quantile heads are trained on one-step-ahead residuals and never see the
error that accumulates over a 14-day recursive horizon.

Fix: conformalized quantile regression (CQR), applied sequentially so it never
looks forward. For each day in a fold we compute the conformity score

    E = max(p10 - y, y - p90)

(positive when the actual falls outside the interval, negative when inside
with room to spare). For fold k, the correction Q is the finite-sample 80%
quantile of the scores pooled from all chronologically EARLIER folds across
stores, and the served interval becomes [p10 - Q, p90 + Q]. Q can be negative,
so the same mechanism would shrink over-wide intervals; here it widens.

Fold 1 has no earlier folds to calibrate on, so it is reported raw-only: a
deployed system spends its first cycle collecting scores before correcting.

    python conformal.py     # ~2-3 minutes, same fold structure as backtest.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from models import gbm_with_quantiles

DATA = Path(__file__).parent / "data" / "demand.csv"
HORIZON = 14
N_FOLDS = 4
STEP = 28
TARGET = 0.80


def conformal_quantile(scores: np.ndarray, level: float = TARGET) -> float:
    """Finite-sample-corrected empirical quantile (the standard CQR form)."""
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    stores = sorted(df["store_id"].unique())
    last_date = df["date"].max()
    # chronological order: earliest cutoff first, so "past" means past
    cutoffs = sorted(last_date - pd.Timedelta(days=HORIZON + i * STEP)
                     for i in range(N_FOLDS))

    records = []   # (fold_idx, actual, p10, p90) per day
    for i, store in enumerate(stores, 1):
        print(f"forecasting store {i}/{len(stores)}...", flush=True)
        s = df[df.store_id == store].set_index("date")["units"].asfreq("D")
        for k, cutoff in enumerate(cutoffs):
            train = s[s.index <= cutoff]
            actual = s[s.index > cutoff][:HORIZON].to_numpy()
            q = gbm_with_quantiles(train, HORIZON)
            for y, lo, hi in zip(actual, q["p10"], q["p90"]):
                records.append((k, y, lo, hi))

    rec = pd.DataFrame(records, columns=["fold", "y", "p10", "p90"])
    rec["score"] = np.maximum(rec.p10 - rec.y, rec.y - rec.p90)

    print(f"\nSequential CQR: correction for fold k calibrated on folds < k "
          f"(pooled across {len(stores)} stores)\n")
    print("| fold | coverage raw | coverage conformal | mean width raw | mean width conformal |")
    print("|---|---|---|---|---|")
    raw_all, conf_all = [], []
    for k in range(N_FOLDS):
        fold = rec[rec.fold == k]
        raw_cov = ((fold.y >= fold.p10) & (fold.y <= fold.p90)).mean()
        raw_w = (fold.p90 - fold.p10).mean()
        if k == 0:
            print(f"| 1 (no calibration yet) | {raw_cov:.1%} | - | {raw_w:.1f} | - |")
            continue
        cal_scores = rec[rec.fold < k]["score"].to_numpy()
        Q = conformal_quantile(cal_scores)
        lo, hi = fold.p10 - Q, fold.p90 + Q
        conf_cov = ((fold.y >= lo) & (fold.y <= hi)).mean()
        conf_w = (hi - lo).mean()
        print(f"| {k + 1} (Q={Q:+.1f}) | {raw_cov:.1%} | {conf_cov:.1%} "
              f"| {raw_w:.1f} | {conf_w:.1f} |")
        raw_all.append(raw_cov)
        conf_all.append(conf_cov)

    print(f"\nfolds 2-{N_FOLDS} overall: raw {np.mean(raw_all):.1%} -> "
          f"conformal {np.mean(conf_all):.1%}  (target {TARGET:.0%})")


if __name__ == "__main__":
    main()
