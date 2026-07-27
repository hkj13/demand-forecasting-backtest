#!/usr/bin/env python3
"""Generate synthetic daily demand for a fleet of retail stores.

Signal components per store: base level, slow trend, weekly seasonality,
annual seasonality, random promo uplifts, and noise. Deterministic (seeded)
so results in the README are reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_STORES = 12
N_DAYS = 730  # two years
START = "2024-07-01"
OUT = Path(__file__).parent / "data" / "demand.csv"


def main() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    day = np.arange(N_DAYS)
    frames = []

    for s in range(N_STORES):
        base = rng.uniform(80, 400)
        trend = rng.uniform(-0.02, 0.06) * day
        weekly = base * 0.25 * np.sin(2 * np.pi * (day + rng.integers(0, 7)) / 7)
        weekend_lift = base * 0.15 * (pd.Series(dates).dt.dayofweek >= 5).to_numpy()
        annual = base * 0.12 * np.sin(2 * np.pi * day / 365 + rng.uniform(0, 2 * np.pi))

        promo = np.zeros(N_DAYS)
        for _ in range(10):  # ~10 promo weeks over two years
            start = rng.integers(0, N_DAYS - 7)
            promo[start : start + 7] += base * rng.uniform(0.2, 0.5)

        noise = rng.normal(0, base * 0.08, N_DAYS)
        demand = np.maximum(base + trend + weekly + weekend_lift + annual + promo + noise, 0)

        frames.append(pd.DataFrame({
            "date": dates,
            "store_id": f"s{s + 1:02d}",
            "units": np.round(demand).astype(int),
        }))

    df = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {N_STORES} stores x {N_DAYS} days")


if __name__ == "__main__":
    main()
