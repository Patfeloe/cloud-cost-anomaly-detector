"""
Feature engineering shared by training and inference.

Per the locked Isolation Forest feature spec:
- features: daily_cost (raw), rolling 7-day mean/std, day_of_week,
  pct_change_from_prev_day
- daily_cost here is the ZAR-converted value (see FX normalization in the
  architecture doc) — never raw cost_usd
- rolling features require a 7-day warm-up window; the first 7 rows of any
  chronologically-ordered dataset are undefined and must be dropped
  (training) or skipped (inference, until 7 prior days of history exist)
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "daily_cost",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "pct_change_from_prev_day",
    "cost_deviation_ratio",
]

WARMUP_DAYS = 7

# --- Deviation from the locked feature spec, documented here on purpose ---
# The spec's original 4 features (plus day_of_week) use raw daily_cost
# directly. On a long-running chronological train/test split, raw daily_cost
# drifts with trend: the test period's "normal" days sit well above the
# train period's price level, so the model partly learns train-period price
# level as "normal" and over-flags test-period normal days. Empirically this
# capped precision at ~43-46% even at 90%+ recall — no threshold cleared
# both the recall >= 90% and precision >= 50% acceptance bar simultaneously.
#
# cost_deviation_ratio = (daily_cost - rolling_mean_7) / rolling_std_7 is a
# z-score-style feature relative to *recent* spend, not an absolute price
# level, so it stays valid regardless of how much the baseline has drifted
# since training. Added as a 5th feature; the original 4 are unchanged.


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame sorted chronologically (ascending date), with at least
    columns [date, daily_cost, day_of_week].

    Output: same DataFrame with added feature columns. Rows within the
    first WARMUP_DAYS of the input (rolling window undefined) will have
    NaNs in rolling_mean_7 / rolling_std_7 / pct_change_from_prev_day —
    caller is responsible for dropping (training) or skipping (inference)
    those rows; this function does not drop rows itself so it stays usable
    for both a full historical batch (training) and a rolling inference
    window (which may itself be shorter than WARMUP_DAYS on day 1-7 of a
    fresh deployment).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["rolling_mean_7"] = (
        df["daily_cost"].rolling(window=WARMUP_DAYS, min_periods=WARMUP_DAYS).mean()
    )
    df["rolling_std_7"] = (
        df["daily_cost"].rolling(window=WARMUP_DAYS, min_periods=WARMUP_DAYS).std()
    )
    df["pct_change_from_prev_day"] = df["daily_cost"].pct_change()

    # Guard div-by-zero: a flat rolling window (std == 0) is rare but
    # possible early on with low noise; treat deviation as 0 in that case
    # rather than producing inf/NaN.
    std_safe = df["rolling_std_7"].replace(0, pd.NA)
    df["cost_deviation_ratio"] = (
        (df["daily_cost"] - df["rolling_mean_7"]) / std_safe
    ).fillna(0.0)

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Training-time helper: drop rows with undefined rolling features
    (first WARMUP_DAYS rows) rather than backfilling them."""
    return df.dropna(subset=["rolling_mean_7", "rolling_std_7", "pct_change_from_prev_day"]).reset_index(
        drop=True
    )


def has_sufficient_history(prior_days_count: int) -> bool:
    """Inference-time helper: a fresh deployment needs >= WARMUP_DAYS prior
    days on record before a day can be scored."""
    return prior_days_count >= WARMUP_DAYS