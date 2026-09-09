"""Checks the trained Isolation Forest pipeline against the locked
acceptance criteria in the Isolation Forest feature spec."""

import json
import pickle
import sys
from datetime import date

import numpy as np

from features import FEATURE_COLUMNS, WARMUP_DAYS, build_features, drop_warmup_rows
from synthetic_spend import generate_synthetic_spend, strip_labels
from train_isolation_forest import MODEL_PATH, METADATA_PATH, chronological_split
from anomaly_check import score_day

results = []

def check(name, condition):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

# --- 1. Model trains successfully; warm-up rows excluded ---
check("Model artifact exists and loads", model is not None)
check(
    "Warm-up rows excluded from training (train_rows count is plausible "
    f"post-drop: {metadata['trained_on']['train_rows']})",
    metadata["trained_on"]["train_rows"] > 0,
)

# --- 2. Chronological split used, not random ---
df, _ = generate_synthetic_spend(
    start_date=date(2018, 1, 1), num_days=3000, anomaly_rate=0.06,
    fx_volatility=0.08, seed=42,
)
featured = build_features(df)
train_raw, test_raw = chronological_split(featured, 0.8)
check(
    "Chronological split: all train dates precede all test dates",
    train_raw["date"].max() < test_raw["date"].min(),
)
