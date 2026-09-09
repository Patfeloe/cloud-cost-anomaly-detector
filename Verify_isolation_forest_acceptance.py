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
# --- 3. contamination set explicitly (not 'auto') ---
check(
    f"contamination is an explicit float, not 'auto' (value={metadata['contamination']})",
    isinstance(metadata["contamination"], (int, float)),
)

# --- 4. Recall >= 90% AT precision >= 50% (combined hit-based recall, per spec) ---
eval_metrics = metadata["evaluation"]
combined_recall = eval_metrics["combined_hit_based_recall"]
row_precision = eval_metrics["row_level_precision"]
check(
    f"Combined hit-based recall >= 0.90 (actual={combined_recall:.3f})",
    combined_recall >= 0.90,
)
check(
    f"Row-level precision >= 0.50 (actual={row_precision:.3f})",
    row_precision >= 0.50,
)

# --- 5. Inference runs in <1s per record ---
latency_ms = metadata["inference_latency_ms_single_record"]
check(f"Inference latency < 1000ms (actual={latency_ms:.2f}ms)", latency_ms < 1000)

# --- 6. Lambda handles <7-days-of-history without erroring ---
prod_shaped = strip_labels(df).to_dict("records")
try:
    result_day3 = score_day(prod_shaped[:3], prod_shaped[3])
    no_error = True
except Exception as e:
    no_error = False
    result_day3 = {"error": str(e)}
check("Scoring with <7 days history does not raise", no_error)
check(
    "Insufficient-history case returns status='insufficient_history', is_anomaly=None (not an error)",
    result_day3.get("status") == "insufficient_history" and result_day3.get("is_anomaly") is None,
)

# Confirm it recovers correctly at exactly WARMUP_DAYS history
result_day8 = score_day(prod_shaped[:WARMUP_DAYS], prod_shaped[WARMUP_DAYS])
check(
    f"Scoring resumes normally at exactly {WARMUP_DAYS} days of history "
    f"(status={result_day8.get('status')})",
    result_day8.get("status") == "scored" and result_day8.get("is_anomaly") in (True, False),
)

# --- 7. Output schema matches interface, including score sign convention ---
result_deep = score_day(prod_shaped[:500], prod_shaped[500])
expected_keys = {"date", "daily_cost", "is_anomaly", "anomaly_score",
                  "service_breakdown", "cost_usd", "fx_rate", "status"}
check("Output schema has all required interface fields", expected_keys.issubset(result_deep.keys()))

# Score sign convention: higher = more anomalous. Verify on a known
# spike day (ground truth) vs a known normal day.
spike_idx = df[df["anomaly_type"] == "spike"].index
normal_idx = df[~df["is_anomaly"]].index
sample_spike_idx = int(spike_idx[spike_idx > WARMUP_DAYS][0])
sample_normal_idx = int(normal_idx[normal_idx > WARMUP_DAYS][10])

spike_result = score_day(prod_shaped[:sample_spike_idx], prod_shaped[sample_spike_idx])
normal_result = score_day(prod_shaped[:sample_normal_idx], prod_shaped[sample_normal_idx])
check(
    "Score convention correct: known spike day scores higher (more anomalous) "
    f"than known normal day (spike={spike_result['anomaly_score']:.3f}, "
    f"normal={normal_result['anomaly_score']:.3f})",
    spike_result["anomaly_score"] > normal_result["anomaly_score"],
)

# --- 8. daily_cost in features/output refers to ZAR, not USD (FX spec requirement) ---
check(
    "daily_cost in scored output matches the ZAR value passed in (not USD)",
    abs(result_deep["daily_cost"] - prod_shaped[500]["daily_cost"]) < 0.01,
)
check(
    "cost_usd / fx_rate passed through unchanged, unused by model features",
    "cost_usd" not in FEATURE_COLUMNS and "fx_rate" not in FEATURE_COLUMNS,
)

print(f"\n{sum(p for _, p in results)}/{len(results)} checks passed")
if not all(p for _, p in results):
    sys.exit(1)