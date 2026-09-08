"""
Train and evaluate the Isolation Forest anomaly detection model.

Implements the locked "Feature Spec: Isolation Forest Anomaly Detection
Step": chronological split, explicit contamination, sign-flipped score
convention (higher = more anomalous), and event-level recall scoring for
sustained_increase anomalies (point anomalies stay row-level).
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from features import FEATURE_COLUMNS, build_features, drop_warmup_rows
from synthetic_spend import generate_synthetic_spend


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by position in date order — NOT random. Rolling/pct-change
    features depend on prior rows, so a random split would leak future
    information into training and inflate reported recall."""
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def group_sustained_events(df: pd.DataFrame) -> list[list[int]]:
    """Group consecutive rows labeled anomaly_type == 'sustained_increase'
    into events (list of positional index lists), for event-level scoring.
    Assumes df is sorted chronologically with a contiguous positional index
    (0..n-1), i.e. call this on a reset_index'd, date-sorted frame."""
    events: list[list[int]] = []
    current: list[int] = []
    prev_pos = None

    for pos, atype in enumerate(df["anomaly_type"].values):
        if atype == "sustained_increase":
            if current and pos == prev_pos + 1:
                current.append(pos)
            else:
                if current:
                    events.append(current)
                current = [pos]
            prev_pos = pos
        else:
            if current:
                events.append(current)
                current = []
            prev_pos = None

    if current:
        events.append(current)
    return events


def evaluate(test_df: pd.DataFrame, y_pred_is_anomaly: np.ndarray) -> dict:
    """
    Row-level metrics (all anomaly types) + point-only row recall +
    sustained-increase event-level recall, per the spec's eval section.
    """
    y_true = test_df["is_anomaly"].values

    row_precision = precision_score(y_true, y_pred_is_anomaly, zero_division=0)
    row_recall = recall_score(y_true, y_pred_is_anomaly, zero_division=0)
    row_f1 = f1_score(y_true, y_pred_is_anomaly, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_is_anomaly).tolist()

    point_mask = test_df["anomaly_type"].isin(["spike", "drop"]).values
    point_total = point_mask.sum()
    point_hits = int((y_pred_is_anomaly[point_mask] & y_true[point_mask]).sum()) if point_total else 0
    point_recall = point_hits / point_total if point_total else None

    events = group_sustained_events(test_df)
    event_total = len(events)
    event_hits = sum(1 for ev in events if y_pred_is_anomaly[ev].any())
    event_recall = event_hits / event_total if event_total else None

    # Combined "hit-based" recall used for the acceptance bar: point anomalies
    # scored per-row, sustained_increase scored per-event (spec-mandated).
    combined_hits = point_hits + event_hits
    combined_total = point_total + event_total
    combined_recall = combined_hits / combined_total if combined_total else None

    return {
        "row_level_precision": row_precision,
        "row_level_recall": row_recall,
        "row_level_f1": row_f1,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "point_anomaly_row_recall": point_recall,
        "point_anomaly_row_count": int(point_total),
        "sustained_event_recall": event_recall,
        "sustained_event_count": event_total,
        "combined_hit_based_recall": combined_recall,
        "combined_total_anomalies": combined_total,
    }


def train_and_evaluate():
    # --- Generate synthetic data (same generator used for both demo + training) ---
    df, gen_config = generate_synthetic_spend(
        start_date=date(2018, 1, 1),
        num_days=3000,
        anomaly_rate=0.06,
        anomaly_types=["spike", "drop", "sustained_increase"],
        fx_volatility=0.08,
        seed=42,
    )

    # --- Feature engineering (shared with inference) ---
    featured = build_features(df)

    # --- Chronological split BEFORE warm-up drop, so the split boundary
    # reflects real calendar time, then drop warm-up rows from each side
    # independently (train's first 7 rows and test's first 7 rows both
    # have undefined rolling features, since test's rolling window at its
    # own start doesn't have access to train's tail in this simplified
    # local-eval setup — matches "drop first 7 rows per dataset" in spec) ---
    train_raw, test_raw = chronological_split(featured, train_frac=0.8)
    train_df = drop_warmup_rows(train_raw)
    test_df = drop_warmup_rows(test_raw)

    print(f"Train rows (post warm-up drop): {len(train_df)}")
    print(f"Test rows (post warm-up drop): {len(test_df)}")

    # --- Contamination selection: carve an inner validation slice out of
    # TRAIN (chronologically, same rule as the outer split) and sweep
    # candidate contamination values there. Selecting on test_df directly
    # would be leakage — the acceptance numbers reported below come from a
    # single untouched evaluation on test_df using whatever contamination
    # wins on the inner validation slice. ---
    inner_train_raw, val_raw = chronological_split(train_df, train_frac=0.8)
    inner_train_df = drop_warmup_rows(inner_train_raw)
    val_df = drop_warmup_rows(val_raw)
    print(f"Inner-train rows: {len(inner_train_df)}, validation rows: {len(val_df)}")

    empirical_rate = train_df["is_anomaly"].mean()
    candidates = sorted(set(
        round(empirical_rate * mult, 4) for mult in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    ))
    candidates = [min(max(c, 0.001), 0.5) for c in candidates]

    best_contamination = None
    best_val_score = (-1, -1.0)
    print("\nContamination sweep (selected on validation slice, not test):")
    for c in candidates:
        cand_model = IsolationForest(n_estimators=100, contamination=c, random_state=42)
        cand_model.fit(inner_train_df[FEATURE_COLUMNS].values)
        val_pred = cand_model.predict(val_df[FEATURE_COLUMNS].values) == -1
        val_metrics = evaluate(val_df, val_pred)
        r = val_metrics["combined_hit_based_recall"] or 0.0
        p = val_metrics["row_level_precision"]
        # Selection rule mirrors the acceptance bar: among candidates that
        # hit recall >= 0.90, pick the one with highest precision. If none
        # clear 0.90 recall, fall back to best combined recall.
        meets_recall = r >= 0.90
        score = (1, p) if meets_recall else (0, r)
        print(f"  contamination={c:.4f}  val_recall={r:.3f}  val_precision={p:.3f}"
              f"  {'[meets recall bar]' if meets_recall else ''}")
        if score > best_val_score:
            best_val_score = score
            best_contamination = c

    contamination = best_contamination
    print(f"\nSelected contamination (from validation, not test): {contamination:.4f}")

    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)

    # --- Inference + score convention: sklearn decision_function is
    # higher = more normal; flip sign so higher = more anomalous ---
    raw_predictions = model.predict(X_test)  # -1 anomaly, 1 normal
    y_pred_is_anomaly = raw_predictions == -1

    raw_scores = model.decision_function(X_test)  # higher = more normal
    anomaly_scores = -raw_scores  # higher = more anomalous

    metrics = evaluate(test_df, y_pred_is_anomaly)

    print("\n--- Evaluation ---")
    print(f"Row-level precision: {metrics['row_level_precision']:.3f}")
    print(f"Row-level recall:    {metrics['row_level_recall']:.3f}")
    print(f"Row-level F1:        {metrics['row_level_f1']:.3f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")
    print(f"Point-anomaly row recall (spike/drop): {metrics['point_anomaly_row_recall']}"
          f" ({metrics['point_anomaly_row_count']} point-anomaly rows)")
    print(f"Sustained-increase EVENT recall: {metrics['sustained_event_recall']}"
          f" ({metrics['sustained_event_count']} events)")
    print(f"Combined hit-based recall (spec acceptance metric): "
          f"{metrics['combined_hit_based_recall']:.3f}"
          f" ({metrics['combined_total_anomalies']} total anomalies)")

    # --- Manual sanity check: inspect a few flagged and non-flagged days ---
    print("\n--- Sample flagged days ---")
    flagged = test_df[y_pred_is_anomaly].head(3)
    print(flagged[["date", "daily_cost", "anomaly_type", "is_anomaly"]].to_string(index=False))
    print("\n--- Sample non-flagged days ---")
    not_flagged = test_df[~y_pred_is_anomaly].head(3)
    print(not_flagged[["date", "daily_cost", "anomaly_type", "is_anomaly"]].to_string(index=False))

    # --- Inference latency check (<1s per record) ---
    single_row = X_test[:1]
    start = time.perf_counter()
    model.predict(single_row)
    model.decision_function(single_row)
    elapsed = time.perf_counter() - start
    print(f"\nSingle-record inference latency: {elapsed*1000:.2f} ms")

    # --- Save model + metadata ---
    import os
    os.makedirs("/home/claude/model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "n_estimators": 100,
        "score_convention": "higher_is_more_anomalous",
        "trained_on": {
            "generator_config": gen_config.__dict__,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "evaluation": metrics,
        "inference_latency_ms_single_record": elapsed * 1000,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

    return model, metadata, test_df, y_pred_is_anomaly, anomaly_scores


if __name__ == "__main__":
    train_and_evaluate()