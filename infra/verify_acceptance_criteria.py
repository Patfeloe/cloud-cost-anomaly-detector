"""
Train and evaluate the Isolation Forest anomaly detection model.

Implements the locked "Feature Spec: Isolation Forest Anomaly Detection
Step": chronological split, explicit contamination, sign-flipped score
convention (higher = more anomalous), and event-level recall scoring for
sustained_increase anomalies (point anomalies stay row-level).
"""

from _future_ import annotations

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

MODEL_PATH = "/home/claude/model/isolation_forest.pkl"
METADATA_PATH = "/home/claude/model/model_metadata.json"


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

