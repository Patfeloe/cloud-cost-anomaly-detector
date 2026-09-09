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

