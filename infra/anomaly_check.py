def _load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache


def score_day(history: list[dict], today: dict) -> dict:
    """
    history: prior daily spend records, chronologically ordered, each shaped
        like {date, daily_cost, day_of_week, ...}
        today: the record being scored, shape:
        {date, daily_cost, service_breakdown, cost_usd, fx_rate, day_of_week}

 Returns the anomaly-check Lambda output shape. If history is
    insufficient (< 7 prior days), returns is_anomaly=None with a
    "insufficient history" note instead of a score.
    """
    if not has_sufficient_history(len(history)):
        return {
            "date": today["date"],
            "daily_cost": today["daily_cost"],
            "is_anomaly": None,
            "anomaly_score": None,
            "service_breakdown": today.get("service_breakdown"),
            "cost_usd": today.get("cost_usd"),
            "fx_rate": today.get("fx_rate"),
            "status": "insufficient_history",
            "note": f"Need >= {WARMUP_DAYS} prior days, have {len(history)}. Scoring skipped.",
        }

    window_records = history[-WARMUP_DAYS:] + [today]
    window_df = pd.DataFrame(window_records)
    featured = build_features(window_df)

    today_row = featured.iloc[[-1]]  # last row = today, after build_features
    if today_row[FEATURE_COLUMNS].isna().any(axis=None):
        # Shouldn't happen if history length check passed, but guard anyway
        # rather than silently scoring on NaNs.
        return {
            "date": today["date"],
            "daily_cost": today["daily_cost"],
            "is_anomaly": None,
            "anomaly_score": None,
            "service_breakdown": today.get("service_breakdown"),
            "cost_usd": today.get("cost_usd"),
            "fx_rate": today.get("fx_rate"),
            "status": "insufficient_history",
            "note": "Rolling features still NaN despite history length check; skipping.",
        }
    model = _load_model()
    X = today_row[FEATURE_COLUMNS].values

    raw_pred = model.predict(X)[0]  # -1 anomaly, 1 normal
    is_anomaly = bool(raw_pred == -1)

    raw_score = model.decision_function(X)[0]  # higher = more normal
    anomaly_score = float(-raw_score)  # flipped: higher = more anomalous

    return {
        "date": today["date"],
        "daily_cost": today["daily_cost"],
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "service_breakdown": today.get("service_breakdown"),
        "cost_usd": today.get("cost_usd"),
        "fx_rate": today.get("fx_rate"),
        "status": "scored",
    }

def lambda_handler(event: dict, context=None) -> dict:
    """
    AWS Lambda entrypoint shape. Expects:
        event = {"history": [...], "today": {...}}
    Returns the JSON-serializable output dict.
    """
    history = event.get("history", [])
    today = event["today"]
    result = score_day(history, today)
    return {"statusCode": 200, "body": json.dumps(result, default=str)}


if _name_ == "_main_":
    from datetime import date
    from synthetic_spend import generate_synthetic_spend, strip_labels

    df, _ = generate_synthetic_spend(
        start_date=date(2018, 1, 1), num_days=3000, anomaly_rate=0.06,
        fx_volatility=0.08, seed=42,
    )
    prod_shaped = strip_labels(df) 
    records = prod_shaped.to_dict("records")

    print("--- Fresh deployment: day 3 (insufficient history) ---")
    result = score_day(records[:3], records[3])
    print(json.dumps(result, indent=2, default=str))

    print("\n--- Fresh deployment: day 8 (exactly enough history) ---")
    result = score_day(records[:8], records[8])
    print(json.dumps(result, indent=2, default=str))

    print("\n--- Normal steady-state scoring (deep into history) ---")
    idx = 500
    result = score_day(records[:idx], records[idx])
    print(json.dumps(result, indent=2, default=str))

    # Compare against ground truth for that record (generator's own label,
    print(f"\nGround truth for that day: is_anomaly={df.iloc[idx]['is_anomaly']}, "
          f"anomaly_type={df.iloc[idx]['anomaly_type']}")