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