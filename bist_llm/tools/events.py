"""Technical reversal / state-change event detection with forward-return statistics."""
import pandas as pd
from .core import calculate_indicators, get_ohlcv, _clamp

# event -> (expected direction: +1 up / -1 down, description)
EVENTS = {
    "lower_bollinger_touch": (+1, "first day of a run where low <= lower Bollinger(20,2); success = price higher after h days"),
    "upper_bollinger_touch": (-1, "first day of a run where high >= upper Bollinger(20,2); success = price lower after h days"),
    "middle_bollinger_test": (+1, "close was above middle band, low touches middle band, close stays above; success = continuation up"),
    "rsi_trough": (+1, "RSI(14) crosses back above 30 from below; success = price higher"),
    "rsi_peak": (-1, "RSI(14) crosses back below 70 from above; success = price lower"),
    "supertrend_bullish_flip": (+1, "Supertrend(10,3) turns bullish; success = price higher"),
    "supertrend_bearish_flip": (-1, "Supertrend(10,3) turns bearish; success = price lower"),
    "kama_cross_up": (+1, "close crosses above KAMA(10,2,30); success = price higher"),
    "kama_cross_down": (-1, "close crosses below KAMA(10,2,30); success = price lower"),
    "ichimoku_cloud_breakout": (+1, "close crosses above the Ichimoku cloud top; success = price higher"),
    "ichimoku_cloud_breakdown": (-1, "close crosses below the Ichimoku cloud bottom; success = price lower"),
}


def _event_mask(ind, event):
    p = lambda s: s.shift(1)
    if event == "lower_bollinger_touch":
        t = ind.low <= ind.bb_lo; return t & ~p(t).fillna(False).astype(bool)
    if event == "upper_bollinger_touch":
        t = ind.high >= ind.bb_up; return t & ~p(t).fillna(False).astype(bool)
    if event == "middle_bollinger_test":
        return (p(ind.close) > p(ind.bb_mid)) & (ind.low <= ind.bb_mid) & (ind.close > ind.bb_mid)
    if event == "rsi_trough":
        return (p(ind.rsi14) < 30) & (ind.rsi14 >= 30)
    if event == "rsi_peak":
        return (p(ind.rsi14) > 70) & (ind.rsi14 <= 70)
    if event == "supertrend_bullish_flip":
        return (p(ind.supertrend) == -1) & (ind.supertrend == 1)
    if event == "supertrend_bearish_flip":
        return (p(ind.supertrend) == 1) & (ind.supertrend == -1)
    if event == "kama_cross_up":
        return (p(ind.close) <= p(ind.kama)) & (ind.close > ind.kama)
    if event == "kama_cross_down":
        return (p(ind.close) >= p(ind.kama)) & (ind.close < ind.kama)
    if event == "ichimoku_cloud_breakout":
        return (p(ind.close) <= p(ind.cloud_top)) & (ind.close > ind.cloud_top)
    if event == "ichimoku_cloud_breakdown":
        return (p(ind.close) >= p(ind.cloud_bot)) & (ind.close < ind.cloud_bot)
    raise ValueError(f"unknown event; allowed: {sorted(EVENTS)}")


def detect_reversal_events(ticker, event="lower_bollinger_touch", start="2024-01-01", end="2026-09-10",
                           horizons=(1, 3, 5, 10), success_horizon=5, high_volume_ratio=1.5) -> dict:
    if event not in EVENTS:
        raise ValueError(f"unknown event; allowed: {sorted(EVENTS)}")
    direction, desc = EVENTS[event]
    end = _clamp(end)
    ind = calculate_indicators(ticker, "2023-06-01", end).copy()
    xu = get_ohlcv("XU100", "2023-06-01", end).close
    ind["xu_up"] = (xu / xu.shift(20) - 1).reindex(ind.index) > 0
    ind["hi_vola"] = ind.atr_pct > ind.atr_pct.rolling(120, min_periods=40).median()
    mask = _event_mask(ind, event).fillna(False).astype(bool) & (ind.index >= pd.Timestamp(start))
    ev = ind[mask].copy()
    ev["rsi_trough_state"] = ((ind.rsi14 <= ind.rsi14.rolling(10).min()) & (ind.rsi14 < 40))[ev.index]  # causal
    ev["high_volume"] = (ind.vol_ratio >= high_volume_ratio)[ev.index]
    ev["st_bear"] = (ind.supertrend == -1)[ev.index]
    ev["below_cloud"] = (ind.close < ind.cloud_bot)[ev.index]
    ev["flat_kama"] = (ind.kama_slope_pct.abs() < 0.3)[ev.index]
    for h in horizons:
        ev[f"f{h}"] = (ind.close.shift(-h) / ind.close - 1)[ev.index] * direction  # signed: >0 = success
        ev[f"raw{h}"] = (ind.close.shift(-h) / ind.close - 1)[ev.index]

    def stats(sub):
        done = sub.dropna(subset=[f"f{success_horizon}"])
        s = {"events": len(sub), "events_with_complete_window": len(done),
             "excluded_incomplete_window": len(sub) - len(done)}
        if len(done):
            s["success_count"] = int((done[f"f{success_horizon}"] > 0).sum())
            s["failure_count"] = int((done[f"f{success_horizon}"] <= 0).sum())
            s["success_rate_pct"] = round(s["success_count"] / len(done) * 100, 1)
        for h in horizons:
            x = sub[f"raw{h}"].dropna()
            s[f"mean_return_{h}d_pct"] = round(float(x.mean()) * 100, 2) if len(x) else None
            s[f"median_return_{h}d_pct"] = round(float(x.median()) * 100, 2) if len(x) else None
        if len(done) < 10:
            s["warning"] = "fewer than 10 events; low statistical power"
        return s

    groups = {"all_events": stats(ev)}
    conds = {
        "with_rsi_trough_and_high_volume": ev.rsi_trough_state & ev.high_volume,
        "with_bearish_supertrend_and_below_cloud": ev.st_bear & ev.below_cloud,
        "with_rsi_trough_high_volume_flat_kama": ev.rsi_trough_state & ev.high_volume & ev.flat_kama,
        "with_high_volume": ev.high_volume,
        "regime_xu100_rising": ev.xu_up.astype(bool),
        "regime_xu100_falling": ~ev.xu_up.astype(bool),
        "regime_high_volatility": ev.hi_vola.astype(bool),
        "regime_low_volatility": ~ev.hi_vola.astype(bool),
    }
    for k, m in conds.items():
        groups[k] = stats(ev[m])
    excluded_dates = [str(d.date()) for d in ev.index[ev[f"f{success_horizon}"].isna()]]
    warnings = []
    zero_vol = float((ind.volume[ind.index >= pd.Timestamp(start)] == 0).mean())
    if zero_vol > 0.05:
        warnings.append(f"{zero_vol*100:.0f}% of days have zero volume in source data; volume-based conditions unreliable")
    if excluded_dates:
        warnings.append(f"{len(excluded_dates)} recent event(s) excluded because the forward window is incomplete as of {end.date()}: {excluded_dates}")
    # decision inputs computed here (trading-day arithmetic is not left to the language model)
    last_ev = ev.index[-1] if len(ev) else None
    days_since = int((ind.index > last_ev).sum()) if last_ev is not None else None
    a = groups["all_events"]
    summary = {"last_event_date": str(last_ev.date()) if last_ev is not None else None,
               "trading_days_since_last_event": days_since,
               "event_in_last_10_trading_days": bool(days_since is not None and days_since <= 10),
               "events_with_complete_window": a["events_with_complete_window"],
               "at_least_10_complete_events": a["events_with_complete_window"] >= 10,
               "success_rate_pct": a.get("success_rate_pct"),
               "success_rate_at_least_60": bool((a.get("success_rate_pct") or 0) >= 60),
               "success_rate_below_50": bool(a.get("success_rate_pct") is not None and a["success_rate_pct"] < 50)}
    return {"tool": "detect_reversal_events", "ticker": ticker.upper(), "event": event, "event_definition": desc,
            "summary": summary,
            "expected_direction": "up" if direction > 0 else "down",
            "period": [str(pd.Timestamp(start).date()), str(end.date())],
            "success_definition": f"price moves in expected direction over {success_horizon} trading days (close to close)",
            "groups": groups, "last_event_dates": [str(d.date()) for d in ev.index[-5:]],
            "notes": ["all conditions are causal (known at event close)",
                      "mean/median returns are raw price returns (not direction-signed)",
                      "regime: XU100 20-day return sign; volatility: ATR% vs its 120-day median"],
            "warnings": warnings}
