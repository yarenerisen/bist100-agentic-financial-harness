"""Support / resistance detection: pivot clustering, touch counts, recency, volume and reversal strength."""
import numpy as np
import pandas as pd
from .core import get_ohlcv, _clamp


def _pivots(df, w):
    """Pivot highs/lows confirmed by `w` bars on each side. A pivot at i is only known at i+w (no look-ahead
    beyond the analysis date: pivots whose confirmation bar is after the last bar are ignored)."""
    lo, hi = df.low.values, df.high.values
    out = []
    for i in range(w, len(df) - w):
        if lo[i] == lo[i - w:i + w + 1].min():
            out.append((df.index[i], float(lo[i]), "low"))
        if hi[i] == hi[i - w:i + w + 1].max():
            out.append((df.index[i], float(hi[i]), "high"))
    return out


def _cluster(pivots, tol):
    pivots = sorted(pivots, key=lambda p: p[1])
    clusters, cur = [], []
    for p in pivots:
        if cur and p[1] > np.mean([x[1] for x in cur]) * (1 + tol):
            clusters.append(cur); cur = []
        cur.append(p)
    if cur:
        clusters.append(cur)
    return clusters


def find_support_resistance(ticker, start="2024-01-01", end="2026-09-10", tolerance_pct=1.5, pivot_window=5,
                            min_touches=2, horizon=5, max_levels=3) -> dict:
    end = _clamp(end)
    df = get_ohlcv(ticker, start, end)
    if len(df) < 4 * pivot_window:
        raise ValueError("not enough history for pivot detection")
    tol = tolerance_pct / 100
    vol_ratio = df.volume / df.volume.rolling(20, min_periods=5).mean()
    last = float(df.close.iloc[-1])
    fwd = df.close.shift(-horizon) / df.close - 1
    levels = []
    for c in _cluster(_pivots(df, pivot_window), tol):
        if len(c) < min_touches:
            continue
        lvl = float(np.mean([x[1] for x in c]))
        kind = "support" if lvl < last else "resistance"
        lo_b, hi_b = lvl * (1 - tol), lvl * (1 + tol)
        # touch events: first day price enters the zone from the expected side
        if kind == "support":
            inz = (df.low <= hi_b) & (df.low >= lo_b * (1 - tol))
        else:
            inz = (df.high >= lo_b) & (df.high <= hi_b * (1 + tol))
        first = inz & ~inz.shift(1, fill_value=False)
        ev = fwd[first]
        done = ev.dropna()
        succ = (done > 0) if kind == "support" else (done < 0)
        last_touch = max(x[0] for x in c)
        levels.append({
            "type": kind, "level": round(lvl, 4), "zone": [round(min(x[1] for x in c), 4), round(max(x[1] for x in c), 4)],
            "distance_from_close_pct": round((lvl / last - 1) * 100, 2), "pivot_touches": len(c),
            "last_pivot_date": str(last_touch.date()),
            "days_since_last_pivot": int((df.index[-1] - last_touch).days),
            "zone_touch_events": int(first.sum()), "events_with_complete_window": int(len(done)),
            "reversal_rate_pct": round(float(succ.mean()) * 100, 1) if len(done) else None,
            "mean_forward_return_pct": round(float(done.mean()) * 100, 2) if len(done) else None,
            "avg_volume_ratio_at_pivots": round(float(np.nanmean([vol_ratio.get(x[0], np.nan) for x in c])), 2),
        })
    for L in levels:   # strength: more touches, more recent, stronger reversals
        rec = 1 / (1 + L["days_since_last_pivot"] / 90)
        L["strength_score"] = round(L["pivot_touches"] * rec * (0.5 + (L["reversal_rate_pct"] or 50) / 100), 2)
    sup = sorted([l for l in levels if l["type"] == "support"], key=lambda l: -l["level"])[:max_levels]
    res = sorted([l for l in levels if l["type"] == "resistance"], key=lambda l: l["level"])[:max_levels]
    out = {"tool": "find_support_resistance", "ticker": ticker.upper(), "period": [str(df.index[0].date()), str(df.index[-1].date())],
           "last_close": round(last, 4), "tolerance_pct": tolerance_pct, "pivot_window": pivot_window,
           "nearest_supports": sup, "nearest_resistances": res,
           "method": f"pivot highs/lows (±{pivot_window} bars) clustered within {tolerance_pct}%; reversal = price {horizon}d after a zone touch moves away from the level",
           "warnings": []}
    if not sup:
        out["warnings"].append("no support level with enough touches below the last close")
    if not res:
        out["warnings"].append("no resistance level with enough touches above the last close (price near period high)")
    if any((l["events_with_complete_window"] or 0) < 10 for l in sup + res):
        out["warnings"].append("some levels have fewer than 10 touch events; reversal rates are fragile")
    return out
