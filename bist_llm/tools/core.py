"""Deterministic analysis tools (data access, sector, calendar patterns, indicators).
All functions return JSON-serialisable dicts. Every tool clamps dates to CUTOFF."""
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-09-10")
DATA = Path(__file__).resolve().parent.parent / "data" / "prices.parquet"
EXCLUDE: list[str] = []  # tickers removed from the study universe (fill in when announced)

# Simplified sector map (coursework). First 30 = official study universe; others added so
# questions about peers (e.g. GARAN, ISCTR, YKBNK) can be answered.
SECTORS = {
    "banking": ["AKBNK", "GARAN", "ISCTR", "YKBNK", "HALKB", "VAKBN", "TSKB", "SKBNK"],
    "retail": ["BIMAS", "MGROS", "MAVI", "SOKM"],
    "holding": ["KCHOL", "SAHOL", "DOHOL", "AGHOL", "ALARK"],
    "steel": ["EREGL", "KRDMD", "BRSAN"],
    "automotive": ["FROTO", "TOASO", "DOAS", "OTKAR", "TTRAK"],
    "food_beverage": ["CCOLA", "AEFES", "ULKER", "TUKAS", "TABGD"],
    "airlines": ["THYAO", "PGSUS"],
    "aviation_services": ["TAVHL", "CLEBI"],
    "telecom": ["TCELL", "TTKOM"],
    "utilities": ["ENJSA", "AKSEN", "ODAS", "ZOREN", "ENERY", "ALFAS"],
    "energy_equipment": ["CWENE", "EUPWR", "ASTOR", "SMRTG", "GESAN", "KONTR"],
    "chemicals": ["SASA", "AKSA", "HEKTS", "GUBRF"],
    "petroleum": ["TUPRS", "PETKM", "AYGAZ"],
    "cement": ["OYAKC", "CIMSA", "BTCIM", "AKCNS"],
    "glass": ["SISE"],
    "defense": ["ASELS"],
    "mining": ["TRALT"],
    "insurance": ["TURSG", "ANSGR", "AGESA"],
    "brokerage": ["ISMEN"],
    "healthcare": ["MPARK", "ECILC"],
    "reit": ["EKGYO"],
    "durables": ["ARCLK", "VESTL"],
    "construction": ["ENKAI", "TKFEN"],
    "technology": ["MIATK", "REEDR"],
}
SECTOR_ALIASES = {"bank": "banking", "banks": "banking", "perakende": "retail", "bankacilik": "banking",
                  "airline": "airlines", "havayolu": "airlines", "otomotiv": "automotive", "celik": "steel",
                  "demir_celik": "steel", "telekom": "telecom", "enerji": "utilities", "elektrik": "utilities",
                  "kimya": "chemicals", "rafineri": "petroleum", "cimento": "cement", "sigorta": "insurance",
                  "savunma": "defense", "saglik": "healthcare", "insaat": "construction", "gida": "food_beverage",
                  "food": "food_beverage", "holdings": "holding"}
SECTOR_OF = {t: s for s, ts in SECTORS.items() for t in ts}
SECTOR_OF["PETKM"] = "petroleum"


@lru_cache(maxsize=1)
def _prices() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[df.date <= CUTOFF]


def available_tickers():
    return sorted(_prices().ticker.unique())


def _clamp(d) -> pd.Timestamp:
    return min(pd.Timestamp(d), CUTOFF)


def get_ohlcv(ticker: str, start="2024-01-01", end="2026-09-10") -> pd.DataFrame:
    ticker = ticker.upper().replace(".IS", "")
    df = _prices()
    x = df[(df.ticker == ticker) & (df.date >= pd.Timestamp(start)) & (df.date <= _clamp(end))]
    if x.empty:
        raise ValueError(f"no data for {ticker}")
    return x.set_index("date")[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def _close_wide(tickers, start="2023-06-01", end="2026-09-10") -> pd.DataFrame:
    df = _prices()
    x = df[df.ticker.isin(tickers) & (df.date >= pd.Timestamp(start)) & (df.date <= _clamp(end))]
    missing = set(tickers) - set(x.ticker.unique())
    if missing:
        raise ValueError(f"no data for {sorted(missing)}")
    return x.pivot(index="date", columns="ticker", values="close").dropna(how="any")


def _resolve(sector_or_tickers):
    if isinstance(sector_or_tickers, str):
        s = sector_or_tickers.strip()
        if "," in s or " " in s:
            t = [x for x in s.replace(",", " ").split()]
        else:
            key = SECTOR_ALIASES.get(s.lower(), s.lower())
            if key in SECTORS:
                t = SECTORS[key]
            elif s.upper().replace(".IS", "") in available_tickers():
                t = [s]
            else:
                raise KeyError(s)
    else:
        t = list(sector_or_tickers)
    return [x.upper().replace(".IS", "") for x in t if x.upper() not in EXCLUDE]


def _label(sector):
    return sector if isinstance(sector, str) else ",".join(sector)


# ------------------------------------------------------------------ basic history
def get_stock_history(ticker, start="2024-01-01", end="2026-09-10") -> dict:
    df = get_ohlcv(ticker, start, end)
    c = df.close
    dd = (c / c.cummax() - 1).min()
    r = c.pct_change().dropna()
    return {"tool": "get_stock_history", "ticker": ticker.upper(), "sector": SECTOR_OF.get(ticker.upper()),
            "period": [str(df.index[0].date()), str(df.index[-1].date())], "trading_days": len(df),
            "first_close": round(float(c.iloc[0]), 4), "last_close": round(float(c.iloc[-1]), 4),
            "total_return_pct": round(float(c.iloc[-1] / c.iloc[0] - 1) * 100, 2),
            "max_close": round(float(c.max()), 4), "max_close_date": str(c.idxmax().date()),
            "min_close": round(float(c.min()), 4), "min_close_date": str(c.idxmin().date()),
            "annualised_volatility_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
            "max_drawdown_pct": round(float(dd) * 100, 2),
            "avg_daily_volume": round(float(df.volume.mean()), 0),
            "zero_volume_days": int((df.volume == 0).sum()),
            "note": "prices are split/dividend adjusted (Yahoo Finance)"}


# ------------------------------------------------------------------ Scenario 1
def _tech_state(ticker, end):
    ind = calculate_indicators(ticker, "2024-01-01", end)
    last = ind.iloc[-1]
    return {"supertrend": "bullish" if last.supertrend == 1 else "bearish",
            "close_above_kama": bool(last.close > last.kama),
            "kama_slope_pct_3d": None if pd.isna(last.kama_slope_pct) else round(float(last.kama_slope_pct), 3)}


def rank_sector_relative_strength(sector, horizon=20, as_of="2026-09-10") -> dict:
    tk = _resolve(sector)
    as_of = _clamp(as_of)
    w = _close_wide(tk, end=as_of)
    if len(w) <= horizon:
        raise ValueError("not enough common history for this horizon")
    r = w.iloc[-1] / w.iloc[-1 - horizon] - 1
    med = float(r.median())
    xu = get_ohlcv("XU100", "2023-06-01", as_of).close
    d0, d1 = w.index[-1 - horizon], w.index[-1]
    xr = float(xu.loc[d1] / xu.loc[d0] - 1)
    rows = []
    for i, t in enumerate(r.sort_values().index, 1):
        row = {"rank_from_weakest": i, "ticker": t, "return_pct": round(float(r[t]) * 100, 2),
               "lag_vs_median_pp": round((float(r[t]) - med) * 100, 2),
               "vs_xu100_pp": round((float(r[t]) - xr) * 100, 2)}
        row.update(_tech_state(t, as_of))
        rows.append(row)
    warn = []
    if len(tk) < 3:
        warn.append("peer group has fewer than 3 stocks; median is weak evidence")
    strong = med > 0 and med > xr
    return {"tool": "rank_sector_relative_strength", "sector": _label(sector), "peers": tk,
            "horizon_trading_days": horizon, "window_start": str(d0.date()), "window_end": str(d1.date()),
            "sector_median_return_pct": round(med * 100, 2), "xu100_return_pct": round(xr * 100, 2),
            "sector_strong": bool(strong),
            "sector_strong_definition": "sector median return > 0 and > XU100 return over the window",
            "weakest": rows[0]["ticker"], "ranking": rows, "warnings": warn}


def run_backtest_catch_up(sector, start="2024-01-01", end="2026-09-10", lag_window=20,
                          lag_threshold_pp=5.0, horizons=(5, 10, 20), cost_pct=0.2,
                          require_strong_sector=False, tickers_only=None) -> dict:
    """Event study: stock lags the peer median by >= threshold over lag_window days.
    Outcome: relative return vs peer median over the next h days. Non-overlapping events;
    forward windows past the cutoff are excluded."""
    tk = _resolve(sector)
    end = _clamp(end)
    w = _close_wide(tk, start="2023-06-01", end=end)
    ret_l = w / w.shift(lag_window) - 1
    med_l = ret_l.median(axis=1)
    lag = ret_l.sub(med_l, axis=0) * 100
    strong = med_l > 0
    idx = w.index
    out = {}
    tot = {h: [] for h in horizons}
    for t in (tickers_only or tk):
        res = {}
        for h in horizons:
            fwd = w[t].shift(-h) / w[t] - 1
            fwd_med = (w.shift(-h) / w - 1).median(axis=1)
            rel = (fwd - fwd_med) * 100
            sig = (lag[t] <= -lag_threshold_pp) & (idx >= pd.Timestamp(start))
            if require_strong_sector:
                sig &= strong
            ev, last = [], -10**9
            for i in np.where(sig.values)[0]:
                if i - last >= h:
                    ev.append(i); last = i
            n_all = len(ev)
            ev = [i for i in ev if not np.isnan(rel.iloc[i])]
            r = np.array([rel.iloc[i] for i in ev]); gap = np.array([-lag[t].iloc[i] for i in ev])
            tot[h].extend(r.tolist())
            res[f"{h}d"] = {
                "events": len(ev), "excluded_incomplete_window": n_all - len(ev),
                "hit_rate_pct": round(float((r > 0).mean() * 100), 1) if len(r) else None,
                "gap_closed_rate_pct": round(float((r >= gap).mean() * 100), 1) if len(r) else None,
                "mean_rel_return_pp": round(float(r.mean()), 2) if len(r) else None,
                "median_rel_return_pp": round(float(np.median(r)), 2) if len(r) else None,
                "mean_after_cost_pp": round(float(r.mean() - cost_pct), 2) if len(r) else None,
                "false_positives": int((r <= 0).sum()),
            }
        out[t] = {"current_lag_vs_median_pp": round(float(lag[t].iloc[-1]), 2), "by_horizon": res}
    pooled = {f"{h}d": {"events": len(v), "hit_rate_pct": round(float(np.mean(np.array(v) > 0) * 100), 1) if v else None,
                        "mean_rel_return_pp": round(float(np.mean(v)), 2) if v else None,
                        "false_positive_rate_pct": round(float(np.mean(np.array(v) <= 0) * 100), 1) if v else None}
              for h, v in tot.items()}
    warn = []
    if any(v["by_horizon"][f"{horizons[-1]}d"]["events"] < 10 for v in out.values()):
        warn.append("fewer than 10 non-overlapping events for some stocks; low statistical power")
    if len(tk) < 3:
        warn.append("peer group has fewer than 3 stocks")
    return {"tool": "run_backtest", "strategy": "sector_catch_up", "sector": _label(sector),
            "period": [str(pd.Timestamp(start).date()), str(end.date())], "lag_window": lag_window,
            "lag_threshold_pp": lag_threshold_pp, "require_strong_sector": require_strong_sector,
            "cost_pct_round_trip": cost_pct, "results": out, "pooled_all_stocks": pooled,
            "notes": ["hit = stock beat peer median over next h days; gap_closed = it recovered the full lag",
                      "false positive = lagging stock that kept underperforming",
                      "events non-overlapping; incomplete forward windows excluded"], "warnings": warn}


# ------------------------------------------------------------------ Scenario 2
def _binom_sf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _daily_ret(ticker, start, end):
    px = get_ohlcv(ticker, "2023-06-01", end)["close"]
    ret = px.pct_change().dropna()
    return ret[(ret.index >= pd.Timestamp(start)) & (ret.index <= end)]


def test_weekday_pattern(ticker, pattern="DDDUU", start="2024-01-01", end="2025-12-31", cost_pct=0.2) -> dict:
    """pattern: 5 chars Mon..Fri, U=up D=down (X = any). Evaluated on complete 5-day weeks."""
    pattern = pattern.upper()
    if len(pattern) != 5 or not set(pattern) <= {"U", "D", "X"}:
        raise ValueError("pattern must be 5 letters from U/D/X for Mon..Fri, e.g. DDDUU")
    end = _clamp(end)
    ret = _daily_ret(ticker, start, end)
    iso = ret.index.isocalendar()
    d = pd.DataFrame({"r": ret.values, "wk": iso.year.astype(str).values + "-" + iso.week.astype(str).values,
                      "dow": ret.index.dayofweek}, index=ret.index)
    weeks = d.groupby("wk").filter(lambda g: sorted(g.dow) == [0, 1, 2, 3, 4])
    m = weeks.pivot(index="wk", columns="dow", values="r")
    if m.empty:
        raise ValueError("no complete trading weeks in period")
    cols = [i for i, c in enumerate(pattern) if c != "X"]
    want = {i: pattern[i] == "U" for i in cols}
    match = np.all([(m[i].values > 0) == want[i] for i in cols], axis=0)
    n, k = len(m), int(match.sum())
    p_exp = float(np.prod([(m[i] > 0).mean() if want[i] else (m[i] <= 0).mean() for i in cols]))
    pval = _binom_sf(k, n, p_exp)
    tf_all = (1 + m[3]) * (1 + m[4]) - 1
    tf_cond = tf_all[match]
    pre_cols = [i for i in cols if i <= 2]
    pre = np.all([(m[i].values > 0) == want[i] for i in pre_cols], axis=0) if pre_cols else np.ones(n, bool)
    tf_pre = tf_all[pre]
    per_day = {}
    for i, dn in enumerate(DOW):
        x = d[d.dow == i].r
        t = float(x.mean() / (x.std() / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else None
        per_day[dn] = {"n": len(x), "mean_pct": round(float(x.mean()) * 100, 3), "median_pct": round(float(x.median()) * 100, 3),
                       "up_days_pct": round(float((x > 0).mean()) * 100, 1), "t_stat": None if t is None else round(t, 2)}
    res = {
        "tool": "test_weekday_pattern", "ticker": ticker.upper(), "pattern_Mon_to_Fri": pattern,
        "period": [str(pd.Timestamp(start).date()), str(end.date())], "complete_weeks": n,
        "pattern_occurrences": k, "pattern_frequency_pct": round(k / n * 100, 1),
        "expected_frequency_if_independent_pct": round(p_exp * 100, 1),
        "binomial_p_value_one_sided": round(pval, 4),
        "weekday_stats": per_day,
        "unconditional_thu_fri_mean_pct": round(float(tf_all.mean()) * 100, 3),
        "tradable_signal_weeks": int(pre.sum()),
        "tradable_thu_fri_mean_pct": round(float(tf_pre.mean()) * 100, 3) if pre.sum() else None,
        "tradable_thu_fri_median_pct": round(float(tf_pre.median()) * 100, 3) if pre.sum() else None,
        "tradable_thu_fri_hit_rate_pct": round(float((tf_pre > 0).mean()) * 100, 1) if pre.sum() else None,
        "tradable_thu_fri_mean_after_cost_pct": round(float(tf_pre.mean()) * 100 - cost_pct, 3) if pre.sum() else None,
        "pattern_weeks_thu_fri_mean_pct": round(float(tf_cond.mean()) * 100, 3) if k else None,
        "notes": ["tradable_* = weeks where the Mon-Wed part of the pattern happened (known at Wednesday close); outcome = Thu+Fri return",
                  "pattern_weeks_* conditions on the Thu-Fri outcome itself: descriptive only, not tradable",
                  f"cost {cost_pct}% round trip"],
        "multiple_testing_note": "2^5=32 possible Mon-Fri up/down patterns; Bonferroni threshold p<0.0016 for alpha 0.05",
        "warnings": [],
    }
    if k < 10:
        res["warnings"].append(f"only {k} full-pattern occurrences; statistically fragile")
    if pval > 0.05 / 32:
        res["warnings"].append("full pattern not significant after multiple-testing correction")
    return res


def detect_multiday_patterns(ticker, pattern_type="consecutive_down", n_days=3, threshold_pct=-3.0,
                             horizon=2, start="2024-01-01", end="2026-09-10", cost_pct=0.2) -> dict:
    """pattern_type: consecutive_down | consecutive_up | large_down_day | large_up_day.
    Signal known at close t; outcome = return from close t to close t+horizon."""
    end = _clamp(end)
    px = get_ohlcv(ticker, "2023-06-01", end).close
    r = px.pct_change()
    if pattern_type == "consecutive_down":
        sig = (r < 0).rolling(n_days).sum() == n_days
    elif pattern_type == "consecutive_up":
        sig = (r > 0).rolling(n_days).sum() == n_days
    elif pattern_type == "large_down_day":
        sig = r * 100 <= threshold_pct
    elif pattern_type == "large_up_day":
        sig = r * 100 >= abs(threshold_pct)
    else:
        raise ValueError("pattern_type must be consecutive_down, consecutive_up, large_down_day or large_up_day")
    fwd = px.shift(-horizon) / px - 1
    inwin = (px.index >= pd.Timestamp(start))
    ev = sig & inwin
    ev_idx = px.index[ev.fillna(False).values]
    f = fwd[ev_idx]
    done = f.dropna()
    base = fwd[inwin].dropna()
    out = {"tool": "detect_multiday_patterns", "ticker": ticker.upper(), "pattern_type": pattern_type,
           "n_days": n_days if "consecutive" in pattern_type else None,
           "threshold_pct": threshold_pct if "large" in pattern_type else None, "horizon_days": horizon,
           "period": [str(pd.Timestamp(start).date()), str(end.date())],
           "events": int(len(f)), "excluded_incomplete_window": int(len(f) - len(done)),
           "mean_forward_return_pct": round(float(done.mean()) * 100, 3) if len(done) else None,
           "median_forward_return_pct": round(float(done.median()) * 100, 3) if len(done) else None,
           "up_rate_pct": round(float((done > 0).mean()) * 100, 1) if len(done) else None,
           "unconditional_mean_forward_return_pct": round(float(base.mean()) * 100, 3),
           "unconditional_up_rate_pct": round(float((base > 0).mean()) * 100, 1),
           "mean_after_cost_pct": round(float(done.mean()) * 100 - cost_pct, 3) if len(done) else None,
           "last_events": [str(x.date()) for x in ev_idx[-5:]],
           "warnings": []}
    if len(done) < 10:
        out["warnings"].append("fewer than 10 events; low statistical power")
    out["warnings"].append("events can overlap for consecutive-day patterns; treat as descriptive")
    return out


# ------------------------------------------------------------------ Indicators
def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _atr(df, n=10):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _kama(c, n=10, fast=2, slow=30):
    ch = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    er = (ch / vol).fillna(0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(c), np.nan)
    v = c.values
    if len(c) <= n:
        return pd.Series(out, index=c.index)
    out[n] = v[n]
    for i in range(n + 1, len(c)):
        out[i] = out[i - 1] + sc.iloc[i] * (v[i] - out[i - 1])
    return pd.Series(out, index=c.index)


def _supertrend(df, n=10, mult=3.0):
    atr = _atr(df, n)
    hl2 = (df.high + df.low) / 2
    ub, lb = (hl2 + mult * atr).values, (hl2 - mult * atr).values
    c = df.close.values
    fu, fl = ub.copy(), lb.copy()
    trend = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < fl[i] else 1
        else:
            trend[i] = 1 if c[i] > fu[i] else -1
    return pd.Series(trend, index=df.index)  # 1 bullish, -1 bearish


@lru_cache(maxsize=256)
def _indicators_cached(ticker, end_str):
    df = get_ohlcv(ticker, "2023-06-01", end_str)
    c = df.close
    o = pd.DataFrame(index=df.index)
    o["open"], o["close"], o["low"], o["high"], o["volume"] = df.open, c, df.low, df.high, df.volume
    o["sma20"], o["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    o["ema20"] = c.ewm(span=20, adjust=False).mean()
    sd = c.rolling(20).std(ddof=0)
    o["bb_mid"], o["bb_up"], o["bb_lo"] = o.sma20, o.sma20 + 2 * sd, o.sma20 - 2 * sd
    o["rsi14"] = _rsi(c)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    o["macd"] = e12 - e26
    o["macd_signal"] = o.macd.ewm(span=9, adjust=False).mean()
    o["atr10"] = _atr(df)
    o["atr_pct"] = o.atr10 / c * 100
    o["kama"] = _kama(c)
    o["kama_slope_pct"] = o.kama.pct_change(3) * 100
    o["supertrend"] = _supertrend(df)
    tenkan = (df.high.rolling(9).max() + df.low.rolling(9).min()) / 2
    kijun = (df.high.rolling(26).max() + df.low.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26)
    spb = ((df.high.rolling(52).max() + df.low.rolling(52).min()) / 2).shift(26)
    o["ichi_tenkan"], o["ichi_kijun"] = tenkan, kijun
    o["cloud_top"] = pd.concat([spa, spb], axis=1).max(axis=1)
    o["cloud_bot"] = pd.concat([spa, spb], axis=1).min(axis=1)
    o["vol_ratio"] = df.volume / df.volume.rolling(20).mean()
    return o


def calculate_indicators(ticker, start="2024-01-01", end="2026-09-10") -> pd.DataFrame:
    """Indicator DataFrame computed with warm-up history from 2023-06, then sliced. Causal only."""
    end = _clamp(end)
    o = _indicators_cached(ticker.upper(), str(end.date()))
    return o[o.index >= pd.Timestamp(start)]
