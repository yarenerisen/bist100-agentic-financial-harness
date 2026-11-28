"""Single entry point for all MCP-style tools: validates arguments, enforces the 2026-09-10 cutoff,
and returns JSON-serialisable dicts. Errors are returned as {"error": CODE, "message": ...}, never raised."""
import json, re
import pandas as pd
from . import core, macro_fund, events, backtest, quality, levels, context
from .core import CUTOFF

DATE = {"type": "string", "description": "YYYY-MM-DD, must be <= 2026-09-10"}
TICKER = {"type": "string", "description": "BIST ticker without .IS, e.g. THYAO"}
SECTOR = {"type": ["string", "array"], "description": "sector name (" + ", ".join(core.SECTORS) + ") or list of tickers"}
TICKERS = {"type": ["string", "array"], "description": "one ticker, list of tickers, or a sector name"}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


def pattern_by_day(letters):
    """'DDDUU' -> 'Mon=D,Tue=D,Wed=D,Thu=U,Fri=U' (the day-by-day form the model writes)."""
    return ",".join(f"{d}={c}" for d, c in zip(DAYS, letters.upper()))


def normalize_pattern(p):
    """Accepts 'DDDUU' or the day-by-day form 'Mon=D,Tue=D,Wed=D,Thu=U,Fri=U'; returns the 5 letters (or p unchanged)."""
    if not isinstance(p, str):
        return p
    m = re.fullmatch(r"\s*" + r"\s*[,;]\s*".join(fr"{d}\s*[=:]\s*([UDXudx])" for d in DAYS) + r"\s*", p)
    return "".join(m.groups()).upper() if m else p.strip().upper()


def _ind_summary(ticker, start="2024-01-01", end="2026-09-10"):
    ind = core.calculate_indicators(ticker, start, end)
    last = ind.iloc[-1]
    f = lambda v: None if pd.isna(v) else round(float(v), 4)
    lo = ind.low <= ind.bb_lo
    up = ind.high >= ind.bb_up
    return {"tool": "calculate_indicators", "ticker": ticker.upper(), "period": [str(ind.index[0].date()), str(ind.index[-1].date())],
            "as_of": str(ind.index[-1].date()), "close": f(last.close), "sma20": f(last.sma20), "sma50": f(last.sma50),
            "ema20": f(last.ema20), "rsi14": f(last.rsi14), "macd": f(last.macd), "macd_signal": f(last.macd_signal),
            "bb_lower": f(last.bb_lo), "bb_mid": f(last.bb_mid), "bb_upper": f(last.bb_up), "atr10": f(last.atr10),
            "kama": f(last.kama), "kama_slope_pct_3d": f(last.kama_slope_pct),
            "supertrend": "bullish" if last.supertrend == 1 else "bearish",
            "ichimoku_position": "above_cloud" if last.close > last.cloud_top else ("below_cloud" if last.close < last.cloud_bot else "inside_cloud"),
            "ichimoku_tenkan": f(last.ichi_tenkan), "ichimoku_kijun": f(last.ichi_kijun),
            "volume_ratio_20d": f(last.vol_ratio),
            "period_counts": {"lower_band_touch_runs": int((lo & ~lo.shift(1, fill_value=False)).sum()),
                              "upper_band_touch_runs": int((up & ~up.shift(1, fill_value=False)).sum()),
                              "days_rsi_below_30": int((ind.rsi14 < 30).sum()), "days_rsi_above_70": int((ind.rsi14 > 70).sum()),
                              "supertrend_flips": int((ind.supertrend.diff().abs() > 0).sum()),
                              "days_below_cloud": int((ind.close < ind.cloud_bot).sum())},
            "parameters": "SMA20/50, EMA20, RSI14, MACD(12,26,9), Bollinger(20,2), ATR10, KAMA(10,2,30), Supertrend(10,3), Ichimoku(9,26,52)"}


def _backtest(strategy, sector=None, tickers=None, start="2024-01-01", end="2026-09-10", lag_window=20, lag_threshold_pp=5.0,
              horizons=(5, 10, 20), require_strong_sector=False, hold_days=None, cost_pct=0.2):
    if strategy == "sector_catch_up":
        if sector is None:
            raise ValueError("sector_catch_up needs 'sector' (sector name or ticker list)")
        es = core.run_backtest_catch_up(sector, start, end, lag_window, lag_threshold_pp, tuple(horizons), cost_pct, require_strong_sector)
        pb = backtest.run_strategy_backtest("sector_catch_up", sector, start, end, hold_days or 10, cost_pct, sector=sector,
                                            lag_window=lag_window, lag_threshold_pp=lag_threshold_pp, require_strong_sector=require_strong_sector)
        es["portfolio_backtest"] = {k: pb[k] for k in ("execution", "trades", "win_rate_pct", "mean_trade_return_net_pct",
                                                       "performance", "excess_vs_buy_and_hold_pp", "excess_vs_xu100_pp", "warnings")}
        return es
    if tickers is None:
        raise ValueError("this strategy needs 'tickers'")
    return backtest.run_strategy_backtest(strategy, tickers, start, end, hold_days or 5, cost_pct)


STRATEGIES = ["sector_catch_up", "weekday_mon_wed_down"] + sorted(events.EVENTS)

def _index_history(index="XU100", start="2024-01-01", end="2026-09-10"):
    if index.upper() not in ("XU100", "USDTRY", "EURTRY"):
        raise ValueError("index must be XU100 (USDTRY/EURTRY also accepted)")
    r = core.get_stock_history(index.upper(), start, end)
    r["tool"] = "get_index_history"; r["index"] = r.pop("ticker"); r.pop("sector", None); r.pop("zero_volume_days", None)
    return r


TOOLS = {
    "get_stock_history": (core.get_stock_history, {"ticker": TICKER, "start": DATE, "end": DATE}, ["ticker"]),
    "get_index_history": (_index_history, {"index": {"type": "string", "enum": ["XU100", "USDTRY", "EURTRY"]}, "start": DATE, "end": DATE}, []),
    "find_support_resistance": (levels.find_support_resistance, {"ticker": TICKER, "start": DATE, "end": DATE,
                                                                 "tolerance_pct": {"type": "number", "default": 1.5},
                                                                 "pivot_window": {"type": "integer", "default": 5}}, ["ticker"]),
    "preview_strategy_variants": (backtest.preview_strategy_variants, {"tickers": TICKERS, "start": DATE, "end": DATE, "oos_start": DATE,
                                                                       "hold_days": {"type": "integer", "default": 5}}, ["tickers"]),
    "get_news_events": (context.get_news_events, {"ticker": TICKER, "topic": {"type": "string", "enum": ["earnings", "rates"]},
                                                  "start": DATE, "end": DATE}, []),
    "get_speech_features": (context.get_speech_features, {"source_id": {"type": "string"}, "topic": {"type": "string"}, "ticker": TICKER}, []),
    "rank_sector_relative_strength": (core.rank_sector_relative_strength, {"sector": SECTOR, "horizon": {"type": "integer", "default": 20}, "as_of": DATE}, ["sector"]),
    "run_backtest": (_backtest, {"strategy": {"type": "string", "enum": STRATEGIES}, "sector": SECTOR, "tickers": TICKERS,
                                 "start": DATE, "end": DATE, "lag_window": {"type": "integer", "default": 20},
                                 "lag_threshold_pp": {"type": "number", "default": 5.0},
                                 "horizons": {"type": "array", "items": {"type": "integer"}, "default": [5, 10, 20]},
                                 "require_strong_sector": {"type": "boolean", "default": False},
                                 "hold_days": {"type": "integer"}, "cost_pct": {"type": "number", "default": 0.2}}, ["strategy"]),
    "test_weekday_pattern": (core.test_weekday_pattern, {"ticker": TICKER, "pattern": {"type": "string", "description": "one letter per day Mon..Fri: U=up, D=down, X=any, written day by day, e.g. Mon=D,Tue=D,Wed=D,Thu=U,Fri=U (the short form DDDUU is also accepted)"},
                                                         "start": DATE, "end": DATE, "cost_pct": {"type": "number", "default": 0.2}}, ["ticker", "pattern", "start", "end"]),
    "detect_multiday_patterns": (core.detect_multiday_patterns, {"ticker": TICKER, "pattern_type": {"type": "string", "enum": ["consecutive_down", "consecutive_up", "large_down_day", "large_up_day"]},
                                                                 "n_days": {"type": "integer", "default": 3}, "threshold_pct": {"type": "number", "default": -3.0},
                                                                 "horizon": {"type": "integer", "default": 2}, "start": DATE, "end": DATE}, ["ticker", "pattern_type"]),
    "calculate_indicators": (_ind_summary, {"ticker": TICKER, "start": DATE, "end": DATE}, ["ticker"]),
    "detect_reversal_events": (events.detect_reversal_events, {"ticker": TICKER, "event": {"type": "string", "enum": sorted(events.EVENTS)},
                                                               "start": DATE, "end": DATE, "success_horizon": {"type": "integer", "default": 5}}, ["ticker", "event"]),
    "get_fundamental_metrics": (macro_fund.get_fundamental_metrics, {"ticker": TICKER, "start": DATE, "end": DATE}, ["ticker"]),
    "calculate_fundamental_change": (macro_fund.calculate_fundamental_change, {"ticker": TICKER, "start": DATE, "end": DATE}, ["ticker"]),
    "get_macro_context": (macro_fund.get_macro_context, {"start": DATE, "end": DATE, "series": {"type": "array", "items": {"type": "string", "enum": ["USDTRY", "EURTRY"]}}, "ticker": TICKER}, ["start", "end"]),
    "check_data_quality": (quality.check_data_quality, {"tickers": TICKERS, "start": DATE, "end": DATE,
                                                        "need_fundamentals": {"type": "boolean", "default": False},
                                                        "need_macro": {"type": "boolean", "default": False}}, ["tickers"]),
}
DESCRIPTIONS = {
    "get_index_history": "XU100 (or USDTRY/EURTRY) summary: return, high/low, volatility, max drawdown.",
    "find_support_resistance": "Support/resistance levels from clustered pivots: zone, touches, recency, reversal rate after zone touches.",
    "preview_strategy_variants": "Strategy variants A (technical) .. D (technical+sector+fundamental+macro), E (news, unavailable) with identical costs; in-sample vs out-of-sample.",
    "get_news_events": "Timestamped official events: KAP financial-report disclosures, TCMB/FOMC rate decisions (contextual only).",
    "get_speech_features": "Transcript-derived topics/claims/stance from pre-approved finance videos (if loaded).",
    "get_stock_history": "Price summary for one stock: first/last close, total return, high/low, volatility, max drawdown.",
    "rank_sector_relative_strength": "Ranks stocks in a sector by return over the last `horizon` trading days vs the sector median and XU100; flags whether the sector is strong; adds Supertrend/KAMA state.",
    "run_backtest": "Historical verification. sector_catch_up = event study of laggards (hit rate, false positives over 5/10/20d) + portfolio backtest; other strategies = trade backtest with Sharpe, max drawdown, win rate vs buy-and-hold and XU100.",
    "test_weekday_pattern": "Tests a Mon..Fri up/down pattern: occurrences, expected frequency, binomial p-value, weekday stats, tradable Thu-Fri result after costs. Call separately for in-sample and out-of-sample periods.",
    "detect_multiday_patterns": "Consecutive up/down days or large-move days and the forward return afterwards vs unconditional.",
    "calculate_indicators": "Latest technical indicator values and states plus counts over the period.",
    "detect_reversal_events": "Finds technical events (Bollinger touches, RSI troughs/peaks, Supertrend flips, KAMA crosses, Ichimoku breakouts) and forward 1/3/5/10d returns, success rates, combined-condition and regime comparisons.",
    "get_fundamental_metrics": "Point-in-time quarterly metrics (EBITDA margin, net margin, Net Debt/EBITDA) using KAP disclosure dates.",
    "calculate_fundamental_change": "Quarter-over-quarter changes and post-disclosure 1/5/20d returns vs XU100 and sector peers; flags incomplete windows.",
    "get_macro_context": "USD/TRY, EUR/TRY changes, TCMB policy rate, Fed funds rate; optional stock-vs-USDTRY correlation.",
    "check_data_quality": "Data-quality gate: missing data, zero volume, stale prices, look-ahead, small peer groups. Returns PASS / PASS_WITH_WARNINGS / INVESTIGATE / HARD_BLOCK.",
}
TOOL_SPECS = [{"name": n, "description": DESCRIPTIONS[n], "parameters": {"type": "object", "properties": p, "required": r}}
              for n, (_, p, r) in TOOLS.items()]
DATE_KEYS = {"start", "end", "as_of", "oos_start"}


def _err(tool, msg, code="INVALID_CALL"):
    return {"tool": tool, "error": code, "message": msg}


# Turkish case suffixes that can stick to a ticker when it is typed without an apostrophe ("akbnka" = AKBNK'a)
TR_SUFFIXES = {"A", "E", "I", "İ", "U", "Ü", "YA", "YE", "YI", "Yİ", "YU", "YÜ", "DA", "DE", "TA", "TE", "DAN", "DEN", "TAN",
               "TEN", "IN", "İN", "UN", "ÜN", "NIN", "NİN", "NUN", "NÜN", "LA", "LE", "YLA", "YLE"}


def resolve_ticker(raw):
    """(ticker, note): exact ticker, or a known ticker followed by an apostrophe / Turkish case suffix
    ("AKBNK'A", "AKBNKA" -> AKBNK) when exactly one known ticker matches; otherwise (raw, None)."""
    t = str(raw).strip().upper().replace(".IS", "")
    known = set(core.available_tickers())
    if t in known:
        return t, None
    for sep in ("'", "’"):
        if sep in t and t.split(sep)[0] in known:
            return t.split(sep)[0], f"{raw} interpreted as {t.split(sep)[0]}"
    hits = {k for k in known if t.startswith(k) and t[len(k):] in TR_SUFFIXES}
    if len(hits) == 1:
        k = hits.pop()
        return k, f"{raw} interpreted as {k} (Turkish suffix '{t[len(k):]}')"
    return t, None


DROP_KEYS = {"notes", "method", "parameters", "point_in_time_rule", "relationship_note", "sector_strong_definition",
             "pattern_weeks_note", "success_definition", "execution_note", "source"}
DROP_SUFFIX = ("median_return_1d_pct", "median_return_3d_pct", "median_return_10d_pct", "median_rel_return_pp",
               "mean_rel_return_pp", "xu100_return_pct_unused")


def compact(obj, parent=None):
    """Model-facing view of a tool result: removes verbose/duplicate fields to keep prompts short.
    Used identically by the training-data generator and the inference harness."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS or k in DROP_SUFFIX:
                continue
            if parent == "post" and k in ("xu100_return_pct", "sector_peer_median_return_pct", "stock_return_pct"):
                continue
            if k == "revenue_growth_yoy_pct":
                continue
            if parent == "quarter_metrics" and k in ("revenue", "ebitda", "net_profit"):
                continue
            nxt = "post" if k in ("1d", "5d", "20d") and parent == "post_disclosure" else ("post_disclosure" if k == "post_disclosure" else None)
            out[k] = compact(v, nxt)
        return out
    if isinstance(obj, list):
        return [compact(x, parent) for x in obj]
    return obj


def call_tool(name, args=None, full=False):
    """full=False returns the compact model-facing view (default for harness and training)."""
    r = _call_tool(name, args)
    if full or "error" in r:
        return r
    if name == "get_fundamental_metrics":
        r = dict(r); r["quarters"] = [compact(q, "quarter_metrics") for q in r.get("quarters", [])]
    return compact(r)


def _call_tool(name, args=None):
    args = dict(args or {})
    if name not in TOOLS:
        return _err(name, f"unknown tool; allowed: {sorted(TOOLS)}", "UNKNOWN_TOOL")
    fn, props, req = TOOLS[name]
    unknown = set(args) - set(props)
    if unknown:
        return _err(name, f"unknown arguments: {sorted(unknown)}")
    miss = [r for r in req if r not in args]
    if miss:
        return _err(name, f"missing required arguments: {miss}")
    for k in DATE_KEYS & set(args):
        if not isinstance(args[k], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args[k]):
            return _err(name, f"{k} must be YYYY-MM-DD")
        try:
            d = pd.Timestamp(args[k])
        except Exception:
            return _err(name, f"{k} is not a valid date")
        if d > CUTOFF:
            return _err(name, f"{k}={args[k]} is after the analysis cutoff 2026-09-10; refusing (look-ahead)", "LOOKAHEAD_BLOCKED")
    if "start" in args and "end" in args and args["start"] > args["end"]:
        return _err(name, "start is after end")
    notes = []
    for k in ("ticker",):
        if k in args:
            t, note = resolve_ticker(args[k])
            if note:
                notes.append(note)
            if t in core.EXCLUDE:
                return _err(name, f"{t} excluded from study universe", "TICKER_EXCLUDED")
            if t not in core.available_tickers():
                return _err(name, f"no data for ticker {t}", "UNKNOWN_ENTITY")
            args[k] = t
    for k in ("tickers", "sector"):          # ticker lists (a sector name is left to core._resolve)
        v = args.get(k)
        if isinstance(v, list):
            fixed = []
            for x in v:
                if isinstance(x, str) and x.lower() not in core.SECTORS and x.lower() not in core.SECTOR_ALIASES:
                    x, note = resolve_ticker(x)
                    if note:
                        notes.append(note)
                fixed.append(x)
            args[k] = fixed
    if name == "test_weekday_pattern" and "pattern" in args:
        args["pattern"] = normalize_pattern(args["pattern"])
    if "event" in props and "event" in args and args["event"] not in events.EVENTS:
        return _err(name, f"unknown event; allowed: {sorted(events.EVENTS)}")
    if name == "run_backtest" and args.get("strategy") not in STRATEGIES:
        return _err(name, f"unknown strategy; allowed: {STRATEGIES}")
    try:
        out = json.loads(json.dumps(fn(**args), default=str))
        if notes and isinstance(out, dict):
            out["input_notes"] = notes
        return out
    except KeyError as e:
        return _err(name, f"unknown sector or ticker: {e}", "UNKNOWN_ENTITY")
    except Exception as e:
        return _err(name, f"{type(e).__name__}: {e}", "TOOL_ERROR")
