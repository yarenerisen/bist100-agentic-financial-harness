"""Long-only educational backtest engine with benchmark comparison and strategy-variant preview.
Signal at close t -> enter at OPEN t+1 -> exit at CLOSE t+hold. No information from t+1 is used to decide."""
import math
import numpy as np
import pandas as pd
from . import core
from .core import _clamp, _resolve, get_ohlcv, SECTOR_OF, SECTORS, EXCLUDE
from .events import _event_mask, EVENTS


# ------------------------------------------------------------------ signals
def _signals(strategy, ticker, params, end, peers=None):
    ind = core.calculate_indicators(ticker, "2023-06-01", end)
    if strategy in EVENTS:
        return _event_mask(ind, strategy).fillna(False).astype(bool), ind
    if strategy == "sector_catch_up":
        w = core._close_wide(peers, end=end)
        lw = params.get("lag_window", 20)
        rl = w / w.shift(lw) - 1
        med = rl.median(axis=1)
        lag = (rl[ticker] - med) * 100
        s = lag <= -params.get("lag_threshold_pp", 5.0)
        if params.get("require_strong_sector", False):
            s &= med > 0
        return s.reindex(ind.index).fillna(False).astype(bool), ind
    if strategy == "weekday_mon_wed_down":
        r = ind.close.pct_change()
        dow = ind.index.dayofweek
        down = (r < 0).values
        s = np.zeros(len(ind), bool)
        for i in range(2, len(ind)):
            s[i] = dow[i] == 2 and dow[i - 1] == 1 and dow[i - 2] == 0 and down[i] and down[i - 1] and down[i - 2]
        return pd.Series(s, index=ind.index), ind
    raise ValueError("unknown strategy")


# ------------------------------------------------------------------ engine
def _simulate(sig, ind, start, hold_days, cost_pct, slippage_pct):
    """Non-overlapping trades for one ticker. Returns trades, daily returns, position series (from start)."""
    o, c, idx = ind.open, ind.close, ind.index
    n = len(idx)
    dret = np.zeros(n); pos = np.zeros(n)
    trades, i = [], 0
    sv = sig.reindex(idx).fillna(False).values
    side = (cost_pct / 2 + slippage_pct) / 100
    while i < n:
        if sv[i] and idx[i] >= start:
            ent, ext = i + 1, i + hold_days
            if ext >= n:
                trades.append({"signal_date": str(idx[i].date()), "status": "excluded_incomplete"}); break
            entry = float(o.iloc[ent])
            net = float(c.iloc[ext]) / entry - 1 - 2 * side
            trades.append({"signal_date": str(idx[i].date()), "entry_date": str(idx[ent].date()),
                           "exit_date": str(idx[ext].date()), "return_net_pct": round(net * 100, 2)})
            dret[ent] = c.iloc[ent] / entry - 1 - side
            for j in range(ent + 1, ext + 1):
                dret[j] = c.iloc[j] / c.iloc[j - 1] - 1
            dret[ext] -= side
            pos[ent:ext + 1] = 1
            i = ext + 1
        else:
            i += 1
    m = idx >= start
    return trades, pd.Series(dret[m], idx[m]), pd.Series(pos[m], idx[m])


def _perf(daily, label):
    daily = daily.fillna(0)
    if daily.empty:
        return {"series": label, "cumulative_return_pct": None, "annualised_return_pct": None, "sharpe_ratio": None, "max_drawdown_pct": None}
    eq = (1 + daily).cumprod()
    n, sd = len(daily), daily.std()
    return {"series": label, "cumulative_return_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
            "annualised_return_pct": round(float(eq.iloc[-1] ** (252 / n) - 1) * 100, 2) if n > 20 else None,
            "sharpe_ratio": round(float(daily.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None,
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2)}


def _portfolio(per_ticker, start, end, tk):
    rets = pd.DataFrame({t: v[1] for t, v in per_ticker.items()}).fillna(0)
    poss = pd.DataFrame({t: v[2] for t, v in per_ticker.items()}).fillna(0)
    active = poss.sum(axis=1)
    strat = ((rets * poss).sum(axis=1) / active.replace(0, np.nan)).fillna(0)
    bh = pd.DataFrame({t: core.get_ohlcv(t, "2023-06-01", end).close for t in tk})
    bh = bh[bh.index >= start].pct_change().mean(axis=1)
    xu = get_ohlcv("XU100", "2023-06-01", end).close
    xu = xu[xu.index >= start].pct_change()
    done = [x for v in per_ticker.values() for x in v[0] if "return_net_pct" in x]
    r = np.array([x["return_net_pct"] for x in done])
    sp, b, x = _perf(strat, "strategy"), _perf(bh, "buy_and_hold_equal_weight"), _perf(xu, "XU100")
    ex = lambda a, bb: None if a["cumulative_return_pct"] is None or bb["cumulative_return_pct"] is None else round(a["cumulative_return_pct"] - bb["cumulative_return_pct"], 2)
    return {"trades": len(done), "win_rate_pct": round(float((r > 0).mean()) * 100, 1) if len(r) else None,
            "mean_trade_return_net_pct": round(float(r.mean()), 2) if len(r) else None,
            "time_in_market_pct": round(float((active > 0).mean()) * 100, 1) if len(active) else None,
            "performance": [sp, b, x], "excess_vs_buy_and_hold_pp": ex(sp, b), "excess_vs_xu100_pp": ex(sp, x),
            "excluded_incomplete_trades": sum(1 for v in per_ticker.values() for t in v[0] if t.get("status") == "excluded_incomplete"),
            "last_trades": [dict(t, ticker=k) for k, v in per_ticker.items() for t in v[0] if "return_net_pct" in t][-5:]}


def run_strategy_backtest(strategy, tickers, start="2024-01-01", end="2026-09-10", hold_days=5,
                          cost_pct=0.2, slippage_pct=0.05, **params) -> dict:
    end = _clamp(end); start = pd.Timestamp(start)
    tk = _resolve(tickers)
    peers = _resolve(params.get("sector")) if strategy == "sector_catch_up" and params.get("sector") else tk
    per = {}
    for t in tk:
        sig, ind = _signals(strategy, t, params, end, peers)
        per[t] = _simulate(sig, ind, start, hold_days, cost_pct, slippage_pct)
    out = {"tool": "run_backtest", "strategy": strategy, "tickers": tk, "period": [str(start.date()), str(end.date())],
           "execution": f"signal at close t, enter next open, exit close after {hold_days} trading days",
           "costs": {"commission_round_trip_pct": cost_pct, "slippage_per_side_pct": slippage_pct}}
    out.update(_portfolio(per, start, end, tk))
    out["warnings"] = []
    if out["trades"] < 20:
        out["warnings"].append(f"only {out['trades']} trades; high return from few trades is unreliable")
    out["warnings"].append("in-sample backtest; no guarantee of future performance")
    return out


# ------------------------------------------------------------------ strategy variants A-E
def _pit_fundamental_improving(ticker, idx):
    """Point-in-time flag: latest statement disclosed on/before t had a higher EBITDA margin than the previous one."""
    from .macro_fund import FUND_CSV
    f = pd.read_csv(FUND_CSV, parse_dates=["quarter_end", "disclosure_date"])
    f = f[(f.ticker == ticker)].dropna(subset=["disclosure_date", "revenue", "ebitda"]).sort_values("quarter_end")
    if len(f) < 2:
        return None
    f["margin"] = f.ebitda / f.revenue
    f["impr"] = f.margin.diff() > 0
    s = pd.Series(np.nan, index=idx)
    for _, r in f.iloc[1:].iterrows():
        s[s.index >= r.disclosure_date] = float(r.impr)
    return s.fillna(0).astype(bool)


VARIANT_DESC = {
    "A_technical": "lower Bollinger touch with RSI14 < 40",
    "B_technical_sector": "A + stock lags its sector median by >= 3 pp over 20 days",
    "C_technical_sector_fundamental": "B + latest disclosed quarter had a higher EBITDA margin (point-in-time)",
    "D_full_structured": "C + no FX shock (USD/TRY 20-day change < 5%)",
    "E_full_contextual": "D + verified news/speech context",
}


def preview_strategy_variants(tickers, start="2024-01-01", end="2026-09-10", oos_start="2026-01-01", hold_days=5,
                              cost_pct=0.2, slippage_pct=0.05) -> dict:
    """Runs variants A-D with identical execution/cost rules, split into in-sample and out-of-sample periods.
    Rules are fixed before looking at the out-of-sample period (walk-forward check)."""
    end = _clamp(end); start = pd.Timestamp(start); oos = pd.Timestamp(oos_start)
    if not (start < oos <= end):
        raise ValueError("oos_start must lie inside (start, end]")
    tk = _resolve(tickers)
    fx = get_ohlcv("USDTRY", "2023-06-01", end).close
    fx20 = fx / fx.shift(20) - 1
    sigs, notes = {k: {} for k in VARIANT_DESC if k != "E_full_contextual"}, []
    for t in tk:
        ind = core.calculate_indicators(t, "2023-06-01", end)
        a = _event_mask(ind, "lower_bollinger_touch").fillna(False).astype(bool) & (ind.rsi14 < 40)
        sec = SECTOR_OF.get(t)
        peers = [x for x in SECTORS.get(sec, []) if x not in EXCLUDE and x in core.available_tickers()]
        if len(peers) >= 2:
            w = core._close_wide(peers, end=end); rl = w / w.shift(20) - 1
            lag = ((rl[t] - rl.median(axis=1)) * 100).reindex(ind.index)
            b = a & (lag <= -3).fillna(False)
        else:
            b = a.copy(); notes.append(f"{t}: no sector peers, B = A")
        fi = _pit_fundamental_improving(t, ind.index)
        if fi is None:
            c = None; notes.append(f"{t}: no fundamentals, C/D not available")
        else:
            c = b & fi
        d = None if c is None else c & (fx20.reindex(ind.index).ffill() < 0.05).fillna(False)
        for k, s in zip(sigs, (a, b, c, d)):
            if s is not None:
                sigs[k][t] = (s, ind)
    out = {"tool": "preview_strategy_variants", "tickers": tk, "in_sample": [str(start.date()), str((oos - pd.Timedelta(days=1)).date())],
           "out_of_sample": [str(oos.date()), str(end.date())],
           "execution": f"signal at close t, enter next open, exit close after {hold_days} trading days; costs {cost_pct}% + slippage {slippage_pct}%/side",
           "variants": {}, "warnings": []}
    for k, d in sigs.items():
        if not d:
            out["variants"][k] = {"definition": VARIANT_DESC[k], "available": False}; continue
        res = {"definition": VARIANT_DESC[k], "available": True, "tickers_used": sorted(d)}
        for tag, s0, e0 in (("in_sample", start, oos - pd.Timedelta(days=1)), ("out_of_sample", oos, end)):
            per = {}
            for t, (s, ind) in d.items():
                ii = ind[ind.index <= e0]
                per[t] = _simulate(s[s.index <= e0], ii, s0, hold_days, cost_pct, slippage_pct)
            p = _portfolio(per, s0, e0, sorted(d))
            perf = {x["series"]: x for x in p["performance"]}
            res[tag] = {"trades": p["trades"], "win_rate_pct": p["win_rate_pct"], "mean_trade_return_net_pct": p["mean_trade_return_net_pct"],
                        "cumulative_return_pct": perf["strategy"]["cumulative_return_pct"], "sharpe_ratio": perf["strategy"]["sharpe_ratio"],
                        "max_drawdown_pct": perf["strategy"]["max_drawdown_pct"],
                        "buy_and_hold_return_pct": perf["buy_and_hold_equal_weight"]["cumulative_return_pct"],
                        "xu100_return_pct": perf["XU100"]["cumulative_return_pct"]}
            if p["trades"] < 10:
                out["warnings"].append(f"{k} {tag}: only {p['trades']} trades")
        out["variants"][k] = res
    out["variants"]["E_full_contextual"] = {"definition": VARIANT_DESC["E_full_contextual"], "available": False,
                                            "reason": "no verified, timestamped news/speech feed is loaded"}
    out["warnings"] += sorted(set(notes))[:6]
    out["warnings"].append("variants are educational backtests; more filters usually mean fewer trades and lower statistical power")
    return out
