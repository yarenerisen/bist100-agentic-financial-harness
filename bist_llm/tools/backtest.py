"""Long-only educational backtest engine with benchmark comparison.
Signal at close t -> enter at OPEN t+1 -> exit at CLOSE t+hold. No information from t+1 is used to decide."""
import math
import numpy as np
import pandas as pd
from . import core
from .core import _clamp, _resolve, _label, get_ohlcv
from .events import _event_mask, EVENTS


def _signals(strategy, ticker, params, start, end, peers=None):
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
        down = r < 0
        s = pd.Series(False, index=ind.index)
        for i in range(2, len(ind)):
            if dow[i] == 2 and dow[i - 1] == 1 and dow[i - 2] == 0 and down.iloc[i] and down.iloc[i - 1] and down.iloc[i - 2]:
                s.iloc[i] = True
        return s, ind
    raise ValueError("unknown strategy")


def _perf(daily, label):
    daily = daily.fillna(0)
    eq = (1 + daily).cumprod()
    n = len(daily)
    sd = daily.std()
    return {"series": label, "cumulative_return_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
            "annualised_return_pct": round(float(eq.iloc[-1] ** (252 / n) - 1) * 100, 2) if n > 20 else None,
            "sharpe_ratio": round(float(daily.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None,
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2)}


def run_strategy_backtest(strategy, tickers, start="2024-01-01", end="2026-09-10", hold_days=5,
                          cost_pct=0.2, slippage_pct=0.05, **params) -> dict:
    end = _clamp(end)
    start = pd.Timestamp(start)
    tk = _resolve(tickers)
    peers = None
    if strategy == "sector_catch_up":
        peers = _resolve(params.get("sector")) if params.get("sector") else tk
    trades, daily_cols = [], {}
    for t in tk:
        sig, ind = _signals(strategy, t, params, start, end, peers)
        o, c = ind.open, ind.close
        idx = ind.index
        pos = pd.Series(0.0, index=idx)
        dret = pd.Series(0.0, index=idx)
        i, n = 0, len(idx)
        sig_v = sig.values
        while i < n:
            if sig_v[i] and idx[i] >= start:
                ent, ext = i + 1, i + hold_days
                if ext >= n:  # incomplete trade at the cutoff: excluded
                    trades.append({"ticker": t, "signal_date": str(idx[i].date()), "status": "excluded_incomplete"}); break
                entry, exitp = float(o.iloc[ent]), float(c.iloc[ext])
                gross = exitp / entry - 1
                net = gross - (cost_pct + 2 * slippage_pct) / 100
                trades.append({"ticker": t, "signal_date": str(idx[i].date()), "entry_date": str(idx[ent].date()),
                               "exit_date": str(idx[ext].date()), "return_net_pct": round(net * 100, 2)})
                # daily path: open->close on entry day, then close->close
                dret.iloc[ent] = c.iloc[ent] / entry - 1
                for j in range(ent + 1, ext + 1):
                    dret.iloc[j] = c.iloc[j] / c.iloc[j - 1] - 1
                dret.iloc[ent] -= (cost_pct / 2 + slippage_pct) / 100
                dret.iloc[ext] -= (cost_pct / 2 + slippage_pct) / 100
                pos.iloc[ent:ext + 1] = 1
                i = ext + 1
            else:
                i += 1
        m = idx >= start
        daily_cols[t] = (dret[m], pos[m])
    # equal-weight across tickers that are in a position each day; flat days earn 0
    rets = pd.DataFrame({t: v[0] for t, v in daily_cols.items()}).fillna(0)
    poss = pd.DataFrame({t: v[1] for t, v in daily_cols.items()}).fillna(0)
    active = poss.sum(axis=1)
    strat = (rets * poss).sum(axis=1) / active.replace(0, np.nan)
    strat = strat.fillna(0)
    bh = pd.DataFrame({t: core.get_ohlcv(t, "2023-06-01", end).close for t in tk})
    bh = bh[bh.index >= start].pct_change().mean(axis=1)
    xu = get_ohlcv("XU100", "2023-06-01", end).close
    xu = xu[xu.index >= start].pct_change()
    done = [t for t in trades if "return_net_pct" in t]
    r = np.array([t["return_net_pct"] for t in done])
    sp, b, x = _perf(strat, "strategy"), _perf(bh, "buy_and_hold_equal_weight"), _perf(xu, "XU100")
    out = {"tool": "run_backtest", "strategy": strategy, "tickers": tk, "period": [str(start.date()), str(end.date())],
           "execution": f"signal at close t, enter next open, exit close after {hold_days} trading days",
           "costs": {"commission_round_trip_pct": cost_pct, "slippage_per_side_pct": slippage_pct},
           "trades": len(done), "win_rate_pct": round(float((r > 0).mean()) * 100, 1) if len(r) else None,
           "mean_trade_return_net_pct": round(float(r.mean()), 2) if len(r) else None,
           "time_in_market_pct": round(float((active > 0).mean()) * 100, 1),
           "performance": [sp, b, x],
           "excess_vs_buy_and_hold_pp": round(sp["cumulative_return_pct"] - b["cumulative_return_pct"], 2),
           "excess_vs_xu100_pp": round(sp["cumulative_return_pct"] - x["cumulative_return_pct"], 2),
           "excluded_incomplete_trades": sum(1 for t in trades if t.get("status") == "excluded_incomplete"),
           "last_trades": done[-5:], "warnings": []}
    if len(done) < 20:
        out["warnings"].append(f"only {len(done)} trades; high return from few trades is unreliable")
    out["warnings"].append("in-sample backtest; no guarantee of future performance")
    return out
