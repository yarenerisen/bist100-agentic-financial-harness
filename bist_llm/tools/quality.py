"""Risk & data-quality gate. Output status: PASS | PASS_WITH_WARNINGS | INVESTIGATE | ANALYSIS_UNSAFE | HARD_BLOCK."""
import pandas as pd
from . import core
from .core import CUTOFF, EXCLUDE, SECTOR_OF, SECTORS


def check_data_quality(tickers, start="2024-01-01", end="2026-09-10", need_fundamentals=False, need_macro=False) -> dict:
    from .macro_fund import FUND_CSV, RATE_CSV
    tk = core._resolve(tickers)
    blocks, warns, info = [], [], {}
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e > CUTOFF or s > CUTOFF:
        blocks.append(f"requested period extends past analysis cutoff {CUTOFF.date()} (look-ahead)")
    if s > e:
        blocks.append("start after end")
    e = min(e, CUTOFF)
    xu = core.get_ohlcv("XU100", str(s.date()), e).index
    avail = set(core.available_tickers())
    for t in tk:
        d = {}
        if t in EXCLUDE:
            blocks.append(f"{t}: excluded from study universe"); continue
        if t not in avail:
            blocks.append(f"{t}: no price data (wrong ticker or not downloaded)"); continue
        df = core.get_ohlcv(t, str(s.date()), e)
        d["first_date"], d["last_date"], d["rows"] = str(df.index[0].date()), str(df.index[-1].date()), len(df)
        miss = len(set(xu) - set(df.index))
        d["missing_days_vs_xu100"] = miss
        if df.index[0] > s + pd.Timedelta(days=10):
            warns.append(f"{t}: history starts {df.index[0].date()}, later than requested {s.date()}")
        if miss > 0.05 * len(xu):
            warns.append(f"{t}: {miss} trading days missing vs XU100 calendar")
        zv = int((df.volume == 0).sum()); d["zero_volume_days"] = zv
        if zv > 0.05 * len(df):
            warns.append(f"{t}: {zv} zero-volume days; volume data unreliable")
        jumps = df.close.pct_change().abs()
        big = jumps[jumps > 0.25]
        if len(big):
            warns.append(f"{t}: {len(big)} daily moves >25% ({[str(x.date()) for x in big.index[:3]]}); check corporate actions")
        stale = (df.close.diff() == 0).astype(int).groupby((df.close.diff() != 0).cumsum()).sum().max()
        if stale >= 5:
            warns.append(f"{t}: price unchanged for {int(stale)} consecutive days (stale data?)")
        if df.index[-1] < e - pd.Timedelta(days=7):
            warns.append(f"{t}: last price {df.index[-1].date()} is stale relative to {e.date()}")
        sec = SECTOR_OF.get(t)
        d["sector"] = sec
        if sec and len(SECTORS[sec]) < 3:
            warns.append(f"{t}: sector peer group '{sec}' has fewer than 3 stocks")
        info[t] = d
    status_extra = None
    if need_fundamentals:
        f = pd.read_csv(FUND_CSV).dropna(subset=["disclosure_date", "revenue"])
        for t in tk:
            n = int(((f.ticker == t) & (pd.to_datetime(f.disclosure_date) <= e)).sum())
            info.setdefault(t, {})["fundamental_quarters_available"] = n
            if n < 2:
                warns.append(f"{t}: fewer than 2 disclosed quarters in fundamentals file"); status_extra = "INVESTIGATE"
    if need_macro:
        r = pd.read_csv(RATE_CSV)
        if r.empty:
            warns.append("policy-rate series not loaded")
    if blocks:
        status = "HARD_BLOCK"
    elif status_extra:
        status = status_extra
    elif warns:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return {"tool": "check_data_quality", "tickers": tk, "period": [str(s.date()), str(e.date())],
            "status": status, "hard_blocks": blocks, "warnings": warns, "details": info}
