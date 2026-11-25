"""Macro and fundamental tools (CSV-backed where no free API exists)."""
import numpy as np
import pandas as pd
from pathlib import Path
from . import core
from .core import CUTOFF, _clamp, _prices, get_ohlcv, SECTORS, SECTOR_OF, EXCLUDE

D = Path(__file__).resolve().parent.parent / "data"
FUND_CSV = D / "fundamentals.csv"
RATE_CSV = D / "policy_rate.csv"
CPI_CSV = D / "cpi.csv"
FED_CSV = D / "fed_rate.csv"


def _fx(series, start, end):
    df = _prices()
    x = df[(df.ticker == series) & (df.date >= pd.Timestamp(start)) & (df.date <= end)]
    return x.set_index("date").close.dropna()


def get_macro_context(start="2024-01-01", end="2026-09-10", series=("USDTRY", "EURTRY"), ticker=None) -> dict:
    end = _clamp(end); start = pd.Timestamp(start)
    out = {"tool": "get_macro_context", "period": [str(start.date()), str(end.date())], "series": {}, "warnings": []}
    for s in series:
        px = _fx(s, start, end)
        if px.empty:
            out["warnings"].append(f"{s}: no data"); continue
        r = px.pct_change().dropna()
        out["series"][s] = {"first": round(float(px.iloc[0]), 4), "last": round(float(px.iloc[-1]), 4),
                            "change_pct": round(float(px.iloc[-1] / px.iloc[0] - 1) * 100, 2),
                            "daily_vol_pct": round(float(r.std()) * 100, 3), "last_date": str(px.index[-1].date()),
                            "source": "Yahoo Finance market quote (not official TCMB rate)"}
    if ticker:
        fx = _fx("USDTRY", start - pd.Timedelta(days=10), end)
        st = get_ohlcv(ticker, str(start.date()), end).close
        d = pd.concat([fx.rename("fx"), st.rename("st")], axis=1).dropna()
        for freq, name in (("W-FRI", "weekly"), ("ME", "monthly")):
            r = d.resample(freq).last().pct_change().dropna()
            if len(r) >= 8:
                c = float(r.fx.corr(r.st)); beta = float(np.polyfit(r.fx, r.st, 1)[0])
                same = float(((r.fx > 0) == (r.st > 0)).mean())
                out[f"{ticker.upper()}_vs_USDTRY_{name}"] = {
                    "observations": len(r), "correlation": round(c, 3), "beta": round(beta, 3),
                    "same_direction_pct": round(same * 100, 1),
                    "mean_stock_return_when_usdtry_up_pct": round(float(r.st[r.fx > 0].mean()) * 100, 2),
                    "mean_stock_return_when_usdtry_down_pct": round(float(r.st[r.fx <= 0].mean()) * 100, 2)}
        out["relationship_note"] = "association only; correlation is not causation and is not a forecast"
    for name, path, dcol, vcol in (("tcmb_policy_rate", RATE_CSV, "effective_date", "policy_rate_pct"),
                                   ("fed_funds_rate", FED_CSV, "effective_date", "fed_funds_upper_pct"),
                                   ("tuik_cpi", CPI_CSV, "publication_date", "cpi_yoy_pct")):
        t = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if t.empty or t.dropna(how="all").empty:
            out["warnings"].append(f"{name}: data not loaded (macro series unavailable)")
            continue
        t[dcol] = pd.to_datetime(t[dcol]); t = t.sort_values(dcol)
        before_s, before_e = t[t[dcol] <= start], t[t[dcol] <= end]
        inwin = t[(t[dcol] > start) & (t[dcol] <= end)]
        rec = {"level_at_start": None if before_s.empty else float(before_s[vcol].iloc[-1]),
               "level_at_end": None if before_e.empty else float(before_e[vcol].iloc[-1]),
               "changes_in_period": [{"date": r[dcol].strftime("%Y-%m-%d"), "value": float(r[vcol])} for _, r in inwin.iterrows()],
               "source": str(t["source"].iloc[-1]) if "source" in t else None}
        if rec["level_at_start"] is not None and rec["level_at_end"] is not None:
            rec["change_pp"] = round(rec["level_at_end"] - rec["level_at_start"], 2)
        last = t[dcol].max()
        if last < end - pd.Timedelta(days=120) and name != "tcmb_policy_rate":
            out["warnings"].append(f"{name}: last record {last.date()}; later values not loaded")
        out[name] = rec
    out["notes"] = ["values aligned by decision/publication date; nothing after the analysis date is used"]
    return out


def _load_fund(ticker, as_of):
    t = pd.read_csv(FUND_CSV, parse_dates=["quarter_end", "disclosure_date"])
    t = t[t.ticker == ticker.upper()].dropna(subset=["disclosure_date", "revenue", "ebitda"]).sort_values("quarter_end")
    return t[t.disclosure_date <= as_of].copy()   # point-in-time: only already-disclosed statements


def get_fundamental_metrics(ticker, start="2024-01-01", end="2026-09-10") -> dict:
    end = _clamp(end)
    t = _load_fund(ticker, end)
    t = t[t.disclosure_date >= pd.Timestamp(start)]
    out = {"tool": "get_fundamental_metrics", "ticker": ticker.upper(), "period": [start, str(end.date())],
           "point_in_time_rule": "only statements with disclosure_date <= end are included", "warnings": []}
    if t.empty:
        out["quarters"] = []
        if SECTOR_OF.get(ticker.upper()) == "banking":
            out["warnings"].append("bank: EBITDA / Net Debt-EBITDA are not meaningful for banks; industrial ratios not applied")
        else:
            out["warnings"].append("fundamental data not loaded for this ticker/period; cannot analyse")
        return out
    if t.disclosure_date.min() > pd.Timestamp(start) + pd.Timedelta(days=120):
        out["warnings"].append(f"earliest available statement is {t.quarter_end.min().date()} (disclosed {t.disclosure_date.min().date()}); earlier quarters of the requested period are not in the data")
    t["ebitda_margin_pct"] = t.ebitda / t.revenue * 100
    t["net_margin_pct"] = t.net_profit / t.revenue * 100
    t["net_debt_to_ebitda"] = t.net_debt / t.ebitda_ltm
    out["quarters"] = [{"quarter_end": r.quarter_end.strftime("%Y-%m-%d"), "disclosure_date": r.disclosure_date.strftime("%Y-%m-%d"),
                        "revenue": r.revenue, "ebitda": r.ebitda, "net_profit": r.net_profit,
                        "ebitda_margin_pct": round(r.ebitda_margin_pct, 2), "net_margin_pct": round(r.net_margin_pct, 2),
                        "net_debt_to_ebitda": None if pd.isna(r.net_debt_to_ebitda) else round(r.net_debt_to_ebitda, 2)}
                       for r in t.itertuples()]
    return out


def _peers(ticker):
    sec = SECTOR_OF.get(ticker)
    return [x for x in SECTORS.get(sec, []) if x != ticker and x not in EXCLUDE]


def calculate_fundamental_change(ticker, start="2024-01-01", end="2026-09-10", horizons=(1, 5, 20)) -> dict:
    ticker = ticker.upper(); end = _clamp(end)
    t = _load_fund(ticker, end)
    out = {"tool": "calculate_fundamental_change", "ticker": ticker, "period": [start, str(end.date())], "warnings": [],
           "method": "base=last close on/before disclosure_date; return over next h trading days; abnormal = stock - benchmark",
           "quarters": []}
    if len(t) < 2:
        out["warnings"].append("bank: EBITDA-based ratios not applicable" if SECTOR_OF.get(ticker) == "banking"
                               else "fundamental data not loaded or fewer than 2 quarters; cannot compute changes"); return out
    if t.disclosure_date.min() > pd.Timestamp(start) + pd.Timedelta(days=120):
        out["warnings"].append(f"earliest available statement is {t.quarter_end.min().date()} (disclosed {t.disclosure_date.min().date()}); earlier quarters not in the data")
    t = t.reset_index(drop=True)
    t["ebitda_margin"] = t.ebitda / t.revenue * 100
    t["nd_ebitda"] = t.net_debt / t.ebitda_ltm
    px = get_ohlcv(ticker, "2023-06-01", end).close
    xu = get_ohlcv("XU100", "2023-06-01", end).close
    peers = _peers(ticker)
    pw = core._close_wide(peers, end=end) if peers else None
    if not peers:
        out["warnings"].append("no sector peers in universe; sector-adjusted return unavailable")
    for i, r in t.iterrows():
        if r.disclosure_date < pd.Timestamp(start):
            continue
        prev = t.iloc[i - 1] if i >= 1 else None
        yoy = t.iloc[i - 4] if i >= 4 else None
        row = {"quarter_end": r.quarter_end.strftime("%Y-%m-%d"), "disclosure_date": r.disclosure_date.strftime("%Y-%m-%d"),
               "ebitda_margin_pct": round(r.ebitda_margin, 2),
               "ebitda_margin_change_qoq_pp": None if prev is None else round(r.ebitda_margin - prev.ebitda_margin, 2),
               "net_debt_to_ebitda": None if pd.isna(r.nd_ebitda) else round(r.nd_ebitda, 2),
               "net_debt_to_ebitda_change_qoq": None if prev is None or pd.isna(r.nd_ebitda) or pd.isna(prev.nd_ebitda) else round(r.nd_ebitda - prev.nd_ebitda, 2),
               "revenue_growth_yoy_pct": None if yoy is None else round((r.revenue / yoy.revenue - 1) * 100, 2)}
        row["improving"] = bool(prev is not None and row["ebitda_margin_change_qoq_pp"] is not None and row["ebitda_margin_change_qoq_pp"] > 0
                                and (row["net_debt_to_ebitda_change_qoq"] is None or row["net_debt_to_ebitda_change_qoq"] <= 0))
        pos = px.index.searchsorted(r.disclosure_date, side="right") - 1
        reac = {}
        for h in horizons:
            if pos < 0 or pos + h >= len(px):
                reac[f"{h}d"] = {"complete": False, "note": "forward window extends past analysis date; excluded"}; continue
            d0, d1 = px.index[pos], px.index[pos + h]
            sr = px.iloc[pos + h] / px.iloc[pos] - 1
            xr = xu.loc[d1] / xu.loc[d0] - 1
            e = {"complete": True, "stock_return_pct": round(float(sr) * 100, 2), "xu100_return_pct": round(float(xr) * 100, 2),
                 "abnormal_vs_xu100_pp": round(float(sr - xr) * 100, 2)}
            if pw is not None and d0 in pw.index and d1 in pw.index:
                pr = float((pw.loc[d1] / pw.loc[d0] - 1).median())
                e["sector_peer_median_return_pct"] = round(pr * 100, 2); e["abnormal_vs_sector_pp"] = round((float(sr) - pr) * 100, 2)
            reac[f"{h}d"] = e
        row["post_disclosure"] = reac
        out["quarters"].append(row)
        if any(not v["complete"] for v in reac.values()):
            out["warnings"].append(f"{row['quarter_end']}: incomplete post-disclosure window (data ends {CUTOFF.date()})")
    grp = {}
    for flag in (True, False):
        sub = [q for q in out["quarters"] if q["improving"] == flag and q["post_disclosure"]["20d"]["complete"]]
        grp["improving" if flag else "not_improving"] = {
            "quarters": len(sub),
            "mean_abnormal_20d_vs_xu100_pp": round(float(np.mean([q["post_disclosure"]["20d"]["abnormal_vs_xu100_pp"] for q in sub])), 2) if sub else None}
    out["improving_vs_not_20d"] = grp
    if out["quarters"]:   # decision inputs computed here so the model reads them instead of counting nested fields
        out["summary"] = {"quarters_disclosed": len(out["quarters"]),
                          "quarters_with_complete_20d_window": sum(q["post_disclosure"]["20d"]["complete"] for q in out["quarters"]),
                          "latest_quarter_end": out["quarters"][-1]["quarter_end"],
                          "latest_quarter_improving": out["quarters"][-1]["improving"]}
    out["notes"] = ["association only; no causal claim", "small sample (few quarters)"]
    return out
