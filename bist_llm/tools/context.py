"""News / event and speech-intelligence context layer (contextual evidence only, never a substitute for numbers).

Event sources (all timestamped, verifiable):
  - KAP financial-report disclosures (data/fundamentals.csv, board-approval date)
  - TCMB policy-rate decisions (data/policy_rate.csv), FOMC decisions (data/fed_rate.csv)
  - optional curated, legally accessible headlines: data/news_events.csv
    (timestamp,ticker,topic,event_type,headline,source,url)
Speech features: data/speech/features.jsonl produced by data/speech_pipeline.py from pre-approved videos."""
import json
from pathlib import Path
import pandas as pd
from .core import _clamp, SECTOR_OF

D = Path(__file__).resolve().parent.parent / "data"
NEWS_CSV = D / "news_events.csv"
SPEECH = D / "speech" / "features.jsonl"
MACRO_TOPICS = {"rates", "fx", "inflation", "macro", "monetary_policy"}


def _official_events(start, end):
    from .macro_fund import FUND_CSV, RATE_CSV, FED_CSV
    ev = []
    if FUND_CSV.exists():
        f = pd.read_csv(FUND_CSV).dropna(subset=["disclosure_date"])
        for r in f.itertuples():
            ev.append({"date": r.disclosure_date, "ticker": r.ticker, "topic": "earnings", "event_type": "financial_report_disclosure",
                       "headline": f"{r.ticker} financial statements for quarter ending {r.quarter_end} approved/disclosed",
                       "source": "KAP financial report (board approval date)"})
    for path, name, col in ((RATE_CSV, "TCMB policy rate decision", "policy_rate_pct"), (FED_CSV, "FOMC fed funds decision", "fed_funds_upper_pct")):
        if path.exists():
            t = pd.read_csv(path)
            prev = None
            for r in t.itertuples():
                v = getattr(r, col)
                if prev is not None:
                    move = "hold" if v == prev else ("cut" if v < prev else "hike")
                    ev.append({"date": r.effective_date, "ticker": None, "topic": "rates", "event_type": f"policy_rate_{move}",
                               "headline": f"{name}: {move} to {v}%", "source": name.split()[0]})
                prev = v
    if NEWS_CSV.exists():
        n = pd.read_csv(NEWS_CSV)
        for r in n.itertuples():
            ev.append({"date": str(r.timestamp)[:10], "ticker": r.ticker if isinstance(r.ticker, str) else None, "topic": r.topic,
                       "event_type": r.event_type, "headline": r.headline, "source": r.source, "url": r.url})
    e = pd.DataFrame(ev)
    if e.empty:
        return e
    e["date"] = pd.to_datetime(e["date"])
    return e[(e.date >= pd.Timestamp(start)) & (e.date <= end)].sort_values("date")


def get_news_events(ticker=None, topic=None, start="2024-01-01", end="2026-09-10", limit=15) -> dict:
    end = _clamp(end)
    e = _official_events(start, end)
    out = {"tool": "get_news_events", "ticker": ticker.upper() if ticker else None, "topic": topic,
           "period": [str(pd.Timestamp(start).date()), str(end.date())], "events": [], "counts_by_type": {}, "warnings": []}
    if not e.empty:
        m = pd.Series(True, index=e.index)
        if ticker:
            m &= (e.ticker == ticker.upper()) | (e.topic.isin(MACRO_TOPICS) & e.ticker.isna())
        if topic:
            m &= e.topic == topic
        e = e[m]
        out["counts_by_type"] = e.event_type.value_counts().to_dict()
        out["total_events"] = int(len(e))
        out["events"] = [{k: (v.strftime("%Y-%m-%d") if k == "date" else v) for k, v in r.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}
                         for r in e.tail(limit).to_dict("records")]
    if not NEWS_CSV.exists():
        out["warnings"].append("news coverage is sparse: only official calendar events (KAP disclosures, TCMB/FOMC decisions); no licensed news feed loaded")
    if ticker and SECTOR_OF.get(ticker.upper()) and not any(x.get("ticker") == ticker.upper() for x in out["events"]):
        out["warnings"].append(f"no company-specific events for {ticker.upper()} in the period")
    out["note"] = "events are contextual evidence with timestamps; they do not explain or predict price moves by themselves"
    return out


def get_speech_features(source_id=None, topic=None, ticker=None) -> dict:
    out = {"tool": "get_speech_features", "source_id": source_id, "topic": topic, "ticker": ticker, "segments": [], "warnings": []}
    if not SPEECH.exists():
        out["warnings"].append("no pre-approved video transcripts are loaded; speech context unavailable")
        return out
    rows = [json.loads(l) for l in SPEECH.read_text().splitlines() if l.strip()]
    for r in rows:
        if source_id and r.get("source_id") != source_id:
            continue
        if topic and topic not in r.get("topics", []):
            continue
        if ticker and ticker.upper() not in r.get("tickers", []):
            continue
        out["segments"].append({k: r.get(k) for k in ("source_id", "url", "speaker", "channel", "published", "segment_start",
                                                      "segment_end", "topics", "stance", "claims")})
    out["segments"] = out["segments"][:10]
    unverified = sum(1 for s in out["segments"] for c in (s.get("claims") or []) if c.get("numeric") and not c.get("verified"))
    if unverified:
        out["warnings"].append(f"{unverified} numeric claim(s) are not yet verified against deterministic tools; do not use them as facts")
    if not out["segments"]:
        out["warnings"].append("no matching transcript segments")
    out["note"] = "expert commentary is contextual evidence, not ground truth"
    return out
