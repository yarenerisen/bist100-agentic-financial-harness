"""Episodic memory: past analysis cases  state -> evidence -> action -> gate -> human decision, in SQLite.
Temporal memory = data/ files (prices, fundamentals, macro); procedural memory = training/prompts.py + labeling rules + registry."""
import json, sqlite3, time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "logs" / "memory.sqlite"


def _conn():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, time TEXT, question TEXT, tickers TEXT, tools TEXT,
                 state TEXT, quality_gate TEXT, evidence TEXT, human_decision TEXT, human_reason TEXT)""")
    return c


def record(trace, tickers):
    hr = trace.get("human_review") or {}
    ev = [e["text"] for e in (trace.get("evidence_bundle") or {}).get("committed_evidence", [])]
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (trace["id"], time.strftime("%Y-%m-%d %H:%M:%S"), trace["question"], ",".join(sorted(set(tickers))),
                   ",".join(x.get("tool", "INVALID") for x in trace["tool_calls"]), trace.get("state"),
                   (trace.get("quality_gate") or {}).get("status"), json.dumps(ev, ensure_ascii=False),
                   hr.get("decision"), hr.get("reason")))


def similar(tickers, limit=3):
    if not tickers or not DB.exists():
        return []
    with _conn() as c:
        rows = c.execute("SELECT id,time,question,tickers,state,human_decision,human_reason FROM cases ORDER BY time DESC").fetchall()
    tk = set(tickers)
    hits = [r for r in rows if tk & set((r[3] or "").split(","))]
    return [dict(zip(("id", "time", "question", "tickers", "state", "human_decision", "human_reason"), r)) for r in hits[:limit]]
