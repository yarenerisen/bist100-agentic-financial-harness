"""Builds release/knowledge_base.json: the documents the system retrieves from, with source metadata and SHA-256.

Retrieval in this system is deterministic and tool-based (no vector store): every tool reads specific knowledge items
(listed under `used_by_tools`), so each number in an answer can be traced to a document below.
Usage (from bist_llm/): python -m release.build_knowledge_base [--pdf-dir ../../raw_fin/data_econ]"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from tools import core   # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def file_doc(doc_id, path, kind, source, used_by, **extra):
    d = {"id": doc_id, "type": kind, "path": rel(path), "sha256": sha(path), "bytes": Path(path).stat().st_size,
         "source": source, "used_by_tools": used_by}
    d.update(extra)
    return d


def build(pdf_dir=None):
    D = ROOT / "data"
    docs = []
    px = pd.read_parquet(D / "prices.parquet")
    docs.append(file_doc("market_prices", D / "prices.parquet", "market_data",
                         {"provider": "Yahoo Finance via yfinance", "adjustment": "split and dividend adjusted (auto_adjust=True)",
                          "script": "bist_llm/data/download_prices.py"},
                         ["get_stock_history", "get_index_history", "rank_sector_relative_strength", "run_backtest", "test_weekday_pattern",
                          "detect_multiday_patterns", "calculate_indicators", "detect_reversal_events", "find_support_resistance",
                          "preview_strategy_variants", "get_macro_context", "calculate_fundamental_change", "check_data_quality"],
                         series=sorted(px.ticker.unique()), date_range=[str(pd.to_datetime(px.date).min().date()), str(pd.to_datetime(px.date).max().date())],
                         rows=int(len(px))))
    f = pd.read_csv(D / "fundamentals.csv")
    docs.append(file_doc("fundamentals_quarterly", D / "fundamentals.csv", "fundamentals",
                         {"provider": "KAP financial report PDFs", "parser": "bist_llm/data/parse_financials.py",
                          "point_in_time_key": "disclosure_date (board-approval date in the report)"},
                         ["get_fundamental_metrics", "calculate_fundamental_change", "preview_strategy_variants", "get_news_events"],
                         companies=sorted(f.ticker.unique()), rows=int(len(f))))
    docs.append(file_doc("fundamentals_parsed_ytd", D / "fundamentals_parsed_ytd.csv", "fundamentals_audit_trail",
                         {"provider": "KAP financial report PDFs", "parser": "bist_llm/data/parse_financials.py"}, []))
    # one entry per KAP report (the underlying documents)
    ytd = pd.read_csv(D / "fundamentals_parsed_ytd.csv")
    pdfs = Path(pdf_dir) if pdf_dir else None
    reports = []
    for r in ytd.itertuples():
        rep = {"id": f"kap_{r.ticker}_{r.quarter}", "type": "kap_financial_report", "ticker": r.ticker, "period": r.quarter,
               "quarter_end": r.quarter_end, "disclosure_date": r.disclosure_date, "disclosure_date_source": r.disclosure_source,
               "extracted_ytd": {k: (None if pd.isna(getattr(r, k)) else float(getattr(r, k)))
                                 for k in ("revenue_ytd", "ebit_ytd", "da_ytd", "net_profit_ytd", "cash", "st_borrowings",
                                           "lt_current_portion", "lt_borrowings", "lease_liabilities")},
               "source": {"provider": "KAP (Public Disclosure Platform) financial report", "file": f"{r.ticker}/{r.quarter}.pdf"}}
        if pdfs and (pdfs / r.ticker / f"{r.quarter}.pdf").exists():
            rep["source"]["sha256"] = sha(pdfs / r.ticker / f"{r.quarter}.pdf")
        reports.append(rep)
    for name, used, src in (("policy_rate.csv", ["get_macro_context", "get_news_events"], "CBRT (TCMB) MPC decisions"),
                            ("fed_rate.csv", ["get_macro_context", "get_news_events"], "Federal Reserve FOMC decisions"),
                            ("cpi.csv", ["get_macro_context"], "TurkStat CPI (template, not loaded)")):
        t = pd.read_csv(D / name)
        docs.append(file_doc(f"macro_{name.split('.')[0]}", D / name, "macro", {"provider": src},
                             used, records=t.to_dict("records")))
    docs.append({"id": "sector_map", "type": "reference", "path": "bist_llm/tools/core.py", "source": {"provider": "project (simplified sector map)"},
                 "used_by_tools": ["rank_sector_relative_strength", "run_backtest", "calculate_fundamental_change", "check_data_quality"],
                 "content": core.SECTORS})
    for doc_id, p, kind in (("labeling_rules", ROOT / "training" / "labeling_rules.md", "reference"),
                            ("data_dictionary", REPO / "docs" / "data_dictionary.md", "reference"),
                            ("study_universe", REPO / "docs" / "universe.md", "reference")):
        if p.exists():
            docs.append(file_doc(doc_id, p, kind, {"provider": "project documentation"}, []))
    for p in sorted((REPO / "docs" / "results").glob("*.md")):
        docs.append(file_doc(f"research_{p.stem}", p, "research_report", {"provider": "bist_llm/research/run_scenarios.py"}, []))
    return {"name": "BIST-Analyst knowledge base", "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "analysis_date": "2026-09-10",
            "retrieval": "deterministic tool-based retrieval over structured, timestamped sources (no vector store); "
                         "point-in-time filters ensure nothing after 2026-09-10 is returned",
            "documents": docs, "kap_reports": reports, "counts": {"documents": len(docs), "kap_reports": len(reports)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=None, help="folder with <TICKER>/<YYYYQn>.pdf to hash the original reports")
    ap.add_argument("--out", default=str(ROOT / "release" / "knowledge_base.json"))
    a = ap.parse_args()
    kb = build(a.pdf_dir)
    Path(a.out).write_text(json.dumps(kb, ensure_ascii=False, indent=1, default=str))
    print("saved", a.out, kb["counts"])
