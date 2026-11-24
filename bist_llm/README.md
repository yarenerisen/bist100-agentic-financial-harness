# BIST-Analyst: fine-tuned local LLM + deterministic tools + stateful harness

**Core principle:** the LLM reasons and explains. Python tools calculate. The harness controls. Backtests verify. The model runs locally.

## Layout
```
data/      prices.parquet (76 BIST stocks + XU100 + USDTRY/EURTRY, 2023-06 → 2026-09-10), policy_rate.csv, fed_rate.csv,
           fundamentals.csv (KAP data, filled by hand), cpi.csv, download_prices.py, README_fill_data.md
tools/     core.py (data, sector ranking, catch-up event study, weekday & multi-day patterns, indicators)
           events.py (11 technical event families + regime breakdown), backtest.py (Sharpe, MDD, B&H, XU100, costs)
           macro_fund.py (macro context, point-in-time fundamentals, post-disclosure abnormal returns)
           quality.py (data-quality gate), registry.py (11 MCP-style tools: validation, look-ahead block, compact output)
training/  prompts.py (system prompt), generate_dataset.py → train/val/test.jsonl (1,195 / 95 / 120),
           labeling_rules.md, finetune_qlora.py, RUN_ON_KAGGLE.md
harness/   run_agent.py (state machine, tool execution, quality gate, answer validation, human review, decision log, replay)
           validate.py (numeric hallucination / format / action-space / advice checks), evaluate.py (Section 30 metrics)
```

## Quick commands (from `bist_llm/`)
```bash
python -m training.generate_dataset                    # rebuild training data from real tool outputs (~3 min)
python -m harness.evaluate --backend replay            # pipeline sanity check (oracle answers → all metrics perfect)
python -m harness.run_agent --backend openai "..."     # ask the local model (llama.cpp / Ollama server)
python -m harness.run_agent --replay logs/<id>.json    # reproducibility check of a logged analysis
```
Python: `pip install pandas pyarrow numpy yfinance` (tools) + `transformers peft bitsandbytes accelerate` (training).

## Guarantees enforced in code
- No date after 2026-09-10: data is cut at download, every tool clamps, the registry rejects later dates (`LOOKAHEAD_BLOCKED`).
- Fundamentals are point-in-time by **KAP disclosure date**; post-disclosure windows past the cutoff are flagged incomplete.
- Forward-return windows not complete at the analysis date are excluded and reported.
- Backtests: signal at close t → enter next open → exit later close; commission + slippage; benchmark vs buy-and-hold and XU100.
- Answers: every number must appear in a tool output (validator); allowed action space only; no advice language;
  failed answers get one retry, then a safe `ANALYSIS_UNSAFE` fallback.
- Endpoint guard: the harness refuses non-localhost model endpoints (no cloud AI APIs).

## Known limitations
- Sector map is simplified; several sectors have only 2 stocks (warned).
- Yahoo volume data is unreliable for some tickers (e.g. SASA has many zero-volume days; warned).
- Fundamentals: parsed from the instructor's KAP report PDFs (`data/parse_financials.py`, 13 non-bank companies,
  reports 2024Q4–2026Q2 → 2024Q1–Q3 not available). Quarterly values = YTD differences (approximate under TMS 29
  inflation restatement); 2024Q4 row = full-year 2024; disclosure date = board-approval date in the report;
  net debt includes lease liabilities; CCOLA 2024Q4 income statement is unreadable (garbled font) and is omitted.
  AKBNK (bank) is excluded: EBITDA-based ratios are not meaningful for banks.
- TÜİK CPI not loaded; TCMB/Fed rate files should be re-verified against official tables.
- News / speech-intelligence layer not implemented.
- Event studies are in-sample; few events for combined conditions → low statistical power (always reported).
