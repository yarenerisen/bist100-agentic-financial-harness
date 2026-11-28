# BIST-Analyst: fine-tuned local LLM + deterministic tools + stateful harness

**Core principle:** the LLM reasons and explains. Python tools calculate. The harness controls. Backtests verify. The model runs locally.

## Layout
```
data/      prices.parquet (76 BIST stocks + XU100 + USDTRY/EURTRY, 2023-06 → 2026-09-10), policy_rate.csv, fed_rate.csv,
           fundamentals.csv (KAP data, filled by hand), cpi.csv (TÜİK), MACRO_SOURCES.md, download_prices.py, README_fill_data.md
tools/     core.py (data, sector ranking, catch-up event study, weekday & multi-day patterns, indicators)
           events.py (11 technical event families + regime breakdown), backtest.py (Sharpe, MDD, B&H, XU100, costs)
           macro_fund.py (macro context, point-in-time fundamentals, post-disclosure abnormal returns)
           quality.py (data-quality gate), levels.py (support/resistance), context.py (official event calendar, speech features)
           registry.py (16 tools: validation, look-ahead block, compact output); ../mcp_server.py serves them over MCP (stdio)
training/  prompts.py (system prompt), generate_dataset.py → train/val/test.jsonl (1,710 / 109 / 143) + test_ext.jsonl (390),
           labeling_rules.md, finetune_qlora.py, RUN_ON_KAGGLE.md
harness/   run_agent.py (16-state workflow with agent roles, tool execution, quality and risk gates, answer validation,
           human review, decision log, replay,
           ablation modes A–E), evidence.py (observed context vs committed evidence), memory.py (episodic memory, SQLite)
           validate.py (numeric hallucination / format / action-space / advice checks), evaluate.py (evaluation metrics, per-tool
           table), compare.py (side-by-side report), test_workflow.py (state-machine self-check)
research/  run_scenarios.py → docs/results/*.md (scenario reports), plot_signals.py → docs/figures/*.png
```

## Running the released model (offline, from `bist_llm/`)
Released files are in `release/`: `manifest.json` (SHA-256 of every file), `training_report.json` (hyperparameters, loss
curve), `execution_traces.json` (reference scenarios), `knowledge_base.json`. The LoRA adapter
(`adapter_model.safetensors` + `adapter_config.json`, SHA-256 in the manifest) is submitted separately.
```bash
pip install -r ../requirements.txt
# one question
python -m harness.run_agent --backend hf --model <adapter folder> --load-4bit "<question>"
# the 5 reference scenarios (TR + EN) -> release/execution_traces.json
python -m release.make_traces --backend hf --model <adapter folder> --load-4bit
# a live question with a full trace
python -m release.make_traces --backend hf --model <adapter folder> --load-4bit --question "<question>" --out release/live_trace.json
```
Needs a CUDA GPU with about 6 GB of memory; set `HF_HUB_OFFLINE=1` after the base model has been downloaded once.

## Quick commands (from `bist_llm/`)
```bash
python -m training.generate_dataset                    # rebuild training data from real tool outputs (~3 min)
python -m harness.evaluate --backend replay            # pipeline sanity check (oracle answers → all metrics perfect)
python -m harness.run_agent --backend openai "..."     # ask the local model (llama.cpp / Ollama server)
python -m harness.run_agent --replay logs/<id>.json    # reproducibility check of a logged analysis
python -m harness.run_agent --bundle logs/<id>.json    # evidence bundle of a logged analysis
python -m harness.evaluate --backend openai --mode C   # harness ablation A..E
python -m research.run_scenarios                       # regenerate docs/results/ (~30 s)
python -m research.plot_signals ASELS lower_bollinger_touch   # chart → docs/figures/
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
- Fundamentals: parsed from KAP financial report PDFs (`data/parse_financials.py`, 13 non-bank companies,
  reports 2024Q4–2026Q2 → 2024Q1–Q3 not available). Quarterly values = YTD differences (approximate under TMS 29
  inflation restatement); 2024Q4 row = full-year 2024; disclosure date = board-approval date in the report;
  net debt includes lease liabilities; CCOLA 2024Q4 income statement is unreadable (garbled font) and is omitted.
  AKBNK (bank) is excluded: EBITDA-based ratios are not meaningful for banks.
- TÜİK CPI (2023-06 → 2026-08, by publication date) and the TCMB/FOMC rate files are verified against the official
  bulletins and decision announcements; sources in `data/MACRO_SOURCES.md`. TÜİK switched CPI to 2025=100 from the
  January 2026 bulletin (published annual rates are used as released).
- News layer = official timestamped events only (KAP disclosures, TCMB/FOMC); no licensed news feed. Speech pipeline exists
  (`data/speech_pipeline.py`, local Whisper) but no pre-approved videos are loaded yet; strategy variant E is unavailable.
- Event studies are in-sample; few events for combined conditions → low statistical power (always reported).
