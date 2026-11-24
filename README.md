# BIST 100 Agentic Financial Harness

**Evidence-first AI research harness for BIST 100 stocks: a local fine-tuned
LLM orchestrates deterministic MCP tools for technical, sector, fundamental and
macro analysis, with backtest verification and no investment advice.**

> The LLM interprets, Python tools compute, the harness controls, backtests
> verify, evidence is logged, and a human makes the final call.

This is a research system for checking market hypotheses on a diversified
universe of 30 BIST 100 stocks. It doesn't forecast prices with one black-box
model. Instead, a small fine-tuned open-source LLM orchestrates deterministic
MCP tools through a stateful harness. It's only allowed to state what the
tools return, and only after the quality gate and backtest steps have run.

---

## Research scenarios

| # | Scenario | Question | Core tools |
|---|----------|----------|------------|
| 1 | **Sector catch-up** | Do stocks lagging a strong sector later close the gap over 5/10/20 days? Is the lag structural or temporary? | `rank_sector_relative_strength`, `run_backtest` |
| 2 | **Weekday / multi-day patterns** | Are day-of-week and multi-day sequences stable out of sample, after costs and multiple-testing control? | `test_weekday_pattern` |
| 3 | **Technical reversals** | Do Bollinger, RSI, Supertrend, KAMA, Ichimoku and S/R events repeat, alone or combined? | `calculate_indicators`, `detect_reversal_events`, `find_support_resistance` |
| 4 | **Fundamentals → price reaction** | How are quarterly changes associated with 1/5/20-day abnormal returns after the disclosure date? | `get_fundamental_metrics`, `calculate_fundamental_change` |

Context layers: macro data (USD/TRY, EUR/TRY, CBRT and Fed policy rates,
TurkStat CPI), timestamped public disclosures (KAP) and news metadata, and speech-to-text
features from selected public finance commentary. These layers are
**contextual evidence**. They never replace numeric data, and any numeric
claim they contain is checked against the deterministic tools first.

## Design principles

- **No invented numbers.** Every figure in an answer has to come from a tool output.
- **Point-in-time data.** Each query has a cutoff date. Financial statements
  are aligned to their public disclosure (KAP) time and macro series to their release
  time. Signals are shifted so that a decision at *t* can't see *t+1*.
- **Finite action space.** Answers end with one analytical state, never a
  buy/sell call or price target:
  `HOLD` · `WATCH` · `INVESTIGATE` · `POTENTIAL_CATCH_UP_CANDIDATE` ·
  `TECHNICAL_REVERSAL_CANDIDATE` · `FUNDAMENTAL_IMPROVEMENT_CANDIDATE` ·
  `REJECT_SIGNAL` · `ANALYSIS_UNSAFE`
- **Mandatory verification.** The quality-gate and backtest states can't be skipped.
- **Reproducible.** Fixed model, seeds and data versions. Every run is logged
  and can be replayed.
- **Local only.** No cloud LLM APIs. Runs offline via Ollama / llama.cpp / Transformers.

## Architecture

```
                      ┌──────────────────────────┐
                      │  Orchestrator (local LLM) │
                      └────────────┬─────────────┘
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
 Market / Sector           Fundamental / Macro         News / Speech
         └─────────────┬───────────┴─────────────┬───────────┘
                       ▼                         ▼
               Evidence Builder        Risk & Data Quality Gate
                       └────────────┬────────────┘
                                    ▼
                       Strategy / Backtest Verification
                                    ▼
                       Human Review · Decision Log · Replay
 ───────────────────────────────────────────────────────────────────
  Deterministic MCP tools: OHLCV · indicators · sector ranking ·
  fundamentals · macro · event detection · news · STT · backtest
```

### Harness states

Each state exposes only a fixed subset of tools.

```
START → SELECT_UNIVERSE → LOAD_MARKET_DATA → CHECK_DATA_QUALITY
→ CALCULATE_TECHNICALS → RUN_SECTOR_RELATIVE_ANALYSIS
→ RUN_PATTERN_AND_REVERSAL_ANALYSIS → LOAD_FUNDAMENTALS → LOAD_MACRO_CONTEXT
→ LOAD_NEWS_AND_SPEECH_CONTEXT → BUILD_EVIDENCE → PREVIEW_STRATEGIES
→ RUN_BACKTEST → RISK_GATE → GENERATE_EXPLANATION → HUMAN_REVIEW → LOG_AND_CLOSE
```

Gate outputs: `PASS` · `PASS_WITH_WARNINGS` · `INVESTIGATE` · `ANALYSIS_UNSAFE` · `HARD_BLOCK`

### MCP tools

| Tool | Output |
|------|--------|
| `get_stock_history` / `get_index_history` | OHLCV and benchmark series |
| `calculate_indicators` | SMA, EMA, KAMA, RSI, MACD, Bollinger, ATR, Supertrend, Ichimoku |
| `detect_reversal_events` | Timestamped technical events with forward returns (1/3/5/10d) |
| `find_support_resistance` | Levels, zones, touch counts, strength |
| `rank_sector_relative_strength` | Peer returns, lag score, rank |
| `test_weekday_pattern` | Occurrences, conditional returns, significance diagnostics |
| `get_fundamental_metrics` / `calculate_fundamental_change` | Point-in-time metrics and QoQ/YoY changes |
| `get_macro_context` | FX, rates, inflation with publication metadata |
| `get_news_events` / `get_speech_features` | Timestamped event metadata, transcript topics, claims and stance |
| `run_backtest` | Performance metrics and trade/event log |
| `check_data_quality` | Warnings, blocks, stale-data and leakage flags |
| `build_evidence_bundle` | Full observed context and committed evidence set |

### Strategy variants

| Variant | Evidence |
|---------|----------|
| A | Technical only |
| B | A + sector-relative |
| C | B + fundamentals |
| D | C + macro |
| E | D + verified news/speech context |

All variants run through the same backtest (benchmarks, costs, slippage,
walk-forward) and data-quality rules.

## Harness ablation

The base model stays the same at every stage. Only the architecture changes:

```
A Raw LLM → B + Tools → C + Stateful Harness → D + Evidence / Quality Gate → E + Memory / Human Review
```

**Metrics:** numeric hallucination rate · invalid tool-call rate ·
unsupported-conclusion rate · evidence completeness · reproducibility ·
backtest correctness · tool-call count · latency

## Study universe

30 stocks across 20+ sectors, frozen for each experiment period:

`ASELS` `TUPRS` `BIMAS` `THYAO` `AKBNK` `KCHOL` `EREGL` `TCELL` `CCOLA` `MGROS`
`SASA` `TRALT` `FROTO` `ENKAI` `PGSUS` `MPARK` `EKGYO` `GUBRF` `TURSG` `ENJSA`
`TTKOM` `OYAKC` `AKSEN` `ISMEN` `ARCLK` `ECILC` `MAVI` `PETKM` `TOASO` `KRDMD`

Index membership is re-checked against each analysis date. The full sector
map is in [`docs/universe.md`](docs/universe.md).

## Data sources

| Layer | Source |
|-------|--------|
| Market | Yahoo Finance (`yfinance`) and other legally accessible sources |
| Fundamentals | Fintables (quarterly statements and ratios) |
| Disclosures | Public Disclosure Platform (KAP): official announcement timestamps |
| Macro (Turkiye) | Central Bank of the Republic of Turkiye (CBRT, EVDS), Turkish Statistical Institute (TurkStat) |
| Macro (global) | Federal Reserve (FOMC, fed funds target) |
| News / speech | Licensed or legally accessible feeds, and selected public videos (STT) |

The main fundamental analysis uses the period after the IAS 29
inflation-accounting switch, so all quarters are on a consistent basis.
Banks are handled with sector-specific metrics. Adjustments and missing-data
handling are documented in `docs/`.

## Repository layout

```
bist100-agentic-financial-harness/
├── data/            # raw & processed datasets (not committed)
├── src/bist_harness/
│   ├── data/        # loaders, cleaning, point-in-time alignment
│   ├── tools/       # deterministic analysis tools (exposed via MCP)
│   ├── harness/     # state machine, evidence store, gates, memory, replay
│   └── llm/         # local inference wrappers
├── finetune/        # dataset builder, LoRA/QLoRA configs, training
├── eval/            # test questions, metrics, ablation runs
├── notebooks/       # exploration
└── docs/            # methodology, data dictionary, universe
```

## Quick start

> 🚧 Work in progress

```bash
git clone https://github.com/yarenerisen/bist100-agentic-financial-harness.git
cd bist100-agentic-financial-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Team

- [@yarenerisen](https://github.com/yarenerisen)
- [@cissygc](https://github.com/](https://github.com/cissygc)

## Disclaimer

This is a research and educational tool. It doesn't connect to any broker,
doesn't place orders and doesn't manage money. Nothing it outputs is
investment advice or a recommendation to buy, sell or hold any security.
Historical evidence doesn't guarantee future results.

## License

[MIT](LICENSE)
