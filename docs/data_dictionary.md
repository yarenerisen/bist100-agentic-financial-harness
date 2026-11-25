# Data dictionary

Analysis date **2026-09-10**; data period 2024-01-01 – 2026-09-10 (price history from 2023-06-01 for indicator warm-up).
Nothing after 2026-09-10 is stored or used. All money amounts are in the reporting currency/unit of the source.

## 1. Raw / stored data (`bist_llm/data/`)

### `prices.parquet` — daily market data (Yahoo Finance via `download_prices.py`)
| Field | Type | Description |
|---|---|---|
| `date` | date | Trading day (Europe/Istanbul session), tz-naive |
| `ticker` | str | BIST code without `.IS` (76 stocks), `XU100`, `USDTRY`, `EURTRY` |
| `open`, `high`, `low`, `close` | float | Split- and dividend-adjusted prices (`auto_adjust=True`); FX = market quote |
| `volume` | float | Shares traded (0 for FX; some Yahoo series have zero-volume gaps, e.g. SASA, flagged by the quality gate) |

### `fundamentals.csv` — point-in-time quarterly fundamentals (from KAP report PDFs via `parse_financials.py`)
| Field | Description |
|---|---|
| `ticker` | Company (13 non-bank companies; AKBNK excluded, bank ratios differ) |
| `quarter_end` | Fiscal quarter end (2024-12-31 … 2026-06-30) |
| `disclosure_date` | **Public availability date**: board-approval date written in the report (KAP publication is the same or next day). Used for all point-in-time logic |
| `revenue` | Quarterly revenue (Hasılat) = YTD difference within the fiscal year; 2024Q4 row = full year 2024 |
| `ebitda` | Quarterly EBITDA = operating profit (Esas faaliyet kârı) + depreciation & amortisation (cash-flow adjustment) |
| `net_profit` | Quarterly net profit (Dönem kârı) |
| `net_debt` | Short-term + long-term borrowings + current portion of long-term + lease liabilities − cash & equivalents (balance sheet) |
| `ebitda_ltm` | Last-twelve-month EBITDA: sum of last 4 quarters, full year for Q4, else annualised YTD (method in `source`) |
| `source` | Provenance + approximations (TMS 29 YTD differencing, LTM method) |

`fundamentals_parsed_ytd.csv` keeps the raw year-to-date values extracted per report (audit trail).

### Macro files
| File | Fields | Notes |
|---|---|---|
| `policy_rate.csv` | `effective_date`, `policy_rate_pct`, `source` | TCMB one-week repo rate decisions (changes) |
| `fed_rate.csv` | `effective_date`, `fed_funds_lower_pct`, `fed_funds_upper_pct`, `source` | FOMC target range changes |
| `cpi.csv` | `reference_month`, `publication_date`, `cpi_mom_pct`, `cpi_yoy_pct`, `source` | TÜİK CPI (template; not loaded) |

### Unstructured context
| File | Fields |
|---|---|
| `news_events.csv` (optional) | `timestamp, ticker, topic, event_type, headline, source, url`: legally accessible headline metadata only |
| `speech/features.jsonl` (optional) | `source_id, url, speaker, channel, published, language, segment_start, segment_end, text, topics[], stance, claims[{text, numeric, verified}], tickers[]` produced by `speech_pipeline.py` |

## 2. Engineered features (computed causally inside the tools)
| Feature | Definition | Tool |
|---|---|---|
| Daily / horizon return | close_t / close_{t-h} − 1 | all |
| Sector lag score | stock return − peer-median return over `horizon` days (pp) | `rank_sector_relative_strength`, `run_backtest` |
| `sector_strong` | peer-median return > 0 and > XU100 return over the window | `rank_sector_relative_strength` |
| SMA20/50, EMA20 | simple / exponential moving averages of close | `calculate_indicators` |
| Bollinger (20, 2) | SMA20 ± 2 population std; touch = low ≤ lower (high ≥ upper) | `calculate_indicators`, `detect_reversal_events` |
| RSI14 | Wilder RSI; trough event = cross back above 30, peak = back below 70; trough *state* = lowest RSI of last 10 days and < 40 | events |
| MACD (12, 26, 9) | EMA12 − EMA26, signal EMA9 | indicators |
| ATR10, ATR% | Wilder ATR; ATR / close; high-volatility regime = ATR% above its 120-day median | events |
| KAMA (10, 2, 30) | Kaufman adaptive MA; slope = 3-day % change; flat = abs(slope) < 0.3% | indicators, events |
| Supertrend (10, 3) | ATR band trend state (bullish / bearish); flips are events | indicators, events |
| Ichimoku (9, 26, 52) | Tenkan, Kijun, Senkou A/B shifted 26 (cloud known at t); position above / inside / below | indicators, events |
| Volume ratio | volume / 20-day mean volume; high volume ≥ 1.5 | events |
| Market regime | sign of XU100 20-day return | events |
| Weekday pattern | Mon..Fri up/down string on complete 5-day weeks; tradable version = Mon–Wed signal, Thu+Fri outcome | `test_weekday_pattern` |
| Multi-day pattern | n consecutive up/down days; large down/up day ≤ / ≥ threshold % | `detect_multiday_patterns` |
| Support / resistance | pivot highs/lows (±5 bars, confirmed only after 5 bars) clustered within 1.5% | `find_support_resistance` |
| Forward return | close_{t+h} / close_t − 1 for h ∈ {1, 3, 5, 10, 20}; **excluded when t+h is after 2026-09-10** | events, backtests, fundamentals |
| EBITDA margin | EBITDA / revenue (%) | fundamentals |
| Net Debt / EBITDA | net_debt / ebitda_ltm | fundamentals |
| Improving quarter | EBITDA margin up QoQ and Net Debt/EBITDA not up | `calculate_fundamental_change` |
| Abnormal return | stock return − XU100 return (and − sector peer median) after disclosure | `calculate_fundamental_change` |
| FX relationship | weekly / monthly correlation and beta of stock returns on USD/TRY returns | `get_macro_context` |

## 3. Backtest conventions
Signal at close *t* → enter at **open t+1** → exit at close *t+hold*. Commission 0.2% round trip + slippage 0.05% per side.
Non-overlapping trades per stock; equal weight across stocks in position. Benchmarks: equal-weight buy-and-hold of the same
stocks and XU100. Metrics: cumulative and annualised return, Sharpe (daily, √252), maximum drawdown, win rate, number of
trades (warning below 20), excess return vs benchmarks. Strategy variants A–D use the same rules with an in-sample /
out-of-sample split.

## 4. Analytical outputs
| Field | Values |
|---|---|
| Analytical state | `HOLD, WATCH, INVESTIGATE, POTENTIAL_CATCH_UP_CANDIDATE, TECHNICAL_REVERSAL_CANDIDATE, FUNDAMENTAL_IMPROVEMENT_CANDIDATE, REJECT_SIGNAL, ANALYSIS_UNSAFE` |
| Quality gate | `PASS, PASS_WITH_WARNINGS, INVESTIGATE, ANALYSIS_UNSAFE, HARD_BLOCK` |
| Tool error codes | `INVALID_CALL, UNKNOWN_TOOL, UNKNOWN_ENTITY, LOOKAHEAD_BLOCKED, TICKER_EXCLUDED, TOOL_ERROR` |
| Decision log (`logs/*.json`) | question, mode, states visited, tool calls + outputs, quality gate, evidence bundle (observed context vs committed evidence), answer, validation, human review, latency |
| Episodic memory (`logs/memory.sqlite`) | id, time, question, tickers, tools, state, gate, committed evidence, human decision and reason |
