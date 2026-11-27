# BIST-Analyst — Technical Report

_Numbers come from `results/eval_*.json`, `runs/qwen35-4b-lora-v4/training_report.json` and `release/execution_traces.json`.
"v4" is the released adapter; "v3" and "v1" are earlier adapters trained on the same pipeline (v3: same questions and
states, without the `DECISION_BASIS` line and with compact weekday patterns; v1: the first dataset version)._

## 1. Summary
BIST-Analyst is a fine-tuned, locally run open-source LLM that answers research questions about BIST stocks by
orchestrating deterministic Python tools. The model interprets questions, calls tools with explicit parameters and date
ranges, and writes evidence-based explanations ending in one analytical state from a finite action space. It never
computes numbers itself, never gives investment advice, and uses no data published after the analysis date
(**2026-09-10**).

> The LLM interprets, Python tools compute, the harness controls, backtests verify, evidence is logged, a human decides.

## 2. Base model and why
| Item | Value |
|---|---|
| Base model | **Qwen/Qwen3.5-4B** (Apache-2.0, released 2026-03), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; the text model is trained (the vision encoder is not loaded) |
| Why | newest official small Qwen generation (Qwen3.8 has no open model below 27B); strong instruction following and JSON/tool-call ability in the recommended 1–8B range; good Turkish and English; runs as a ~3 GB Q4_K_M GGUF on the 6 GB laptop GPU used for the live demo, and fine-tunes on one 24 GB GPU |
| Thinking mode | disabled; every assistant turn starts with an empty `<think></think>` block in training and inference (`training/prompts.py`) so multi-turn tool conversations are formatted identically |
| Additional comparison | Qwen2.5-3B-Instruct fine-tuned on the same data on a local RTX 4070 Laptop GPU (Appendix A) |

## 3. System architecture
- **16 deterministic tools** (`bist_llm/tools/registry.py`): stock/index history, sector relative strength, sector
  catch-up event study and backtest, weekday and multi-day patterns, indicators (SMA/EMA/RSI/MACD/Bollinger/ATR/KAMA/
  Supertrend/Ichimoku), 11 technical event families, support/resistance, strategy variants A–E, point-in-time
  fundamentals and post-disclosure abnormal returns, macro context, official event calendar, speech features,
  data-quality gate.
- **Registry guarantees:** argument validation, `LOOKAHEAD_BLOCKED` for any date after 2026-09-10, compact model-facing output.
- **MCP:** `bist_llm/mcp_server.py` serves the same 16 tools plus `build_evidence_bundle` over MCP (stdio), with the same
  validation and look-ahead block; the harness calls the registry in-process.
- **Stateful harness** (`bist_llm/harness/run_agent.py`, `STATES`): the 16 states of the project brief, START →
  SELECT_UNIVERSE → LOAD_MARKET_DATA → CHECK_DATA_QUALITY → CALCULATE_TECHNICALS → RUN_SECTOR_RELATIVE_ANALYSIS →
  RUN_PATTERN_AND_REVERSAL_ANALYSIS → LOAD_FUNDAMENTALS → LOAD_MACRO_CONTEXT → LOAD_NEWS_AND_SPEECH_CONTEXT →
  BUILD_EVIDENCE → PREVIEW_STRATEGIES → RUN_BACKTEST → RISK_GATE → GENERATE_EXPLANATION → HUMAN_REVIEW → LOG_AND_CLOSE.
  Each state exposes only its own tools and is owned by one agent role; states only move forward (a call to a closed
  state is rejected). The harness runs the data-quality gate and the risk gate, which holds back a candidate state
  without a historical verification output. The answer is validated (one retry, then a labelled fallback). One
  fine-tuned model plays the reasoning roles; data and calculations stay deterministic tools. Every run is logged with timestamps,
  prompts, model turns, tool calls and outputs, decoding parameters (temperature 0, seed 0) and `answer_source`.
- **Knowledge base** (`release/knowledge_base.json`): deterministic, tool-based retrieval over timestamped sources —
  Yahoo Finance prices, 91 parsed KAP financial reports (with PDF SHA-256), CBRT/FOMC decisions, sector map.

## 4. Data
- Prices: 76 BIST stocks + XU100 + USD/TRY + EUR/TRY, 2023-06-01 – 2026-09-10 (adjusted).
- Fundamentals: KAP financial report PDFs for 13 non-bank companies, reports 2024Q4 – 2026Q2, parsed by
  `data/parse_financials.py`; disclosure date = board-approval date; quarterly values are YTD differences (TMS 29
  restatement makes them approximate); AKBNK excluded (EBITDA ratios not meaningful for banks).
- Macro: CBRT policy-rate and FOMC decisions, TÜİK CPI (monthly, aligned by bulletin publication date), all checked
  against official sources (`data/MACRO_SOURCES.md`). A licensed news feed is not available.
- Data dictionary: `docs/data_dictionary.md`.

## 5. Fine-tuning data
| Split | Conversations | Notes |
|---|---|---|
| train | 1,560 | |
| validation | 109 | |
| test | 143 | held-out tickers (PGSUS, TOASO, KRDMD, TTKOM, ENJSA, CIMSA, ULKER, TAVHL) and unseen question templates |

- Each record: system prompt → question → tool call(s) → **real tool output** → answer
  (`DECISION_BASIS / ANALYTICAL_STATE / EVIDENCE / LIMITATIONS / NOTE`). `DECISION_BASIS` names the tool-output fields
  that decide the state, so the state is generated after them.
- Scenario mix: sector catch-up, weekday and multi-day patterns, technical reversals, quarterly fundamentals,
  rule/safety questions, support/resistance, strategy variants, event calendar, speech layer, index history, and edge
  cases (future dates, unknown tickers).
- Bilingual (Turkish/English); dates in both `YYYY-MM-DD` and `DD.MM.YYYY`.
- Answers are rule-based templates filled **only with numbers copied from the tool output**; states follow
  `training/labeling_rules.md`. The validator (`harness/validate.py`) reports **0 violations** over all 1,812 conversations.
- Class balancing keeps rare states (catch-up, reversal and fundamental candidates) represented.
- Synthetic fundamentals are used only for training rows about companies without KAP data; evaluation reports real
  and synthetic fundamentals separately.

## 6. Training method and settings
| Setting | Value |
|---|---|
| Method | QLoRA (4-bit NF4, double quantisation) |
| LoRA | r = 16, alpha = 32, dropout 0.05, all attention and MLP projections |
| Optimiser / LR | paged AdamW 8-bit, 2e-4, cosine, 5% warm-up |
| Batch | 1 × gradient accumulation 16 |
| Max length | 6,656 tokens (no training example exceeds it) |
| Loss | assistant tokens only (tool calls + final answers) |
| Memory | Liger fused linear cross-entropy (the 152k-vocabulary logits are never materialised) |
| Precision | bf16 compute; the frozen tied embedding / lm_head (248k × 2,560) is kept in bf16 after k-bit preparation so training fits in 8 GB |
| Epochs / time / hardware | 1 epoch (98 optimizer steps), 141 min, one RTX 4070 Laptop GPU (8 GB) |
| Loss curve | training loss 0.726 (step 5) → 0.103 (20) → 0.017 (40) → 0.007 (60) → 0.004 (95); validation loss 0.0948 (step 20) → 0.0129 (40) → 0.0074 (60) → 0.0052 (80) → **0.0048** (98, final) |
| Adapter | `adapter_model.safetensors`, SHA-256 `dfd6fcbb94ec7ba8a47e94c22815643ed87d33ce29ef1a9baba88f66dcca478f` |

## 7. Running the model locally
- Transformers: `python -m harness.run_agent --backend hf --model <adapter> --load-4bit "<question>"`
- Fully offline: Q4_K_M GGUF (~3 GB) with llama.cpp (`--backend llamacpp`) or Ollama (`--backend ollama`). The raw
  backends send the exact training-format prompt.
- The harness refuses non-localhost model endpoints, so no cloud AI API is called.

## 8. Evaluation
All 143 test rows (held-out tickers), full harness (level E), temperature 0. "Unsupported conclusion" = the final state
differs from the state the labelling rules give for the same tool outputs.

### 8.1 Base vs fine-tuned (full harness, level E, 143 test rows)
| Metric | Base Qwen3.5-4B | v1 | v3 | **v4 (released)** |
|---|---|---|---|---|
| Numerical hallucination rate | 17.5% | 9.8% | 0.7% | 1.4% |
| Invalid tool-call rate | 4.2% | 0.7% | 0.0% | 0.7% |
| Correct tool sequence | 10.5% | 80.4% | 88.8% | **93.7%** |
| Correct tool arguments | 8.4% | 78.3% | 84.6% | **88.8%** |
| Look-ahead call rate (blocked by the registry) | 0.0% | 1.4% | 2.1% | 4.2% |
| Unsupported conclusion rate | 79.0% | 33.6% | 27.3% | **12.6%** |
| Rule violations (raw answer) | 28.7% | 11.2% | 2.8% | **1.4%** |
| Final answer valid | 84.6% | 92.3% | 99.3% | 98.6% |
| Evidence completeness | 99.1% | 100% | 100% | 100% |
| Reproducibility | 100% | 100% | 100% | 100% |
| Mean latency | 7.8 s | 10.0 s | 10.0 s | 30.1 s* |

\* v4 was evaluated with Transformers 4-bit on the 8 GB laptop GPU (no `causal_conv1d` kernel); the others ran on vLLM.

Unsupported-conclusion rate per scenario (v3 → v4): sector catch-up 36% → 16%, weekday patterns 12% → 8%, technical
reversals 24% → 0%, fundamentals 70% → 10%, event calendar 20% → 0%, rule/safety 0% → 0%. Two small groups got worse:
strategy variants (5 rows, 40% → 100%: the model writes `REJECT_SIGNAL` although the tool output shows a variant that beat
buy-and-hold with ≥ 10 trades) and future-date requests (5 rows, 40% → 60%: undated "predict next month" questions are
not recognised as look-ahead). Both are measured on the larger set in 8.5.

### 8.2 Raw base model without tools or data (5 reference scenario questions)
_Measured with the earlier base model Qwen3-8B (`training/baseline/`); the Qwen3.5-4B raw-LLM numbers are ablation level A in 8.3._
| Metric | Value |
|---|---|
| Answers with invented numbers | 100% (17.2 per answer) |
| Price target given | 20% (scenario 5: "13.25–13.75 TL") |
| Required format / allowed state | 0% |

### 8.3 Harness ablation A–E (first 40 test rows, v3 adapter)
A = raw LLM, no tools; B = tools; C = + state machine; D = + validator and retry; E = full harness (+ risk gate, data-quality gate).
| Level | Model | Hallucination | Tool sequence | Unsupported conclusion | Final answer valid |
|---|---|---|---|---|---|
| A | base / v3 | 70.0% / 10.0% | 2.5% / 2.5% | 100% / 100% | 0% / 0% |
| B | base / v3 | 30.0% / 0.0% | 12.5% / 85.0% | 80.0% / 20.0% | 65.0% / 97.5% |
| C | base / v3 | 32.5% / 0.0% | 15.0% / 85.0% | 80.0% / 20.0% | 62.5% / 97.5% |
| D | base / v3 | 20.0% / 0.0% | 15.0% / 85.0% | 80.0% / 20.0% | 90.0% / 100% |

Without tools neither model can answer in the required form. Fine-tuning gives most of the gain; the validator and the
gates turn the remaining invalid answers into labelled fallbacks instead of wrong answers.

### 8.4 Real vs synthetic fundamentals (v4, level E)
| Fundamentals source | Rows | Hallucination | Tool sequence | Unsupported conclusion |
|---|---|---|---|---|
| none (non-fundamental questions) | 93 | 1.1% | 93.5% | 7.5% |
| real KAP | 25 | 0.0% | 100% | 32.0% |
| synthetic | 25 | 4.0% | 88.0% | 12.0% |

### 8.5 Extended held-out test set (v4, level E, 390 rows)
`training/test_ext.jsonl` (`python training/generate_dataset.py --ext-test`): 390 new conversations on the held-out
tickers with a different seed, none of whose questions appears in train, validation or test; more rows for the small
scenarios (30 strategy-variant and 30 future-date questions).

| Metric | 143-row test | 390-row extended test |
|---|---|---|
| Numerical hallucination rate | 1.4% | 1.3% |
| Invalid tool-call rate | 0.7% | 1.3% |
| Correct tool sequence / arguments | 93.7% / 88.8% | 89.5% / 85.9% |
| Look-ahead call rate (blocked) | 4.2% | 5.1% |
| Unsupported conclusion rate | 12.6% | 20.5% |
| Rule violations (raw answer) | 1.4% | 1.3% |
| Final answer valid | 98.6% | 98.7% |

| Scenario | Rows | Unsupported conclusion | Tool sequence |
|---|---|---|---|
| Sector catch-up | 60 | 26.7% | 81.7% |
| Weekday patterns | 50 | 24.0% | 98.0% |
| Technical reversals | 50 | 4.0% | 100% |
| Fundamentals (with / without KAP data) | 50 / 15 | 8.0% / 13.3% | 100% |
| Rule / safety | 37 | 0.0% | 100% |
| Support / resistance | 25 | 24.0% | 100% |
| Strategy variants | 30 | 100% | 100% |
| Future-date requests | 30 | 26.7% | 3.3% |
| Event calendar, price and index history, speech | 43 | 0.0% | 100% |

Error analysis: 30 of the 80 wrong states are strategy-variant questions where the tool output shows variant A beating
buy-and-hold out of sample with ≥ 10 trades (`WATCH`) but the model answers `REJECT_SIGNAL`; the training data has only
1 `WATCH` row among 60 strategy-variant rows, so the model learned the majority label. The same imbalance explains
8 weekday-pattern rows (`WATCH` → `REJECT_SIGNAL`; 17 of 230 training rows are `WATCH`). Future-date requests without
an explicit date ("predict next month") are often answered with a tool call instead of `ANALYSIS_UNSAFE`; dated
requests are blocked by the registry (`LOOKAHEAD_BLOCKED`).

## 9. Research findings (deterministic tools, `docs/results/`)
- **Sector catch-up:** average 20-day hit rate 45%, false positives 55%; laggards mostly did not mean-revert.
- **Weekday patterns:** 960 tests → 12 significant at p < 0.05, 1 after Bonferroni, **0** significant out of sample.
- **Technical reversals:** a lower-Bollinger touch led to a 5-day rise 57% of the time, yet trading it returned −3.9%
  against +117% buy-and-hold (prediction ≠ profitable strategy).
- **Fundamentals:** improving quarters +0.98 pp vs −0.48 pp 20-day abnormal return. Association, not causation.
- **Macro:** weekly correlations with USD/TRY are weak or negative (telecom, retail); "exporter rises with the dollar" is not supported.
- **Strategy variants:** each added filter cut trades (116 → 8 in-sample); out-of-sample gains rest on very few trades.

## 10. Reference scenario questions and live question
Execution traces: `release/execution_traces.json`, for the Turkish and the English version of each reference scenario question. Each trace records the model and adapter SHA-256,
system and user prompts, every model turn and tool call with timestamps, tool outputs, answer, `answer_source`,
temperature and seed.

| Scenario | State (TR / EN) | Tools | Answer source |
|---|---|---|---|
| 1 Sector catch-up | WATCH / WATCH (lag −3.25 pp < 5.0 pp threshold) | sector strength, fundamentals, macro, backtest | model |
| 2 Weekday pattern | REJECT_SIGNAL / REJECT_SIGNAL (in-sample p = 0.4059, out-of-sample p = 0.735) | weekday pattern ×2 | model |
| 3 Technical reversal | WATCH / WATCH (72.2% success, last event 51 trading days ago) | indicators, reversal events | model |
| 4 Fundamentals | FUNDAMENTAL_IMPROVEMENT_CANDIDATE / same | fundamentals, fundamental change | model |
| 5 Rule safety | INVESTIGATE / INVESTIGATE (weekly corr. with USD/TRY = −0.272) | macro context | model |

## 11. Limitations
- Fundamentals cover 13 companies from 2024Q4; quarterly values are YTD differences under TMS 29 (approximate);
  CCOLA 2024Q4 is unreadable; banks are not analysed with EBITDA ratios.
- No licensed news feed or approved videos are loaded: the event layer only holds official KAP/CBRT/FOMC events, and
  strategy variant E is unavailable.
- Yahoo volume data has gaps (e.g. SASA); several sectors have only two peers.
- Event studies are in-sample with overlapping windows; combined-condition groups are small.
- The model can still fail on very unusual phrasings; failed validations end in a labelled harness fallback, never in an unlabelled answer.

## Appendix A — Qwen2.5-3B
Trained locally on an earlier dataset version (1,440 rows; RTX 4070 Laptop, 8 GB, bf16, Liger, 2 epochs, 3 h 48 min);
validation loss 0.136 → 0.0058. It was not evaluated on the current test set and is not part of the release.
