# BIST-Analyst — Technical Report

_Numbers come from `results/eval_*.json`, `release/training_report.json` and `release/execution_traces.json`.
**v5 is the released adapter.** Earlier adapters on the same pipeline: v1 (first dataset version), v3 (no `DECISION_BASIS`
line, compact weekday patterns), v4 (same 1,560 training rows as v3 with `DECISION_BASIS`); v5 = v4 data + 150 rows for
under-represented labels (section 5)._

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
| train | 1,710 | 1,560 base rows + 150 rows for under-represented labels |
| validation | 109 | |
| test | 143 | held-out tickers (PGSUS, TOASO, KRDMD, TTKOM, ENJSA, CIMSA, ULKER, TAVHL) and unseen question templates |
| extended test | 390 | same held-out tickers, different seed, no question shared with any other split (section 8.5) |

- Each record: system prompt → question → tool call(s) → **real tool output** → answer
  (`DECISION_BASIS / ANALYTICAL_STATE / EVIDENCE / LIMITATIONS / NOTE`). `DECISION_BASIS` names the tool-output fields
  that decide the state, so the state is generated after them.
- Scenario mix: sector catch-up, weekday and multi-day patterns, technical reversals, quarterly fundamentals,
  rule/safety questions, support/resistance, strategy variants, event calendar, speech layer, index history, and edge
  cases (future dates, unknown tickers).
- Bilingual (Turkish/English); dates in both `YYYY-MM-DD` and `DD.MM.YYYY`.
- Answers are rule-based templates filled **only with numbers copied from the tool output**; states follow
  `training/labeling_rules.md`. The validator (`harness/validate.py`) reports **0 violations** over all 2,352 conversations (train, validation, test and extended test).
- Class balancing keeps rare states (catch-up, reversal and fundamental candidates) represented.
- Augmentation for v5 (`python training/generate_dataset.py --augment`): the error analysis of v4 on the extended test
  set showed labels the model had rarely seen, so 150 training rows were added on training tickers only: strategy-variant
  `WATCH` (1 → 41 rows), weekday-pattern `WATCH` (17 → 57), support/resistance `TECHNICAL_REVERSAL_CANDIDATE` (17 → 47) and
  undated forecast requests answered with `ANALYSIS_UNSAFE` (40 new rows). Validation and test files are unchanged.
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
| Epochs / time / hardware | 1 epoch (107 optimizer steps), 151 min, one RTX 4070 Laptop GPU (8 GB) |
| Loss curve | training loss 0.727 (step 5) → 0.128 (20) → 0.018 (40) → 0.008 (60) → 0.002 (105); validation loss 0.1028 (step 20) → 0.0154 (40) → 0.0059 (60) → 0.0043 (80) → **0.0039** (107, final) |
| Adapter (v5) | `adapter_model.safetensors`, SHA-256 `e2b2f84c71d1035b4a82904a53c49629d1df2d532ba67600f79c2f6eb829a104` |

The validation loss is very low because answers are templates; it shows that the format is learned, not that the
decisions generalise. Decision quality is measured on held-out tickers in section 8.

## 7. Running the model locally
- Transformers (used for every number in this report): `python -m harness.run_agent --backend hf --model <adapter> --load-4bit "<question>"`
  loads Qwen/Qwen3.5-4B in 4-bit with the LoRA adapter; it needs ~6 GB of GPU memory and no internet once the base model is cached.
- Optional: a merged Q4_K_M GGUF (~3 GB) with llama.cpp (`--backend llamacpp`) or Ollama (`--backend ollama`); the raw
  backends send the exact training-format prompt.
- The harness refuses non-localhost model endpoints, so no cloud AI API is called.

## 8. Evaluation
All 143 test rows (held-out tickers), full harness (level E), temperature 0. "Unsupported conclusion" = the final state
differs from the state the labelling rules give for the same tool outputs.

### 8.1 Base vs fine-tuned (full harness, level E, 143 test rows)
| Metric | Base Qwen3.5-4B | v1 | v3 | v4 | **v5 (released)** |
|---|---|---|---|---|---|
| Numerical hallucination rate | 17.5% | 9.8% | 0.7% | 1.4% | 1.4% |
| Invalid tool-call rate | 4.2% | 0.7% | 0.0% | 0.7% | 1.4% |
| Correct tool sequence | 10.5% | 80.4% | 88.8% | 93.7% | **95.1%** |
| Correct tool arguments | 8.4% | 78.3% | 84.6% | 88.8% | **93.0%** |
| Look-ahead call rate (blocked by the registry) | 0.0% | 1.4% | 2.1% | 4.2% | 1.4% |
| Unsupported conclusion rate | 79.0% | 33.6% | 27.3% | 12.6% | 13.3% |
| Rule violations (raw answer) | 28.7% | 11.2% | 2.8% | 1.4% | 1.4% |
| Final answer valid | 84.6% | 92.3% | 99.3% | 98.6% | 98.6% |
| Evidence completeness | 99.1% | 100% | 100% | 100% | 100% |
| Reproducibility | 100% | 100% | 100% | 100% | 100% |
| Mean latency | 7.8 s | 10.0 s | 10.0 s | 30.1 s* | 30.3 s* |

\* v4 and v5 were evaluated with Transformers 4-bit on the 8 GB laptop GPU (no `causal_conv1d` kernel); the others ran on vLLM.

Unsupported-conclusion rate per scenario (v3 → v4 → v5): sector catch-up 36% → 16% → 28%, weekday patterns 12% → 8% → 8%,
technical reversals 24% → 0% → 0%, fundamentals 70% → 10% → 10%, rule/safety 0% → 0% → 0%, future-date requests
40% → 60% → 0%, strategy variants 40% → 100% → 80% (5 rows), support/resistance 25% → 25% → 50% (8 rows). On this
set v4 and v5 are within one or two rows of each other; the larger extended set (8.5) separates them.

Overlap: 10 of the 143 test questions also occur verbatim in the training data (5 unknown-ticker, 3 index-history and
2 speech-layer questions, which have only a few templates). Without them (133 rows): v4 unsupported conclusions 12.8%,
tool sequence 93.2%, hallucination 1.5%; v5 14.3%, 94.7%, 1.5%. The extended test set (8.5) shares no question with the
training data.

### 8.2 Raw base model without tools or data (5 reference scenario questions)
_Measured with the earlier base model Qwen3-8B (`training/baseline/`); the Qwen3.5-4B raw-LLM numbers are ablation level A in 8.3._
| Metric | Value |
|---|---|
| Answers with invented numbers | 100% (17.2 per answer) |
| Price target given | 20% (scenario 5: "13.25–13.75 TL") |
| Required format / allowed state | 0% |

### 8.3 Harness ablation A–E (first 40 test rows)
A = raw LLM, no tools; B = tools; C = + state machine; D = + validator and retry; E = full harness (+ risk gate, data-quality gate).
<!-- ABLATION_TABLE -->

Without tools neither model can answer in the required form. Fine-tuning gives most of the gain; the validator and the
gates turn the remaining invalid answers into labelled fallbacks instead of wrong answers.

### 8.4 Real vs synthetic fundamentals (v5, level E, 143 test rows)
| Fundamentals source | Rows | Hallucination | Tool sequence | Unsupported conclusion |
|---|---|---|---|---|
| none (non-fundamental questions) | 93 | 0.0% | 98.9% | 6.5% |
| real KAP | 25 | 0.0% | 100% | 28.0% |
| synthetic | 25 | 8.0% | 76.0% | 24.0% |

### 8.5 Extended held-out test set (390 rows, level E)
`training/test_ext.jsonl` (`python training/generate_dataset.py --ext-test`): 390 new conversations on the held-out
tickers with a different seed, none of whose questions appears in train, validation or test; more rows for the small
scenarios (30 strategy-variant and 30 future-date questions).

| Metric | v4 | **v5** |
|---|---|---|
| Numerical hallucination rate | 1.3% | 1.5% |
| Invalid tool-call rate | 1.3% | 0.3% |
| Correct tool sequence / arguments | 89.5% / 85.9% | **95.9% / 93.8%** |
| Look-ahead call rate (blocked) | 5.1% | **0.3%** |
| Unsupported conclusion rate | 20.5% | **16.7%** |
| Rule violations (raw answer) | 1.3% | 1.5% |
| Final answer valid | 98.7% | 98.5% |

| Scenario | Rows | Unsupported conclusion v4 → v5 | Tool sequence v4 → v5 |
|---|---|---|---|
| Sector catch-up | 60 | 26.7% → 28.3% | 81.7% → 75.0% |
| Weekday patterns | 50 | 24.0% → 14.0% | 98.0% → 100% |
| Technical reversals | 50 | 4.0% → 2.0% | 100% |
| Fundamentals (with / without KAP data) | 50 / 15 | 8.0% → 6.0% / 13.3% → 13.3% | 100% |
| Rule / safety | 37 | 0.0% → 0.0% | 100% |
| Support / resistance | 25 | 24.0% → 40.0% | 100% → 96.0% |
| Strategy variants | 30 | 100% → 83.3% | 100% |
| Future-date requests | 30 | 26.7% → **0.0%** | 3.3% → 100% |
| Event calendar, price and index history, speech | 43 | 0.0% → 0.0% | 100% |

Error analysis.
- v4's strategy-variant errors came from label imbalance (1 `WATCH` row among 60 training rows); with 41 `WATCH` rows
  v5 still answers `REJECT_SIGNAL` in 22 of 30 cases. Unlike the scenarios that improved, the strategy-variant and
  support/resistance `DECISION_BASIS` lines are verbal ("no variant with enough trades beat buy-and-hold") instead of
  `field = value` pairs, so the state is written before any number is; the evidence lines that follow are correct.
- In 6–9 sector questions v5 passes invented tickers inside the `sector` argument (e.g. `ASLS`, `CEKUR`); the registry
  returns `no data`, and the model answers `ANALYSIS_UNSAFE`. No number is invented, but the question is not answered.
- Undated forecast requests ("predict next month") are now refused without tool calls (30/30).

### 8.6 Live-style check (20 hand-written questions, v5)
`release/live_check_questions.json`: 20 questions written by hand in the style of the sample questions but with other
stocks, sectors, dates and wordings (4 per reference scenario, Turkish and English), run through the full harness and
checked by hand against the tool outputs (`release/live_check_traces.json`).

| Check | Result |
|---|---|
| Invented numbers (validator) | 0 / 20 |
| Correct tools in the correct order | 20 / 20 |
| Correct arguments (tickers, dates, pattern, window) | 19 / 20 (a "last 10 trading days" question used a 30-day window) |
| State consistent with the labelling rules for the tool outputs | 20 / 20 |
| Investment advice or price target | 0 / 20 |
| Harness fallback | 0 / 20 |

Weakness: for rule/safety questions whose claim is about something other than the exchange rate ("rates will fall, so
banks will rally", "oil is falling, so THYAO will rise") the state is correct and no advice is given, but the decision
basis is still the USD/TRY correlation; the model does not say that oil prices are not in the data, and it reports the
CBRT rate path without linking it to the claim. All rule/safety training rows use an exchange-rate claim.

## 9. Research findings (deterministic tools, `docs/results/`)
- **Sector catch-up:** average 20-day hit rate 45%, false positives 55%; laggards mostly did not mean-revert.
- **Weekday patterns:** 960 tests → 12 significant at p < 0.05, 1 after Bonferroni, **0** significant out of sample.
- **Technical reversals:** a lower-Bollinger touch led to a 5-day rise 57% of the time, yet trading it returned −3.9%
  against +117% buy-and-hold (prediction ≠ profitable strategy).
- **Fundamentals:** improving quarters +0.98 pp vs −0.48 pp 20-day abnormal return. Association, not causation.
- **Macro:** weekly correlations with USD/TRY are weak or negative (telecom, retail); "exporter rises with the dollar" is not supported.
- **Strategy variants:** each added filter cut trades (116 → 8 in-sample); out-of-sample gains rest on very few trades.

## 10. Reference scenario questions and live question
Execution traces of the released v5 adapter: `release/execution_traces.json`, for the Turkish and the English version of each reference scenario question. Each trace records the model and adapter SHA-256,
system and user prompts, every model turn and tool call with timestamps, tool outputs, answer, `answer_source`,
temperature and seed.

| Scenario | State (TR / EN) | Tools | Answer source |
|---|---|---|---|
| 1 Sector catch-up | WATCH / WATCH (lag −3.25 pp < 5.0 pp threshold) | sector strength, fundamentals, macro, backtest | model |
| 2 Weekday pattern | REJECT_SIGNAL / REJECT_SIGNAL (in-sample p = 0.4059, out-of-sample p = 0.735) | weekday pattern ×2 | model |
| 3 Technical reversal | WATCH / WATCH (72.2% success, last event 51 trading days ago) | indicators, reversal events | model |
| 4 Fundamentals | FUNDAMENTAL_IMPROVEMENT_CANDIDATE / same | fundamentals, fundamental change | model |
| 5 Rule safety | INVESTIGATE / INVESTIGATE (weekly corr. with USD/TRY = −0.272) | macro context | model |

All 10 states agree with the labelling rules for the tool outputs (v4 gave the same 10 states; v3 got 6 of 10).

Live question (offline, from `bist_llm/`):
```bash
python -m release.make_traces --backend hf --model <adapter folder> --load-4bit \
    --question "<question>" --out release/live_trace.json
```

## 11. Limitations
- Fundamentals cover 13 companies from 2024Q4; quarterly values are YTD differences under TMS 29 (approximate);
  CCOLA 2024Q4 is unreadable; banks are not analysed with EBITDA ratios.
- No licensed news feed or approved videos are loaded: the event layer only holds official KAP/CBRT/FOMC events, and
  strategy variant E is unavailable.
- Yahoo volume data has gaps (e.g. SASA); several sectors have only two peers.
- Event studies are in-sample with overlapping windows; combined-condition groups are small.
- Decision quality: about one answer in six on the extended test set ends in a state that the labelling rules do not
  give for the same tool outputs (section 8.5); strategy-variant and support/resistance questions are the weakest.
- Sector questions sometimes contain invented tickers in the tool arguments; the registry rejects them, so no numbers are
  invented, but the answer becomes `ANALYSIS_UNSAFE`.
- Rule/safety questions about drivers other than the exchange rate (rates, oil, gold) are answered with the USD/TRY
  relationship; oil and commodity prices are not in the data.
- Test questions are generated from templates similar to the training data (with held-out tickers and unseen
  templates); the 20 hand-written questions in 8.6 are the closest measure of unseen phrasings.
- The model can still fail on very unusual phrasings; failed validations end in a labelled harness fallback, never in an unlabelled answer.

## Appendix A — Qwen2.5-3B
Trained locally on an earlier dataset version (1,440 rows; RTX 4070 Laptop, 8 GB, bf16, Liger, 2 epochs, 3 h 48 min);
validation loss 0.136 → 0.0058. It was not evaluated on the current test set and is not part of the release.
