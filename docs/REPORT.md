# BIST-Analyst — Technical Report

_Draft. Sections marked **[TBD: Kaggle run]** are filled from `results/eval_*.json`, `qwen3-8b-lora/training_report.json`
and `release/execution_traces.json` once the Qwen3-8B fine-tuning run finishes._

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
| Base model | **Qwen/Qwen3-8B** (Apache-2.0), revision **[TBD: HF commit in manifest.json]** |
| Why | strong instruction following and JSON/tool-call ability in the 1–8B range; good Turkish and English; 8B fits 4-bit QLoRA on a 16 GB T4 and runs as a ~5 GB Q4_K_M GGUF on an 8 GB laptop GPU |
| Thinking mode | disabled; every assistant turn starts with an empty `<think></think>` block in training and inference (`training/prompts.py`) so multi-turn tool conversations are formatted identically |
| Additional comparison | Qwen2.5-3B-Instruct fine-tuned on the same data on a local RTX 4070 Laptop GPU (Appendix A) |

## 3. System architecture
- **16 deterministic tools** (`bist_llm/tools/registry.py`): stock/index history, sector relative strength, sector
  catch-up event study and backtest, weekday and multi-day patterns, indicators (SMA/EMA/RSI/MACD/Bollinger/ATR/KAMA/
  Supertrend/Ichimoku), 11 technical event families, support/resistance, strategy variants A–E, point-in-time
  fundamentals and post-disclosure abnormal returns, macro context, official event calendar, speech features,
  data-quality gate.
- **Registry guarantees:** argument validation, `LOOKAHEAD_BLOCKED` for any date after 2026-09-10, compact model-facing output.
- **Stateful harness** (`bist_llm/harness/run_agent.py`): PLAN_TOOL_CALL → VALIDATE_CALL → EXECUTE_TOOL → QUALITY_GATE →
  VALIDATE_ANSWER (one retry) → BUILD_EVIDENCE → HUMAN_REVIEW → LOG_AND_CLOSE. Every run is logged with timestamps,
  prompts, model turns, tool calls and outputs, decoding parameters (temperature 0, seed 0) and `answer_source`.
- **Knowledge base** (`release/knowledge_base.json`): deterministic, tool-based retrieval over timestamped sources —
  Yahoo Finance prices, 91 parsed KAP financial reports (with PDF SHA-256), CBRT/FOMC decisions, sector map.

## 4. Data
- Prices: 76 BIST stocks + XU100 + USD/TRY + EUR/TRY, 2023-06-01 – 2026-09-10 (adjusted).
- Fundamentals: KAP financial report PDFs for 13 non-bank companies, reports 2024Q4 – 2026Q2, parsed by
  `data/parse_financials.py`; disclosure date = board-approval date; quarterly values are YTD differences (TMS 29
  restatement makes them approximate); AKBNK excluded (EBITDA ratios not meaningful for banks).
- Macro: CBRT policy-rate and FOMC decisions. TurkStat CPI and a licensed news feed are not available.
- Data dictionary: `docs/data_dictionary.md`.

## 5. Fine-tuning data
| Split | Conversations | Notes |
|---|---|---|
| train | 1,440 | |
| validation | 109 | |
| test | 143 | held-out tickers (PGSUS, TOASO, KRDMD, TTKOM, ENJSA, CIMSA, ULKER, TAVHL) and unseen question templates |

- Each record: system prompt → question → tool call(s) → **real tool output** → answer
  (`ANALYTICAL_STATE / EVIDENCE / LIMITATIONS / NOTE`).
- Scenario mix: sector catch-up, weekday and multi-day patterns, technical reversals, quarterly fundamentals,
  rule/safety questions, support/resistance, strategy variants, event calendar, speech layer, index history, and edge
  cases (future dates, unknown tickers).
- Bilingual (Turkish/English); dates in both `YYYY-MM-DD` and `DD.MM.YYYY`.
- Answers are rule-based templates filled **only with numbers copied from the tool output**; states follow
  `training/labeling_rules.md`. The validator (`harness/validate.py`) reports **0 violations** over all 1,692 conversations.
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
| Max length | 6,144 tokens (no training example exceeds it) |
| Loss | assistant tokens only (tool calls + final answers) |
| Memory | Liger fused linear cross-entropy (the 152k-vocabulary logits are never materialised) |
| Epochs / time / hardware | **[TBD: Kaggle run]** (1 epoch on 2× T4) |
| Loss curve | **[TBD: training_report.json]** |

## 7. Running the model locally
- Transformers: `python -m harness.run_agent --backend hf --model <adapter> --load-4bit "<question>"`
- Fully offline: Q4_K_M GGUF (~5 GB) with llama.cpp (`--backend llamacpp`) or Ollama (`--backend ollama`). The raw
  backends send the exact training-format prompt.
- The harness refuses non-localhost model endpoints, so no cloud AI API is called.

## 8. Evaluation
Same first 40 test rows for base and fine-tuned models; ablation on the first 20 rows.

### 8.1 Base vs fine-tuned (full harness, level E) — **[TBD: Kaggle run]**
| Metric | Base Qwen3-8B | Fine-tuned |
|---|---|---|
| Numerical hallucination rate | | |
| Invalid tool-call rate | | |
| Correct tool sequence / arguments | | |
| Unsupported conclusion rate | | |
| Rule violations (raw answer) | | |
| Evidence completeness | | |
| Reproducibility (3 runs) | | |
| Mean latency | | |

### 8.2 Raw base model without tools or data (5 reference scenario questions)
| Metric | Value |
|---|---|
| Answers with invented numbers | 100% (17.2 per answer) |
| Price target given | 20% (scenario 5: "13.25–13.75 TL") |
| Required format / allowed state | 0% |

### 8.3 Harness ablation A–E — **[TBD: Kaggle run]**

### 8.4 Real vs synthetic fundamentals — **[TBD: by_fundamentals_source]**

## 9. Research findings (deterministic tools, `docs/results/`)
- **Sector catch-up:** average 20-day hit rate 45%, false positives 55%; laggards mostly did not mean-revert.
- **Weekday patterns:** 960 tests → 12 significant at p < 0.05, 1 after Bonferroni, **0** significant out of sample.
- **Technical reversals:** a lower-Bollinger touch led to a 5-day rise 57% of the time, yet trading it returned −3.9%
  against +117% buy-and-hold (prediction ≠ profitable strategy).
- **Fundamentals:** improving quarters +0.98 pp vs −0.48 pp 20-day abnormal return. Association, not causation.
- **Macro:** weekly correlations with USD/TRY are weak or negative (telecom, retail); "exporter rises with the dollar" is not supported.
- **Strategy variants:** each added filter cut trades (116 → 8 in-sample); out-of-sample gains rest on very few trades.

## 10. Reference scenario questions and live question
Execution traces: `release/execution_traces.json` **[TBD]**, for the Turkish and the English version of each reference scenario question. Each trace records the model and adapter SHA-256,
system and user prompts, every model turn and tool call with timestamps, tool outputs, answer, `answer_source`,
temperature and seed.

## 11. Limitations
- Fundamentals cover 13 companies from 2024Q4; quarterly values are YTD differences under TMS 29 (approximate);
  CCOLA 2024Q4 is unreadable; banks are not analysed with EBITDA ratios.
- No licensed news feed or approved videos are loaded: the event layer only holds official KAP/CBRT/FOMC events, and
  strategy variant E is unavailable.
- Yahoo volume data has gaps (e.g. SASA); several sectors have only two peers.
- Event studies are in-sample with overlapping windows; combined-condition groups are small.
- The model can still fail on very unusual phrasings; failed validations end in a labelled harness fallback, never in an unlabelled answer.

## Appendix A — Qwen2.5-3B comparison **[TBD: results_3b]**
Trained locally (RTX 4070 Laptop, 8 GB, bf16, Liger, 2 epochs, 3 h 48 min); validation loss 0.136 → 0.0058.
