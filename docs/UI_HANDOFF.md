# Handoff: local web UI for the live demo

## Goal
A small browser UI on top of the existing harness, used in the live presentation. The person asking types a
question; the page shows the tool calls, the analytical state, the decision basis, the evidence, the limitations and
whether the answer passed validation. The terminal demo (`bist_llm/release/demo.py`) already does this in text; the UI
is a nicer view of the same thing.

The demo runs on the team laptop with the NVIDIA GPU (the model needs it). The UI is developed in this repository and
pulled onto that laptop for the presentation. You can build and test it fully without a GPU (see "Developing without
the model").

## Rules (from the project brief and the report)
1. **Local only.** The page is served on `127.0.0.1`; nothing is sent to the internet (no CDN fonts or scripts, no
   telemetry, no cloud API). The demo laptop may be offline.
2. **Display only.** Show `trace["answer"]` and the tool outputs exactly as the harness returns them. Do not shorten,
   rewrite, translate or recompute anything: the report states that every number in an answer comes from a tool output.
3. **One model, loaded once.** The GPU has 8 GB; load the backend once per process (`st.cache_resource` in Streamlit)
   and answer one question at a time.
4. **Keep the traces.** Save every answer as a full trace JSON in `bist_llm/release/live_traces/`, as `demo.py` does.
5. **Label the source.** If `answer_source` is `harness_fallback`, say so visibly ("validation failed twice, the
   harness returned a safe notice"); in replay mode, show that the answer is a reference answer, not the model.

## Interface
Everything the UI needs already exists; run from `bist_llm/`.

```python
from harness.run_agent import run, make_backend, ReplayBackend
from release.answer_format import parse_answer, tool_rows, STATE_KIND

# real model (demo laptop only): build once, reuse for every question
class Args: backend = "hf"; model = "runs/qwen35-4b-lora-v5"; load_4bit = True; allow_remote = False; base_url = ""
backend = make_backend(Args)

trace = run(question, backend, log=True, mode="E")     # one question through the full harness (20-60 s)
```

`trace` (dict) fields used by the display:

| Field | Meaning |
|---|---|
| `question` | the question as typed |
| `state` | final analytical state (one of the 8 allowed states) |
| `answer` | full answer text: `DECISION_BASIS`, `ANALYTICAL_STATE`, `EVIDENCE` (`- E1: ...`), `LIMITATIONS`, `NOTE` |
| `tool_calls` | list of `{"tool", "arguments", "result", "started_at", "finished_at", "latency_s"}`; a failed call has `result["error"]` (`TOOL_ERROR`, `LOOKAHEAD_BLOCKED`, ...) and `result["message"]` |
| `answer_source` | `model`, `model_after_validation_retry` or `harness_fallback` |
| `answer_validation` | `{"ok": bool, "state": str, "issues": [...]}` |
| `latency_s` | seconds for the whole question |
| `quality_gate`, `risk_gate`, `workflow`, `states` | optional: data-quality result, risk-gate decision, state sequence with agent roles |

Helpers in `bist_llm/release/answer_format.py` (no dependencies):
- `parse_answer(trace["answer"])` → `{"decision_basis", "state", "evidence": [("E1", text), ...], "limitations", "note", "other"}`
- `tool_rows(trace)` → `[(tool, arguments, "ok" | "TOOL_ERROR: ..."), ...]`
- `STATE_KIND[state]` → `candidate` / `watch` / `investigate` / `reject` / `unsafe` (for colours; the terminal uses green / yellow / cyan / red / magenta)

## Developing without the model (Mac, no GPU)
The tools run on CPU; only the model needs the GPU. Two ways to get real traces:

1. **Saved traces** (instant): `bist_llm/release/live_check_traces.json` (20 hand-written questions) and
   `bist_llm/release/execution_traces.json` (5 reference scenarios, TR + EN) contain complete traces in exactly the format
   above: `json.load(open(path))["traces"]`.
2. **Replay backend** (real harness and real tool calls, reference answers instead of the model):
   ```python
   row = json.loads(open("training/test.jsonl").readline())              # any row of test.jsonl / test_ext.jsonl
   q = next(m["content"] for m in row["messages"] if m["role"] == "user")
   trace = run(q, ReplayBackend(row), log=False)
   ```
   Note: in replay mode `answer_source` still reads `model`; the UI must label it as a reference answer.

Suggested switch: `--mode model --model <adapter>` on the demo laptop, `--mode traces <file>` or `--mode replay` for
development.

## Suggested implementation
- File: `bist_llm/release/ui.py`, Streamlit (`pip install streamlit`, add it to `requirements.txt`).
- Start (demo laptop):
  `HF_HUB_OFFLINE=1 streamlit run release/ui.py --server.address 127.0.0.1 --browser.gatherUsageStats false -- --model runs/qwen35-4b-lora-v5`
- Layout, top to bottom: question box (+ the 5 reference questions as examples); tool-call table with status;
  state badge + decision basis; evidence list; limitations; footer with answer source, validation result, latency and
  the trace file name; an expander with the raw answer text and, optionally, the raw tool outputs.
- Keep the terminal demo working; the UI is an addition.

## Done when
- [ ] Runs from `bist_llm/` with the command above and opens on `http://127.0.0.1:8501`, with the network off
- [ ] Shows all 20 traces in `release/live_check_traces.json` and the 10 in `release/execution_traces.json` correctly
      (state badge = `trace["state"]`, every evidence line present, no text changed)
- [ ] Failed tool calls are visibly marked (replay the `TSLA` row of `test.jsonl`: `UNKNOWN_ENTITY`) and so is a
      fallback (no saved trace has one; test by setting `answer_source = "harness_fallback"` on a copy of a trace)
- [ ] Each live answer is written to `release/live_traces/`
- [ ] New dependency listed in `requirements.txt`
- [ ] Pushed on a branch with a pull request into `main`
