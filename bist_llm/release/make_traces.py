"""Runs questions through the full harness (mode E) with the submitted model and writes complete execution traces.
Each trace contains: model name/version (base model, adapter path + SHA-256, decoding: temperature, seed), system and
user prompts, every model turn with timestamps and raw output, every tool call with arguments, date ranges, timestamps
and outputs, the quality gate, the evidence bundle, the final answer, answer_source (model / model_after_validation_retry
/ harness_fallback) and code version.
Usage (from bist_llm/):
  python -m release.make_traces --backend hf --model /path/to/qwen3-8b-lora --load-4bit
  python -m release.make_traces --backend llamacpp                          # local GGUF server
  python -m release.make_traces --backend hf --model ... --question "live question" --out release/live_trace.json"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.run_agent import run, make_backend   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["openai", "hf", "llamacpp", "ollama"])
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--questions", default=str(ROOT / "release" / "scenario_questions.json"))
    ap.add_argument("--question", help="run a single (live) question instead of the scenario file")
    ap.add_argument("--languages", default="tr,en", help="which question versions to run from the scenario file")
    ap.add_argument("--out", default=str(ROOT / "release" / "execution_traces.json"))
    a = ap.parse_args()
    if a.question:
        qs = [{"id": "live", "question": a.question}]
    else:
        qs = [{"id": f"{x['id']}_{lang}", "question": x[f"question_{lang}"]}
              for x in json.load(open(a.questions))["questions"] for lang in a.languages.split(",")]
    be = make_backend(a)
    traces = []
    for q in qs:
        print(f"[{time.strftime('%H:%M:%S')}] {q['id']} ...", flush=True)
        t = run(q["question"], be, log=True, mode="E")
        t["scenario_id"] = q["id"]
        traces.append(t)
        print(f"   -> {t['state']} | source {t['answer_source']} | tools {[c.get('tool') for c in t['tool_calls']]} | {t['latency_s']}s", flush=True)
    out = {"generated_at": traces[0]["timestamp_start"] if traces else None, "harness_mode": "E",
           "model": traces[0]["model"] if traces else None, "n": len(traces),
           "summary": [{"scenario_id": t["scenario_id"], "state": t["state"], "answer_source": t["answer_source"],
                        "tools": [c.get("tool") for c in t["tool_calls"]], "answer_valid": t["answer_validation"]["ok"],
                        "latency_s": t["latency_s"]} for t in traces],
           "traces": traces}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print("saved", a.out)
