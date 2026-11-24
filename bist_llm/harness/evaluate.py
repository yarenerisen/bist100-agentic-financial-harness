"""Evaluation with the course metrics (Section 30): runs test questions through the harness and compares to references.
Usage:
  python -m harness.evaluate --backend replay                       # sanity check of the pipeline (should be ~perfect)
  python -m harness.evaluate --backend openai --model bist-analyst  # fine-tuned model on local llama.cpp/Ollama
  python -m harness.evaluate --backend openai --model qwen2.5:7b-instruct --tag base   # base model for comparison
Writes results/eval_<tag>.json"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.run_agent import run, OpenAIBackend, HFBackend, ReplayBackend   # noqa: E402
from harness.validate import numbers_not_in_evidence, check_answer          # noqa: E402

KEY_ARGS = ("ticker", "event", "pattern", "pattern_type", "strategy", "start", "end", "as_of", "horizon")


def ref_calls(row):
    return [json.loads(m["content"]) for m in row["messages"][2:-1] if m["role"] == "assistant"]


def eval_row(row, backend_factory, repeats=1):
    q = row["messages"][1]["content"]
    refs = ref_calls(row)
    traces = [run(q, backend_factory(row), log=False) for _ in range(repeats)]
    t = traces[0]
    calls = [c for c in t["tool_calls"] if "tool" in c]
    tool_ok = [c["tool"] for c in calls] == [r["tool"] for r in refs]
    args_ok = tool_ok and all(all(str(c["arguments"].get(k)) == str(r["arguments"].get(k)) for k in KEY_ARGS if k in r["arguments"])
                              for c, r in zip(calls, refs))
    lookahead = any(str(v) > "2026-09-10" for c in calls for k, v in c["arguments"].items() if k in ("start", "end", "as_of"))
    evid = [json.dumps(c["result"], ensure_ascii=False, separators=(",", ":")) for c in calls] + [q]
    raw_answer = t.get("first_answer", t["answer"])          # judge the model's own first answer, before harness fallback
    bad_nums = numbers_not_in_evidence(raw_answer, evid)
    raw_chk = check_answer(raw_answer, evid)
    return {"scenario": row["meta"]["scenario"], "ref_state": row["meta"]["state"], "state": t["state"],
            "raw_state": raw_chk["state"],
            "tool_sequence_correct": tool_ok, "tool_args_correct": args_ok,
            "invalid_tool_calls": t["invalid_tool_calls"], "n_tool_calls": len(calls), "lookahead_call": lookahead,
            "hallucinated_numbers": bad_nums, "raw_answer_valid": raw_chk["ok"], "raw_issues": raw_chk["issues"],
            "final_answer_valid": t["answer_validation"]["ok"],
            "reproducible": len({x["answer"] for x in traces}) == 1, "latency_s": t["latency_s"]}


def summarize(res):
    n = len(res)
    pct = lambda k: round(100 * sum(1 for r in res if r[k]) / n, 1)
    return {"n": n,
            "numerical_hallucination_rate_pct": round(100 * sum(1 for r in res if r["hallucinated_numbers"]) / n, 1),
            "invalid_tool_call_rate_pct": round(100 * sum(1 for r in res if r["invalid_tool_calls"]) / n, 1),
            "correct_tool_sequence_pct": pct("tool_sequence_correct"), "correct_tool_args_pct": pct("tool_args_correct"),
            "lookahead_call_rate_pct": pct("lookahead_call"),
            "unsupported_conclusion_rate_pct": round(100 * sum(1 for r in res if r["raw_state"] != r["ref_state"]) / n, 1),
            "raw_answer_rule_violation_rate_pct": round(100 * sum(1 for r in res if not r["raw_answer_valid"]) / n, 1),
            "final_answer_valid_pct": pct("final_answer_valid"),
            "reproducibility_pct": pct("reproducible"),
            "mean_tool_calls": round(sum(r["n_tool_calls"] for r in res) / n, 2),
            "mean_latency_s": round(sum(r["latency_s"] for r in res) / n, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="replay", choices=["replay", "openai", "hf"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--data", default=str(ROOT / "training" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="run each question N times to measure reproducibility")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--real-fundamentals", action="store_true",
                    help="use data/fundamentals.csv instead of the synthetic file the test set was generated with")
    a = ap.parse_args()
    if not a.real_fundamentals:
        from tools import macro_fund
        macro_fund.FUND_CSV = ROOT / "training" / "synthetic_fundamentals_TRAINING_ONLY.csv"
    rows = [json.loads(l) for l in open(a.data)]
    rows = rows[:a.limit] if a.limit else rows
    if a.backend == "replay":
        fac = lambda row: ReplayBackend(row)
    else:
        be = OpenAIBackend(a.base_url, a.model, a.allow_remote) if a.backend == "openai" else HFBackend(a.model)
        fac = lambda row: be
    res = []
    for i, r in enumerate(rows, 1):
        res.append(eval_row(r, fac, a.repeats))
        if i % 20 == 0 or i == len(rows):
            print(f"{i}/{len(rows)}", flush=True)
    s = summarize(res)
    by = {}
    for sc in sorted({r["scenario"] for r in res}):
        by[sc] = summarize([r for r in res if r["scenario"] == sc])
    out = {"tag": a.tag or a.backend, "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M"), "overall": s, "by_scenario": by, "rows": res}
    Path(ROOT / "results").mkdir(exist_ok=True)
    p = ROOT / "results" / f"eval_{out['tag']}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + json.dumps(s, indent=1)); print("saved", p)
