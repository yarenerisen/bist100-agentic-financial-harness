"""Evaluation with the evaluation metrics: runs test questions through the harness and compares to references.
Usage:
  python -m harness.evaluate --backend replay                       # sanity check of the pipeline (should be ~perfect)
  python -m harness.evaluate --backend openai --model bist-analyst  # fine-tuned model on local llama.cpp/Ollama
  python -m harness.evaluate --backend openai --model qwen2.5:7b-instruct --tag base   # base model for comparison
Writes results/eval_<tag>.json"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.run_agent import run, ReplayBackend, make_backend   # noqa: E402
from harness.validate import numbers_not_in_evidence, check_answer          # noqa: E402

import pandas as pd
REAL_FUND = set(pd.read_csv(ROOT / "data" / "fundamentals.csv").dropna(subset=["revenue"]).ticker)
FUND_TOOLS = {"get_fundamental_metrics", "calculate_fundamental_change", "preview_strategy_variants"}


def fundamentals_source(refs):
    """real = all fundamentals come from parsed KAP reports; synthetic = training-only synthetic file; n/a = no fundamentals used."""
    tk = set()
    for r in refs:
        if r["tool"] in FUND_TOOLS:
            a = r["arguments"]
            v = a.get("ticker") or a.get("tickers") or []
            tk |= set([v] if isinstance(v, str) else v)
    if not tk:
        return "no_fundamentals"
    return "real_KAP" if tk <= REAL_FUND else ("synthetic" if not (tk & REAL_FUND) else "mixed")


KEY_ARGS = ("ticker", "event", "pattern", "pattern_type", "strategy", "start", "end", "as_of", "horizon")


def ref_calls(row):
    return [json.loads(m["content"]) for m in row["messages"][2:-1] if m["role"] == "assistant"]


def eval_row(row, backend_factory, repeats=1, mode="E"):
    q = row["messages"][1]["content"]
    refs = ref_calls(row)
    raw = "\n".join(m["content"] for m in row["messages"] if m["role"] == "tool") if mode == "A" else None
    traces = [run(q, backend_factory(row), log=False, mode=mode, raw_data=raw) for _ in range(repeats)]
    t = traces[0]
    calls = [c for c in t["tool_calls"] if "tool" in c]
    tool_ok = [c["tool"] for c in calls] == [r["tool"] for r in refs]
    args_ok = tool_ok and all(all(str(c["arguments"].get(k)) == str(r["arguments"].get(k)) for k in KEY_ARGS if k in r["arguments"])
                              for c, r in zip(calls, refs))
    lookahead = any(str(v) > "2026-09-10" for c in calls for k, v in c["arguments"].items() if k in ("start", "end", "as_of"))
    evid = [json.dumps(c["result"], ensure_ascii=False, separators=(",", ":")) for c in calls] + [q] + ([raw] if raw else [])
    raw_answer = t.get("first_answer", t["answer"])          # judge the model's own first answer, before harness fallback
    bad_nums = numbers_not_in_evidence(raw_answer, evid)
    raw_chk = check_answer(raw_answer, evid)
    return {"scenario": row["meta"]["scenario"], "fundamentals_source": fundamentals_source(refs),
            "ref_state": row["meta"]["state"], "state": t["state"],
            "raw_state": raw_chk["state"],
            "tool_sequence_correct": tool_ok, "tool_args_correct": args_ok,
            "invalid_tool_calls": t["invalid_tool_calls"], "n_tool_calls": len(calls), "lookahead_call": lookahead,
            "hallucinated_numbers": bad_nums, "raw_answer_valid": raw_chk["ok"], "raw_issues": raw_chk["issues"],
            "final_answer_valid": t["answer_validation"]["ok"],
            "reproducible": len({x["answer"] for x in traces}) == 1, "latency_s": t["latency_s"],
            "evidence_completeness_pct": (t.get("evidence_bundle") or {}).get("evidence_completeness_pct"),
            # kept for the per-tool table and side-by-side output comparisons
            "question": q, "ref_calls": refs, "calls": [{"tool": c["tool"], "arguments": c["arguments"]} for c in calls],
            "answer": t["answer"], "raw_answer": raw_answer, "ref_answer": row["messages"][-1]["content"],
            "answer_source": t["answer_source"]}


def per_tool(res):
    """Per expected tool: rows that needed it, rows where it was called, and called with the reference key arguments."""
    out = {}
    for r in res:
        for ref in r["ref_calls"]:
            d = out.setdefault(ref["tool"], {"expected": 0, "called": 0, "args_correct": 0, "extra_calls": 0})
            d["expected"] += 1
            got = [c for c in r["calls"] if c["tool"] == ref["tool"]]
            d["called"] += bool(got)
            d["args_correct"] += any(all(str(c["arguments"].get(k)) == str(ref["arguments"].get(k)) for k in KEY_ARGS if k in ref["arguments"])
                                     for c in got)
        for c in r["calls"]:
            if c["tool"] not in {x["tool"] for x in r["ref_calls"]}:
                out.setdefault(c["tool"], {"expected": 0, "called": 0, "args_correct": 0, "extra_calls": 0})["extra_calls"] += 1
    return dict(sorted(out.items()))


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
            "evidence_completeness_pct": (lambda v: round(sum(v) / len(v), 1) if v else None)([r["evidence_completeness_pct"] for r in res if r["evidence_completeness_pct"] is not None]),
            "mean_latency_s": round(sum(r["latency_s"] for r in res) / n, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="replay", choices=["replay", "openai", "hf", "llamacpp", "ollama", "vllm"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--data", default=str(ROOT / "training" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="run each question N times to measure reproducibility")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mode", default="E", choices=list("ABCDE"), help="harness ablation level")
    ap.add_argument("--load-4bit", action="store_true", help="hf backend: load the model in 4-bit (7B on a T4)")
    ap.add_argument("--out-dir", default=None, help="where to write eval_<tag>.json (e.g. a Google Drive folder)")
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel questions; only for server backends (vllm, llamacpp, openai)")
    ap.add_argument("--real-fundamentals", action="store_true",
                    help="run every row on data/fundamentals.csv instead of the file its reference was generated with")
    a = ap.parse_args()
    from tools import macro_fund
    FUND_FILES = {"real": macro_fund.FUND_CSV, "synthetic": ROOT / "training" / "synthetic_fundamentals_TRAINING_ONLY.csv"}
    rows = [json.loads(l) for l in open(a.data)]
    rows = rows[:a.limit] if a.limit else rows
    if a.backend == "replay":
        fac = lambda row: ReplayBackend(row)
    else:
        be = make_backend(a)
        fac = lambda row: be
    if a.workers > 1 and a.backend == "hf":
        raise SystemExit("--workers > 1 needs a server backend; the hf backend runs one generation at a time")
    from concurrent.futures import ThreadPoolExecutor
    # each row runs with the fundamentals file its reference was generated with (meta.fundamentals); the file is a
    # module global, so rows are run in one batch per file
    res, done = [None] * len(rows), 0
    for kind, path in FUND_FILES.items():
        idx = [i for i, r in enumerate(rows) if ("real" if a.real_fundamentals else r["meta"].get("fundamentals", "synthetic")) == kind]
        macro_fund.FUND_CSV = path
        with ThreadPoolExecutor(max(1, a.workers)) as ex:
            for i, r in zip(idx, ex.map(lambda j: eval_row(rows[j], fac, a.repeats, a.mode), idx)):
                res[i] = r; done += 1
                if done % 20 == 0 or done == len(rows):
                    print(f"{done}/{len(rows)}", flush=True)
    macro_fund.FUND_CSV = FUND_FILES["real"]
    s = summarize(res)
    by = {}
    for sc in sorted({r["scenario"] for r in res}):
        by[sc] = summarize([r for r in res if r["scenario"] == sc])
    by_src = {k: summarize([r for r in res if r["fundamentals_source"] == k])
              for k in sorted({r["fundamentals_source"] for r in res})}
    pt = per_tool(res)
    out = {"tag": a.tag or f"{a.backend}_{a.mode}", "mode": a.mode, "by_fundamentals_source": by_src, "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M"), "overall": s, "by_scenario": by, "per_tool": pt, "rows": res}
    rdir = Path(a.out_dir) if a.out_dir else ROOT / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    p = rdir / f"eval_{out['tag']}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + json.dumps(s, indent=1))
    print("by fundamentals source:", json.dumps({k: {"n": v["n"], "halluc_%": v["numerical_hallucination_rate_pct"],
                                                    "unsupported_%": v["unsupported_conclusion_rate_pct"]} for k, v in by_src.items()}))
    print("per tool (expected / called / args correct / extra):")
    for k, v in pt.items():
        print(f"  {k:30} {v['expected']:4} {v['called']:4} {v['args_correct']:4} {v['extra_calls']:4}")
    print("saved", p)
