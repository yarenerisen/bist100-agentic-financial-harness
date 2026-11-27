"""Scores free-form answers produced WITHOUT tools or data (e.g. the raw base-model baseline) with the same checks
as harness/evaluate.py. Input: JSONL with at least {"question", "answer"} (optional q_id, model, stage).
Because no tool output exists, the only evidence is the question itself: every other number counts as invented.
Usage (from bist_llm/): python -m harness.score_raw training/baseline/base_model_raw_outputs.jsonl"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.validate import check_answer, numbers_not_in_evidence, FORBIDDEN, ALLOW_ADVICE_CONTEXT   # noqa: E402
from training.prompts import ACTIONS   # noqa: E402

TARGET_RX = re.compile(r"(hedef fiyat|price target|target price)[^\n]{0,40}?\d", re.I)


def score_row(r):
    ans = r["answer"]
    bad = numbers_not_in_evidence(ans, [r["question"]])
    chk = check_answer(ans, [r["question"]])
    advice = [l.strip()[:120] for l in ans.splitlines()
              if any(re.search(p, l, re.I) for p in FORBIDDEN) and not ALLOW_ADVICE_CONTEXT.search(l)]
    return {"q_id": r.get("q_id"), "model": r.get("model"), "stage": r.get("stage"),
            "invented_numbers": len(bad), "invented_number_examples": bad[:10],
            "has_numerical_hallucination": bool(bad),
            "format_valid": not any(i.startswith("missing") for i in chk["issues"]),
            "state": chk["state"], "state_in_action_space": chk["state"] in ACTIONS,
            "mentions_allowed_state": any(a in ans for a in ACTIONS),
            "advice_violation": bool(advice), "advice_lines": advice[:3],
            "gives_price_target": bool(TARGET_RX.search(ans)), "tool_calls": 0, "answer_chars": len(ans)}


def summarize(rows):
    n = len(rows)
    pct = lambda k: round(100 * sum(1 for r in rows if r[k]) / n, 1)
    return {"n": n, "numerical_hallucination_rate_pct": pct("has_numerical_hallucination"),
            "mean_invented_numbers_per_answer": round(sum(r["invented_numbers"] for r in rows) / n, 1),
            "format_valid_pct": pct("format_valid"), "state_in_action_space_pct": pct("state_in_action_space"),
            "advice_violation_rate_pct": pct("advice_violation"), "price_target_rate_pct": pct("gives_price_target"),
            "invalid_tool_call_rate_pct": None, "note": "no tools and no data were given; evidence = question text only"}


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "training" / "baseline" / "base_model_raw_outputs.jsonl")
    rows = [score_row(json.loads(l)) for l in src.read_text().splitlines() if l.strip()]
    out = {"source": str(src.name), "overall": summarize(rows), "rows": rows}
    dst = src.with_name(src.stem.replace("_outputs", "") + "_scores.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["overall"], indent=1))
    for r in rows:
        print(f"Q{r['q_id']}: invented numbers {r['invented_numbers']} {r['invented_number_examples'][:6]} | state {r['state']} | "
              f"advice {r['advice_violation']} | price target {r['gives_price_target']}")
    print("saved", dst)
