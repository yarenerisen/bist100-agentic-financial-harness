"""Side-by-side report of evaluation runs on the same test rows: headline metrics, per-tool table, example outputs.
Usage (from bist_llm/): python -m harness.compare results/eval_base_E.json results/eval_finetuned_E.json > results/compare.md"""
import json, sys

METRICS = ["n", "correct_tool_sequence_pct", "correct_tool_args_pct", "invalid_tool_call_rate_pct", "numerical_hallucination_rate_pct",
           "unsupported_conclusion_rate_pct", "raw_answer_rule_violation_rate_pct", "final_answer_valid_pct", "evidence_completeness_pct",
           "mean_tool_calls", "mean_latency_s"]


def cut(s, n=700):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    tags = [r["tag"] for r in runs]
    n = min(len(r["rows"]) for r in runs)
    print(f"# Comparison: {' vs '.join(tags)}\n\nSame first {n} test rows, harness mode {runs[0]['mode']}.\n")
    print("| metric | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
    for m in METRICS:
        print(f"| {m} | " + " | ".join(str(r["overall"].get(m)) for r in runs) + " |")
    print("\n## Per tool (expected / called / arguments correct / extra calls)\n")
    print("| tool | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
    for tool in sorted(set().union(*[r.get("per_tool", {}) for r in runs])):
        cells = [r.get("per_tool", {}).get(tool) for r in runs]
        print(f"| {tool} | " + " | ".join(f"{c['expected']} / {c['called']} / {c['args_correct']} / {c['extra_calls']}" if c else "-" for c in cells) + " |")
    print("\n## Example outputs (first row of each scenario where the runs differ in state or tools)\n")
    seen = set()
    for i in range(n):
        rows = [r["rows"][i] for r in runs]
        sc = rows[0]["scenario"]
        if sc in seen or len({(x["state"], json.dumps(x["calls"])) for x in rows}) == 1:
            continue
        seen.add(sc)
        print(f"### {sc} — reference state {rows[0]['ref_state']}\n\n**Question:** {rows[0]['question']}\n")
        print("**Reference tools:** " + ", ".join(f"`{c['tool']}`" for c in rows[0]["ref_calls"]) + "\n")
        for tag, x in zip(tags, rows):
            calls = ", ".join(f"`{c['tool']}({', '.join(f'{k}={v}' for k, v in c['arguments'].items())})`" for c in x["calls"]) or "none"
            print(f"**{tag}** — state `{x['state']}` ({x['answer_source']}), tools: {calls}\n\n```text\n{cut(x['raw_answer'])}\n```\n")
        print(f"**Reference answer:**\n\n```text\n{cut(rows[0]['ref_answer'])}\n```\n")


if __name__ == "__main__":
    main(sys.argv[1:])
