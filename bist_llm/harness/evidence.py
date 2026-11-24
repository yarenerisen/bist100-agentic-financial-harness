"""Evidence bundle (Section 25): full observed context vs. the committed evidence set actually used in the answer."""
import json, re
from harness.validate import NUM


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


CONSTANTS = {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 32.0, 0.0, 100.0}


def build_evidence_bundle(tool_calls, answer, question=""):
    observed = []
    for c in tool_calls:
        if "tool" in c:
            observed += [{"tool": c["tool"], "field": p, "value": v} for p, v in _leaves(c["result"])]
    index = {}
    for o in observed:
        for v in (o["value"], abs(o["value"])):
            index.setdefault(round(float(v), 6), []).append(f"{o['tool']}:{o['field']}")
    for c in tool_calls:                      # numbers inside text fields (definitions, headlines)
        if "tool" in c:
            for n in NUM.findall(json.dumps(c["result"], ensure_ascii=False)):
                index.setdefault(round(abs(float(n)), 6), []).append(f"{c['tool']}:text")
    for n in NUM.findall(question):
        index.setdefault(round(abs(float(n)), 6), []).append("question")
    committed = []
    for line in answer.splitlines():
        m = re.match(r"-\s*(E\d+):\s*(.*)", line.strip())
        if not m:
            continue
        nums = [n for n in NUM.findall(re.sub(r"\d{4}-\d{2}-\d{2}", "", m.group(2)))]
        sources = {}
        for n in nums:
            v = round(abs(float(n)), 6)
            if v in index:
                sources[n] = index[v][:2]
            elif v in CONSTANTS:
                sources[n] = ["definition_constant"]
        committed.append({"id": m.group(1), "text": m.group(2), "numbers": nums, "sources": sources,
                          "unsourced_numbers": [n for n in nums if n not in sources]})
    st = re.search(r"^ANALYTICAL_STATE:\s*(\S+)", answer, re.M)
    return {"observed_context": {"tools": [c["tool"] for c in tool_calls if "tool" in c], "numeric_fields": len(observed)},
            "committed_evidence": committed, "analytical_state": st.group(1) if st else None,
            "evidence_completeness_pct": round(100 * sum(1 for c in committed if not c["unsourced_numbers"]) / len(committed), 1) if committed else None,
            "warning": "Historical evidence only; no guarantee of future return."}
