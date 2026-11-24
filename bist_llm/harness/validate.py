"""Answer validators shared by evaluation and the inference harness.
- numbers_not_in_evidence: every number in the answer must appear in a tool output or the user question
- format / action-space / forbidden-advice checks"""
import json, re
from training.prompts import ACTIONS

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
FORBIDDEN = [r"\bbuy\b(?![- ]and[- ]hold| signal)", r"\bsell\b", r"price target", r"target price", r"will definitely", r"guaranteed",
             r"\balın\b", r"\bsatın\b", r"hedef fiyat(?!.*(veremem|vermiyorum|verilemez))", r"kesin(likle)? (yükselir|düşer|çıkar)"]
ALLOW_ADVICE_CONTEXT = re.compile(r"(cannot|can't|not|no|never|veremem|değil|yapamam|yasak|izin verilen kapsamın dışında)", re.I)


def _numbers(text):
    out = []
    for m in NUM.finditer(text):
        s = m.group()
        out.append(s)
    return out


def _evidence_numbers(texts):
    pool = set()
    for t in texts:
        for s in _numbers(t):
            pool.add(s.lstrip("+"))
            try:
                v = float(s); pool.add(repr(v)); pool.add(str(abs(v)).rstrip("0").rstrip(".") if "." in str(abs(v)) else str(abs(v)))
                pool.add(str(abs(v)))
                if v == int(v):
                    pool.add(str(int(v))); pool.add(str(abs(int(v))))
            except ValueError:
                pass
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", t):      # dates: allow their parts and the dd.mm.yyyy form
            y, mth, dd = d.split("-"); pool.update({y, mth, dd, str(int(mth)), str(int(dd))})
        for d in re.findall(r"\d{2}\.\d{2}\.\d{4}", t):
            dd, mth, y = d.split("."); pool.update({y, mth, dd, f"{y}-{mth}-{dd}"})
    return pool


GIVEN_FACTS = "analysis date 2026-09-10, data period 2024-01-01 to 2026-09-10"


def numbers_not_in_evidence(answer, evidence_texts):
    pool = _evidence_numbers(list(evidence_texts) + [GIVEN_FACTS])
    body = re.sub(r"\bE\d+\b", "", answer)                    # evidence labels E1, E2...
    body = re.sub(r"\d{4}-\d{2}-\d{2}", lambda m: " ".join(m.group().split("-")), body)
    bad = []
    for s in _numbers(body):
        k = s.lstrip("+")
        if k in pool or k.lstrip("-") in pool:
            continue
        if k in {"1", "3", "5", "10", "20", "0", "2", "4", "32", "100"}:   # horizons/structural constants used in definitions
            continue
        bad.append(s)
    return bad


def check_answer(answer, evidence_texts):
    issues = []
    m = re.search(r"^ANALYTICAL_STATE:\s*(\S+)", answer, re.M)
    if not m:
        issues.append("missing ANALYTICAL_STATE")
    elif m.group(1) not in ACTIONS:
        issues.append(f"state not in action space: {m.group(1)}")
    for sec in ("EVIDENCE:", "LIMITATIONS:", "NOTE:"):
        if sec not in answer:
            issues.append(f"missing {sec}")
    bad = numbers_not_in_evidence(answer, evidence_texts)
    if bad:
        issues.append(f"numbers not found in tool outputs: {bad[:8]}")
    for line in answer.splitlines():
        for pat in FORBIDDEN:
            if re.search(pat, line, re.I) and not ALLOW_ADVICE_CONTEXT.search(line):
                issues.append(f"forbidden advice language: '{line[:80]}'")
                break
    return {"ok": not issues, "state": m.group(1) if m else None, "issues": issues}


def parse_tool_call(text):
    """Returns (dict or None, error). Accepts a bare JSON object, optionally fenced."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    if not t.startswith("{"):
        return None, "not a JSON object"
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict) or "tool" not in obj or not isinstance(obj.get("arguments", {}), dict):
        return None, "JSON must have 'tool' and 'arguments'"
    obj.setdefault("arguments", {})
    return obj, None
