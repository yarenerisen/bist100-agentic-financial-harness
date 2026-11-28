"""Splits a harness answer into its fields for display (terminal demo, web UI). Display only: the text is not changed.

    parse_answer(text) -> {"decision_basis": str, "state": str, "evidence": [(label, text), ...],
                           "limitations": str, "note": str, "other": [lines not matched]}
"""
import re

_FIELD = re.compile(r"^(DECISION_BASIS|ANALYTICAL_STATE|EVIDENCE|LIMITATIONS|NOTE):\s*(.*)$")
_EVID = re.compile(r"^-\s*(E\d+):\s*(.*)$")

# display groups of the finite action space
STATE_KIND = {
    "POTENTIAL_CATCH_UP_CANDIDATE": "candidate", "TECHNICAL_REVERSAL_CANDIDATE": "candidate",
    "FUNDAMENTAL_IMPROVEMENT_CANDIDATE": "candidate", "WATCH": "watch", "HOLD": "watch",
    "INVESTIGATE": "investigate", "REJECT_SIGNAL": "reject", "ANALYSIS_UNSAFE": "unsafe",
}


def parse_answer(text):
    out = {"decision_basis": "", "state": "", "evidence": [], "limitations": "", "note": "", "other": []}
    key = None
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = _FIELD.match(s)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key == "DECISION_BASIS":
                out["decision_basis"] = val
            elif key == "ANALYTICAL_STATE":
                out["state"] = val
            elif key == "LIMITATIONS":
                out["limitations"] = val
            elif key == "NOTE":
                out["note"] = val
            continue
        e = _EVID.match(s)
        if key == "EVIDENCE" and e:
            out["evidence"].append((e.group(1), e.group(2)))
        elif key == "EVIDENCE" and out["evidence"]:          # wrapped evidence line
            lab, txt = out["evidence"][-1]
            out["evidence"][-1] = (lab, txt + " " + s)
        elif key == "LIMITATIONS":
            out["limitations"] = (out["limitations"] + " " + s).strip()
        elif key == "NOTE":
            out["note"] = (out["note"] + " " + s).strip()
        else:
            out["other"].append(s)
    return out


def tool_rows(trace):
    """(tool, arguments, status) for every executed tool call of a trace."""
    rows = []
    for c in trace.get("tool_calls", []):
        if "tool" not in c:
            continue
        r = c.get("result")
        err = r.get("error") if isinstance(r, dict) else None
        msg = r.get("message", "") if err else ""
        rows.append((c["tool"], c.get("arguments", {}), f"{err}: {msg}" if err else "ok"))
    return rows
