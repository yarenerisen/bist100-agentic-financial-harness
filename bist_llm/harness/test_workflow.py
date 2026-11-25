"""Self-check of the workflow state machine with a scripted model. Run from bist_llm/: python -m harness.test_workflow"""
import json
from harness.run_agent import run, STATE_INDEX


class Scripted:
    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def chat(self, messages):
        return self.outputs.pop(0)


call = lambda tool, **args: json.dumps({"tool": tool, "arguments": args})
ANSWER = "ANALYTICAL_STATE: {}\nEVIDENCE:\n- E1: see tool output.\nLIMITATIONS: none.\nNOTE: Historical, educational analysis only."


def test_states_only_move_forward():
    t = run("THYAO catch-up?", Scripted(call("run_backtest", strategy="sector_catch_up", sector="airlines"),
                                        call("get_fundamental_metrics", ticker="THYAO"), ANSWER.format("WATCH")), log=False)
    assert t["invalid_tool_calls"] == 1 and "already closed" in t["tool_calls"][-1]["error"], t["tool_calls"]
    idx = [STATE_INDEX[s] for s in t["states"]]
    assert idx == sorted(idx) and t["states"][-1] == "LOG_AND_CLOSE", t["states"]


def test_unverified_candidate_is_held_by_risk_gate():
    t = run("THYAO reversal?", Scripted(call("calculate_indicators", ticker="THYAO"), ANSWER.format("TECHNICAL_REVERSAL_CANDIDATE")), log=False)
    assert t["risk_gate"]["status"] == "INVESTIGATE" and t["state"] == "INVESTIGATE" and t["answer_source"] == "harness_fallback", t["risk_gate"]


def test_verified_candidate_passes():
    t = run("THYAO reversal?", Scripted(call("detect_reversal_events", ticker="THYAO", event="lower_bollinger_touch"),
                                        ANSWER.format("TECHNICAL_REVERSAL_CANDIDATE")), log=False)
    assert t["risk_gate"]["verification_tools_run"] == ["detect_reversal_events"], t["risk_gate"]
    assert "CHECK_DATA_QUALITY" in t["states"] and "RUN_BACKTEST" in t["states"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
