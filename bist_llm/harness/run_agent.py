"""Stateful inference harness: model proposes tool calls, harness validates & executes them, then validates the answer.

States: START -> PLAN_TOOL_CALL -> VALIDATE_CALL -> EXECUTE_TOOL -> (repeat) -> QUALITY_GATE -> GENERATE_EXPLANATION
        -> VALIDATE_ANSWER -> HUMAN_REVIEW (optional) -> LOG_AND_CLOSE
Backends (all local, no cloud AI APIs):
  --backend openai   : OpenAI-compatible local server (llama.cpp `llama-server`, Ollama `/v1`, vLLM), default http://127.0.0.1:8080/v1
  --backend hf       : Hugging Face transformers model path (e.g. merged fine-tuned model on Colab/Kaggle)
  --backend replay   : replays reference answers from a dataset row (pipeline test without a model)
Usage:
  python -m harness.run_agent --backend openai --model bist-analyst "THYAO ... ?"
"""
import argparse, datetime, json, platform, re, sys, time, uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.registry import call_tool, TOOLS            # noqa: E402
from tools import quality, core                        # noqa: E402
from training.prompts import SYSTEM_PROMPT, chat_style, render_prompt   # noqa: E402
from harness.validate import check_answer, parse_tool_call  # noqa: E402
from harness.evidence import build_evidence_bundle  # noqa: E402
from harness import memory  # noqa: E402

LOG_DIR = ROOT / "logs"
MAX_TOOL_CALLS = 6
TOOL_JSON = lambda r: json.dumps(r, ensure_ascii=False, separators=(",", ":"))
# state machine: which tools the model may call (all analysis tools; quality gate is run by the harness itself)
ALLOWED_TOOLS = set(TOOLS) - {"check_data_quality"}


DECODING = {"temperature": 0.0, "do_sample": False, "top_p": 1.0, "seed": 0, "max_new_tokens": 1500}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_version():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ backends
class OpenAIBackend:
    def __init__(self, base_url, model, allow_remote=False):
        host = urlparse(base_url).hostname
        if host not in ("127.0.0.1", "localhost", "0.0.0.0") and not allow_remote:
            raise SystemExit(f"refusing non-local endpoint {base_url}: the system must not call cloud AI APIs")
        self.url, self.model = base_url.rstrip("/") + "/chat/completions", model

    def describe(self):
        return {"backend": "openai-compatible local server", "endpoint": self.url, "model": self.model, "decoding": DECODING}

    def chat(self, messages):
        import urllib.request
        # tool messages are sent as user turns wrapped in <tool_response>, identical to the Qwen chat templates
        msgs = [{"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"} if m["role"] == "tool" else m for m in messages]
        body = json.dumps({"model": self.model, "messages": msgs, "temperature": 0, "top_p": 1, "seed": 0, "max_tokens": 1500}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class HFBackend:
    def __init__(self, path, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {}
        if load_4bit:   # fits a 7B model on a 16 GB T4
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        # one device: avoids "Expected all tensors to be on the same device" when several GPUs are visible
        dmap = {"": 0} if torch.cuda.is_available() else None
        adapter = Path(path) / "adapter_config.json"
        if adapter.exists():   # LoRA adapter directory: load base model + adapter without merging
            from peft import PeftModel
            base = json.loads(adapter.read_text())["base_model_name_or_path"]
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map=dmap, **kw), path)
        else:
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map=dmap, **kw)
        self.model.eval()
        self.style = chat_style(path + " " + str(getattr(self.tok, "chat_template", "")))
        self.torch = torch
        self.info = {"backend": "transformers", "model_path": str(path), "load_4bit": load_4bit, "chat_style": self.style,
                     "decoding": DECODING}
        if adapter.exists():
            cfg = json.loads(adapter.read_text())
            w = Path(path) / "adapter_model.safetensors"
            self.info.update({"base_model": cfg.get("base_model_name_or_path"), "adapter": str(path),
                              "adapter_sha256": _sha256(w) if w.exists() else None,
                              "lora": {k: cfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules")}})
        else:
            self.info["base_model"] = str(path)
        try:
            import transformers, peft
            self.info["libraries"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
        except Exception:
            pass

    def describe(self):
        return self.info

    def chat(self, messages):
        text = render_prompt(messages, self.style, self.tok)
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
        self.torch.manual_seed(DECODING["seed"])
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=DECODING["max_new_tokens"], do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class RawBackend:
    """Sends the exact training-format prompt to a local llama.cpp server (/completion) or Ollama (/api/generate raw).
    Use this for Qwen3 models so multi-turn tool conversations are rendered exactly as in training."""
    def __init__(self, base_url, kind, model="bist-analyst", style="qwen3", allow_remote=False):
        host = urlparse(base_url).hostname
        if host not in ("127.0.0.1", "localhost", "0.0.0.0") and not allow_remote:
            raise SystemExit(f"refusing non-local endpoint {base_url}: the system must not call cloud AI APIs")
        self.url, self.kind, self.model, self.style = base_url.rstrip("/"), kind, model, style

    def describe(self):
        return {"backend": self.kind, "endpoint": self.url, "model": self.model, "chat_style": self.style, "decoding": DECODING}

    def chat(self, messages):
        import urllib.request
        prompt = render_prompt(messages, self.style)
        if self.kind == "llamacpp":
            url, body = self.url + "/completion", {"prompt": prompt, "temperature": 0, "seed": 0, "n_predict": 1500,
                                                    "stop": ["<|im_end|>"], "cache_prompt": True}
        else:
            url, body = self.url + "/api/generate", {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                                                     "options": {"temperature": 0, "seed": 0, "num_predict": 1500,
                                                                 "num_ctx": 8192, "stop": ["<|im_end|>"]}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        return d.get("content") if self.kind == "llamacpp" else d.get("response")


class ReplayBackend:
    """Returns the reference assistant messages of a dataset row in order (tests the harness end to end)."""
    def __init__(self, row):
        self.queue = [m["content"] for m in row["messages"][2:] if m["role"] == "assistant"]

    def describe(self):
        return {"backend": "replay (reference answers, pipeline test only - NOT a model)", "decoding": None}

    def chat(self, messages):
        return self.queue.pop(0)


# ------------------------------------------------------------------ harness
def _tickers_in(args):
    out = []
    for k in ("ticker", "tickers", "sector"):
        v = args.get(k)
        if isinstance(v, list):
            out += v
        elif isinstance(v, str):
            try:
                out += core._resolve(v)
            except Exception:
                out.append(v)
    return out


def safe_fallback(reason, lang_tr):
    if lang_tr:
        return ("ANALYTICAL_STATE: ANALYSIS_UNSAFE\nEVIDENCE:\n- E1: Sistem bu soru için doğrulanmış bir analiz üretemedi.\n"
                f"LIMITATIONS: {reason}\nNOTE: Yalnızca tarihsel ve eğitim amaçlı analizdir; yatırım tavsiyesi veya tahmin değildir.")
    return ("ANALYTICAL_STATE: ANALYSIS_UNSAFE\nEVIDENCE:\n- E1: The system could not produce a validated analysis for this question.\n"
            f"LIMITATIONS: {reason}\nNOTE: Historical, educational analysis only; not investment advice and not a forecast.")


RAW_SYSTEM = "You are a financial analysis assistant for Borsa Istanbul stocks. Answer the question using the data provided."
MODES = {  # harness ablation levels
    "A": "raw LLM: data passed as text, no tools, no checks",
    "B": "LLM + tools: tool loop, no call/answer validation",
    "C": "tools + stateful harness: validated calls, allowed tools per state, call limit",
    "D": "C + evidence bundle + quality gate + answer validation/fallback",
    "E": "D + episodic memory + human review + decision log",
}


def run(question, backend, review=False, log=True, mode="E", raw_data=None):
    t0 = time.time()
    lang_tr = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", question))
    trace = {"id": str(uuid.uuid4())[:8], "timestamp_start": _now(), "analysis_date": "2026-09-10", "mode": mode,
             "model": backend.describe() if hasattr(backend, "describe") else {"backend": type(backend).__name__},
             "code_version": _code_version(), "python": platform.python_version(),
             "system_prompt": RAW_SYSTEM if mode == "A" else SYSTEM_PROMPT, "user_prompt": question,
             "question": question, "states": ["START"], "model_turns": [], "tool_calls": [],
             "invalid_tool_calls": 0, "quality_gate": None, "answer_retries": 0}
    tool_texts, tickers = [], []
    source = {"value": "model"}

    def ask(messages, purpose):
        """Every model call is recorded with timestamps and its raw output."""
        t_start, ts = _now(), time.time()
        raw = backend.chat(messages)
        trace["model_turns"].append({"turn": len(trace["model_turns"]) + 1, "purpose": purpose, "started_at": t_start,
                                     "finished_at": _now(), "latency_s": round(time.time() - ts, 2), "raw_output": raw})
        return re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", raw or "").strip()

    def fallback(reason):
        source["value"] = "harness_fallback"
        trace["fallback_reason"] = reason
        return safe_fallback(reason, lang_tr)

    if mode == "A":
        msgs = [{"role": "system", "content": RAW_SYSTEM}, {"role": "user", "content": question + "\n\nDATA:\n" + (raw_data or "")}]
        answer = ask(msgs, "answer")
        tool_texts = [raw_data or ""]
        msgs_final = msgs
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        answer = None
        strict = mode in ("C", "D", "E")
        for step in range(MAX_TOOL_CALLS + 1):
            trace["states"].append("PLAN_TOOL_CALL")
            out = ask(msgs, "tool_call_or_answer")
            if out.startswith("{") or out.startswith("```"):
                call, err = parse_tool_call(out)
                if strict:
                    trace["states"].append("VALIDATE_CALL")
                    if err is None and call["tool"] not in ALLOWED_TOOLS:
                        err = f"tool {call['tool']} not allowed in this state"
                if err:
                    trace["invalid_tool_calls"] += 1
                    trace["tool_calls"].append({"raw": out[:500], "error": err})
                    if not strict:
                        answer = out; break
                    if trace["invalid_tool_calls"] >= 2:
                        answer = fallback(f"invalid tool call: {err}"); break
                    msgs += [{"role": "assistant", "content": out},
                             {"role": "tool", "content": TOOL_JSON({"error": "INVALID_CALL", "message": err})}]
                    continue
                trace["states"].append("EXECUTE_TOOL")
                t_start, ts = _now(), time.time()
                res = call_tool(call["tool"], call["arguments"])
                if "error" in res:
                    trace["invalid_tool_calls"] += res["error"] in ("INVALID_CALL", "UNKNOWN_TOOL")
                trace["tool_calls"].append({"tool": call["tool"], "arguments": call["arguments"], "started_at": t_start,
                                            "finished_at": _now(), "latency_s": round(time.time() - ts, 3), "result": res})
                tickers += _tickers_in(call["arguments"])
                txt = TOOL_JSON(res); tool_texts.append(txt)
                msgs += [{"role": "assistant", "content": json.dumps(call, ensure_ascii=False)}, {"role": "tool", "content": txt}]
                if not strict and step >= MAX_TOOL_CALLS:
                    break
                continue
            answer = out
            break
        if answer is None:
            answer = fallback("too many tool calls")
        msgs_final = msgs
    if mode in ("D", "E"):
        # ---- quality gate (deterministic, run by the harness, not the model)
        trace["states"].append("QUALITY_GATE")
        valid_t = [t for t in dict.fromkeys(tickers) if t in core.available_tickers()]
        if valid_t:
            qg = quality.check_data_quality(valid_t)
            trace["quality_gate"] = {"status": qg["status"], "hard_blocks": qg["hard_blocks"], "warnings": qg["warnings"]}
            if qg["status"] == "HARD_BLOCK":
                trace["model_answer_before_gate"] = answer
                answer = fallback("data-quality gate HARD_BLOCK: " + "; ".join(qg["hard_blocks"]))
        # ---- validate answer; one retry, then safe fallback
        trace["states"].append("VALIDATE_ANSWER")
        chk = check_answer(answer, tool_texts + [question])
        if not chk["ok"] and source["value"] != "harness_fallback":
            trace["answer_retries"] += 1
            trace["first_answer"], trace["first_answer_issues"] = answer, chk["issues"]
            retry = msgs_final + [{"role": "assistant", "content": answer},
                                  {"role": "user", "content": "Your answer failed validation: " + "; ".join(chk["issues"]) +
                                   ". Rewrite it using only numbers that appear in the tool results, the required format and an allowed state."}]
            answer2 = ask(retry, "validation_retry") if not isinstance(backend, ReplayBackend) else answer
            chk2 = check_answer(answer2, tool_texts + [question])
            if chk2["ok"]:
                answer, chk = answer2, chk2
                source["value"] = "model_after_validation_retry"
            else:
                answer = fallback("answer failed validation: " + "; ".join(chk["issues"]))
        trace["states"].append("BUILD_EVIDENCE")
        trace["evidence_bundle"] = build_evidence_bundle(trace["tool_calls"], answer, question)
    trace["answer_validation"] = check_answer(answer, tool_texts + [question])
    trace["answer"] = answer
    trace["answer_source"] = source["value"]   # model | model_after_validation_retry | harness_fallback
    trace["state"] = trace["answer_validation"]["state"]
    # ---- human review + episodic memory
    if mode == "E" and review:
        trace["states"].append("HUMAN_REVIEW")
        past = memory.similar([t for t in tickers if t in core.available_tickers()])
        if past:
            print("\nSimilar past cases (episodic memory):")
            for p in past:
                print(f"  {p['time']} {p['tickers']} -> {p['state']} | human: {p['human_decision']} {p['human_reason'] or ''}")
        print("\n" + answer + "\n")
        dec = input("Human review [a]ccept / [r]eject / [m]odify: ").strip().lower()[:1] or "a"
        reason = input("Reason: ").strip() if dec in ("r", "m") else ""
        trace["human_review"] = {"decision": {"a": "ACCEPT", "r": "REJECT", "m": "MODIFY"}.get(dec, "ACCEPT"), "reason": reason}
    trace["states"].append("LOG_AND_CLOSE")
    trace["timestamp_end"] = _now()
    trace["latency_s"] = round(time.time() - t0, 2)
    trace["n_tool_calls"] = len([c for c in trace["tool_calls"] if "tool" in c])
    if log and mode == "E":
        LOG_DIR.mkdir(exist_ok=True)
        (LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{trace['id']}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1, default=str))
        memory.record(trace, [t for t in tickers if t in core.available_tickers()])
    return trace


def make_backend(a):
    if a.backend == "openai":
        return OpenAIBackend(a.base_url, a.model, a.allow_remote)
    if a.backend in ("llamacpp", "ollama"):
        url = a.base_url if "8080/v1" not in a.base_url else ("http://127.0.0.1:8080" if a.backend == "llamacpp" else "http://127.0.0.1:11434")
        return RawBackend(url, a.backend, a.model, getattr(a, "chat_style", "qwen3"), a.allow_remote)
    return HFBackend(a.model, a.load_4bit)


def replay_log(path):
    """Re-executes the logged tool calls and checks the outputs are identical (reproducibility)."""
    t = json.loads(Path(path).read_text())
    same = all(call_tool(c["tool"], c["arguments"]) == c["result"] for c in t["tool_calls"] if "tool" in c)
    return {"id": t["id"], "tool_outputs_reproduced": same, "state": t.get("state")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "llamacpp", "ollama"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--allow-remote", action="store_true", help="allow a non-localhost endpoint you host yourself (e.g. Colab tunnel)")
    ap.add_argument("--review", action="store_true", help="ask for human review before logging")
    ap.add_argument("--replay", help="re-run tool calls of a decision log and check reproducibility")
    ap.add_argument("--bundle", help="print the evidence bundle of a decision log")
    ap.add_argument("--mode", default="E", choices=list(MODES), help="harness ablation level (default E = full harness)")
    ap.add_argument("--load-4bit", action="store_true", help="hf backend: load the model in 4-bit (7B on a T4)")
    a = ap.parse_args()
    if a.replay:
        print(json.dumps(replay_log(a.replay), indent=1)); sys.exit()
    if a.bundle:
        print(json.dumps(json.loads(Path(a.bundle).read_text()).get("evidence_bundle"), indent=1, ensure_ascii=False)); sys.exit()
    be = make_backend(a)
    q = a.question or input("Question: ")
    tr = run(q, be, review=a.review, mode=a.mode)
    print("\n" + tr["answer"])
    print(f"\n[tools: {[c.get('tool') for c in tr['tool_calls']]} | quality: {tr['quality_gate'] and tr['quality_gate']['status']} | {tr['latency_s']}s | log id {tr['id']}]")
