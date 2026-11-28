"""Simple local web UI: type a question, get the harness answer. Same run as release/demo.py, in the browser.
Display only: the answer text is shown exactly as the harness returns it. Every answer is saved in release/live_traces/.

Run from bist_llm/ (opens on http://127.0.0.1:8501; nothing is sent to the internet):

    HF_HUB_OFFLINE=1 streamlit run release/ui.py -- --model runs/qwen35-4b-lora-v5

Without a GPU, to check the page with saved answers:  streamlit run release/ui.py -- --traces
Options after "--": --model, --backend hf|llamacpp|ollama|vllm, --base-url, --no-4bit, --traces [file], --out-dir
"""
import argparse, html, json, os, sys, threading, time, warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from release.answer_format import parse_answer, tool_rows, STATE_KIND  # noqa: E402

COLOR = {"candidate": "#1a7f37", "watch": "#b58105", "investigate": "#0a7ea4", "reject": "#c62828",
         "unsafe": "#8e24aa"}
esc = html.escape

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--model", help="LoRA adapter folder (or model name for llamacpp/ollama/vllm)")
ap.add_argument("--mode", choices=["model", "traces"], help="'traces' is the same as --traces")
ap.add_argument("--backend", default="hf", choices=["hf", "llamacpp", "ollama", "vllm"])
ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
ap.add_argument("--no-4bit", action="store_true", help="hf backend: load in bf16 instead of 4-bit")
ap.add_argument("--traces", nargs="?", const=str(ROOT / "release" / "live_check_traces.json"),
                help="show saved answers from this file instead of running the model (no GPU needed)")
ap.add_argument("--out-dir", default=str(ROOT / "release" / "live_traces"))
A = ap.parse_args(sys.argv[1:])
if A.mode == "traces" and not A.traces:
    A.traces = str(ROOT / "release" / "live_check_traces.json")


@st.cache_resource(show_spinner="Loading the model (about a minute)...")
def get_backend():
    from harness.run_agent import make_backend
    return make_backend(argparse.Namespace(backend=A.backend, model=A.model, base_url=A.base_url,
                                           load_4bit=not A.no_4bit, allow_remote=False))


@st.cache_resource
def run_lock():
    return threading.Lock()   # one question at a time on the GPU


CSS = """<style>
.bx-badge{display:inline-block;color:#fff;font-weight:700;padding:4px 12px;border-radius:6px;
          font-family:ui-monospace,Menlo,monospace}
.bx-p{white-space:pre-wrap;margin:8px 0}
.bx-h{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.6;margin-top:18px}
.bx-ev{margin:4px 0;white-space:pre-wrap}.bx-ev b{display:inline-block;min-width:2.4em}
.bx-small{font-size:.85rem;opacity:.75}
.bx-warn{border:1px solid #8e24aa;background:rgba(142,36,170,.12);border-radius:8px;padding:8px 12px;font-weight:600}
.bx-tbl{width:100%;border-collapse:collapse;font-size:.85rem}
.bx-tbl td{padding:4px 6px;border-bottom:1px solid rgba(128,128,128,.2);vertical-align:top;word-break:break-word}
.bx-ok{color:#1a7f37}.bx-err{color:#c62828;font-weight:700}
</style>"""


def show_answer(t, saved=None):
    p = parse_answer(t.get("answer", ""))
    state = t.get("state") or p["state"]
    color = COLOR.get(STATE_KIND.get(state, ""), "#555")
    out = [CSS]
    if t.get("answer_source") == "harness_fallback":
        out.append('<div class="bx-warn">The model answer failed validation twice; the harness returned a safe '
                   'notice instead.</div>')
    out.append(f'<span class="bx-badge" style="background:{color}">{esc(state)}</span>')
    if p["decision_basis"]:
        out.append(f'<div class="bx-p">{esc(p["decision_basis"])}</div>')
    if p["evidence"]:
        out.append('<div class="bx-h">Evidence</div>')
        out += [f'<div class="bx-ev"><b>{esc(lab)}</b> {esc(txt)}</div>' for lab, txt in p["evidence"]]
    if p["limitations"]:
        out.append(f'<div class="bx-h">Limitations</div><div class="bx-p">{esc(p["limitations"])}</div>')
    out += [f'<div class="bx-p">{esc(x)}</div>' for x in p["other"]]
    if p["note"]:
        out.append(f'<div class="bx-p bx-small"><i>{esc(p["note"])}</i></div>')
    ok = (t.get("answer_validation") or {}).get("ok")
    foot = (f'answer source {esc(t.get("answer_source", ""))} · validation '
            f'{"passed" if ok else "failed"} · {esc(str(t.get("latency_s", "")))} s')
    if saved:
        foot += f" · saved as {esc(str(saved))}"
    out.append(f'<div class="bx-small" style="margin-top:14px">{foot}</div>')
    st.html("".join(out))

    rows = tool_rows(t)
    with st.expander(f"Tool calls ({len(rows)})"):
        tbl = "".join(f'<tr><td><b>{esc(tool)}</b></td><td>{esc(json.dumps(args, ensure_ascii=False))}</td>'
                      f'<td class="{"bx-ok" if s == "ok" else "bx-err"}">{esc(s)}</td></tr>'
                      for tool, args, s in rows)
        st.html(CSS + f'<table class="bx-tbl">{tbl}</table>' if rows else "no tool call")
    with st.expander("Raw answer text"):
        st.code(t.get("answer", ""), language=None, wrap_lines=True)


st.set_page_config(page_title="BIST 100 Analyst", layout="centered")
st.title("BIST 100 Analyst")
st.caption("Analysis date 2026-09-10 · local model and tools · historical analysis only, not investment advice")

if A.traces:      # no GPU: pick a saved answer
    traces = json.loads(Path(A.traces).read_text())["traces"]
    k = st.selectbox("Saved question", range(len(traces)), format_func=lambda i: traces[i]["question"])
    st.caption("Saved answer from an earlier run, not computed now.")
    show_answer(traces[k])
    st.stop()

if not A.model:
    st.error('Start with --model <adapter folder>, e.g.  streamlit run release/ui.py -- --model runs/qwen35-4b-lora-v5')
    st.stop()
backend = get_backend()

with st.form("ask"):
    q = st.text_area("Question (Turkish or English)", height=120)
    go = st.form_submit_button("Ask", type="primary")

if go and q.strip():
    from harness.run_agent import run
    with st.spinner("Thinking (tools, model, checks)... 20-60 s"):
        with run_lock():
            t = run(q.strip(), backend, log=True, mode="E")
    out = Path(A.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"live_{time.strftime('%Y%m%d_%H%M%S')}.json"
    f.write_text(json.dumps(t, ensure_ascii=False, indent=1, default=str))
    st.session_state["last"] = (t, f.name)

if st.session_state.get("last"):
    show_answer(*st.session_state["last"])
