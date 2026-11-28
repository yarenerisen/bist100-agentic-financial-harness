"""Live demo: loads the model once, then answers questions typed in the terminal (full harness, level E, offline).
Every answer is saved as a complete execution trace in release/live_traces/.
Run from bist_llm/:
    HF_HUB_OFFLINE=1 python -m release.demo --model <adapter folder>
Type a question and press Enter; an empty line or "q" quits.

Without a GPU (to work on the display): python -m release.demo --from-traces release/live_check_traces.json
shows saved traces instead of running the model (pick one by number)."""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from release.answer_format import parse_answer, tool_rows, STATE_KIND   # noqa: E402

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:          # plain-text fallback
    Console = None

COLOR = {"candidate": "green", "watch": "yellow", "investigate": "cyan", "reject": "red", "unsafe": "magenta"}


def show_plain(t, saved):
    print("\nTool calls:")
    for tool, args, status in tool_rows(t):
        print(f"  - {tool}({json.dumps(args, ensure_ascii=False)})  [{status}]")
    print("\n" + t["answer"])
    print(f"\n[state {t['state']} | answer source {t['answer_source']} | answer valid {t['answer_validation']['ok']} | "
          f"{t['latency_s']} s | trace {saved}]\n")


def show_rich(con, t, saved):
    p = parse_answer(t["answer"])
    state = p["state"] or t.get("state", "")
    color = COLOR.get(STATE_KIND.get(state, ""), "white")

    tools = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False)
    tools.add_column("#", style="dim", width=3)
    tools.add_column("Tool", style="bold")
    tools.add_column("Arguments", overflow="fold")
    tools.add_column("Result", width=12, overflow="fold")
    rows = tool_rows(t)
    for i, (tool, args, status) in enumerate(rows, 1):
        arg = ", ".join(f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}" for k, v in args.items())
        tools.add_row(str(i), tool, arg, Text(status, style="green" if status == "ok" else "red"))
    con.print(Panel(tools if rows else Text("no tool call (answered without data)", style="dim"),
                    title="Tool calls", title_align="left", border_style="blue"))

    head = Text()
    head.append(f" {state} ", style=f"bold black on {color}")
    if p["decision_basis"]:
        head.append("\n\n" + p["decision_basis"])
    con.print(Panel(head, title="Analytical state  ·  decision basis", title_align="left", border_style=color))

    if p["evidence"]:
        ev = Table(box=None, show_header=False, expand=True, padding=(0, 1))
        ev.add_column(style="bold", width=4)
        ev.add_column(overflow="fold")
        for lab, txt in p["evidence"]:
            ev.add_row(lab, txt)
        con.print(Panel(ev, title="Evidence (numbers from tool outputs)", title_align="left", border_style="white"))
    if p["limitations"]:
        con.print(Panel(p["limitations"], title="Limitations", title_align="left", border_style="dim"))
    for line in p["other"]:
        con.print(line)
    if p["note"]:
        con.print(Text(p["note"], style="italic dim"))

    v = t["answer_validation"]
    src = t["answer_source"]
    foot = Text()
    foot.append("answer source ", style="dim"); foot.append(src, style="green" if src.startswith("model") else "magenta")
    foot.append("  ·  validation ", style="dim"); foot.append("passed" if v["ok"] else "failed", style="green" if v["ok"] else "red")
    foot.append(f"  ·  {t['latency_s']} s  ·  trace {saved}", style="dim")
    con.print(foot)
    con.print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="LoRA adapter folder")
    ap.add_argument("--backend", default="hf", choices=["hf", "llamacpp", "ollama", "vllm"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no-4bit", action="store_true", help="hf backend: load in bf16 instead of 4-bit")
    ap.add_argument("--out-dir", default=str(ROOT / "release" / "live_traces"))
    ap.add_argument("--from-traces", help="show saved traces from this file instead of running the model (no GPU needed)")
    ap.add_argument("--plain", action="store_true", help="plain text output")
    a = ap.parse_args()
    con = Console() if (Console and not a.plain) else None
    show = (lambda t, saved: show_rich(con, t, saved)) if con else show_plain

    if a.from_traces:
        traces = json.loads(Path(a.from_traces).read_text())["traces"]
        for i, t in enumerate(traces, 1):
            print(f"{i:2d}. {t['question'][:110]}")
        while True:
            k = input("\nTrace number> ").strip()
            if not k.isdigit() or not 1 <= int(k) <= len(traces):
                break
            t = traces[int(k) - 1]
            print("\n" + t["question"] + "\n")
            show(t, a.from_traces)
        raise SystemExit

    if not a.model:
        ap.error("--model is required (or use --from-traces)")
    from harness.run_agent import run, make_backend
    a.load_4bit, a.allow_remote = not a.no_4bit, False
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("loading the model (about a minute)...", flush=True)
    be = make_backend(a)
    print("ready. Analysis date 2026-09-10. Type a question (empty line or q to quit).\n", flush=True)
    while True:
        try:
            q = input("Soru / Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("q", "quit", "exit", "çık"):
            break
        if con:
            with con.status("running the harness (tools, model, validation)..."):
                t = run(q, be, log=True, mode="E")
        else:
            t = run(q, be, log=True, mode="E")
        f = out / f"live_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(t, ensure_ascii=False, indent=1, default=str))
        show(t, f.relative_to(ROOT))
