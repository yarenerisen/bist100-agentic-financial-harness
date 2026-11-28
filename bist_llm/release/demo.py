"""Live demo: loads the model once, then answers questions typed in the terminal (full harness, level E, offline).
Every answer is saved as a complete execution trace in release/live_traces/.
Run from bist_llm/:
    HF_HUB_OFFLINE=1 python -m release.demo --model <adapter folder>
Type a question and press Enter; an empty line or "q" quits."""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.run_agent import run, make_backend   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="LoRA adapter folder")
    ap.add_argument("--backend", default="hf", choices=["hf", "llamacpp", "ollama", "vllm"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no-4bit", action="store_true", help="hf backend: load in bf16 instead of 4-bit")
    ap.add_argument("--out-dir", default=str(ROOT / "release" / "live_traces"))
    a = ap.parse_args()
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
        t = run(q, be, log=True, mode="E")
        print("\nTool calls:")
        for c in t["tool_calls"]:
            if "tool" in c:
                print(f"  - {c['tool']}({json.dumps(c['arguments'], ensure_ascii=False)})")
        if not any("tool" in c for c in t["tool_calls"]):
            print("  (none)")
        print("\n" + t["answer"])
        f = out / f"live_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(t, ensure_ascii=False, indent=1, default=str))
        print(f"\n[state {t['state']} | answer source {t['answer_source']} | answer valid {t['answer_validation']['ok']} | "
              f"{t['latency_s']} s | trace {f.relative_to(ROOT)}]\n")
