#!/usr/bin/env bash
# Full run on one rented NVIDIA GPU (used on RunPod, RTX 4090 24 GB): QLoRA training (train_local.sh), vLLM evaluation
# of the base and the fine-tuned model on every test row, reproducibility, harness ablation A-D, execution traces,
# manifest and a Q4_K_M GGUF for offline llama.cpp / Ollama.
#   bash bist_llm/training/run_on_runpod.sh        (re-running skips finished steps)
# Needs ~80 GB disk and CUDA in /usr/local/cuda (RunPod pytorch images). Results: runs/qwen35-4b-lora/, results/,
# release/, runs/*.gguf
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> bist_llm/
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
OUT="${OUT:-runs/qwen35-4b-lora}"
WORKERS="${WORKERS:-8}"
ABL_LIMIT="${ABL_LIMIT:-40}"
export MODEL OUT
mkdir -p runs results
log() { echo -e "\n[$(date +%H:%M:%S)] $*"; }

# vLLM pins its own torch, so it gets its own environment; installed while the model trains
if [ ! -x .venv-vllm/bin/vllm ]; then
  (python3 -m venv .venv-vllm && .venv-vllm/bin/pip install -q vllm) > runs/vllm_install.log 2>&1 &
  VINST=$!
fi

STEPS="setup" bash training/train_local.sh
if ! .venv/bin/python -c "import fla" 2>/dev/null; then
  log "fast kernels for Qwen3.5's linear-attention layers (flash-linear-attention, causal-conv1d)"
  .venv/bin/pip install -q flash-linear-attention ninja packaging || log "WARNING: flash-linear-attention not installed (slow fallback)"
  PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda .venv/bin/pip install -q --no-build-isolation causal-conv1d \
    > runs/causal_conv1d.log 2>&1 || log "WARNING: causal-conv1d not installed (slow fallback), see runs/causal_conv1d.log"
fi
STEPS="smoke" bash training/train_local.sh
[ -z "${VINST:-}" ] || wait "$VINST" || { tail -30 runs/vllm_install.log; exit 1; }

serve() {   # serve <model path or id> <served name>: start vLLM and wait until it answers
  log "vLLM server: $2"
  .venv-vllm/bin/vllm serve "$1" --served-model-name "$2" --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --generation-config vllm --seed 0 --port 8000 > "runs/vllm_$2.log" 2>&1 &
  VPID=$!
  until curl -sf localhost:8000/v1/models > /dev/null; do
    kill -0 $VPID 2>/dev/null || { tail -40 "runs/vllm_$2.log"; return 1; }; sleep 5
  done
}
stop() { kill $VPID 2>/dev/null || true; wait $VPID 2>/dev/null || true; }
trap stop EXIT
merge() { [ -f "$2/config.json" ] || .venv/bin/python training/finetune_qlora.py --merge --model "$MODEL" --out "$1" --merged-out "$2"; }

# fail early: base model and a merged smoke adapter must both serve and answer before the long training run
if [ ! -f runs/serve_check.ok ]; then
  merge runs/smoke runs/smoke-merged
  for spec in "$MODEL base" "runs/smoke-merged smoke"; do
    set -- $spec; serve "$1" "$2"
    .venv/bin/python -m harness.run_agent --backend vllm --base-url http://127.0.0.1:8000 --model "$2" \
      "THYAO 2024-01-01 ile 2026-09-10 arasında nasıl performans gösterdi?" | tail -3
    stop
  done
  rm -rf runs/smoke-merged; touch runs/serve_check.ok
fi

STEPS="train" bash training/train_local.sh
merge "$OUT" runs/merged

EV() { local tag=$1; shift; [ -f "results/eval_$tag.json" ] && return 0; log "evaluate $tag"
       .venv/bin/python -m harness.evaluate --backend vllm --base-url http://127.0.0.1:8000 --tag "$tag" "$@"; }
for who in base finetuned; do
  if [ $who = base ]; then serve "$MODEL" base; else serve runs/merged finetuned; fi
  EV "${who}_E" --model $who --workers "$WORKERS"
  [ $who = base ] || EV finetuned_repro --model finetuned --limit 15 --repeats 3
  for m in A B C D; do EV "${who}_$m" --model $who --mode $m --limit "$ABL_LIMIT" --workers "$WORKERS"; done
  stop
done
.venv/bin/python -m harness.compare results/eval_base_E.json results/eval_finetuned_E.json > results/compare_base_vs_finetuned.md

# execution traces with the HF backend (records the adapter SHA-256) and the manifest, as in train_local.sh
STEPS="traces manifest" bash training/train_local.sh

GG="runs/bist-analyst-qwen35-4b-Q4_K_M.gguf"
if [ ! -f "$GG" ]; then
  log "GGUF export (Q4_K_M); optional, the submission needs only the adapter"
  ( [ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
    [ -x .venv-gguf/bin/python ] || { python3 -m venv .venv-gguf && .venv-gguf/bin/pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt cmake; }
    .venv-gguf/bin/python llama.cpp/convert_hf_to_gguf.py runs/merged --outfile runs/bist-bf16.gguf --outtype bf16
    .venv-gguf/bin/cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
    .venv-gguf/bin/cmake --build llama.cpp/build --target llama-quantize -j > /dev/null
    llama.cpp/build/bin/llama-quantize runs/bist-bf16.gguf "$GG" Q4_K_M && rm -f runs/bist-bf16.gguf
  ) > runs/gguf.log 2>&1 || log "WARNING: GGUF export failed, see runs/gguf.log"
fi
log "done. Results: $OUT/ results/ release/ $GG"
