#!/usr/bin/env bash
# One-command local run on a single NVIDIA GPU with >= 16 GB VRAM (Linux or Windows WSL2).
#   bash bist_llm/training/train_local.sh            # everything: setup, smoke test, training, evaluation, traces, manifest
#   STEPS="train eval" bash bist_llm/training/train_local.sh   # only some steps
# Every step is skipped if its output already exists, so the script can simply be re-run after an interruption.
# Needs: ~45 GB free disk (Qwen3.5-4B download 9 GB + environment 6 GB + outputs), 32 GB RAM recommended,
#        internet for the first run only.
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> bist_llm/
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
OUT="${OUT:-runs/qwen35-4b-lora}"
EVAL_LIMIT="${EVAL_LIMIT:-40}"
STEPS="${STEPS:-setup smoke train eval traces manifest}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
has() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo -e "\n[$(date +%H:%M:%S)] $*"; }

if has setup; then
  log "setup"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  [ -x "$PY" ] || python3 -m venv .venv
  $PY -m pip install -q --upgrade pip
  $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  $PY -m pip install -q -r ../requirements.txt
  $PY -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())"
fi

if has smoke && [ ! -f runs/smoke/training_report.json ]; then
  log "smoke test: 30 examples (checks memory and the Qwen3 chat format)"
  $PY training/finetune_qlora.py --model "$MODEL" --out runs/smoke --limit 30 --epochs 1 --save-steps 1000 --liger
fi

if has train && [ ! -f "$OUT/training_report.json" ]; then
  log "training: 1 epoch, checkpoint every 20 steps (re-run the script to resume)"
  $PY training/finetune_qlora.py --model "$MODEL" --out "$OUT" --epochs 1 --liger --resume
fi

if has eval; then
  for spec in "base_E $MODEL" "finetuned_E $OUT"; do
    set -- $spec
    [ -f "results/eval_$1.json" ] && { log "skip $1 (exists)"; continue; }
    log "evaluate $1 on the first $EVAL_LIMIT test rows"
    $PY -m harness.evaluate --backend hf --model "$2" --load-4bit --mode E --limit "$EVAL_LIMIT" --tag "$1"
  done
  [ -f results/eval_finetuned_repro.json ] || $PY -m harness.evaluate --backend hf --model "$OUT" --load-4bit --limit 10 --repeats 3 --tag finetuned_repro
fi

if has traces && [ ! -f release/execution_traces.json ]; then
  log "execution traces: 5 reference scenarios x (TR, EN)"
  $PY -m release.make_traces --backend hf --model "$OUT"
fi

if has manifest; then
  log "knowledge base + manifest"
  $PY -m release.build_knowledge_base
  REV=$($PY -c "from huggingface_hub import model_info; print(model_info('$MODEL').sha)" 2>/dev/null || echo "")
  $PY -m release.build_manifest --adapter "$OUT" --base-model "$MODEL" ${REV:+--base-revision $REV}
fi
log "done. Send back: $OUT/ (adapter_model.safetensors, adapter_config.json, training_report.json), results/, release/execution_traces.json, release/manifest.json"
