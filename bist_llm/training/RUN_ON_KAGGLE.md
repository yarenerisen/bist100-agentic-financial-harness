# Training and running the model

Base model for the project: **Qwen/Qwen3-8B** (fixed for baseline, fine-tuning and ablation; `unsloth/Qwen3-8B` is the
same weights). On Kaggle use the two notebooks: `notebooks/kaggle_finetune.ipynb` (training) and
`notebooks/kaggle_eval_export.ipynb` (evaluation, ablation, GGUF).
Qwen3 chats are rendered by `training/prompts.py` (empty think block on every assistant turn) so training and
inference match; for llama.cpp / Ollama use the raw backends (`--backend llamacpp` / `--backend ollama`).

Three ways to train, all using `training/finetune_qlora.py` (QLoRA, loss on assistant tokens only):

| Where | Notes |
|---|---|
| **Local GPU, one command** (≥ 16 GB VRAM, Linux/WSL2) | `bash bist_llm/training/train_local.sh`: setup, smoke test, training, evaluation, traces and manifest; re-run to resume |
| Local NVIDIA GPU (≥ 8 GB) | Tested on an RTX 4070 Laptop GPU (8 GB): Qwen2.5-3B, bf16, `--liger`, 2 epochs ≈ 3 h 50 min |
| Google Colab (free T4) | Open `notebooks/colab_finetune.ipynb` (File → Open notebook → GitHub tab → repo URL) and run the cells in order. Checkpoints go to Google Drive; after a disconnect run cells 1–3 again and continue with `--resume` |
| Kaggle (T4 x2) — **team default: Qwen3-8B** | Open `notebooks/kaggle_finetune.ipynb`. Choose T4, not P100 (`--liger` needs compute capability ≥ 7.0). Use Save & Run All for the long run (≈ 3 h per epoch). Add `--load-4bit` when evaluating 8B |

## 0) Environment
```bash
pip install -U "transformers>=4.45" "peft>=0.13" "bitsandbytes>=0.44" accelerate pyarrow pandas liger-kernel
# PyTorch with CUDA, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu128
```
On Kaggle/Colab:
```bash
!git clone -q https://github.com/yarenerisen/bist100-agentic-financial-harness.git /kaggle/working/repo
%cd /kaggle/working/repo/bist_llm
```

## 1) Baseline: evaluate the BASE model first (needed for the base vs fine-tuned comparison)
```bash
python -m harness.evaluate --backend hf --model Qwen/Qwen3-8B --load-4bit --limit 40 --tag base_E
```

## 2) Smoke test, then full training
```bash
python training/finetune_qlora.py --model Qwen/Qwen3-8B --out runs/smoke --limit 30 --epochs 1 --liger
python training/finetune_qlora.py --model Qwen/Qwen3-8B --out runs/bist-lora --epochs 1 --liger --resume
```
- `--liger`: fused linear + cross-entropy; the 152k-vocabulary logits are never materialised (needed for 6k-token
  examples on 8 GB GPUs; training and evaluation both fit).
- Out of memory: `--max-len 4096` (longer examples are skipped and counted).
- `--resume` continues from the last checkpoint (saved every `--save-steps`, default 20).
- `training_report.json` holds the settings and the loss history (for the report).

## 3) Evaluate the fine-tuned model (the LoRA adapter can be loaded directly, no merge needed)
```bash
python -m harness.evaluate --backend hf --model runs/bist-lora --load-4bit --limit 40 --tag finetuned_E
python -m harness.evaluate --backend hf --model runs/bist-lora --load-4bit --limit 10 --repeats 3 --tag finetuned_repro
```
Use the same `--limit` for base and fine-tuned runs: `--limit N` always takes the first N rows of `test.jsonl`, so the runs compare identical questions.

## 4) Harness ablation A–E (same model, different harness levels)
A = raw LLM (data as text), B = LLM + tools, C = stateful harness, D = + evidence and quality gate, E = + memory and human review.
```bash
for m in A B C D E; do python -m harness.evaluate --backend hf --model runs/bist-lora --load-4bit --mode $m --limit 20 --tag finetuned_$m; done
for m in A B E; do python -m harness.evaluate --backend hf --model Qwen/Qwen3-8B --load-4bit --mode $m --limit 20 --tag base_$m; done
```

## 5) Export to GGUF for llama.cpp / Ollama (fully offline inference)
```bash
python training/finetune_qlora.py --merge --model Qwen/Qwen3-8B --out runs/bist-lora --merged-out runs/bist-merged
git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py runs/bist-merged --outfile bist-analyst-q8_0.gguf --outtype q8_0
```

## 6) Run locally without internet
llama.cpp:
```bash
llama-server -m bist-analyst-q8_0.gguf -c 8192 --port 8080
python -m harness.run_agent --backend openai --base-url http://127.0.0.1:8080/v1 "Does THYAO show a Mon-Wed down, Thu-Fri up pattern?"
```
Ollama (`Modelfile`):
```
FROM ./bist-analyst-q8_0.gguf
PARAMETER temperature 0
PARAMETER num_ctx 8192
```
```bash
ollama create bist-analyst -f Modelfile
python -m harness.run_agent --backend openai --base-url http://127.0.0.1:11434/v1 --model bist-analyst "..."
```
Or directly with Transformers: `python -m harness.run_agent --backend hf --model runs/bist-lora "..."`.

## 7) Demo
```bash
python -m harness.run_agent --backend hf --model runs/bist-lora --review "question..."   # with human review
python -m harness.run_agent --replay logs/<file>.json                                    # reproducibility check
python -m harness.run_agent --bundle logs/<file>.json                                    # evidence bundle
```
