"""QLoRA fine-tuning on Kaggle (T4/P100) or Colab (T4). Loss only on assistant tokens (tool calls + final answers).

Kaggle/Colab setup (one cell):
    !pip install -q -U "transformers>=4.45" "peft>=0.13" "bitsandbytes>=0.44" accelerate datasets
Run:
    !python training/finetune_qlora.py --model Qwen/Qwen2.5-7B-Instruct --out /kaggle/working/bist-lora
Fast iteration / smaller GPU:
    --model Qwen/Qwen2.5-3B-Instruct
Then merge for GGUF export:
    !python training/finetune_qlora.py --merge --model Qwen/Qwen2.5-7B-Instruct --out /kaggle/working/bist-lora
"""
import argparse, json, math, os, random, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--data", default=str(Path(__file__).resolve().parent))
ap.add_argument("--out", default="bist-lora")
ap.add_argument("--max-len", type=int, default=6144)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="use only N training rows (smoke test)")
ap.add_argument("--merge", action="store_true", help="merge LoRA adapter into fp16 weights (for GGUF export)")
a = ap.parse_args()

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)

tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.pad_token or tok.eos_token

if a.merge:
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cpu")
    m = PeftModel.from_pretrained(base, a.out).merge_and_unload()
    m.save_pretrained(a.out + "-merged", safe_serialization=True); tok.save_pretrained(a.out + "-merged")
    print("merged ->", a.out + "-merged"); raise SystemExit


def _ids(messages, gen=False):
    """token ids of the rendered chat (robust to transformers versions returning a dict)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(messages):
    """input_ids for the whole chat; labels = -100 except assistant message tokens (incl. end-of-turn)."""
    ids = _ids(messages)
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(_ids(messages[:i], gen=True))
        end = len(_ids(messages[:i + 1]))
        assert ids[:end] == _ids(messages[:i + 1]), "chat template is not prefix-stable"
        labels[start:end] = ids[start:end]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load(split):
    rows = [json.loads(l)["messages"] for l in open(Path(a.data) / f"{split}.jsonl")]
    if a.limit and split == "train":
        rows = rows[:a.limit]
    enc, skipped = [], 0
    for r in rows:
        e = encode(r)
        if len(e["input_ids"]) > a.max_len:
            skipped += 1; continue
        enc.append(e)
    print(f"{split}: {len(enc)} examples, skipped {skipped} longer than {a.max_len} tokens")
    return enc


train, val = load("train"), load("val")
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(
    a.model, device_map="auto", torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16))
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
    logging_steps=5, eval_strategy="steps", eval_steps=40, save_strategy="steps", save_steps=40, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", bf16=bf16, fp16=not bf16,
    gradient_checkpointing=True, optim="paged_adamw_8bit", report_to="none", group_by_length=True, seed=42)
t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
trainer.train()
trainer.save_model(a.out); tok.save_pretrained(a.out)
hist = {"base_model": a.model, "train_examples": len(train), "val_examples": len(val), "epochs": a.epochs, "lr": a.lr,
        "lora_rank": a.rank, "lora_alpha": a.alpha, "grad_accum": a.grad_accum, "max_len": a.max_len,
        "minutes": round((time.time() - t0) / 60, 1), "log_history": trainer.state.log_history}
Path(a.out, "training_report.json").write_text(json.dumps(hist, indent=1))
print("saved adapter + training_report.json to", a.out)
