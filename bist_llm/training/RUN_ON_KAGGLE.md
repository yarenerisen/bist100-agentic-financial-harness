# Eğitim ve çalıştırma adımları (Kişi B)

## 0) Kaggle hazırlığı
1. kaggle.com → hesap → **telefon doğrula** (GPU için şart).
2. New Notebook → sağ panel: *Accelerator* = **GPU T4 x2** (veya P100), *Internet* = **On**.
3. `bist_llm.zip` dosyasını *Add Data → Upload* ile yükle (veya notebook'a sürükle). Zip'in içinde `tools/`, `training/`, `harness/`, `data/` olmalı.

## 1) Kurulum hücresi
```bash
!git clone -q https://github.com/yarenerisen/bist100-agentic-financial-harness.git /kaggle/working/repo
%cd /kaggle/working/repo/bist_llm
!pip install -q -U "transformers>=4.45" "peft>=0.13" "bitsandbytes>=0.44" accelerate pyarrow
```

## 2) Önce BASE modeli test et (raporda "base vs fine-tuned" karşılaştırması için zorunlu)
```python
!python -m harness.evaluate --backend hf --model Qwen/Qwen2.5-7B-Instruct --limit 40 --tag base
```
Sonuç: `results/eval_base.json`

## 3) Duman testi (5 dk) → sonra asıl eğitim
```bash
!python training/finetune_qlora.py --model Qwen/Qwen2.5-7B-Instruct --out /kaggle/working/bist-lora --limit 30 --epochs 1
!python training/finetune_qlora.py --model Qwen/Qwen2.5-7B-Instruct --out /kaggle/working/bist-lora --epochs 2
```
- Süre tahmini T4'te: 7B için epoch başına ~1–1.5 saat. Zaman sıkışırsa `--model Qwen/Qwen2.5-3B-Instruct`.
- Bellek hatası (OOM) olursa: `--max-len 4096` (uzun örnekler atlanır, kaç tane atlandığı yazdırılır).
- `training_report.json` → loss eğrisi ve ayarlar (rapora koy).

## 4) Birleştir + fine-tuned modeli değerlendir
```bash
!python training/finetune_qlora.py --merge --model Qwen/Qwen2.5-7B-Instruct --out /kaggle/working/bist-lora
!python -m harness.evaluate --backend hf --model /kaggle/working/bist-lora-merged --tag finetuned
!python -m harness.evaluate --backend hf --model /kaggle/working/bist-lora-merged --limit 20 --repeats 3 --tag finetuned_repro
```

## 5) GGUF'a çevir (laptopta yerel çalıştırmak için)
```bash
!git clone -q https://github.com/ggerganov/llama.cpp && pip install -q -r llama.cpp/requirements.txt
!python llama.cpp/convert_hf_to_gguf.py /kaggle/working/bist-lora-merged --outfile bist-analyst-f16.gguf --outtype f16
# quantize (llama.cpp derlemesi gerekir) ya da f16'yı indirip laptopta quantize et:
!cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j 4
!llama.cpp/build/bin/llama-quantize bist-analyst-f16.gguf bist-analyst-Q4_K_M.gguf Q4_K_M
```
`bist-analyst-Q4_K_M.gguf` (~4.7 GB, 3B modelde ~2 GB) dosyasını indir.

## 6) Laptopta yerel çalıştırma (internetsiz)
Seçenek A, llama.cpp:
```bash
llama-server -m bist-analyst-Q4_K_M.gguf -c 8192 --port 8080
python -m harness.run_agent --backend openai --base-url http://127.0.0.1:8080/v1 "THYAO'da Pzt-Sal-Çar düşüş, Per-Cum yükseliş deseni var mı?"
```
Seçenek B, Ollama: `Modelfile` içeriği:
```
FROM ./bist-analyst-Q4_K_M.gguf
PARAMETER temperature 0
PARAMETER num_ctx 8192
```
```bash
ollama create bist-analyst -f Modelfile
python -m harness.run_agent --backend openai --base-url http://127.0.0.1:11434/v1 --model bist-analyst "..."
```
Not: Ollama GGUF'un içindeki Qwen sohbet şablonunu kullanır; harness araç sonuçlarını `<tool_response>` ile sardığı için eğitimdeki formatla aynıdır.

## 6b) Harness deneyleri A–E (proje dokümanı Bölüm 30, rapor tablosu)
Aynı model, farklı harness seviyeleri: A = ham LLM (veri metin olarak), B = araçlı, C = durum makinesi, D = kanıt ve kalite kapısı, E = hafıza ve insan onayı.
```bash
for m in A B C D E; do python -m harness.evaluate --backend hf --model /kaggle/working/bist-lora-merged --mode $m --limit 40 --tag ft_$m; done
for m in A B D; do python -m harness.evaluate --backend hf --model Qwen/Qwen2.5-7B-Instruct --mode $m --limit 40 --tag base_$m; done
```

## 7) Demo (canlı test)
```bash
python -m harness.run_agent --backend openai --review "soru..."     # human review adımıyla
python -m harness.run_agent --replay logs/<dosya>.json               # tekrar üretilebilirlik kontrolü
```
