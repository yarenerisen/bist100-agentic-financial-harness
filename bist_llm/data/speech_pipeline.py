"""Speech-to-text pipeline for pre-approved public finance videos (runs locally, no cloud APIs).

1. Download the approved video's audio yourself (respect the platform's terms) and note its metadata.
2. python data/speech_pipeline.py --audio talk.m4a --source-id damodaran_2026_01 --url https://... \
       --speaker "Aswath Damodaran" --channel "NYU Stern" --published 2026-01-15 [--start 0 --end 600]
3. Segments are appended to data/speech/features.jsonl with topics, stance and numeric claims (verified=false).
   Numeric claims must be checked against the deterministic tools before they enter a committed evidence set.
Requires: pip install faster-whisper ; ffmpeg on PATH."""
import argparse, json, re, subprocess, tempfile, wave
from pathlib import Path

OUT = Path(__file__).resolve().parent / "speech" / "features.jsonl"
TOPICS = {
    "inflation": ["inflation", "enflasyon", "cpi", "tüfe"], "rates": ["interest rate", "faiz", "policy rate", "fed", "tcmb", "merkez bankası"],
    "fx": ["dollar", "dolar", "kur", "exchange rate", "lira", "euro"], "valuation": ["valuation", "değerleme", "dcf", "multiple", "f/k", "p/e"],
    "earnings": ["earnings", "kâr", "kar", "bilanço", "revenue", "hasılat", "ebitda", "favök"],
    "risk": ["risk", "uncertainty", "belirsizlik", "volatility", "oynaklık"], "banks": ["bank", "banka", "kredi", "credit"],
}
POS = ["opportunity", "fırsat", "cheap", "ucuz", "strong", "güçlü", "improve", "iyileş", "olumlu", "positive"]
NEG = ["expensive", "pahalı", "weak", "zayıf", "risk", "worse", "kötüleş", "olumsuz", "negative", "bubble", "balon"]
NUMC = re.compile(r"[^.?!]*\d+(?:[.,]\d+)?\s*(?:%|percent|yüzde|bp|baz puan|x\b)[^.?!]*[.?!]?", re.I)


def transcribe(audio, start=None, end=None, model="small", lang=None):
    from faster_whisper import WhisperModel
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio), "-ar", "16000", "-ac", "1"]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        subprocess.run(cmd + [str(wav)], check=True)
        w = wave.open(str(wav)); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    segs, info = WhisperModel(model, device="cpu", compute_type="int8").transcribe(a, language=lang, beam_size=5)
    off = start or 0
    return [(s.start + off, s.end + off, s.text.strip()) for s in segs], info.language


def features(text):
    t = text.lower()
    topics = [k for k, ws in TOPICS.items() if any(w in t for w in ws)]
    p, n = sum(t.count(w) for w in POS), sum(t.count(w) for w in NEG)
    stance = "positive" if p > n else ("negative" if n > p else "neutral")
    claims = [{"text": c.strip(), "numeric": True, "verified": False} for c in NUMC.findall(text)]
    return topics, stance, claims


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("audio", "source-id", "url", "speaker", "channel", "published"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--window", type=float, default=60, help="seconds per stored segment")
    ap.add_argument("--tickers", default="", help="comma-separated tickers discussed")
    a = ap.parse_args()
    segs, lang = transcribe(a.audio, a.start, a.end)
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with open(OUT, "a") as f:
        buf, s0 = [], None
        for st, en, tx in segs + [(None, None, None)]:
            if tx is not None:
                s0 = st if s0 is None else s0; buf.append(tx)
            if buf and (tx is None or en - s0 >= a.window):
                text = " ".join(buf); topics, stance, claims = features(text)
                f.write(json.dumps({"source_id": a.source_id, "url": a.url, "speaker": a.speaker, "channel": a.channel,
                                    "published": a.published, "language": lang, "segment_start": round(s0, 1),
                                    "segment_end": round(en if tx is not None else st or s0, 1), "text": text, "topics": topics,
                                    "stance": stance, "claims": claims,
                                    "tickers": [x.strip().upper() for x in a.tickers.split(",") if x.strip()]}, ensure_ascii=False) + "\n")
                n += 1; buf, s0 = [], None
    print(f"wrote {n} segments to {OUT}")
