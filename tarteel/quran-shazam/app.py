#!/usr/bin/env python3
"""
🔊 Quran Shazam — Identify any Quran recitation
Send audio → Get surah, ayah, Arabic text, translation + tafsir
v2: ONNX acceleration, multi-ayah detection, n-gram index, tafsir
"""
import json
import re
import os
import time
import tempfile
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import httpx

# ── Config ──────────────────────────────────────────
MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")
ENRICHED_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "enriched_corpus.json")
SURAH_INTROS_PATH = os.path.join(os.path.dirname(__file__), "data", "surah_intros.json")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "onnx_model")
TOP_K = 5
USE_ONNX = True  # 2-3x faster inference on CPU

# ── Arabic text normalization ───────────────────────
def strip_diacritics(text):
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
    return diacritics.sub('', text)

def normalize_arabic(text):
    text = strip_diacritics(text)
    text = re.sub(r'[\u06D6-\u06FF\uFD3E\uFD3F\uFDFC\uFDFD\uFE70-\uFEFF\u06DE\u06E9]', '', text)
    text = re.sub(r'[۞۩﴾﴿٭]', '', text)
    text = re.sub('[إأآٱا]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('\u0640', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── N-gram index for fast candidate filtering ──────
NGRAM_SIZE = 3

def get_ngrams(text, n=NGRAM_SIZE):
    """Extract character n-grams from text"""
    return set(text[i:i+n] for i in range(len(text) - n + 1)) if len(text) >= n else {text}

def build_ngram_index(corpus):
    """Build inverted index: ngram -> set of corpus indices"""
    index = defaultdict(set)
    for i, entry in enumerate(corpus):
        for ng in get_ngrams(entry['text_normalized']):
            index[ng].add(i)
    return index

def get_candidates(normalized, ngram_index, corpus, max_candidates=200):
    """Use n-gram overlap to quickly find candidate ayahs"""
    if len(normalized) < NGRAM_SIZE:
        return list(range(len(corpus)))  # short text, check all
    
    query_ngrams = get_ngrams(normalized)
    if not query_ngrams:
        return list(range(len(corpus)))
    
    # Count ngram hits per corpus entry
    hits = defaultdict(int)
    for ng in query_ngrams:
        for idx in ngram_index.get(ng, set()):
            hits[idx] += 1
    
    # Sort by hits, take top candidates
    ranked = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:max_candidates]]

# ── Matching engine ─────────────────────────────────
def find_ayah(transcription, corpus, ngram_index, top_k=TOP_K):
    """Find the best matching ayah(s) for a transcription"""
    normalized = normalize_arabic(transcription)
    if not normalized:
        return []
    
    # Get candidates via n-gram index (much faster than brute force)
    candidates = get_candidates(normalized, ngram_index, corpus)
    
    results = []
    for idx in candidates:
        entry = corpus[idx]
        corpus_norm = entry['text_normalized']
        
        if normalized == corpus_norm:
            results.append((1.0, entry))
            continue
        
        if normalized in corpus_norm:
            coverage = len(normalized) / max(len(corpus_norm), 1)
            score = 0.7 + (0.3 * coverage)
            results.append((score, entry))
            continue
        elif corpus_norm in normalized:
            coverage = len(corpus_norm) / max(len(normalized), 1)
            score = 0.6 + (0.3 * coverage)
            results.append((score, entry))
            continue
        
        score = SequenceMatcher(None, normalized, corpus_norm).ratio()
        if score > 0.35:
            results.append((score, entry))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# ── Multi-ayah detection ────────────────────────────
def detect_multi_ayah(transcription, corpus, ngram_index):
    """Detect if transcription spans multiple consecutive ayahs"""
    normalized = normalize_arabic(transcription)
    if not normalized or len(normalized) < 20:
        return None
    
    # First find best single match
    single_matches = find_ayah(transcription, corpus, ngram_index, top_k=1)
    if not single_matches or single_matches[0][0] > 0.9:
        return None  # Single ayah match is good enough
    
    best_score, best_entry = single_matches[0]
    surah = best_entry['surah']
    ayah = best_entry['ayah']
    
    # Try concatenating consecutive ayahs from same surah
    best_multi = None
    best_multi_score = best_score
    
    for start_ayah in range(max(1, ayah - 2), ayah + 3):
        for length in range(2, 6):  # Try 2-5 consecutive ayahs
            concat_text = ""
            ayahs_found = []
            for a in range(start_ayah, start_ayah + length):
                # Find this ayah in corpus
                for entry in corpus:
                    if entry['surah'] == surah and entry['ayah'] == a:
                        concat_text += " " + entry['text_normalized']
                        ayahs_found.append(entry)
                        break
            
            if len(ayahs_found) < 2:
                continue
            
            concat_norm = concat_text.strip()
            score = SequenceMatcher(None, normalized, concat_norm).ratio()
            
            if score > best_multi_score:
                best_multi_score = score
                best_multi = ayahs_found
    
    if best_multi and best_multi_score > best_score + 0.1:
        return best_multi
    return None

# ── Tafsir fetching ─────────────────────────────────
tafsir_cache = {}

async def fetch_tafsir(surah: int, ayah: int) -> str:
    """Fetch tafsir from AlQuran Cloud API (Ibn Kathir)"""
    cache_key = f"{surah}:{ayah}"
    if cache_key in tafsir_cache:
        return tafsir_cache[cache_key]
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            # Try Ibn Kathir English tafsir
            r = await client.get(f"http://api.alquran.cloud/v1/ayah/{surah}:{ayah}/en.ibn-kathir")
            if r.status_code == 200:
                data = r.json()
                if data.get('data') and data['data'].get('text'):
                    tafsir_text = data['data']['text']
                    # Truncate to reasonable length
                    if len(tafsir_text) > 500:
                        tafsir_text = tafsir_text[:497] + "..."
                    tafsir_cache[cache_key] = tafsir_text
                    return tafsir_text
    except Exception:
        pass
    
    return ""

# ── Load model & corpus ────────────────────────────
print("🔊 Quran Shazam v2 — Loading...")
print(f"   Model: {MODEL_NAME}")

t0 = time.time()

# Try ONNX first for faster inference
pipe = None
if USE_ONNX:
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor, AutoConfig
        import numpy as np
        
        if os.path.exists(ONNX_PATH):
            print("   Loading cached ONNX model...")
            ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(ONNX_PATH)
        else:
            print("   Exporting to ONNX (one-time)...")
            ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, export=True)
            ort_model.save_pretrained(ONNX_PATH)
            print(f"   Saved ONNX model to {ONNX_PATH}")
        
        ort_processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=ort_model,
            tokenizer=ort_processor.tokenizer,
            feature_extractor=ort_processor.feature_extractor,
            device="cpu"
        )
        print(f"   ✅ ONNX model loaded ({time.time()-t0:.1f}s) ⚡")
    except Exception as e:
        print(f"   ⚠️ ONNX failed ({e}), falling back to PyTorch")
        pipe = None

if pipe is None:
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("automatic-speech-recognition", model=MODEL_NAME, device="cpu")
    print(f"   ✅ PyTorch model loaded ({time.time()-t0:.1f}s)")

# Prefer enriched corpus if available, fall back to original
_corpus_path = ENRICHED_CORPUS_PATH if os.path.exists(ENRICHED_CORPUS_PATH) else CORPUS_PATH
with open(_corpus_path, 'r') as f:
    corpus = json.load(f)
print(f"   Corpus source: {os.path.basename(_corpus_path)}")

# Ensure normalized text exists
for entry in corpus:
    if 'text_normalized' not in entry:
        entry['text_normalized'] = normalize_arabic(entry['text'])

print(f"   ✅ Corpus loaded ({len(corpus)} ayahs)")

# Build n-gram index
t1 = time.time()
ngram_index = build_ngram_index(corpus)
print(f"   ✅ N-gram index built ({len(ngram_index)} trigrams, {time.time()-t1:.2f}s)")
print("   🚀 Ready!\n")

# ── FastAPI app ─────────────────────────────────────
app = FastAPI(title="Tarteel", description="Discover the Quran through recitation")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load surah intros if available
surah_intros = {}
if os.path.exists(SURAH_INTROS_PATH):
    with open(SURAH_INTROS_PATH, 'r') as f:
        surah_intros = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/identify")
async def identify_endpoint(audio: UploadFile = File(...)):
    """Identify a Quran recitation from audio"""
    t0 = time.time()

    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipe(tmp_path)
        transcription = result['text'].strip()

        if not transcription:
            return {"error": "Could not transcribe audio. Try a clearer recording."}

        # Single ayah matching
        matches = find_ayah(transcription, corpus, ngram_index, top_k=3)

        # Multi-ayah detection
        multi_ayah_result = detect_multi_ayah(transcription, corpus, ngram_index)
        multi_ayah_data = None
        if multi_ayah_result:
            multi_ayah_data = [
                {
                    "surah": e['surah'], "ayah": e['ayah'],
                    "surah_name": e['surahName'], "surah_english": e['surahEnglish'],
                    "text": e['text'], "translation": e['translation'],
                }
                for e in multi_ayah_result
            ]

        # Fetch tafsir for top match
        match_results = []
        for i, (score, entry) in enumerate(matches):
            tafsir = ""
            if i == 0:  # Only fetch tafsir for best match
                tafsir = await fetch_tafsir(entry['surah'], entry['ayah'])
            match_results.append({
                "score": round(score, 4),
                "surah": entry['surah'],
                "ayah": entry['ayah'],
                "surah_name": entry['surahName'],
                "surah_english": entry['surahEnglish'],
                "surah_meaning": entry['surahMeaning'],
                "text": entry['text'],
                "translation": entry['translation'],
                "transliteration": entry.get('transliteration', ''),
                "tafsir": tafsir,
            })

        elapsed = time.time() - t0
        engine = "ONNX ⚡" if USE_ONNX else "PyTorch"

        return {
            "transcription": transcription,
            "elapsed": elapsed,
            "engine": engine,
            "matches": match_results,
            "multi_ayah": multi_ayah_data,
        }
    finally:
        os.unlink(tmp_path)

# ── Verse context endpoint ─────────────────────────
@app.get("/verse/{surah}/{ayah}/context")
async def verse_context(surah: int, ayah: int):
    """Get verse with surrounding context and surah info"""
    # Find the target verse
    target = None
    target_idx = None
    for i, entry in enumerate(corpus):
        if entry['surah'] == surah and entry['ayah'] == ayah:
            target = entry
            target_idx = i
            break

    if not target:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Get surrounding verses (2 before, 2 after) from same surah
    before = []
    after = []
    for offset in [-2, -1]:
        idx = target_idx + offset
        if idx >= 0 and corpus[idx]['surah'] == surah:
            e = corpus[idx]
            before.append({
                "surah": e['surah'], "ayah": e['ayah'],
                "text": e['text'], "translation": e['translation'],
            })
    for offset in [1, 2]:
        idx = target_idx + offset
        if idx < len(corpus) and corpus[idx]['surah'] == surah:
            e = corpus[idx]
            after.append({
                "surah": e['surah'], "ayah": e['ayah'],
                "text": e['text'], "translation": e['translation'],
            })

    # Fetch tafsir for the target verse
    tafsir = await fetch_tafsir(surah, ayah)

    # Surah info
    surah_key = str(surah)
    intro_data = surah_intros.get(surah_key, {})
    surah_info = {
        "name": target.get('surahName', ''),
        "english": target.get('surahEnglish', ''),
        "meaning": target.get('surahMeaning', ''),
        "intro": intro_data.get('intro', ''),
        "revelation": intro_data.get('revelation', ''),
        "verse_count": intro_data.get('verse_count', 0),
    }

    return {
        "verse": {
            "surah": target['surah'], "ayah": target['ayah'],
            "text": target['text'], "translation": target['translation'],
            "transliteration": target.get('transliteration', ''),
            "tafsir": tafsir,
        },
        "before": before,
        "after": after,
        "surah_info": surah_info,
    }

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "corpus_size": len(corpus), "engine": "onnx" if USE_ONNX else "pytorch"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
