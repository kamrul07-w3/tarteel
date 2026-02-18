"""
Quran matching engine — shared core logic.
Normalization, corpus search, multi-ayah detection, tafsir.
"""
import json
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

import httpx

# ── Arabic text normalization ───────────────────────

_DIACRITICS = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670'
    r'\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)
_EXTRA_MARKS = re.compile(r'[\u06D6-\u06FF\uFD3E\uFD3F\uFDFC\uFDFD\uFE70-\uFEFF\u06DE\u06E9]')
_ORNAMENTS = re.compile(r'[۞۩﴾﴿٭]')
_ALEF_VARIANTS = re.compile('[إأآٱا]')
_WHITESPACE = re.compile(r'\s+')


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub('', text)
    text = _EXTRA_MARKS.sub('', text)
    text = _ORNAMENTS.sub('', text)
    text = _ALEF_VARIANTS.sub('ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي').replace('\u0640', '')
    return _WHITESPACE.sub(' ', text).strip()


_BISMILLAH_NORM = normalize_arabic("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")


def strip_bismillah(text: str) -> str:
    if text.startswith(_BISMILLAH_NORM):
        stripped = text[len(_BISMILLAH_NORM):].strip()
        return stripped if stripped else text
    return text


# ── N-gram index ────────────────────────────────────

NGRAM_SIZE = 3


def _get_ngrams(text, n=NGRAM_SIZE):
    return set(text[i:i+n] for i in range(len(text) - n + 1)) if len(text) >= n else {text}


def _build_ngram_index(corpus):
    index = defaultdict(set)
    for i, entry in enumerate(corpus):
        for ng in _get_ngrams(entry['text_normalized']):
            index[ng].add(i)
        no_bis = entry.get('text_no_bismillah', entry['text_normalized'])
        if no_bis != entry['text_normalized']:
            for ng in _get_ngrams(no_bis):
                index[ng].add(i)
    return index


def _get_candidates(normalized, ngram_index, corpus, max_candidates=200):
    if len(normalized) < NGRAM_SIZE:
        return list(range(len(corpus)))
    query_ngrams = _get_ngrams(normalized)
    if not query_ngrams:
        return list(range(len(corpus)))
    hits = defaultdict(int)
    for ng in query_ngrams:
        for idx in ngram_index.get(ng, set()):
            hits[idx] += 1
    ranked = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:max_candidates]]


# ── Match result formatting ─────────────────────────

def format_match(score, entry, tafsir=""):
    return {
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
    }


def format_multi_ayah(entries):
    return [
        {
            "surah": e['surah'], "ayah": e['ayah'],
            "surah_name": e['surahName'], "surah_english": e['surahEnglish'],
            "text": e['text'], "translation": e['translation'],
        }
        for e in entries
    ]


# ── QuranEngine ─────────────────────────────────────

class QuranEngine:
    """All-in-one Quran matching engine with precomputed indices."""

    def __init__(self, *, model_name, corpus_path, enriched_corpus_path,
                 onnx_path, surah_intros_path, use_onnx=True):
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.pipe = None
        self.corpus = []
        self.ngram_index = {}
        self.ayah_index = {}       # (surah, ayah) -> (idx, entry)
        self.surah_intros = {}
        self._tafsir_cache = {}

        self._load_model(model_name, onnx_path, use_onnx)
        self._load_corpus(corpus_path, enriched_corpus_path)
        self._load_surah_intros(surah_intros_path)

    # ── Model loading ──────────────────────────────

    def _load_model(self, model_name, onnx_path, use_onnx):
        print(f"🔊 Tarteel — Loading...")
        print(f"   Model: {model_name}")
        t0 = time.time()

        if use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
                from transformers import WhisperProcessor
                from transformers import pipeline as hf_pipeline

                if os.path.exists(onnx_path):
                    print("   Loading cached ONNX model...")
                    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(onnx_path)
                else:
                    print("   Exporting to ONNX (one-time)...")
                    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)
                    ort_model.save_pretrained(onnx_path)

                processor = WhisperProcessor.from_pretrained(model_name)
                self.pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=ort_model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device="cpu",
                )
                self.use_onnx = True
                print(f"   ✅ ONNX model loaded ({time.time()-t0:.1f}s) ⚡")
                return
            except Exception as e:
                print(f"   ⚠️ ONNX failed ({e}), falling back to PyTorch")

        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline("automatic-speech-recognition", model=model_name, device="cpu")
        self.use_onnx = False
        print(f"   ✅ PyTorch model loaded ({time.time()-t0:.1f}s)")

    # ── Corpus loading ─────────────────────────────

    def _load_corpus(self, corpus_path, enriched_corpus_path):
        path = enriched_corpus_path if os.path.exists(enriched_corpus_path) else corpus_path
        with open(path, 'r') as f:
            self.corpus = json.load(f)
        print(f"   Corpus: {os.path.basename(path)}")

        for entry in self.corpus:
            if 'text_normalized' not in entry:
                entry['text_normalized'] = normalize_arabic(entry['text'])
            entry['text_no_bismillah'] = strip_bismillah(entry['text_normalized'])

        # Precompute ayah index for O(1) lookups
        for i, entry in enumerate(self.corpus):
            self.ayah_index[(entry['surah'], entry['ayah'])] = (i, entry)

        print(f"   ✅ Corpus loaded ({len(self.corpus)} ayahs)")

        t1 = time.time()
        self.ngram_index = _build_ngram_index(self.corpus)
        print(f"   ✅ N-gram index built ({len(self.ngram_index)} trigrams, {time.time()-t1:.2f}s)")

    def _load_surah_intros(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.surah_intros = json.load(f)

    # ── Transcription ──────────────────────────────

    def transcribe(self, audio_path: str) -> str:
        result = self.pipe(audio_path)
        return result['text'].strip()

    # ── Matching ───────────────────────────────────

    def find_ayah(self, transcription: str, top_k=5):
        normalized = normalize_arabic(transcription)
        if not normalized:
            return []

        candidates = _get_candidates(normalized, self.ngram_index, self.corpus)
        results = []

        for idx in candidates:
            entry = self.corpus[idx]
            variants = [entry['text_normalized']]
            no_bis = entry.get('text_no_bismillah', entry['text_normalized'])
            if no_bis != entry['text_normalized']:
                variants.append(no_bis)

            best_score = 0
            for corpus_norm in variants:
                if normalized == corpus_norm:
                    best_score = max(best_score, 1.0)
                    continue
                if normalized in corpus_norm:
                    coverage = len(normalized) / max(len(corpus_norm), 1)
                    best_score = max(best_score, 0.7 + (0.3 * coverage))
                    continue
                if corpus_norm in normalized:
                    coverage = len(corpus_norm) / max(len(normalized), 1)
                    best_score = max(best_score, 0.6 + (0.3 * coverage))
                    continue
                score = SequenceMatcher(None, normalized, corpus_norm).ratio()
                best_score = max(best_score, score)

            if best_score > 0.35:
                results.append((best_score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def detect_multi_ayah(self, transcription: str):
        normalized = normalize_arabic(transcription)
        if not normalized or len(normalized) < 5:
            return None

        single_matches = self.find_ayah(transcription, top_k=10)
        if not single_matches:
            return None

        best_single_score = single_matches[0][0]
        if best_single_score > 0.95:
            return None

        # Collect search anchors from top matches across different surahs
        seen_surahs = set()
        search_points = []
        for _score, entry in single_matches:
            if entry['surah'] not in seen_surahs:
                seen_surahs.add(entry['surah'])
                search_points.append(entry)
            if len(search_points) >= 5:
                break

        best_multi = None
        best_multi_score = best_single_score

        for anchor in search_points:
            surah, ayah = anchor['surah'], anchor['ayah']
            for start_ayah in range(max(1, ayah - 3), ayah + 4):
                for length in range(2, 6):
                    ayahs_found = []
                    concat_parts = []
                    for a in range(start_ayah, start_ayah + length):
                        lookup = self.ayah_index.get((surah, a))
                        if lookup:
                            _, entry = lookup
                            concat_parts.append(entry.get('text_no_bismillah', entry['text_normalized']))
                            ayahs_found.append(entry)
                        else:
                            break
                    if len(ayahs_found) < 2:
                        continue
                    score = SequenceMatcher(None, normalized, " ".join(concat_parts)).ratio()
                    if score > best_multi_score:
                        best_multi_score = score
                        best_multi = ayahs_found

        if best_multi and best_multi_score > best_single_score + 0.05:
            return best_multi
        return None

    # ── Verse context ──────────────────────────────

    def get_verse_context(self, surah: int, ayah: int):
        lookup = self.ayah_index.get((surah, ayah))
        if not lookup:
            return None

        target_idx, target = lookup

        before = []
        for offset in [-2, -1]:
            idx = target_idx + offset
            if idx >= 0 and self.corpus[idx]['surah'] == surah:
                e = self.corpus[idx]
                before.append({"surah": e['surah'], "ayah": e['ayah'],
                               "text": e['text'], "translation": e['translation']})

        after = []
        for offset in [1, 2]:
            idx = target_idx + offset
            if idx < len(self.corpus) and self.corpus[idx]['surah'] == surah:
                e = self.corpus[idx]
                after.append({"surah": e['surah'], "ayah": e['ayah'],
                              "text": e['text'], "translation": e['translation']})

        surah_key = str(surah)
        intro_data = self.surah_intros.get(surah_key, {})
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
            },
            "before": before,
            "after": after,
            "surah_info": surah_info,
        }

    # ── Tafsir ─────────────────────────────────────

    async def fetch_tafsir(self, surah: int, ayah: int) -> str:
        cache_key = f"{surah}:{ayah}"
        if cache_key in self._tafsir_cache:
            return self._tafsir_cache[cache_key]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"http://api.alquran.cloud/v1/ayah/{surah}:{ayah}/en.ibn-kathir"
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get('data') and data['data'].get('text'):
                        text = data['data']['text']
                        if len(text) > 500:
                            text = text[:497] + "..."
                        self._tafsir_cache[cache_key] = text
                        return text
        except Exception:
            pass
        return ""

    # ── Full identify pipeline ─────────────────────

    async def identify(self, audio_path: str):
        """Run full pipeline: transcribe -> match -> tafsir. Returns result dict."""
        t0 = time.time()
        transcription = self.transcribe(audio_path)
        if not transcription:
            return {"error": "Could not transcribe audio. Try a clearer recording."}

        matches = self.find_ayah(transcription, top_k=3)
        multi_ayah = self.detect_multi_ayah(transcription)

        match_results = []
        for i, (score, entry) in enumerate(matches):
            tafsir = await self.fetch_tafsir(entry['surah'], entry['ayah']) if i == 0 else ""
            match_results.append(format_match(score, entry, tafsir))

        return {
            "transcription": transcription,
            "elapsed": time.time() - t0,
            "engine": "ONNX ⚡" if self.use_onnx else "PyTorch",
            "matches": match_results,
            "multi_ayah": format_multi_ayah(multi_ayah) if multi_ayah else None,
        }

    async def match_text(self, transcription: str):
        """Match already-transcribed text. Returns result dict."""
        matches = self.find_ayah(transcription, top_k=3)
        multi_ayah = self.detect_multi_ayah(transcription)

        match_results = []
        for i, (score, entry) in enumerate(matches):
            tafsir = await self.fetch_tafsir(entry['surah'], entry['ayah']) if i == 0 else ""
            match_results.append(format_match(score, entry, tafsir))

        return {
            "transcription": transcription,
            "matches": match_results,
            "multi_ayah": format_multi_ayah(multi_ayah) if multi_ayah else None,
        }


# ── Factory ─────────────────────────────────────────

def create_engine(base_dir=None):
    """Create a QuranEngine with standard paths relative to base_dir."""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    return QuranEngine(
        model_name="tarteel-ai/whisper-base-ar-quran",
        corpus_path=os.path.join(base_dir, "corpus.json"),
        enriched_corpus_path=os.path.join(base_dir, "data", "enriched_corpus.json"),
        onnx_path=os.path.join(base_dir, "onnx_model"),
        surah_intros_path=os.path.join(base_dir, "data", "surah_intros.json"),
    )
