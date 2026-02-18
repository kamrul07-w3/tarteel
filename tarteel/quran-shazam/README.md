# Tarteel — Discover the Quran through recitation

Tarteel identifies Quran recitations from audio. Record or upload a recitation, and it tells you the surah, ayah, translation, transliteration, and scholarly commentary — with words appearing in real-time as you recite.

**Live:** [quran.devtek.uk](https://quran.devtek.uk)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                │
│  ┌─────────────┐    ┌──────────────────────────────┐    │
│  │ Live Mode   │    │ Classic Mode                 │    │
│  │ WebSocket   │    │ Record → POST /identify      │    │
│  │ chunks/1.5s │    │ (batch)                      │    │
│  └──────┬──────┘    └──────────────┬───────────────┘    │
└─────────┼──────────────────────────┼────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (app.py)                                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ /ws/stream│  │ /identify│  │ /verse/{s}/{a}/context│  │
│  └─────┬────┘  └─────┬────┘  └───────────┬───────────┘  │
│        └──────┬──────┘                    │              │
│               ▼                           │              │
│  ┌─────────────────────────┐              │              │
│  │ QuranEngine (core.py)   │◄─────────────┘              │
│  │ ┌─────────────────────┐ │                             │
│  │ │ Whisper ASR (ONNX)  │ │  tarteel-ai/whisper-base    │
│  │ │ N-gram index        │ │  6,236 ayahs indexed        │
│  │ │ SequenceMatcher     │ │  Fuzzy + exact matching      │
│  │ │ Ayah index (O(1))   │ │  Precomputed at startup     │
│  │ │ Bismillah stripping │ │  113/114 surahs handled     │
│  │ └─────────────────────┘ │                             │
│  └─────────────────────────┘                             │
│               │                                          │
│               ▼ (tafsir only)                            │
│  ┌──────────────────────┐                                │
│  │ AlQuran Cloud API    │  Ibn Kathir tafsir, cached     │
│  └──────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.10+
- ~1GB disk (700MB ONNX model + corpus files)

### Install

```bash
pip install fastapi uvicorn httpx transformers optimum[onnxruntime]
```

### Run

```bash
cd quran-shazam
python3 app.py
# → http://localhost:7860
```

First run exports the ONNX model (~2 min). Subsequent starts load from cache (~6s).

### Telegram Bot (optional)

```bash
export QURAN_BOT_TOKEN=your_token
python3 bot.py
```

## API Reference

### POST /identify

Batch audio identification. Upload a recording, get the matching verse.

```bash
curl -F "audio=@recitation.mp3" https://quran.devtek.uk/identify
```

**Response:**
```json
{
  "transcription": "قُلْ هُوَ اللَّهُ أَحَدٌ",
  "elapsed": 1.8,
  "engine": "ONNX ⚡",
  "matches": [
    {
      "score": 1.0,
      "surah": 112,
      "ayah": 1,
      "surah_name": "الإخلاص",
      "surah_english": "Al-Ikhlaas",
      "surah_meaning": "Sincerity",
      "text": "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
      "translation": "Say, \"He is Allah, [who is] One,\"",
      "transliteration": "Qul huwa Allahu ahad",
      "tafsir": "..."
    }
  ],
  "multi_ayah": null
}
```

**Accepted formats:** MP3, WAV, OGG, M4A, WebM

### WebSocket /ws/stream

Real-time streaming transcription. Words appear as you recite.

```
Client connects → ws://host/ws/stream

Client sends:    binary audio chunks (every ~1.5s)
Client sends:    {"type": "stop"}  (when done)

Server responds: {"type": "partial", "text": "قُلْ هُوَ", "new_words": "هُوَ"}
Server responds: {"type": "partial", "text": "قُلْ هُوَ اللَّهُ أَحَدٌ", "new_words": "اللَّهُ أَحَدٌ"}
Server responds: {"type": "match", "transcription": "...", "matches": [...]}
```

**JavaScript example:**
```javascript
const ws = new WebSocket('wss://quran.devtek.uk/ws/stream');
const recorder = new MediaRecorder(stream);

recorder.ondataavailable = e => ws.send(e.data);
recorder.start(1500);  // chunk every 1.5s

ws.onmessage = event => {
  const data = JSON.parse(event.data);
  if (data.type === 'partial') {
    // Words appearing in real-time
    document.getElementById('text').textContent = data.text;
  } else if (data.type === 'match') {
    // Final verse identification
    showResult(data.matches[0]);
  }
};

// When done:
recorder.stop();
ws.send(JSON.stringify({ type: 'stop' }));
```

### GET /verse/{surah}/{ayah}/context

Get a verse with surrounding context, surah intro, and tafsir.

```bash
curl https://quran.devtek.uk/verse/112/1/context
```

**Response:**
```json
{
  "verse": {
    "surah": 112, "ayah": 1,
    "text": "...", "translation": "...",
    "transliteration": "...", "tafsir": "..."
  },
  "before": [],
  "after": [
    {"surah": 112, "ayah": 2, "text": "...", "translation": "..."},
    {"surah": 112, "ayah": 3, "text": "...", "translation": "..."}
  ],
  "surah_info": {
    "name": "الإخلاص",
    "english": "Al-Ikhlaas",
    "meaning": "Sincerity",
    "intro": "Just four verses long, Al-Ikhlas is considered equivalent to...",
    "revelation": "Makkah",
    "verse_count": 4
  }
}
```

### GET /health

```json
{"status": "ok", "model": "tarteel-ai/whisper-base-ar-quran", "corpus_size": 6236, "engine": "onnx"}
```

## How matching works

```
Audio → Whisper ASR → Arabic text → Normalize → Match → Results
```

### 1. Transcription
[tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) — a Whisper Base model fine-tuned on Quran recitation. Runs as ONNX for 2-3x CPU speedup.

### 2. Arabic normalization
Strips diacritics (tashkeel), unifies alef variants (أ إ آ ٱ → ا), normalizes taa marbuta (ة → ه), removes ornamental marks.

### 3. Bismillah handling
113 of 114 surahs have "بسم الله الرحمن الرحيم" prepended to verse 1 in the corpus. The engine strips this prefix and matches against both variants, so reciting without Bismillah still works.

### 4. N-gram candidate filtering
Instead of comparing against all 6,236 verses, character trigrams narrow it to ~200 candidates. This makes matching fast enough for real-time use.

### 5. Scoring
- **Exact match** → 1.0
- **Substring** (recitation contained in verse) → 0.7–1.0 scaled by coverage
- **Superstring** (verse contained in recitation) → 0.6–0.9 scaled by coverage
- **Fuzzy** → `SequenceMatcher.ratio()`, threshold 0.35

### 6. Multi-ayah detection
If no single verse scores > 0.95, the engine tries concatenating 2–5 consecutive verses across the top 5 surah candidates. If the concatenated score beats the single match by 0.05+, it returns the multi-verse result.

## File structure

```
quran-shazam/
├── app.py                  ← FastAPI routes (130 lines)
├── core.py                 ← QuranEngine class (270 lines)
├── bot.py                  ← Telegram bot
├── build_corpus.py         ← Builds corpus.json from source JSONs
├── build_enriched_corpus.py← Adds transliteration via AlQuran Cloud API
├── corpus.json             ← 6,236 ayahs (Arabic + English)
├── data/
│   ├── enriched_corpus.json← Corpus + transliteration field
│   ├── surah_intros.json   ← Curated intros for 20 surahs
│   └── transliteration_cache.json
├── static/
│   └── index.html          ← Web UI (Live + Classic modes)
├── onnx_model/             ← Cached Whisper ONNX export (~700MB)
├── quran_arabic.json       ← Source: Arabic text
└── quran_english.json      ← Source: English translations
```

## External APIs

| API | Used for | Cost | Called when |
|-----|----------|------|------------|
| [AlQuran Cloud](https://alquran.cloud/api) | Ibn Kathir tafsir | Free | User taps "Explore" or best match loads |
| [EveryAyah](https://everyayah.com) | Audio playback (Alafasy) | Free | User taps "Listen" (client-side only) |

All transcription and matching runs locally. No audio leaves the server.
