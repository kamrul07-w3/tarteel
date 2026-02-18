#!/usr/bin/env python3
"""
Tarteel — Discover the Quran through recitation.
Batch identify + real-time streaming transcription.
"""
import os
import json
import time
import asyncio
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core import create_engine, format_match, format_multi_ayah

# ── Init ────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
engine = create_engine()
print("   🚀 Ready!\n")

app = FastAPI(
    title="Tarteel",
    description="Identify Quran recitations from audio. Supports batch upload and real-time streaming.",
    version="2.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Batch identify ──────────────────────────────────

@app.post("/identify", summary="Identify a Quran recitation",
          description="Upload an audio file (MP3, WAV, OGG, M4A, WebM) and get the matching surah, ayah, translation, and tafsir.")
async def identify_endpoint(audio: UploadFile = File(..., description="Audio file of a Quran recitation")):
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        return await engine.identify(tmp_path)
    finally:
        os.unlink(tmp_path)


# ── WebSocket streaming ────────────────────────────

def _detect_suffix(first_chunk: bytes) -> str:
    header = first_chunk[:4] if first_chunk else b''
    if header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xf3'):
        return ".mp3"
    if header == b'RIFF':
        return ".wav"
    if header == b'OggS':
        return ".ogg"
    return ".webm"


@app.websocket("/ws/stream")
async def stream_transcription(ws: WebSocket):
    """
    Real-time streaming transcription.

    Client sends:  binary audio chunks, then JSON {"type": "stop"}
    Server sends:  {"type": "partial", "text": "...", "new_words": "..."}
                   {"type": "match", ...}
    """
    await ws.accept()

    audio_chunks = []
    audio_bytes = 0
    prev_text = ""
    lock = asyncio.Lock()

    async def process_audio():
        nonlocal prev_text
        if not audio_chunks:
            return
        async with lock:
            try:
                suffix = _detect_suffix(audio_chunks[0])
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    for chunk in audio_chunks:
                        tmp.write(chunk)
                    tmp_path = tmp.name

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, engine.pipe, tmp_path)
                text = result['text'].strip()
                os.unlink(tmp_path)

                if text and text != prev_text:
                    prev_words = prev_text.split() if prev_text else []
                    curr_words = text.split()
                    new_words = " ".join(curr_words[len(prev_words):]) if len(curr_words) > len(prev_words) else text
                    prev_text = text
                    await ws.send_json({"type": "partial", "text": text, "new_words": new_words})
            except Exception as e:
                print(f"   Stream error: {e}")

    async def send_match():
        if not prev_text:
            await ws.send_json({"type": "match", "transcription": "", "matches": [], "multi_ayah": None})
            return
        result = await engine.match_text(prev_text)
        result["type"] = "match"
        await ws.send_json(result)

    process_task = None

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    audio_chunks.append(message["bytes"])
                    audio_bytes += len(message["bytes"])
                    if audio_bytes > 16000 and not lock.locked():
                        process_task = asyncio.create_task(process_audio())

                elif "text" in message and message["text"]:
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        if process_task and not process_task.done():
                            await process_task
                        await process_audio()
                        await send_match()
                        break

            elif message["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"   WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Verse context ───────────────────────────────────

@app.get("/verse/{surah}/{ayah}/context", summary="Get verse with context",
         description="Returns the verse, 2 verses before and after, surah intro, and Ibn Kathir tafsir.")
async def verse_context(surah: int, ayah: int):
    ctx = engine.get_verse_context(surah, ayah)
    if not ctx:
        raise HTTPException(status_code=404, detail="Verse not found")
    # Add tafsir (async, can't do in core synchronously)
    tafsir = await engine.fetch_tafsir(surah, ayah)
    ctx["verse"]["tafsir"] = tafsir
    return ctx


# ── Health ──────────────────────────────────────────

@app.get("/health", summary="Service health check")
async def health():
    return {
        "status": "ok",
        "model": engine.model_name,
        "corpus_size": len(engine.corpus),
        "engine": "onnx" if engine.use_onnx else "pytorch",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
