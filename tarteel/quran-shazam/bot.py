#!/usr/bin/env python3
"""
🔊 Quran Shazam — Telegram Bot
Send a voice note or audio file → get ayah identification
"""
import os
import sys
import asyncio
import tempfile
import httpx
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command

# ── Config ──────────────────────────────────────────
TOKEN = os.environ.get("QURAN_BOT_TOKEN", "").strip()
if not TOKEN:
    token_file = os.path.join(os.path.dirname(__file__), ".bot-token")
    if os.path.exists(token_file):
        TOKEN = open(token_file).read().strip()
if not TOKEN:
    print("ERROR: Set QURAN_BOT_TOKEN env var or create .bot-token file")
    sys.exit(1)

API_URL = os.environ.get("QURAN_API_URL", "http://localhost:7860")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# ── Handlers ────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "🔊 *Quran Shazam*\n\n"
        "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ\n\n"
        "Send me a voice note or audio file of a Quran recitation, "
        "and I'll identify the surah, ayah, and show you the translation and tafsir.\n\n"
        "🎤 *Record* a voice note reciting any ayah\n"
        "📁 *Send* an audio file (MP3, WAV, OGG)\n\n"
        "Powered by Tarteel AI 🕌",
        parse_mode=ParseMode.MARKDOWN
    )

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "🔊 *How to use Quran Shazam:*\n\n"
        "1️⃣ Record a voice note of any Quran recitation\n"
        "2️⃣ Or send/forward an audio file\n"
        "3️⃣ I'll identify the ayah and show:\n"
        "   • Surah name & ayah number\n"
        "   • Arabic text\n"
        "   • English translation\n"
        "   • Tafsir (Ibn Kathir)\n\n"
        "🌐 Web version: https://quran.devtek.uk",
        parse_mode=ParseMode.MARKDOWN
    )

@dp.message(F.voice | F.audio | F.document)
async def handle_audio(message: types.Message):
    """Process voice notes, audio files, and documents"""
    processing_msg = await message.answer("🔍 Identifying recitation...")

    try:
        # Get the file
        if message.voice:
            file = await bot.get_file(message.voice.file_id)
            suffix = ".ogg"
        elif message.audio:
            file = await bot.get_file(message.audio.file_id)
            suffix = ".mp3"
        elif message.document:
            file = await bot.get_file(message.document.file_id)
            name = message.document.file_name or "audio.mp3"
            suffix = os.path.splitext(name)[1] or ".mp3"
        else:
            await processing_msg.edit_text("❌ Unsupported file type")
            return

        # Download file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            await bot.download_file(file.file_path, tmp)
            tmp_path = tmp.name

        # Send to API
        async with httpx.AsyncClient(timeout=60) as client:
            with open(tmp_path, "rb") as f:
                resp = await client.post(
                    f"{API_URL}/identify",
                    files={"audio": (f"audio{suffix}", f)}
                )

        os.unlink(tmp_path)

        if resp.status_code != 200:
            await processing_msg.edit_text("❌ Server error. Please try again.")
            return

        data = resp.json()

        if data.get("error"):
            await processing_msg.edit_text(f"❌ {data['error']}")
            return

        # Format response
        text = format_response(data)
        await processing_msg.edit_text(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        await processing_msg.edit_text(f"❌ Error: {str(e)[:200]}")

@dp.message()
async def handle_text(message: types.Message):
    """Handle text messages"""
    await message.answer(
        "🎤 Send me a *voice note* or *audio file* of a Quran recitation to identify it!\n\n"
        "Tip: You can also forward audio messages from other chats.",
        parse_mode=ParseMode.MARKDOWN
    )

def format_response(data: dict) -> str:
    """Format the API response for Telegram"""
    lines = []

    # Transcription
    lines.append(f"🎙 <b>Heard:</b> <i>{data['transcription']}</i>")
    lines.append("")

    # Multi-ayah result
    if data.get("multi_ayah"):
        ma = data["multi_ayah"]
        lines.append(f"📖 <b>Multiple Ayahs Detected</b>")
        lines.append(f"🕌 {ma[0]['surah_english']} — Ayahs {ma[0]['ayah']}-{ma[-1]['ayah']}")
        lines.append("")
        for a in ma:
            lines.append(f"<b>[{a['surah']}:{a['ayah']}]</b>")
            lines.append(f"{a['text']}")
            lines.append(f"<i>{a['translation']}</i>")
            lines.append("")
        lines.append("─" * 20)
        lines.append("")

    # Top match
    if data.get("matches"):
        m = data["matches"][0]
        score_pct = int(m["score"] * 100)
        confidence = "🟢" if score_pct >= 80 else "🟡" if score_pct >= 50 else "🔴"

        lines.append(f"📖 <b>Identified Verse</b> {confidence} {score_pct}%")
        lines.append("")
        lines.append(f"🕌 <b>{m['surah_english']}</b> ({m['surah_name']}) — {m['surah']}:{m['ayah']}")
        lines.append(f"<i>{m['surah_meaning']}</i>")
        lines.append("")
        lines.append(f"{m['text']}")
        lines.append("")
        lines.append(f"📝 <i>{m['translation']}</i>")

        if m.get("tafsir"):
            lines.append("")
            lines.append(f"📚 <b>Tafsir (Ibn Kathir):</b>")
            lines.append(f"<i>{m['tafsir'][:400]}</i>")

        # Other possible matches
        if len(data["matches"]) > 1:
            lines.append("")
            lines.append("─" * 20)
            lines.append("<b>Other possibilities:</b>")
            for m2 in data["matches"][1:3]:
                s2 = int(m2["score"] * 100)
                lines.append(f"  • {m2['surah_english']} {m2['surah']}:{m2['ayah']} ({s2}%)")

    lines.append("")
    lines.append(f"⏱ {data['elapsed']:.1f}s • {data.get('engine', '')}")

    return "\n".join(lines)

async def main():
    print("🔊 Quran Shazam Bot starting...")
    print(f"   API: {API_URL}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
