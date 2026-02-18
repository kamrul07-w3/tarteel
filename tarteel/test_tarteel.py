#!/usr/bin/env python3
"""Tarteel AI - Quran Recitation Recognition"""

from transformers import pipeline
import sys

print("Loading Tarteel AI whisper-base-ar-quran model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="tarteel-ai/whisper-base-ar-quran",
    device="cpu"  # no GPU on this VPS
)
print("✅ Model loaded successfully!")

if len(sys.argv) > 1:
    audio_file = sys.argv[1]
    print(f"\nTranscribing: {audio_file}")
    result = pipe(audio_file)
    print(f"\n📖 Transcription:\n{result['text']}")
else:
    print("\nUsage: python3 test_tarteel.py <audio_file.mp3>")
    print("Send me a Quran recitation audio to test it!")
