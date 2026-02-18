#!/usr/bin/env python3
"""Batch test Tarteel AI model against known ayahs"""
from transformers import pipeline
import time

print("Loading model...")
t0 = time.time()
pipe = pipeline("automatic-speech-recognition", model="tarteel-ai/whisper-base-ar-quran", device="cpu")
print(f"Model loaded in {time.time()-t0:.1f}s\n")

tests = [
    ("fatiha_v2.mp3", "Al-Fatiha 1:2"),
    ("yasin_v1.mp3", "Ya-Sin 36:1"),
    ("rahman_v13.mp3", "Ar-Rahman 55:13"),
    ("ikhlas_v1.mp3", "Al-Ikhlas 112:1"),
]

for audio, label in tests:
    t0 = time.time()
    result = pipe(audio)
    elapsed = time.time() - t0
    print(f"📖 {label} ({elapsed:.1f}s)")
    print(f"   {result['text']}\n")
