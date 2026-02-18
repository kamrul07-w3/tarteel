#!/usr/bin/env python3
"""
Build enriched corpus with transliteration data.
Sources transliteration from AlQuran Cloud API and merges into corpus.json.
"""
import json
import os
import time
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "corpus.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "enriched_corpus.json")
TRANSLITERATION_CACHE = os.path.join(SCRIPT_DIR, "data", "transliteration_cache.json")

API_BASE = "http://api.alquran.cloud/v1"


def fetch_surah_transliteration(surah):
    """Fetch transliteration for a surah from AlQuran Cloud API"""
    url = f"{API_BASE}/surah/{surah}/en.transliteration"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            if data.get('code') == 200 and data.get('data', {}).get('ayahs'):
                return data['data']['ayahs']
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"  Warning: Failed to fetch surah {surah}: {e}")
    return []


def fetch_all_transliterations():
    """Fetch transliteration for all 114 surahs"""
    if os.path.exists(TRANSLITERATION_CACHE):
        print("Loading cached transliterations...")
        with open(TRANSLITERATION_CACHE, 'r') as f:
            return json.load(f)

    print("Fetching transliterations from AlQuran Cloud API...")
    print("(This takes ~2 minutes for all 114 surahs)\n")

    # Map: "surah:ayah" -> transliteration text
    translit_map = {}

    for surah in range(1, 115):
        ayahs = fetch_surah_transliteration(surah)
        for a in ayahs:
            key = f"{surah}:{a['numberInSurah']}"
            translit_map[key] = a['text']

        print(f"  Surah {surah:>3}/114: {len(ayahs)} verses fetched")
        time.sleep(0.2)  # Rate limit

    print(f"\nFetched {len(translit_map)} transliterations total")

    os.makedirs(os.path.dirname(TRANSLITERATION_CACHE), exist_ok=True)
    with open(TRANSLITERATION_CACHE, 'w') as f:
        json.dump(translit_map, f, ensure_ascii=False)
    print(f"Cached to {TRANSLITERATION_CACHE}")

    return translit_map


def build_enriched_corpus():
    """Merge transliteration into corpus"""
    with open(CORPUS_PATH, 'r') as f:
        corpus = json.load(f)

    translit_map = fetch_all_transliterations()

    enriched_count = 0
    for entry in corpus:
        key = f"{entry['surah']}:{entry['ayah']}"
        if key in translit_map:
            entry['transliteration'] = translit_map[key]
            enriched_count += 1

    print(f"\nEnriched {enriched_count}/{len(corpus)} verses with transliteration")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(corpus, f, ensure_ascii=False)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Written to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_enriched_corpus()
