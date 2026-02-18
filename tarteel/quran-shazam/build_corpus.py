#!/usr/bin/env python3
"""Build a searchable Quran corpus from AlQuran Cloud API data"""
import json
import re

def strip_diacritics(text):
    """Remove Arabic diacritics for fuzzy matching"""
    # Arabic diacritics Unicode range
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
    return diacritics.sub('', text)

def normalize_arabic(text):
    """Normalize Arabic text for matching"""
    text = strip_diacritics(text)
    text = re.sub(r'[\u06D6-\u06FF\uFD3E\uFD3F\uFDFC\uFDFD\uFE70-\uFEFF\u06DE\u06E9]', '', text)
    text = re.sub(r'[۞۩﴾﴿٭]', '', text)
    text = re.sub('[إأآٱا]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('\u0640', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Building corpus...")

with open('quran_arabic.json', 'r') as f:
    arabic_data = json.load(f)

with open('quran_english.json', 'r') as f:
    english_data = json.load(f)

# Build lookup
corpus = []
en_lookup = {}

for surah in english_data['data']['surahs']:
    for ayah in surah['ayahs']:
        en_lookup[ayah['number']] = ayah['text']

for surah in arabic_data['data']['surahs']:
    surah_info = {
        'number': surah['number'],
        'name': surah['name'],
        'englishName': surah['englishName'],
        'englishNameTranslation': surah['englishNameTranslation'],
    }
    for ayah in surah['ayahs']:
        corpus.append({
            'number': ayah['number'],           # global ayah number
            'surah': surah_info['number'],
            'surahName': surah_info['name'],
            'surahEnglish': surah_info['englishName'],
            'surahMeaning': surah_info['englishNameTranslation'],
            'ayah': ayah['numberInSurah'],
            'text': ayah['text'],
            'text_normalized': normalize_arabic(ayah['text']),
            'translation': en_lookup.get(ayah['number'], ''),
        })

with open('corpus.json', 'w') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=None)

print(f"✅ Built corpus: {len(corpus)} ayahs across {arabic_data['data']['surahs'][-1]['number']} surahs")
print(f"   File: corpus.json ({len(json.dumps(corpus, ensure_ascii=False)) / 1024 / 1024:.1f} MB)")
