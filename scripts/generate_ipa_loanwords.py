#!/usr/bin/env python3
"""
Generate IPA, diacritization, and uroman for final_loanwords.json
Based on generate_ipa_cognates_v2.py approach.
"""

import json
import subprocess
import sys
import os
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'add_ipa' / 'ipa_tool'))
from arabic_ipa import ArabicToIPA
from phonikud_onnx import OnnxModel
from phonikud import phonemize

INPUT_FILE = 'all_annotation_0522/final_loanwords.json'
CHECKPOINT_FILE = 'all_annotation_0522/loanwords_ipa_checkpoint.json'
UROMAN_SCRIPT = '/tmp/uroman/uroman/uroman.pl'


def load_data():
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading from checkpoint: {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Loading from input: {INPUT_FILE}")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved: {CHECKPOINT_FILE}")


def get_progress(data):
    done = sum(1 for e in data['loanwords'] if e.get('arabic', {}).get('arabic_ipa'))
    return done, len(data['loanwords'])


def arabic_diacritize(text: str) -> str:
    if not text or not text.strip():
        return text
    result = subprocess.run(['camel_diac'], input=text, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return text
    return result.stdout.strip()


def arabic_to_ipa(converter, text: str) -> tuple:
    if not text or not text.strip():
        return text, ''
    diac = arabic_diacritize(text)
    ipa = converter.convert(diac)
    return diac, f'/{ipa}/' if ipa else ''


def hebrew_to_ipa(model, text: str) -> tuple:
    if not text or not text.strip():
        return text, ''
    result = model.predict(text)
    niqqud = result[0] if isinstance(result, list) else result
    ipa = phonemize(niqqud)
    return niqqud, f'/{ipa}/' if ipa else ''


def uromanize(text: str) -> str:
    if not text or not text.strip():
        return text
    result = subprocess.run(
        ['perl', UROMAN_SCRIPT],
        input=text, capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return text
    return result.stdout.strip().lower()


def process_entry(entry, arabic_converter, hebrew_model):
    # Arabic word
    ar_word = entry.get('arabic', {}).get('arabic_undiac', '')
    ar_diac, ar_ipa = arabic_to_ipa(arabic_converter, ar_word)
    entry['arabic']['arabic_diac'] = ar_diac
    entry['arabic']['arabic_uroman'] = uromanize(ar_word)
    entry['arabic']['arabic_ipa'] = ar_ipa

    # Arabic sentence
    ar_sent = entry.get('arabic_sent', {}).get('arabic_sent_undiac', '')
    ar_sent_diac, ar_sent_ipa = arabic_to_ipa(arabic_converter, ar_sent)
    entry['arabic_sent']['arabic_sent_diac'] = ar_sent_diac
    entry['arabic_sent']['arabic_sent_uroman'] = uromanize(ar_sent)
    entry['arabic_sent']['arabic_sent_ipa'] = ar_sent_ipa

    # Hebrew word
    he_word = entry.get('hebrew', {}).get('hebrew_undiac', '')
    he_niqqud, he_ipa = hebrew_to_ipa(hebrew_model, he_word)
    entry['hebrew']['hebrew_diac'] = he_niqqud
    entry['hebrew']['hebrew_uroman'] = uromanize(he_word)
    entry['hebrew']['hebrew_ipa'] = he_ipa

    # Hebrew sentence
    he_sent = entry.get('hebrew_sent', {}).get('hebrew_sent_undiac', '')
    he_sent_niqqud, he_sent_ipa = hebrew_to_ipa(hebrew_model, he_sent)
    entry['hebrew_sent']['hebrew_sent_diac'] = he_sent_niqqud
    entry['hebrew_sent']['hebrew_sent_uroman'] = uromanize(he_sent)
    entry['hebrew_sent']['hebrew_sent_ipa'] = he_sent_ipa

    return entry


def main():
    print("Initializing Arabic IPA converter...")
    arabic_converter = ArabicToIPA()
    print("Initializing Hebrew IPA model...")
    hebrew_model = OnnxModel('/workspace/.cache/huggingface/hub/models--thewh1teagle--phonikud-onnx/snapshots/b806189fe1fc0085b1012b7560ffb5e8ecfd72a2/phonikud-1.0.onnx')
    print("Done.")

    data = load_data()
    done, total = get_progress(data)
    print(f"\nProgress: {done}/{total}")

    entries = data['loanwords']
    print(f"\nProcessing loanwords...")
    for i in range(done, total):
        entry = entries[i]
        try:
            process_entry(entry, arabic_converter, hebrew_model)
            if (i + 1) % 10 == 0 or i == total - 1:
                save_checkpoint(data)
                print(f"  Progress: {i+1}/{total}")
                gc.collect()
        except Exception as e:
            print(f"  Error at entry {i} ({entry.get('id', '?')}): {e}")
            save_checkpoint(data)
            continue

    # Final save
    data['metadata']['ipa_generated'] = True
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nDone! Output saved to: {INPUT_FILE}")

    # Show samples
    print("\nSample results:")
    for entry in data['loanwords'][:2]:
        print(f"\n  {entry['id']}:")
        print(f"    Arabic: {entry['arabic']['arabic_undiac']} → IPA {entry['arabic'].get('arabic_ipa', '')} | uroman: {entry['arabic'].get('arabic_uroman', '')}")
        print(f"    Hebrew: {entry['hebrew']['hebrew_undiac']} → IPA {entry['hebrew'].get('hebrew_ipa', '')} | uroman: {entry['hebrew'].get('hebrew_uroman', '')}")


if __name__ == '__main__':
    main()