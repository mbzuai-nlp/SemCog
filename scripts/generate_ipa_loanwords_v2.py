#!/usr/bin/env python3
"""
Generate IPA/diac/uroman for remaining loanwords entries.
Only processes entries that still need generation.
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
CHECKPOINT_FILE = 'all_annotation_0522/loanwords_ipa_checkpoint2.json'
UROMAN_SCRIPT = '/tmp/uroman/uroman/uroman.pl'


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


def main():
    print("Initializing...")
    arabic_converter = ArabicToIPA()
    hebrew_model = OnnxModel('/workspace/.cache/huggingface/hub/models--thewh1teagle--phonikud-onnx/snapshots/b806189fe1fc0085b1012b7560ffb5e8ecfd72a2/phonikud-1.0.onnx')
    print("Done.")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find entries that still need generation
    need_gen = []
    for e in data['loanwords']:
        # Check if word IPA exists but sentence IPA missing
        has_word_ipa = e.get('arabic', {}).get('arabic_ipa') and e.get('hebrew', {}).get('hebrew_ipa')
        has_sent_ipa = e.get('arabic_sent', {}).get('arabic_sent_ipa') and e.get('hebrew_sent', {}).get('hebrew_sent_ipa')

        if not has_word_ipa or not has_sent_ipa:
            need_gen.append(e)

    print(f"Entries needing generation: {len(need_gen)}")

    for i, entry in enumerate(need_gen):
        eid = entry['id']
        has_word_ipa = entry.get('arabic', {}).get('arabic_ipa') and entry.get('hebrew', {}).get('hebrew_ipa')

        try:
            # Generate word IPA if missing
            if not has_word_ipa:
                ar_word = entry['arabic'].get('arabic_undiac', '')
                ar_diac, ar_ipa = arabic_to_ipa(arabic_converter, ar_word)
                entry['arabic']['arabic_diac'] = ar_diac
                entry['arabic']['arabic_uroman'] = uromanize(ar_word)
                entry['arabic']['arabic_ipa'] = ar_ipa

                he_word = entry['hebrew'].get('hebrew_undiac', '')
                he_niqqud, he_ipa = hebrew_to_ipa(hebrew_model, he_word)
                entry['hebrew']['hebrew_diac'] = he_niqqud
                entry['hebrew']['hebrew_uroman'] = uromanize(he_word)
                entry['hebrew']['hebrew_ipa'] = he_ipa

            # Always generate sentence IPA (this is what differs)
            ar_sent = entry['arabic_sent'].get('arabic_sent_undiac', '')
            ar_sent_diac, ar_sent_ipa = arabic_to_ipa(arabic_converter, ar_sent)
            entry['arabic_sent']['arabic_sent_diac'] = ar_sent_diac
            entry['arabic_sent']['arabic_sent_uroman'] = uromanize(ar_sent)
            entry['arabic_sent']['arabic_sent_ipa'] = ar_sent_ipa

            he_sent = entry['hebrew_sent'].get('hebrew_sent_undiac', '')
            he_sent_niqqud, he_sent_ipa = hebrew_to_ipa(hebrew_model, he_sent)
            entry['hebrew_sent']['hebrew_sent_diac'] = he_sent_niqqud
            entry['hebrew_sent']['hebrew_sent_uroman'] = uromanize(he_sent)
            entry['hebrew_sent']['hebrew_sent_ipa'] = he_sent_ipa

            if (i + 1) % 10 == 0 or i == len(need_gen) - 1:
                with open(INPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Progress: {i+1}/{len(need_gen)}")
                gc.collect()

        except Exception as e:
            print(f"  Error at {eid}: {e}")
            continue

    # Final save
    data['metadata']['ipa_generated'] = True
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Output saved to: {INPUT_FILE}")

    # Verify
    complete = sum(1 for e in data['loanwords']
                   if e.get('arabic', {}).get('arabic_ipa')
                   and e.get('hebrew', {}).get('hebrew_ipa')
                   and e.get('arabic_sent', {}).get('arabic_sent_ipa')
                   and e.get('hebrew_sent', {}).get('hebrew_sent_ipa'))
    print(f"Complete entries: {complete}/{len(data['loanwords'])}")


if __name__ == '__main__':
    main()