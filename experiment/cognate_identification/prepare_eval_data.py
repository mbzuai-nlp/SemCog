"""
Prepare evaluation data for CognateIdentification_0523 experiment.

Reads merged_dataset.json and produces:
  - 3-class JSONL files (TRUE_COGNATE, FALSE_FRIEND, LOANWORD) for each format
  - 2-class JSONL files (TRUE_COGNATE, FALSE_FRIEND only) for each format

Each JSONL line: {"arabic": "<word>", "hebrew": "<word>", "label": "<LABEL>"}
"""

import json
import argparse
from pathlib import Path

FORMAT_FIELDS = {
    "undiac": ("arabic_undiac", "hebrew_undiac"),
    "diac": ("arabic_diac", "hebrew_diac"),
    "uroman": ("arabic_uroman", "hebrew_uroman"),
    "ipa": ("arabic_ipa", "hebrew_ipa"),
}


def load_merged_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for key in ("true_cognates", "false_friends", "loanwords"):
        entries.extend(data.get(key, []))
    return entries


def write_jsonl(entries: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare eval data for cognate identification")
    parser.add_argument("--input", default="../../data/all_annotation_0522/merged_dataset.json",
                        help="Path to merged_dataset.json")
    parser.add_argument("--output_dir", default="data", help="Output directory for JSONL files")
    args = parser.parse_args()

    entries = load_merged_dataset(args.input)
    print(f"Total entries loaded: {len(entries)}")

    for fmt, (ar_field, he_field) in FORMAT_FIELDS.items():
        # 3-class: all entries
        jsonl_3class = []
        for e in entries:
            ar_word = e.get("arabic", {}).get(ar_field, "")
            he_word = e.get("hebrew", {}).get(he_field, "")
            label_raw = e.get("type", "")
            if not label_raw:
                continue
            # Normalize label: TRUE_COGNATE, FALSE_FRIEND stay; Loanword* -> LOANWORD
            label_upper = label_raw.upper()
            if label_upper.startswith("LOANWORD"):
                label = "LOANWORD"
            elif label_upper in ("TRUE_COGNATE", "FALSE_FRIEND"):
                label = label_upper
            else:
                continue
            if ar_word and he_word and label:
                jsonl_3class.append({"arabic": ar_word, "hebrew": he_word, "label": label})

        write_jsonl(jsonl_3class, str(Path(args.output_dir) / f"3class_{fmt}.jsonl"))
        tc = sum(1 for e in jsonl_3class if e["label"] == "TRUE_COGNATE")
        ff = sum(1 for e in jsonl_3class if e["label"] == "FALSE_FRIEND")
        lw = sum(1 for e in jsonl_3class if e["label"] == "LOANWORD")
        print(f"  3-class/{fmt}: {len(jsonl_3class)} entries (TC={tc}, FF={ff}, LW={lw})")

        # 2-class: only TRUE_COGNATE and FALSE_FRIEND
        jsonl_2class = [e for e in jsonl_3class if e["label"] in ("TRUE_COGNATE", "FALSE_FRIEND")]
        write_jsonl(jsonl_2class, str(Path(args.output_dir) / f"2class_{fmt}.jsonl"))
        tc2 = sum(1 for e in jsonl_2class if e["label"] == "TRUE_COGNATE")
        ff2 = sum(1 for e in jsonl_2class if e["label"] == "FALSE_FRIEND")
        print(f"  2-class/{fmt}: {len(jsonl_2class)} entries (TC={tc2}, FF={ff2})")

    print("Done!")


if __name__ == "__main__":
    main()
