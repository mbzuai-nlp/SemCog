#!/usr/bin/env python3
"""Convert semantic disambiguation JSONL data to lm-evaluation-harness JSON format."""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "semantic_disambiguation"
OUTPUT_DIR = EXPERIMENT_DIR / "lm_eval" / "data"

SUB_EXPERIMENTS = {
    "true_cognate": {
        "data_dir": DATA_DIR / "true_cognate",
        "prefix": "semantic_disambiguation_mcq",
    },
    "false_friend": {
        "data_dir": DATA_DIR / "false_friend",
        "prefix": "false_friend_corrupt",
    },
    "loanword": {
        "data_dir": DATA_DIR / "loanword",
        "prefix": "loanword_mcq",
    },
}

FORMATS = ["undiac", "diac", "uroman", "undiac_uroman"]


def convert_jsonl_to_lmeval(input_file: Path, output_file: Path):
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            records.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "answer_idx": item["answer_idx"],
                "correct_answer": item["correct_answer"],
                "choices": ["A", "B", "C"],
            })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records)


def main():
    total = 0
    for sub_name, config in SUB_EXPERIMENTS.items():
        for fmt in FORMATS:
            input_file = config["data_dir"] / f"{config['prefix']}_{fmt}.jsonl"
            output_file = OUTPUT_DIR / sub_name / f"{fmt}.json"

            if not input_file.exists():
                print(f"  SKIP {input_file.name} (not found)")
                continue

            n = convert_jsonl_to_lmeval(input_file, output_file)
            total += n
            print(f"  {sub_name}/{fmt}: {n} records -> {output_file}")

    print(f"\nTotal: {total} records across all sub-experiments and formats")


if __name__ == "__main__":
    main()
