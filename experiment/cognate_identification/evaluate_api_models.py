#!/usr/bin/env python3
"""
Cognate Identification Experiment with API-based models.

Supports: GPT-4o, GPT-5.4, DeepSeek-v4, Qwen3.6-Plus
Runs with concurrent API calls (default concurrency=20).

Usage:
  python evaluate_api_models.py --model gpt-4o --format undiac --task_type 3class --data data/3class_undiac.jsonl --output results/3class/undiac/gpt-4o.json

Environment variables required:
  OPENAI_API_KEY - OpenAI API key for GPT models
  DEEPSEEK_API_KEY - DeepSeek API key
  DASHSCOPE_API_KEY - Qwen/DashScope API key
"""

import json
import os
import sys
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

API_MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5.4": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek-v4": {
        "provider": "deepseek",
        "model_id": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen3.6-plus": {
        "provider": "dashscope",
        "model_id": "qwen3.6-plus",
        "base_url": os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"


def load_prompts(task_type: str) -> dict:
    fname = f"cognate_identification_{task_type}_prompts.json"
    path = PROMPTS_DIR / fname
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_label(response: str, valid_labels: list) -> str:
    response_upper = response.upper()
    for label in valid_labels:
        if label in response_upper:
            return label
    return response.strip().upper()


def call_openai_compatible(prompt: str, model_name: str, model_config: dict, max_retries: int = 3) -> dict:
    api_key = os.environ.get(model_config["api_key_env"], "")
    if not api_key:
        raise ValueError(f"Missing API key: {model_config['api_key_env']}")

    url = f"{model_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_tokens_val = 500 if model_name in ("deepseek-v4", "qwen3.6-plus") else 20
    if model_config["model_id"].startswith("gpt-5"):
        max_tokens_key = "max_completion_tokens"
    else:
        max_tokens_key = "max_tokens"

    data = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        max_tokens_key: max_tokens_val,
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=90)
            if resp.status_code == 200:
                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning_content", "") or ""
                return {"content": content, "reasoning": reasoning}
            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt * 2, 30))
            else:
                print(f"API error: {resp.status_code} - {resp.text[:200]}")
                return {"content": "", "reasoning": ""}
        except Exception as e:
            print(f"API exception: {e}")
            time.sleep(min(2 ** attempt, 10))
    return {"content": "", "reasoning": ""}


def format_prompt(entry: dict, fmt: str, prompt_config: dict) -> str:
    template = prompt_config["prompts"][fmt]["template"]
    field_map = prompt_config["prompts"][fmt]["data_fields"]

    kwargs = {}
    for var_name, field_path in field_map.items():
        parts = field_path.split(".")
        val = entry
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, {})
            else:
                val = {}
                break
        if isinstance(val, str) and val:
            kwargs[var_name] = val
        else:
            flat_key = parts[-1] if len(parts) > 1 else parts[0]
            kwargs[var_name] = entry.get(flat_key, entry.get(parts[0], ""))

    return template.format(**kwargs)


def evaluate_single(entry: dict, prompt: str, model_name: str, model_config: dict,
                   valid_labels: list, delay: float) -> dict:
    result = call_openai_compatible(prompt, model_name, model_config)
    content = result["content"]
    reasoning = result["reasoning"]

    response = content if content.strip() else reasoning
    predicted = extract_label(response, valid_labels)
    true_label = entry["label"]
    time.sleep(delay)
    return {
        "id": entry.get("id", ""),
        "true_label": true_label,
        "predicted_label": predicted,
        "raw_response": content,
        "reasoning": reasoning[:500] if reasoning else "",
        "correct": predicted == true_label,
    }


def evaluate_model(model_name: str, data_path: str, fmt: str, task_type: str,
                   output_path: str = None, delay: float = 0.05,
                   concurrency: int = 50, test_mode: bool = False):
    prompts_config = load_prompts(task_type)
    valid_labels = prompts_config["labels"]

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if test_mode:
        data = data[:10]
        print(f"Test mode: using {len(data)} samples")

    model_config = API_MODELS[model_name]
    print(f"Loaded {len(data)} samples, evaluating {model_name}/{task_type}/{fmt} (concurrency={concurrency})")

    prompts = [format_prompt(entry, fmt, prompts_config) for entry in data]

    results = [None] * len(data)
    if concurrency <= 1:
        for i, (entry, prompt) in enumerate(tqdm(zip(data, prompts), total=len(data),
                                                   desc=f"{model_name}/{fmt}")):
            results[i] = evaluate_single(entry, prompt, model_name, model_config, valid_labels, delay)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for i, (entry, prompt) in enumerate(zip(data, prompts)):
                future = executor.submit(evaluate_single, entry, prompt,
                                         model_name, model_config, valid_labels, delay)
                futures[future] = i
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{model_name}/{fmt}"):
                idx = futures[future]
                results[idx] = future.result()

    sys.path.insert(0, os.path.dirname(__file__))
    from analyze_results import compute_metrics
    metrics = compute_metrics(results, task_type)

    print(f"\nResults for {model_name} ({task_type}/{fmt}):")
    print(f"  Acc: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    if task_type == "3class":
        print(f"  LCR: {metrics.get('lcr', 0):.4f}")
        print(f"  CUR_TC: {metrics.get('cur_tc', 0):.4f}")
        print(f"  CUR_FF: {metrics.get('cur_ff', 0):.4f}")
        print(f"  SDB: {metrics.get('sdb', 0):+.4f}")
        print(f"  CLB: {metrics.get('clb', 0):+.4f}")
    else:
        print(f"  SDB: {metrics.get('sdb', 0):+.4f}")

    if output_path:
        output = {
            "model": model_name,
            "format": fmt,
            "task_type": task_type,
            **metrics,
            "predictions": results,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_path}")

    return metrics["acc"]


def main():
    parser = argparse.ArgumentParser(description="API model evaluation for cognate identification")
    parser.add_argument("--model", required=True,
                        choices=["gpt-4o", "gpt-5.4", "deepseek-v4", "qwen3.6-plus"])
    parser.add_argument("--format", required=True,
                        choices=["undiac", "diac", "uroman", "ipa"])
    parser.add_argument("--task_type", required=True,
                        choices=["3class", "2class"])
    parser.add_argument("--data", required=True, help="Path to JSONL data file")
    parser.add_argument("--output", default=None, help="Path to output JSON file")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between API calls per worker (seconds)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max concurrent API calls (default: 50)")
    parser.add_argument("--test", action="store_true", help="Test mode (10 samples)")
    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.format, args.task_type,
                   args.output, args.delay, args.concurrency, args.test)


if __name__ == "__main__":
    main()
