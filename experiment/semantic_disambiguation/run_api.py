#!/usr/bin/env python3
"""Run API model experiments for all 4 API models x 3 sub-experiments x 4 formats.

Usage:
  python run_api.py --models gpt-4o --sub-experiments true_cognate
  python run_api.py --test --limit 10

Environment variables required:
  OPENAI_API_KEY - OpenAI API key for GPT models
  DEEPSEEK_API_KEY - DeepSeek API key
  DASHSCOPE_API_KEY - Qwen/DashScope API key
  QWEN_BASE_URL - (optional) Qwen API base URL
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "semantic_disambiguation"
RESULTS_DIR = SCRIPT_DIR / "results"

# Load .env file if present
load_dotenv(PROJECT_DIR / ".env")

API_MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "gpt-5.4": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "use_max_completion_tokens": True,
    },
    "deepseek-v4": {
        "provider": "openai_compatible",
        "model_id": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "qwen3.6-plus": {
        "provider": "openai_compatible",
        "model_id": "qwen3.6-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "QWEN_BASE_URL",
    },
}

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


def load_data(sub_exp: str, fmt: str) -> list:
    config = SUB_EXPERIMENTS[sub_exp]
    filepath = config["data_dir"] / f"{config['prefix']}_{fmt}.jsonl"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def extract_answer(text: str) -> str | None:
    text = text.strip()
    match = re.search(r'\b([ABC])\b', text)
    return match.group(1) if match else None


async def call_api(
    session: aiohttp.ClientSession,
    model_config: dict,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    api_key = os.environ.get(model_config["api_key_env"], "")
    if not api_key:
        raise ValueError(f"Missing API key: {model_config['api_key_env']}")

    # Resolve base_url from env var if specified
    if "base_url_env" in model_config:
        base_url = os.environ.get(model_config["base_url_env"], model_config.get("base_url", ""))
    else:
        base_url = model_config["base_url"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    # gpt-5.x uses max_completion_tokens instead of max_tokens
    if model_config.get("use_max_completion_tokens"):
        payload["max_completion_tokens"] = 16
    else:
        payload["max_tokens"] = 16

    url = f"{base_url}/chat/completions"

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(url, headers=headers, json=payload,
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return {"response": content, "answer": extract_answer(content), "status": "ok"}
                    elif resp.status == 429:
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        return {"response": text, "answer": None, "status": f"error_{resp.status}"}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"response": str(e), "answer": None, "status": "exception"}

    return {"response": "max_retries", "answer": None, "status": "max_retries"}


async def run_experiment(
    model_name: str, model_config: dict, sub_exp: str, fmt: str,
    concurrency: int, limit: int | None, skip_existing: bool = False
) -> dict | None:
    output_dir = RESULTS_DIR / sub_exp / "api" / model_name
    output_file = output_dir / f"{fmt}.json"
    if skip_existing and output_file.exists():
        try:
            existing = json.load(open(output_file))
            if existing.get("accuracy") is not None:
                print(f"  {model_name}/{sub_exp}/{fmt}: SKIPPED (exists, acc={existing['accuracy']:.4f})")
                return existing
        except Exception:
            pass

    data = load_data(sub_exp, fmt)
    if limit:
        data = data[:limit]

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            call_api(session, model_config, item["prompt"], semaphore)
            for item in data
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    correct = 0
    predictions = []
    for item, resp in zip(data, responses):
        if isinstance(resp, Exception):
            pred = None
        else:
            pred = resp.get("answer")
        predictions.append({
            "id": item["id"],
            "correct_answer": item["correct_answer"],
            "answer_idx": item["answer_idx"],
            "prediction": pred,
            "permutation": item.get("permutation", []),
        })
        if pred == item["correct_answer"]:
            correct += 1

    accuracy = correct / len(data) if data else 0

    per_label = {"A": {"correct": 0, "total": 0}, "B": {"correct": 0, "total": 0}, "C": {"correct": 0, "total": 0}}
    for pred_item in predictions:
        label = pred_item["correct_answer"]
        per_label[label]["total"] += 1
        if pred_item["prediction"] == label:
            per_label[label]["correct"] += 1

    per_label_acc = {k: v["correct"]/v["total"] if v["total"] > 0 else 0 for k, v in per_label.items()}

    result = {
        "model": model_name,
        "sub_experiment": sub_exp,
        "format": fmt,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(data),
        "per_label_accuracy": per_label_acc,
        "predictions": predictions,
        "timestamp": datetime.now().isoformat(),
    }

    output_dir = RESULTS_DIR / sub_exp / "api" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{fmt}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  {model_name}/{sub_exp}/{fmt}: {accuracy:.4f} ({correct}/{len(data)})")
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--sub-experiments", nargs="*", default=None, dest="sub_experiments")
    parser.add_argument("--formats", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (testing)")
    parser.add_argument("--sequential", action="store_true", help="Run models sequentially instead of in parallel")
    parser.add_argument("--skip-existing", action="store_true", help="Skip conditions that already have results")
    args = parser.parse_args()

    models_to_run = args.models or list(API_MODELS.keys())
    sub_exps = args.sub_experiments or list(SUB_EXPERIMENTS.keys())
    formats = args.formats or FORMATS

    all_results = []
    for model_name in models_to_run:
        if model_name not in API_MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        model_config = API_MODELS[model_name]
        for sub_exp in sub_exps:
            for fmt in formats:
                result = await run_experiment(
                    model_name, model_config, sub_exp, fmt,
                    args.concurrency, args.limit, args.skip_existing
                )
                all_results.append(result)

    print("\n=== API Results Summary ===")
    for r in all_results:
        if r:
            print(f"  {r['model']}/{r['sub_experiment']}/{r['format']}: {r['accuracy']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
