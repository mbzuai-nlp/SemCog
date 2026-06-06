#!/usr/bin/env python3
"""
Orchestrator: run all cognate identification experiments.

Usage:
  # Run all API model experiments
  python scripts/run_all_experiments.py --mode api

  # Run all local model experiments via lm-evaluation-harness
  python scripts/run_all_experiments.py --mode local

  # Run specific model/format/task_type
  python scripts/run_all_experiments.py --mode api --model gpt-4o --format undiac --task_type 3class
  python scripts/run_all_experiments.py --mode local --model Qwen2.5-7B-Instruct --task_type 2class

  # Test mode
  python scripts/run_all_experiments.py --mode api --test
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
SCRIPTS_DIR = SCRIPT_DIR
TASKS_DIR = SCRIPT_DIR / "tasks"

API_MODELS = ["gpt-4o", "gpt-5.4", "deepseek-v4", "qwen3.6-plus"]
LOCAL_MODELS = {
    "Qwen2.5-7B-Instruct": os.environ.get("MODEL_PATH_QWEN25_7B", "Qwen/Qwen2.5-7B-Instruct"),
    "Qwen3-8B": os.environ.get("MODEL_PATH_QWEN3_8B", "Qwen/Qwen3-8B"),
    "Llama-3.1-8B-Instruct": os.environ.get("MODEL_PATH_LLAMA31_8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "GLM-4-9B-Chat": os.environ.get("MODEL_PATH_GLM4_9B", "THUDM/glm-4-9b-chat"),
    "Gemma-2-9B-IT": os.environ.get("MODEL_PATH_GEMMA2_9B", "google/gemma-2-9b-it"),
    "Jais-2-8B-Chat": os.environ.get("MODEL_PATH_JAIS2_8B", "inceptionai/Jais-2-8B-Chat"),
    "Jais-Adapted-7B-Chat": os.environ.get("MODEL_PATH_JAIS_ADAPT_7B", "inceptionai/Jais-Adapted-7B-Chat"),
    "DictaLM-2.0-Instruct": os.environ.get("MODEL_PATH_DICTALM_2", "dicta-il/DictaLM-2.0-Instruct"),
    "Aya-23-8B": os.environ.get("MODEL_PATH_AYA_8B", "CohereForAI/aya-23-8B"),
}

FORMATS = ["undiac", "diac", "uroman", "ipa"]
TASK_TYPES = ["3class", "2class"]


def run_api_experiments(model=None, fmt=None, task_type=None, test=False,
                       delay=0.1, concurrency=20):
    for m in (API_MODELS if model is None else [model]):
        for f in (FORMATS if fmt is None else [fmt]):
            for tt in (TASK_TYPES if task_type is None else [task_type]):
                data_path = DATA_DIR / f"{tt}_{f}.jsonl"
                output_path = RESULTS_DIR / tt / f / f"{m}.json"

                if output_path.exists():
                    print(f"  Skipping (exists): {tt}/{f}/{m}")
                    continue

                cmd = [
                    sys.executable, str(SCRIPTS_DIR / "evaluate_api_models.py"),
                    "--model", m,
                    "--format", f,
                    "--task_type", tt,
                    "--data", str(data_path),
                    "--output", str(output_path),
                    "--delay", str(delay),
                    "--concurrency", str(concurrency),
                ]
                if test:
                    cmd.append("--test")

                print(f"  Running: {m} / {tt} / {f}")
                subprocess.run(cmd, check=True)


def run_local_experiments(model=None, task_type=None, gpu=0):
    for m_name, m_path in (LOCAL_MODELS.items() if model is None else [(model, LOCAL_MODELS.get(model, model))]):
        # Determine batch size and TP based on model size
        is_35b = "35B" in m_name
        batch_size = "8" if is_35b else "32"
        model_args = f"pretrained={m_path},dtype=bfloat16"
        if is_35b:
            model_args += ",tensor_parallel_size=2"

        for tt in (TASK_TYPES if task_type is None else [task_type]):
            if tt == "3class":
                tasks = "cognate_3class_undiac,cognate_3class_diac,cognate_3class_uroman,cognate_3class_ipa"
            else:
                tasks = "cognate_2class_undiac,cognate_2class_diac,cognate_2class_uroman,cognate_2class_ipa"

            output_path = RESULTS_DIR / tt / m_name
            if (output_path / "results.json").exists():
                print(f"  Skipping (exists): {tt}/{m_name}")
                continue

            cmd = [
                "lm_eval", "run",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", tasks,
                "--include_path", str(TASKS_DIR),
                "--batch_size", batch_size,
                "--output_path", str(output_path),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0,1" if is_35b else str(gpu)
            env["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

            print(f"  Running: {m_name} / {tt} (bs={batch_size}, gpu={'0,1' if is_35b else gpu})")
            subprocess.run(cmd, check=True, env=env, cwd=PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run all cognate identification experiments")
    parser.add_argument("--mode", required=True, choices=["api", "local"])
    parser.add_argument("--model", default=None, help="Specific model name")
    parser.add_argument("--format", default=None, choices=FORMATS)
    parser.add_argument("--task_type", default=None, choices=TASK_TYPES)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between API calls per worker (seconds)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API calls (default: 20)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.mode == "api":
        run_api_experiments(args.model, args.format, args.task_type, args.test,
                            args.delay, args.concurrency)
    else:
        run_local_experiments(args.model, args.task_type, args.gpu)


if __name__ == "__main__":
    main()
