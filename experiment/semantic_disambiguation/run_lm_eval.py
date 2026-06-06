#!/usr/bin/env python3
"""Run lm-evaluation-harness experiments for open-source models on dual GPUs.

Strategy: Run 2 models in parallel on 2 GPUs. Each model runs all formats
sequentially (one lm_eval call per format/sub-exp), so GPU memory is freed
between runs.

For Jais-2-8B-Chat, uses the 'hf' model backend with transformers 5.x.
For all other models, uses the 'vllm' backend with transformers 4.x.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
LM_EVAL_DIR = EXPERIMENT_DIR / "lm_eval"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS = {
    "Qwen2.5-7B-Instruct": {
        "path": os.environ.get("MODEL_PATH_QWEN25_7B", "Qwen/Qwen2.5-7B-Instruct"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.80"],
    },
    "Qwen3-8B": {
        "path": os.environ.get("MODEL_PATH_QWEN3_8B", "Qwen/Qwen3-8B"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "tokenizer_mode=slow", "max_model_len=2048", "gpu_memory_utilization=0.80"],
        "extra_args": ["--trust_remote_code"],
    },
    "Llama-3.1-8B-Instruct": {
        "path": os.environ.get("MODEL_PATH_LLAMA31_8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.80"],
    },
    "GLM-4-9B-Chat": {
        "path": os.environ.get("MODEL_PATH_GLM4_9B", "THUDM/glm-4-9b-chat"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.75"],
        "extra_args": ["--trust_remote_code"],
    },
    "Gemma-2-9B-IT": {
        "path": os.environ.get("MODEL_PATH_GEMMA2_9B", "google/gemma-2-9b-it"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.75"],
    },
    "Jais-2-8B-Chat": {
        "path": os.environ.get("MODEL_PATH_JAIS2_8B", "inceptionai/Jais-2-8B-Chat"),
        "batch_size": 8,
        "tp": 1,
        "backend": "hf",
        "extra_args": ["--trust_remote_code"],
    },
    "Jais-Adapted-7B-Chat": {
        "path": os.environ.get("MODEL_PATH_JAIS_ADAPT_7B", "inceptionai/Jais-Adapted-7B-Chat"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.80"],
        "extra_args": ["--trust_remote_code"],
    },
    "DictaLM-2.0-Instruct": {
        "path": os.environ.get("MODEL_PATH_DICTALM_2", "dicta-il/DictaLM-2.0-Instruct"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.80"],
        "extra_args": ["--trust_remote_code"],
    },
    "Aya-23-8B": {
        "path": os.environ.get("MODEL_PATH_AYA_8B", "CohereForAI/aya-23-8B"),
        "batch_size": 4,
        "tp": 1,
        "backend": "vllm",
        "vllm_args": ["enforce_eager=True", "max_model_len=2048", "gpu_memory_utilization=0.80"],
    },
}

SUB_EXPERIMENTS = ["true_cognate", "false_friend", "loanword"]
FORMATS = ["undiac", "diac", "uroman", "undiac_uroman"]


def run_single_format(
    model_name: str, model_config: dict, sub_exp: str, fmt: str, gpu_id: int,
    limit: int | None = None, skip_existing: bool = False
):
    task_name = f"{sub_exp}_disambiguation_{fmt}"
    output_dir = RESULTS_DIR / sub_exp / "opensource" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if results already exist
    if skip_existing:
        existing = list(output_dir.rglob(f"results_*.json"))
        for f in existing:
            try:
                data = json.load(open(f))
                if task_name in data.get("results", {}):
                    acc = data["results"][task_name].get("acc,none")
                    if acc is not None:
                        print(f"[GPU {gpu_id}] {model_name}/{sub_exp}/{fmt} SKIPPED (exists, acc={acc:.4f})")
                        return True
            except Exception:
                pass

    # diac format: reduce max_model_len for vllm to leave room for logprobs allocation
    # vllm's sampler needs ~vocab_size*4 bytes for logprobs, which OOMs when KV cache
    # takes most of GPU memory. Reducing max_model_len frees up GPU memory.
    backend = model_config.get("backend", "vllm")
    batch_size = model_config["batch_size"]

    if backend == "vllm":
        model_args_parts = [f"pretrained={model_config['path']}", f"tensor_parallel_size={model_config['tp']}"]
        for va in model_config.get("vllm_args", []):
            if fmt == "diac" and va.startswith("max_model_len="):
                model_args_parts.append("max_model_len=512")
            else:
                model_args_parts.append(va)
        # trust_remote_code must be in model_args for vllm
        if "trust_remote_code" in model_config.get("extra_args", []):
            model_args_parts.append("trust_remote_code=True")
        model_args_str = ",".join(model_args_parts)
        if fmt == "diac":
            batch_size = 1
    else:
        # hf backend
        model_args_parts = [f"pretrained={model_config['path']}"]
        if "trust_remote_code" in model_config.get("extra_args", []):
            model_args_parts.append("trust_remote_code=True")
        model_args_str = ",".join(model_args_parts)

    cmd = [
        "lm_eval",
        "--model", backend,
        "--model_args", model_args_str,
        "--tasks", task_name,
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
        "--log_samples",
        "--include_path", str(LM_EVAL_DIR / "tasks"),
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    extra = model_config.get("extra_args", [])
    # For vllm, --trust_remote_code must be passed as CLI flag AND in model_args
    # (model_args alone doesn't propagate to tokenizer loading)
    needs_trc = "--trust_remote_code" in extra
    if backend == "vllm" and needs_trc:
        cmd.append("--trust_remote_code")
    cmd.extend([a for a in extra if a != "--trust_remote_code"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[GPU {gpu_id}] {model_name}/{sub_exp}/{fmt} starting ({datetime.now().strftime('%H:%M:%S')})")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=EXPERIMENT_DIR)

    # Brief pause to let GPU memory be released after subprocess exits
    import time
    time.sleep(5)

    # Force GPU memory cleanup
    try:
        cleanup_cmd = [
            sys.executable, "-c",
            f"import torch; torch.cuda.set_device({gpu_id}); torch.cuda.empty_cache(); "
            f"import gc; gc.collect()"
        ]
        cleanup_env = os.environ.copy()
        cleanup_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(cleanup_cmd, env=cleanup_env, capture_output=True, timeout=30)
    except Exception:
        pass

    if result.returncode != 0:
        # Retry once after waiting for GPU cleanup
        print(f"[GPU {gpu_id}] {model_name}/{sub_exp}/{fmt} failed, retrying after 30s GPU cleanup...")
        time.sleep(30)
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=EXPERIMENT_DIR)
        time.sleep(5)
        try:
            cleanup_cmd = [
                sys.executable, "-c",
                f"import torch; torch.cuda.set_device({gpu_id}); torch.cuda.empty_cache(); "
                f"import gc; gc.collect()"
            ]
            cleanup_env = os.environ.copy()
            cleanup_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            subprocess.run(cleanup_cmd, env=cleanup_env, capture_output=True, timeout=30)
        except Exception:
            pass

    if result.returncode != 0:
        print(f"[GPU {gpu_id}] {model_name}/{sub_exp}/{fmt} FAILED (exit {result.returncode})")
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-30:]:
            print(f"  {line}")
        return False

    # Parse accuracy from stdout
    acc = None
    for line in result.stdout.split("\n"):
        if "|acc" in line and "│" in line:
            parts = line.split("│")
            for p in parts:
                p = p.strip()
                try:
                    val = float(p)
                    if 0 <= val <= 1:
                        acc = val
                except ValueError:
                    pass

    print(f"[GPU {gpu_id}] {model_name}/{sub_exp}/{fmt} done - acc={acc} ({datetime.now().strftime('%H:%M:%S')})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Model names to run (default: all)")
    parser.add_argument("--sub-experiments", nargs="*", default=None, dest="sub_experiments")
    parser.add_argument("--formats", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU ID to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip conditions that already have results")
    parser.add_argument("--dual-gpu", action="store_true", help="Run on 2 GPUs in parallel (alternating models)")
    args = parser.parse_args()

    models_to_run = args.models or list(MODELS.keys())
    sub_exps = args.sub_experiments or SUB_EXPERIMENTS
    formats = args.formats or FORMATS
    gpu_id = args.gpu or 0

    if args.dual_gpu:
        # Split models across 2 GPUs and run in parallel
        import threading

        def run_model_on_gpu(model_names, gpu):
            for model_name in model_names:
                if model_name not in MODELS:
                    print(f"Unknown model: {model_name}, skipping")
                    continue
                model_config = MODELS[model_name]
                for sub_exp in sub_exps:
                    for fmt in formats:
                        success = run_single_format(
                            model_name, model_config, sub_exp, fmt, gpu, args.limit,
                            skip_existing=args.skip_existing
                        )
                        if success:
                            with lock:
                                completed[0] += 1
                        else:
                            with lock:
                                failed[0] += 1
                        with lock:
                            done = completed[0] + failed[0]
                            if done % 3 == 0:
                                print(f"  Progress: {done}/{total_runs} ({completed[0]} OK, {failed[0]} FAILED)")

        # Split models evenly across GPUs
        models_gpu0 = models_to_run[::2]  # even indices -> GPU 0
        models_gpu1 = models_to_run[1::2]  # odd indices -> GPU 1
        print(f"Dual-GPU mode: GPU 0 = {models_gpu0}, GPU 1 = {models_gpu1}")

        total_runs = len(models_to_run) * len(sub_exps) * len(formats)
        completed = [0]
        failed = [0]
        lock = threading.Lock()

        print(f"Total runs: {total_runs}")
        print(f"Models: {models_to_run}")
        print(f"Sub-experiments: {sub_exps}")
        print(f"Formats: {formats}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        t0 = threading.Thread(target=run_model_on_gpu, args=(models_gpu0, 0))
        t1 = threading.Thread(target=run_model_on_gpu, args=(models_gpu1, 1))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        print()
        print(f"=== Final Summary ===")
        print(f"Completed: {completed[0]}/{total_runs}")
        print(f"Failed: {failed[0]}/{total_runs}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    total_runs = len(models_to_run) * len(sub_exps) * len(formats)
    completed = 0
    failed = 0

    print(f"Total runs: {total_runs}")
    print(f"Models: {models_to_run}")
    print(f"Sub-experiments: {sub_exps}")
    print(f"Formats: {formats}")
    print(f"GPU: {gpu_id}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        model_config = MODELS[model_name]

        for sub_exp in sub_exps:
            for fmt in formats:
                success = run_single_format(
                    model_name, model_config, sub_exp, fmt, gpu_id, args.limit,
                    skip_existing=args.skip_existing
                )
                if success:
                    completed += 1
                else:
                    failed += 1

                # Progress
                done = completed + failed
                if done % 3 == 0:
                    print(f"  Progress: {done}/{total_runs} ({completed} OK, {failed} FAILED)")

    print()
    print(f"=== Final Summary ===")
    print(f"Completed: {completed}/{total_runs}")
    print(f"Failed: {failed}/{total_runs}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
