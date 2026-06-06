#!/usr/bin/env python3
"""Analyze Semantic Disambiguation experiment results across all sub-experiments and models."""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"

SUB_EXPERIMENTS = ["true_cognate", "false_friend", "loanword"]
FORMATS = ["undiac", "diac", "uroman", "undiac_uroman"]

# Model display names
MODEL_NAMES = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "GLM-4-9B-Chat": "GLM4-9B",
    "Gemma-2-9B-IT": "Gemma2-9B",
    "Jais-2-8B-Chat": "Jais2-8B",
    "Jais-Adapted-7B-Chat": "Jais-Adapt-7B",
    "DictaLM-2.0-Instruct": "DictaLM-2.0",
    "Aya-23-8B": "Aya-23-8B",
    "gpt-4o": "GPT-4o",
    "gpt-5.4": "GPT-5.4",
    "deepseek-v4": "DeepSeek-V4",
    "qwen3.6-plus": "Qwen3.6-Plus",
}


def load_results(results_dir: Path, sub_exp: str, model_type: str = "opensource"):
    """Load all results for a sub-experiment."""
    results = []
    sub_dir = results_dir / sub_exp / model_type
    if not sub_dir.exists():
        return results

    if model_type == "api":
        for model_dir in sorted(sub_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for res_file in sorted(model_dir.glob("*.json")):
                try:
                    data = json.load(open(res_file))
                except Exception:
                    continue
                acc = data.get("accuracy")
                if acc is not None:
                    results.append({
                        "model": model_name,
                        "sub_exp": sub_exp,
                        "format": data.get("format", res_file.stem),
                        "accuracy": acc,
                        "acc_stderr": None,
                    })
        return results

    for model_dir in sorted(sub_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for inner_dir in model_dir.iterdir():
            if not inner_dir.is_dir():
                continue
            for res_file in sorted(inner_dir.glob("results_*.json")):
                try:
                    data = json.load(open(res_file))
                except Exception:
                    continue
                for task_name, vals in data.get("results", {}).items():
                    fmt = next((f for f in sorted(FORMATS, key=len, reverse=True)
                                if task_name.endswith(f"_{f}")), task_name.rsplit("_", 1)[-1])
                    acc = vals.get("acc,none")
                    acc_stderr = vals.get("acc_stderr,none")
                    if acc is not None:
                        results.append({
                            "model": model_name,
                            "sub_exp": sub_exp,
                            "format": fmt,
                            "accuracy": acc,
                            "acc_stderr": acc_stderr,
                        })
    return results


def load_all_results(results_dir: Path):
    """Load results for all sub-experiments and model types."""
    all_results = []
    for sub_exp in SUB_EXPERIMENTS:
        for model_type in ["opensource", "api"]:
            results = load_results(results_dir, sub_exp, model_type)
            all_results.extend(results)
    return all_results


def get_latest_per_condition(results):
    """Keep only the latest result per model/sub_exp/format (dedup)."""
    df = pd.DataFrame(results)
    if df.empty:
        return df
    # Sort by accuracy (keep non-None), then drop duplicates keeping last
    df = df.sort_values("accuracy", na_position="first")
    df = df.drop_duplicates(subset=["model", "sub_exp", "format"], keep="last")
    return df


def print_accuracy_table(df, sub_exp: str):
    """Print accuracy pivot table for a sub-experiment."""
    sub_df = df[df["sub_exp"] == sub_exp].copy()
    if sub_df.empty:
        print(f"\n  No results for {sub_exp}")
        return

    sub_df["display_model"] = sub_df["model"].map(lambda m: MODEL_NAMES.get(m, m))

    pivot = sub_df.pivot_table(
        index="display_model",
        columns="format",
        values="accuracy",
        aggfunc="first",
    )
    col_order = [c for c in FORMATS if c in pivot.columns]
    pivot = pivot[col_order]

    # Add average column
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=False)

    print(f"\n{'='*80}")
    print(f"  {sub_exp.replace('_', ' ').title()} Disambiguation")
    print(f"{'='*80}")
    print(pivot.round(4).to_string())


def print_model_summary(df):
    """Print model performance summary across all sub-experiments."""
    df = df.copy()
    df["display_model"] = df["model"].map(lambda m: MODEL_NAMES.get(m, m))

    model_stats = df.groupby("display_model").agg({
        "accuracy": ["mean", "std", "count"]
    }).round(4)
    model_stats.columns = ["Mean", "Std", "N"]
    model_stats = model_stats.sort_values("Mean", ascending=False)

    print(f"\n{'='*80}")
    print("  Model Performance Summary (across all sub-experiments & formats)")
    print(f"{'='*80}")
    print(model_stats.to_string())


def print_format_summary(df):
    """Print format performance summary."""
    format_stats = df.groupby("format").agg({
        "accuracy": ["mean", "std"]
    }).round(4)
    format_stats.columns = ["Mean", "Std"]
    format_stats = format_stats.sort_values("Mean", ascending=False)

    print(f"\n{'='*80}")
    print("  Format Performance Summary (across all models & sub-experiments)")
    print(f"{'='*80}")
    print(format_stats.to_string())


def print_sub_exp_summary(df):
    """Print sub-experiment performance summary."""
    sub_stats = df.groupby("sub_exp").agg({
        "accuracy": ["mean", "std"]
    }).round(4)
    sub_stats.columns = ["Mean", "Std"]
    sub_stats = sub_stats.sort_values("Mean", ascending=False)

    print(f"\n{'='*80}")
    print("  Sub-Experiment Performance Summary")
    print(f"{'='*80}")
    print(sub_stats.to_string())


def print_cross_table(df):
    """Print model × sub-experiment cross table (averaged over formats)."""
    df = df.copy()
    df["display_model"] = df["model"].map(lambda m: MODEL_NAMES.get(m, m))

    pivot = df.pivot_table(
        index="display_model",
        columns="sub_exp",
        values="accuracy",
        aggfunc="mean",
    )
    col_order = [c for c in SUB_EXPERIMENTS if c in pivot.columns]
    pivot = pivot[col_order]
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=False)

    print(f"\n{'='*80}")
    print("  Model × Sub-Experiment (avg over formats)")
    print(f"{'='*80}")
    print(pivot.round(4).to_string())


def check_missing(results_dir: Path):
    """Check for missing conditions."""
    print(f"\n{'='*80}")
    print("  Missing Conditions")
    print(f"{'='*80}")

    # Load existing results to find which model/sub_exp/format combos exist
    all_results = load_all_results(results_dir)
    existing = set()
    for r in all_results:
        existing.add((r["model"], r["sub_exp"], r["format"]))

    # Check all expected combinations
    all_models_opensource = [
        "Qwen2.5-7B-Instruct", "Qwen3-8B", "Llama-3.1-8B-Instruct", "GLM-4-9B-Chat",
        "Gemma-2-9B-IT", "Jais-2-8B-Chat", "Jais-Adapted-7B-Chat", "DictaLM-2.0-Instruct",
        "Aya-23-8B",
    ]
    all_models_api = ["gpt-4o", "gpt-5.4", "deepseek-v4", "qwen3.6-plus"]

    missing = []
    for model in all_models_opensource + all_models_api:
        for sub_exp in SUB_EXPERIMENTS:
            for fmt in FORMATS:
                if (model, sub_exp, fmt) not in existing:
                    missing.append(f"{model}/{sub_exp}/{fmt}")

    if missing:
        for m in sorted(missing):
            print(f"  MISSING: {m}")
        print(f"\n  Total missing: {len(missing)}")
    else:
        print("  All conditions complete!")


def main():
    parser = argparse.ArgumentParser(description="Analyze Semantic Disambiguation results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Save report as JSON")
    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")

    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} result records")

    if not results:
        print("No results found. Run experiments first.")
        return

    df = get_latest_per_condition(results)
    print(f"After dedup: {len(df)} unique conditions")

    # Print tables
    for sub_exp in SUB_EXPERIMENTS:
        print_accuracy_table(df, sub_exp)

    print_cross_table(df)
    print_model_summary(df)
    print_format_summary(df)
    print_sub_exp_summary(df)
    check_missing(args.results_dir)

    # Save report
    if args.output:
        report = df.to_dict(orient="records")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
