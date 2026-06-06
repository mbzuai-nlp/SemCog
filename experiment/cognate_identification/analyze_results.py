#!/usr/bin/env python3
"""
Analyze results for CognateIdentification_0523 experiment.

Computes three tiers of metrics:
  1. Standard Classification: Acc, MacroF1
  2. Fine-Grained Error Rates: LCR, CUR_TC, CUR_FF
  3. Directional Bias: SDB, CLB

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --task_type 3class
  python scripts/analyze_results.py --model gpt-4o
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

FORMATS = ["undiac", "diac", "uroman", "ipa"]
TASK_TYPES = ["3class", "2class"]

# Short label names for internal computation
LABEL_MAP = {
    "TRUE_COGNATE": "TC",
    "FALSE_FRIEND": "FF",
    "LOANWORD": "LW",
}


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def compute_metrics(predictions: list[dict], task_type: str) -> dict:
    """Compute all evaluation metrics from a list of prediction records."""

    # Build confusion matrix and counts
    labels = sorted(set(p["true_label"] for p in predictions))
    confusion = defaultdict(lambda: defaultdict(int))
    label_total = defaultdict(int)

    for p in predictions:
        tl = p["true_label"]
        pl = p["predicted_label"]
        confusion[tl][pl] += 1
        label_total[tl] += 1

    N = len(predictions)

    # ---- Tier 1: Standard Classification Metrics ----

    correct = sum(confusion[l][l] for l in labels)
    acc = correct / N if N > 0 else 0

    per_label_f1 = {}
    for l in labels:
        tp = confusion[l][l]
        fp = sum(confusion[other][l] for other in labels if other != l)
        fn = sum(confusion[l][other] for other in labels if other != l)
        _, _, f1 = compute_f1(tp, fp, fn)
        per_label_f1[l] = f1

    macro_f1 = sum(per_label_f1.values()) / len(labels) if labels else 0

    per_label_acc = {l: confusion[l][l] / label_total[l] if label_total[l] > 0 else 0 for l in labels}

    metrics = {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_label_accuracy": per_label_acc,
        "per_label_f1": per_label_f1,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "correct": correct,
        "total": N,
    }

    # ---- Tier 2: Fine-Grained Error Rates (3-class only) ----
    if task_type == "3class" and "LOANWORD" in label_total:
        N_lw = label_total.get("LOANWORD", 0)
        N_tc = label_total.get("TRUE_COGNATE", 0)
        N_ff = label_total.get("FALSE_FRIEND", 0)

        # LCR: rate of LW misclassified as TC or FF
        lw_as_gen = confusion["LOANWORD"]["TRUE_COGNATE"] + confusion["LOANWORD"]["FALSE_FRIEND"]
        lcr = lw_as_gen / N_lw if N_lw > 0 else 0

        # CUR_TC: rate of TC misclassified as LW
        tc_as_lw = confusion["TRUE_COGNATE"]["LOANWORD"]
        cur_tc = tc_as_lw / N_tc if N_tc > 0 else 0

        # CUR_FF: rate of FF misclassified as LW
        ff_as_lw = confusion["FALSE_FRIEND"]["LOANWORD"]
        cur_ff = ff_as_lw / N_ff if N_ff > 0 else 0

        metrics["lcr"] = lcr
        metrics["cur_tc"] = cur_tc
        metrics["cur_ff"] = cur_ff

        # ---- Tier 3: Directional Bias Metrics (3-class only) ----

        # SDB: Semantic Drift Bias
        tc_as_ff = confusion["TRUE_COGNATE"]["FALSE_FRIEND"]
        ff_as_tc = confusion["FALSE_FRIEND"]["TRUE_COGNATE"]
        rate_tc_ff = tc_as_ff / N_tc if N_tc > 0 else 0
        rate_ff_tc = ff_as_tc / N_ff if N_ff > 0 else 0
        sdb_denom = rate_tc_ff + rate_ff_tc
        sdb = (rate_tc_ff - rate_ff_tc) / sdb_denom if sdb_denom > 0 else 0

        # CLB: Cognate-Loanword Directional Bias
        N_gen = N_tc + N_ff
        rate_lw_as_gen = lw_as_gen / N_lw if N_lw > 0 else 0
        gen_as_lw = tc_as_lw + ff_as_lw
        rate_gen_as_lw = gen_as_lw / N_gen if N_gen > 0 else 0
        clb_denom = rate_lw_as_gen + rate_gen_as_lw
        clb = (rate_lw_as_gen - rate_gen_as_lw) / clb_denom if clb_denom > 0 else 0

        metrics["sdb"] = sdb
        metrics["clb"] = clb

    elif task_type == "2class":
        # For 2-class: only SDB applies (no LW)
        N_tc = label_total.get("TRUE_COGNATE", 0)
        N_ff = label_total.get("FALSE_FRIEND", 0)
        tc_as_ff = confusion.get("TRUE_COGNATE", {}).get("FALSE_FRIEND", 0)
        ff_as_tc = confusion.get("FALSE_FRIEND", {}).get("TRUE_COGNATE", 0)
        rate_tc_ff = tc_as_ff / N_tc if N_tc > 0 else 0
        rate_ff_tc = ff_as_tc / N_ff if N_ff > 0 else 0
        sdb_denom = rate_tc_ff + rate_ff_tc
        sdb = (rate_tc_ff - rate_ff_tc) / sdb_denom if sdb_denom > 0 else 0
        metrics["sdb"] = sdb

    return metrics


def analyze_result_file(path: str, task_type: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        return None

    return compute_metrics(predictions, task_type)


def main():
    parser = argparse.ArgumentParser(description="Analyze cognate identification results")
    parser.add_argument("--task_type", default=None, choices=TASK_TYPES)
    parser.add_argument("--model", default=None)
    parser.add_argument("--format", default=None, choices=FORMATS)
    args = parser.parse_args()

    task_types = [args.task_type] if args.task_type else TASK_TYPES

    for tt in task_types:
        print(f"\n{'='*120}")
        print(f"  {tt.upper()} RESULTS")
        print(f"{'='*120}")

        # Collect all results
        all_results = {}
        tt_dir = RESULTS_DIR / tt
        if not tt_dir.exists():
            print(f"  No results directory: {tt_dir}")
            continue

        for fmt in FORMATS:
            fmt_dir = tt_dir / fmt
            if not fmt_dir.exists():
                continue
            for result_file in fmt_dir.glob("*.json"):
                model_name = result_file.stem
                if args.model and model_name != args.model:
                    continue
                if args.format and fmt != args.format:
                    continue
                metrics = analyze_result_file(str(result_file), tt)
                if metrics:
                    all_results[(model_name, fmt)] = metrics

        if not all_results:
            print("  No results found.")
            continue

        models = sorted(set(m for m, f in all_results.keys()))

        if tt == "3class":
            header = f"  {'Model':<30} {'Format':<10} {'Acc':>7} {'F1':>7} {'LCR':>7} {'CUR_TC':>7} {'CUR_FF':>7} {'SDB':>7} {'CLB':>7}"
            print(f"\n{header}")
            print(f"  {'-'*30} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
            for model in models:
                for fmt in FORMATS:
                    key = (model, fmt)
                    if key in all_results:
                        m = all_results[key]
                        print(f"  {model:<30} {fmt:<10} {m['acc']:>7.4f} {m['macro_f1']:>7.4f} "
                              f"{m.get('lcr', 0):>7.4f} {m.get('cur_tc', 0):>7.4f} {m.get('cur_ff', 0):>7.4f} "
                              f"{m.get('sdb', 0):>+7.4f} {m.get('clb', 0):>+7.4f}")
        else:
            header = f"  {'Model':<30} {'Format':<10} {'Acc':>7} {'F1':>7} {'SDB':>7}"
            print(f"\n{header}")
            print(f"  {'-'*30} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")
            for model in models:
                for fmt in FORMATS:
                    key = (model, fmt)
                    if key in all_results:
                        m = all_results[key]
                        print(f"  {model:<30} {fmt:<10} {m['acc']:>7.4f} {m['macro_f1']:>7.4f} "
                              f"{m.get('sdb', 0):>+7.4f}")

        # Per-format summary
        print(f"\n  Format comparison (average across models):")
        if tt == "3class":
            print(f"  {'Format':<10} {'Avg Acc':>8} {'Avg F1':>8} {'Avg LCR':>8} {'Avg CUR_TC':>10} {'Avg CUR_FF':>10} {'Avg SDB':>8} {'Avg CLB':>8}")
            for fmt in FORMATS:
                fmt_results = [all_results[(m, fmt)] for m in models if (m, fmt) in all_results]
                if fmt_results:
                    n = len(fmt_results)
                    avg_acc = sum(r["acc"] for r in fmt_results) / n
                    avg_f1 = sum(r["macro_f1"] for r in fmt_results) / n
                    avg_lcr = sum(r.get("lcr", 0) for r in fmt_results) / n
                    avg_cur_tc = sum(r.get("cur_tc", 0) for r in fmt_results) / n
                    avg_cur_ff = sum(r.get("cur_ff", 0) for r in fmt_results) / n
                    avg_sdb = sum(r.get("sdb", 0) for r in fmt_results) / n
                    avg_clb = sum(r.get("clb", 0) for r in fmt_results) / n
                    print(f"  {fmt:<10} {avg_acc:>8.4f} {avg_f1:>8.4f} {avg_lcr:>8.4f} "
                          f"{avg_cur_tc:>10.4f} {avg_cur_ff:>10.4f} {avg_sdb:>+8.4f} {avg_clb:>+8.4f}")
        else:
            print(f"  {'Format':<10} {'Avg Acc':>8} {'Avg F1':>8} {'Avg SDB':>8}")
            for fmt in FORMATS:
                fmt_results = [all_results[(m, fmt)] for m in models if (m, fmt) in all_results]
                if fmt_results:
                    n = len(fmt_results)
                    avg_acc = sum(r["acc"] for r in fmt_results) / n
                    avg_f1 = sum(r["macro_f1"] for r in fmt_results) / n
                    avg_sdb = sum(r.get("sdb", 0) for r in fmt_results) / n
                    print(f"  {fmt:<10} {avg_acc:>8.4f} {avg_f1:>8.4f} {avg_sdb:>+8.4f}")

        # Save analysis
        analysis_path = ANALYSIS_DIR / f"{tt}_analysis.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_data = {}
        for (model, fmt), m in all_results.items():
            analysis_data[f"{model}/{fmt}"] = m
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
