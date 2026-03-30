"""
evaluate_everything.py

Tests ALL models from both projects using identical settings.
No dtype override. No chat template. Same as original correct measurements.

Models tested:
  FROM mlp-surgery:
    - base model
    - finetuned/merged (broken)
    - restored top 5
    - restored top 15
    - restored top 30
    - restored specificity
    - surgical model

  FROM selective-qlora:
    - baseline finetuned (full openhermes)
    - filtered finetuned (top 30% perplexity)

Tasks: GSM8K + ARC Challenge
"""

import subprocess
import json
import os
from datetime import datetime


# ── CONFIG ────────────────────────────────────────────────
DEVICE      = "cuda"
RESULTS_DIR = "results/everything/"
TASKS       = ["gsm8k", "arc_challenge"]

# ── ALL MODELS ────────────────────────────────────────────
# update paths to match your actual folder locations
MODELS = {

    # mlp-surgery models
    "base_model":            "Qwen/Qwen2.5-3B-Instruct",
    "finetuned_broken":      "merged_model/",
    "restore_top5":          "restored_model/",
    "restore_top15":         "restored_model_B/",
    "restore_top30":         "restored_model_C/",
    "restore_specificity":   "restored_model_D/",
    "surgical_model":        "surgical_model/",

    # selective-qlora models
    # update these paths to where you saved them
    "sq_baseline_finetuned": "/home/bart/Desktop/selective-qlora/merged_baseline/",
    "sq_filtered_finetuned": "/home/bart/Desktop/selective-qlora/merged_filtered/",
}


# ── RUN ONE EVALUATION ────────────────────────────────────
def run_eval(model_name, model_path, task):
    output_file = f"{RESULTS_DIR}{model_name}_{task}.json"

    # skip if already done
    if os.path.exists(output_file):
        print(f"  skipping {model_name} — {task} (already done)")
        return output_file

    # skip if model path doesn't exist
    # (only check local paths, not HuggingFace model IDs)
    if not model_path.startswith("Qwen/") and not os.path.exists(model_path):
        print(f"  skipping {model_name} — {task} (path not found: {model_path})")
        return None

    print(f"  running {model_name} — {task}...")

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task,
        "--device", DEVICE,
        "--output_path", output_file,
        "--batch_size", "1",
        # no dtype
        # no chat template
        # identical to original correct measurements
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"  ERROR: {model_name} — {task}")
        return None

    return output_file


# ── PARSE RESULT FILE ─────────────────────────────────────
def parse_score(result_file, task):
    if not result_file or not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    try:
        results = data.get("results", {})

        # find task results
        task_results = None
        if task in results:
            task_results = results[task]
        else:
            for key, val in results.items():
                if task in key:
                    task_results = val
                    break

        if task_results is None:
            print(f"  could not find {task} in keys: {list(results.keys())}")
            return None

        if task == "gsm8k":
            for key in [
                "exact_match,flexible-extract",
                "exact_match,strict-match",
                "exact_match",
            ]:
                if key in task_results:
                    return round(task_results[key] * 100, 2)

        elif task == "arc_challenge":
            for key in [
                "acc_norm,none",
                "acc_norm",
                "acc,none",
                "acc",
            ]:
                if key in task_results:
                    return round(task_results[key] * 100, 2)

    except Exception as e:
        print(f"  parse error: {e}")

    return None


# ── PRINT TABLE ───────────────────────────────────────────
def print_results_table(all_results):
    print("\n")
    print("=" * 76)
    print("COMPLETE RESULTS TABLE — ALL MODELS")
    print("=" * 76)
    print(f"{'Model':<30} {'GSM8K':>8} {'ARC':>8} {'GSM8K vs base':>14}")
    print("-" * 76)

    base_gsm8k = all_results.get("base_model", {}).get("gsm8k")

    # print mlp-surgery models first
    surgery_models = [
        "base_model",
        "finetuned_broken",
        "restore_top5",
        "restore_top15",
        "restore_top30",
        "restore_specificity",
        "surgical_model",
    ]

    print("  — mlp-surgery —")
    for model_name in surgery_models:
        if model_name not in all_results:
            continue
        print_row(model_name, all_results[model_name], base_gsm8k)

    # print selective-qlora models
    qlora_models = [
        "sq_baseline_finetuned",
        "sq_filtered_finetuned",
    ]

    print("\n  — selective-qlora —")
    for model_name in qlora_models:
        if model_name not in all_results:
            continue
        print_row(model_name, all_results[model_name], base_gsm8k)

    print("=" * 76)
    save_markdown(all_results, base_gsm8k)


def print_row(model_name, scores, base_gsm8k):
    gsm8k = scores.get("gsm8k")
    arc   = scores.get("arc_challenge")

    gsm8k_str = f"{gsm8k:.2f}%" if gsm8k is not None else "—"
    arc_str   = f"{arc:.2f}%"   if arc   is not None else "—"

    if base_gsm8k and gsm8k and model_name != "base_model":
        change = gsm8k - base_gsm8k
        change_str = f"{change:+.2f}%"
    else:
        change_str = "—"

    print(f"  {model_name:<28} {gsm8k_str:>8} {arc_str:>8} {change_str:>14}")


# ── SAVE MARKDOWN ─────────────────────────────────────────
def save_markdown(all_results, base_gsm8k):
    lines = []
    lines.append("# Complete Evaluation Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("")
    lines.append(
        "> All models evaluated with identical settings: "
        "GSM8K flexible-extract 5-shot, "
        "ARC Challenge acc_norm 0-shot. "
        "No chat template. No dtype override.\n"
    )
    lines.append("")

    # mlp-surgery section
    lines.append("## mlp-surgery\n")
    lines.append("| Model | GSM8K | ARC Challenge | GSM8K vs base |")
    lines.append("|-------|-------|---------------|---------------|")

    surgery_models = [
        "base_model",
        "finetuned_broken",
        "restore_top5",
        "restore_top15",
        "restore_top30",
        "restore_specificity",
        "surgical_model",
    ]

    for model_name in surgery_models:
        if model_name not in all_results:
            continue
        write_md_row(lines, model_name, all_results[model_name], base_gsm8k)

    lines.append("")

    # selective-qlora section
    lines.append("## selective-qlora\n")
    lines.append("| Model | GSM8K | ARC Challenge | GSM8K vs base |")
    lines.append("|-------|-------|---------------|---------------|")

    qlora_models = [
        "base_model",
        "sq_baseline_finetuned",
        "sq_filtered_finetuned",
    ]

    for model_name in qlora_models:
        if model_name not in all_results:
            continue
        write_md_row(lines, model_name, all_results[model_name], base_gsm8k)

    output = "\n".join(lines)
    md_path = "results/complete_results.md"
    with open(md_path, "w") as f:
        f.write(output)
    print(f"\nSaved to {md_path}")


def write_md_row(lines, model_name, scores, base_gsm8k):
    gsm8k = scores.get("gsm8k")
    arc   = scores.get("arc_challenge")

    gsm8k_str = f"{gsm8k:.2f}%" if gsm8k is not None else "—"
    arc_str   = f"{arc:.2f}%"   if arc   is not None else "—"

    if base_gsm8k and gsm8k and model_name != "base_model":
        change = gsm8k - base_gsm8k
        change_str = f"{change:+.2f}%"
    else:
        change_str = "—"

    lines.append(
        f"| {model_name} | {gsm8k_str} | {arc_str} | {change_str} |"
    )


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Evaluating ALL models.")
    print("Settings: no dtype, no chat template.")
    print(f"Results: {RESULTS_DIR}\n")

    # check which models exist
    print("Model paths:")
    for name, path in MODELS.items():
        if path.startswith("Qwen/"):
            status = "HuggingFace"
        elif os.path.exists(path):
            status = "found"
        else:
            status = "NOT FOUND — will skip"
        print(f"  {name:<30} {status}")
    print()

    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{model_name}")
        print(f"path: {model_path}")

        all_results[model_name] = {}

        for task in TASKS:
            result_file = run_eval(model_name, model_path, task)
            score = parse_score(result_file, task)
            all_results[model_name][task] = score

            score_str = f"{score:.2f}%" if score is not None else "—"
            print(f"  {task}: {score_str}")

    print_results_table(all_results)