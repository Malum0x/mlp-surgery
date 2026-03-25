"""
4_evaluate.py

Goal: run GSM8K and ARC Challenge on all model checkpoints
      and produce a clean results table.

Models tested:
  - base model
  - finetuned (broken)
  - restored top 5
  - restored top 15
  - restored top 30
  - restored specificity top 10
  - surgical model (restored + retrained)
"""

import subprocess
import json
import os
from datetime import datetime


# ── CONFIG ────────────────────────────────────────────────
DEVICE = "cuda"

# all models to evaluate
# name → path
MODELS = {
    "base_model":           "Qwen/Qwen2.5-3B-Instruct",
    "finetuned_broken":     "merged_model/",
    "restore_top5":         "restored_model/",
    "restore_top15":        "restored_model_B/",
    "restore_top30":        "restored_model_C/",
    "restore_specificity":  "restored_model_D/",
    "surgical_model":       "surgical_model/",
}

# tasks to run
TASKS = ["gsm8k", "arc_challenge"]

RESULTS_DIR = "results/"


# ── RUN ONE EVALUATION ────────────────────────────────────
def run_eval(model_name, model_path, task):
    output_file = f"{RESULTS_DIR}{model_name}_{task}.json"

    # skip if already evaluated
    if os.path.exists(output_file):
        print(f"  skipping {model_name} {task} — already done")
        return output_file

    print(f"  evaluating {model_name} on {task}...")

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", task,
        "--device", DEVICE,
        "--output_path", output_file,
        "--batch_size", "1"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"  ERROR on {model_name} {task}")
        return None

    return output_file


# ── PARSE RESULT FILE ─────────────────────────────────────
def parse_score(result_file, task):
    if not result_file or not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    try:
        # navigate the lm-eval output structure
        results = data["results"][task]

        if task == "gsm8k":
            score = results.get("exact_match,flexible-extract", None)
        elif task == "arc_challenge":
            score = results.get("acc_norm,none", None)

        if score is not None:
            return round(score * 100, 2)
    except Exception as e:
        print(f"  parse error for {result_file}: {e}")

    return None


# ── BUILD RESULTS TABLE ───────────────────────────────────
def print_results_table(all_results):
    print("\n")
    print("=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Model':<26} {'GSM8K':>8} {'ARC':>8} {'GSM8K Change':>14}")
    print("-" * 70)

    baseline_gsm8k = None

    for model_name, scores in all_results.items():
        gsm8k = scores.get("gsm8k")
        arc   = scores.get("arc_challenge")

        gsm8k_str  = f"{gsm8k:.2f}%" if gsm8k else "—"
        arc_str    = f"{arc:.2f}%"   if arc   else "—"

        # change vs finetuned broken model
        if model_name == "finetuned_broken" and gsm8k:
            baseline_gsm8k = gsm8k

        if baseline_gsm8k and gsm8k and model_name != "finetuned_broken":
            change = gsm8k - baseline_gsm8k
            change_str = f"{change:+.2f}%"
        else:
            change_str = "—"

        print(f"{model_name:<26} {gsm8k_str:>8} {arc_str:>8} {change_str:>14}")

    print("=" * 70)

    # save to markdown
    save_markdown_table(all_results, baseline_gsm8k)


# ── SAVE MARKDOWN TABLE ───────────────────────────────────
def save_markdown_table(all_results, baseline_gsm8k):
    lines = []
    lines.append("# Final Evaluation Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("")
    lines.append("| Model | GSM8K | ARC Challenge | GSM8K Change |")
    lines.append("|-------|-------|---------------|--------------|")

    for model_name, scores in all_results.items():
        gsm8k = scores.get("gsm8k")
        arc   = scores.get("arc_challenge")

        gsm8k_str = f"{gsm8k:.2f}%" if gsm8k else "—"
        arc_str   = f"{arc:.2f}%"   if arc   else "—"

        if baseline_gsm8k and gsm8k and model_name != "finetuned_broken":
            change = gsm8k - baseline_gsm8k
            change_str = f"{change:+.2f}%"
        else:
            change_str = "—"

        lines.append(
            f"| {model_name} | {gsm8k_str} | {arc_str} | {change_str} |"
        )

    output = "\n".join(lines)

    with open("results/final_results.md", "w") as f:
        f.write(output)

    print(f"\nMarkdown table saved to results/final_results.md")


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for model_name, model_path in MODELS.items():

        # skip surgical model if not trained yet
        if model_name == "surgical_model":
            if not os.path.exists(model_path):
                print(f"\nSkipping {model_name} — not trained yet")
                continue

        print(f"\nEvaluating: {model_name}")
        print(f"Path: {model_path}")

        all_results[model_name] = {}

        for task in TASKS:
            result_file = run_eval(model_name, model_path, task)
            score = parse_score(result_file, task)
            all_results[model_name][task] = score
            print(f"  {task}: {score}%")

    print_results_table(all_results)