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

Note: all evaluations use --apply_chat_template for accurate
      instruct model scoring. Results saved with _chat suffix
      to distinguish from previous runs without chat template.
"""

import subprocess
import json
import os
from datetime import datetime


# ── CONFIG ────────────────────────────────────────────────
DEVICE = "cuda"

MODELS = {
    "base_model":          "Qwen/Qwen2.5-3B-Instruct",
    "finetuned_broken":    "merged_model/",
    "restore_top5":        "restored_model/",
    "restore_top15":       "restored_model_B/",
    "restore_top30":       "restored_model_C/",
    "restore_specificity": "restored_model_D/",
    "surgical_model":      "surgical_model/",
}

TASKS = ["gsm8k", "arc_challenge"]

RESULTS_DIR = "results/chat/"


# ── RUN ONE EVALUATION ────────────────────────────────────
def run_eval(model_name, model_path, task):
    output_file = f"{RESULTS_DIR}{model_name}_{task}_chat.json"

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
        "--batch_size", "1",
        "--apply_chat_template",
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
        results = data["results"][task]

        if task == "gsm8k":
            # try flexible first, fall back to strict
            score = results.get("exact_match,flexible-extract", None)
            if score is None:
                score = results.get("exact_match,strict-match", None)

        elif task == "arc_challenge":
            score = results.get("acc_norm,none", None)
            if score is None:
                score = results.get("acc,none", None)

        if score is not None:
            return round(score * 100, 2)

    except Exception as e:
        print(f"  parse error for {result_file}: {e}")

    return None


# ── PRINT RESULTS TABLE ───────────────────────────────────
def print_results_table(all_results):
    print("\n")
    print("=" * 72)
    print("FINAL RESULTS TABLE — with chat template")
    print("=" * 72)
    print(f"{'Model':<26} {'GSM8K':>8} {'ARC':>8} {'GSM8K vs broken':>16}")
    print("-" * 72)

    baseline_gsm8k = None

    for model_name, scores in all_results.items():
        gsm8k = scores.get("gsm8k")
        arc   = scores.get("arc_challenge")

        gsm8k_str = f"{gsm8k:.2f}%" if gsm8k is not None else "—"
        arc_str   = f"{arc:.2f}%"   if arc   is not None else "—"

        if model_name == "finetuned_broken" and gsm8k:
            baseline_gsm8k = gsm8k

        if baseline_gsm8k and gsm8k and model_name != "finetuned_broken":
            change = gsm8k - baseline_gsm8k
            change_str = f"{change:+.2f}%"
        else:
            change_str = "—"

        print(f"{model_name:<26} {gsm8k_str:>8} {arc_str:>8} {change_str:>16}")

    print("=" * 72)
    save_markdown_table(all_results, baseline_gsm8k)


# ── SAVE MARKDOWN TABLE ───────────────────────────────────
def save_markdown_table(all_results, baseline_gsm8k):
    lines = []
    lines.append("# Final Evaluation Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("")
    lines.append(
        "> All evaluations use `--apply_chat_template` for accurate "
        "instruct model scoring.\n"
    )
    lines.append("")
    lines.append("| Model | GSM8K | ARC Challenge | GSM8K vs broken |")
    lines.append("|-------|-------|---------------|-----------------|")

    for model_name, scores in all_results.items():
        gsm8k = scores.get("gsm8k")
        arc   = scores.get("arc_challenge")

        gsm8k_str = f"{gsm8k:.2f}%" if gsm8k is not None else "—"
        arc_str   = f"{arc:.2f}%"   if arc   is not None else "—"

        if baseline_gsm8k and gsm8k and model_name != "finetuned_broken":
            change = gsm8k - baseline_gsm8k
            change_str = f"{change:+.2f}%"
        else:
            change_str = "—"

        lines.append(
            f"| {model_name} | {gsm8k_str} | "
            f"{arc_str} | {change_str} |"
        )

    output = "\n".join(lines)

    md_path = "results/final_results.md"
    with open(md_path, "w") as f:
        f.write(output)

    print(f"\nMarkdown table saved to {md_path}")


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for model_name, model_path in MODELS.items():

        # skip surgical model if not trained yet
        if model_name == "surgical_model":
            if not os.path.exists(model_path):
                print(f"\nSkipping surgical_model — not trained yet")
                continue

        print(f"\nEvaluating: {model_name}")
        print(f"Path:       {model_path}")

        all_results[model_name] = {}

        for task in TASKS:
            result_file = run_eval(model_name, model_path, task)
            score = parse_score(result_file, task)
            all_results[model_name][task] = score

            score_str = f"{score:.2f}%" if score is not None else "failed"
            print(f"  {task}: {score_str}")

    print_results_table(all_results)
