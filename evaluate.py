# measure GSM8K and ARC Challenge recovery
# pip install lm-eval
# just change model path to test each of the stage 

import torch
import subprocess
import sys

MODEL_PATH = "restored_model/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation(model_path):
    print(f"\nEvaluating: {model_path}")
    
    print("running GSM8K")
    subprocess.run([
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", "gsm8k",
        "--device", DEVICE,
        "--output_path", "results/gsm8k_results.json"

    ])

    print("running ARC Challenge")
    subprocess.run([
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", "arc_challenge",
        "--device", DEVICE,
        "--output_path", "results/arc_results.json"
    ])

if __name__ == "__main__":
    run_evaluation(MODEL_PATH)


