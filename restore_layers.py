#Restore the most damaged MLP layers back to base model weights.

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL_PATH = "merged_model/"
LAYER_SCORES_PATH = "results/specificity_scores.json"
RESTORED_OUTPUT_PATH = "restored_model_D/"
TOP_K_LAYERS = 10

def load_top_layers(scores_path, top_k):
    print(f"loading layer scores from {scores_path}")

    with open(scores_path, "r") as f:
        data = json.load(f)

        top_layers = [
            item["name"]
            for item in data["ranked_layers"][:top_k]
        ]
        print(f"\nTop {top_k} most damaged layers selected:")
        for i, name in enumerate(top_layers):
            print(f" #{i+1:02d} {name}")

        return top_layers
    
def load_models(base_path, finetuned_path):
    print("\n Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("loading finetuned model")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path)

    return base_model, finetuned_model, tokenizer


def restore_layers(base_model, finetuned_model, top_layers):
    """
    for each damaged layer copy the weights from base model into finetuned model
    """
    print("\Restoring damaged layers")

    restored_count = 0

    base_params = dict(base_model.named_parameters())
    finetuned_params = dict(finetuned_model.named_parameters())

    for layer_name in top_layers:
        if layer_name in base_params:
            with torch.no_grad(): 
                finetuned_params[layer_name].copy_(
                    base_params[layer_name]
                )
            print(f" restored {layer_name}")
            restored_count += 1
        else: 
            print(f" not found {layer_name}")

    print(f"\nRestored {restored_count}/{len(top_layers)} layers succesfully")
    return finetuned_model

def save_restored_model(model, tokenizer, output_path):
    print(f"\n Saving restored model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("done")

if __name__ == "__main__":
    top_layers = load_top_layers(LAYER_SCORES_PATH, TOP_K_LAYERS)
    base_model, finetuned_model, tokenizer = load_models(
        BASE_MODEL_PATH,
        FINETUNED_MODEL_PATH
    )

    restored_model = restore_layers(
        base_model,
        finetuned_model,
        top_layers
    )

    save_restored_model(
        restored_model, 
        tokenizer, 
        RESTORED_OUTPUT_PATH
    )

    