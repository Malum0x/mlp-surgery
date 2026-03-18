# run gradient norm scoring on GSM8K error samples to find which MLP layers are most damaged

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


# config

FINETUNED_MODEL_PATH = " PATH "
OUTPUT_PATH = "results/layer_scores.json"
NUM_ERROR_SAMPLES = 100
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

# collect error samples 
def collect_error_samples(model, tokenizer, max_samples=100)
    dataset = load_dataset("gsm8k", "main", split="test")
    error_samples = []

    for item in tqdm(dataset, desc="finding errors"):
        if len(error_samples) >= max_samples:
            break
            
        question = item["question"]
        correct = item["answer"].split("####")[-1].strip()

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        #check if model got it wrong
        if correct not in response:
            error_samples.append({
                "prompt": prompt,
                "correct": correct,
                "response": response
            })

        print(f"Found {len(error_samples)} error samples")
        return error_samples
    
def score_layers(model, tokenizer, error_samples):
    layer_scores = {}
    model.train()
    for sample in tqdm(error_samples, desc="scoring"):
        inputs = tokenizer(
            sample["prompt"],
            return_tensors="pt"

        ).to(DEVICE)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None: 
                continue
            if "mlp" not in name: 
                continue
            norm = param.grad.norm().item()

            if name not in layer_scores:
                layer_scores[name] = 0.0

            layer_scores[name] += norm
        
        model.zero_grad()

    for name in layer_scores:
        layer_scores[name] /= len(error_samples)

    return layer_scores

def save_results(layer_scores, output_path):
    ranked = sorted(
         layer_scores.items(),
         key=lambda x: -x[1]
    )

    output = {
        "ranked_layers": [
            {
            "rank": i + 1,
            "name": name,
            "score": round(score, 6)
            }
            for i, (name, score) in enumerate(ranked)
        ],
        "total_layers_scored": len(ranked)        
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n Top 10 most damaged MLP layers: ")
    for item in output["ranked_layers"][:10]:
        print(f"  #{item['rank']:02d} {item['name']:<45} {item['score']:.4f}")

    print(f"\nFull Results saved to {output_path}")

if __name__ == "__main__":
    model, tokenizer = load_model(FINETUNED_MODEL_PATH)
    error_samples = collect_error_samples(
        model,
        tokenizer,
        max_samples=NUM_ERROR_SAMPLES
    )

    layer_scores = score_layers(model, tokenizer, error_samples)
    save_results(layer_scores, OUTPUT_PATH) 