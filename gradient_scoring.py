# run gradient norm scoring on GSM8K error samples to find which MLP layers are most damaged

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


# config

FINETUNED_MODEL_PATHJ = " PATH "
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
def correct_error_samples(model, tokenizer, max_samples=100)
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