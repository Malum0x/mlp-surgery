from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "/home/bart/Desktop/selective-qlora/Perplexit-weighted-selective-finetuning/filtered_results/final_adapter"
MERGED_OUTPUT_PATH = "merged_model/"

print("loading base model")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype = torch.bfloat16,
    device_map="auto"
)

print("loading adapter")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging weights")
model = model.merge_and_unload()

print("Saving merged model")
model.save_pretrained(MERGED_OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("model is saved to: ", MERGED_OUTPUT_PATH)


