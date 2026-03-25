# apply LoRa only to the restored MLP layers and retrain on perplexity 
# filtered data to recover math reasoning 

import json
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm 
from trl import SFTTrainer, SFTConfig


# config
RESTORED_MODEL_PATH = "restored_model_C/"
DATASET_PATH = "Malum0x/openhermes2.5-Perplexity_filtered_top30"
LAYER_SCORES_PATH = "results/layer_scores.json"
OUTPUT_PATH = "surgical_model/"
TOP_K_LAYERS = 30

# training hyperparameters
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUMULATION = 8
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
DEVICE = "cuda"  if torch.cuda.is_available() else "cpu"

# read which layers to apply lora to
# only the restored layers get adapters
# everything else stays frozen
def get_target_modules(scores_path, top_k):
    
    with open(scores_path, "r") as f:
        data = json.load(f)

    target_modules = []
    seen = set()

    for item in data["ranked_layers"][:top_k]:
        # "model.layers.16.mlp.down_proj.weight" -> "down_proj"
        module_name = item["name"].split(".")[-2]
        if module_name not in seen:
            seen.add(module_name)
            target_modules.append(module_name)

    print(f"Target modules {target_modules}")
    return target_modules

# load model
def load_model(path):
    print("Loading restored model")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    return model, tokenizer

# attach LoRa adapters only to target modules
# in specific layers - not all layers
def apply_lora(model, target_modules, top_k):
    print("Applying LoRA: ")
    with open(LAYER_SCORES_PATH, "r") as f:
        data = json.load(f)
    
    layer_numbers = []
    for item in data["ranked_layers"][:top_k]:
        # "model.layers.16.mlp.down_proj.weight" -> 16
        layer_num = int(item["name"].split(".")[2])
        if layer_num not in layer_numbers:
            layer_numbers.append(layer_num)

    print(f" Applying LoRA to layers: {sorted(layer_numbers)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        layers_to_transform=layer_numbers,# key line, only these layers get adapters
    
    
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_training_data(dataset_path, tokenizer):
    print("Loading dataset...")
    dataset = load_dataset(dataset_path, split="train")

    def format_sample(sample):
        text = ""
        for turn in sample["conversations"]:
            if turn["from"] == "human":
                text += f"### Human: {turn['value']}\n"
            elif turn["from"] == "gpt":
                text += f"### Assistant: {turn['value']}\n"
        return {"text": text}

    # convert to plain text first
    # remove ALL old columns including conversations
    dataset = dataset.map(
        format_sample,
        remove_columns=dataset.column_names
    )

    # tokenize manually so trl doesn't touch conversations at all
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    dataset = dataset.map(
        tokenize,
        remove_columns=["text"]
    )

    return dataset

def train(model, tokenizer, dataset):
    print("starting training")
    training_args = SFTConfig(
        output_dir=OUTPUT_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name="surgical-finetune",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    print("Training complete")
    return trainer


def save_model(trainer, tokenizer): 
    print(f" Saving surgical model to {OUTPUT_PATH}")

    #merge LoRA weights into base model
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("Done")

if __name__ == "__main__":
    target_modules = get_target_modules(
        LAYER_SCORES_PATH,
        TOP_K_LAYERS
    )

    model, tokenizer = load_model(RESTORED_MODEL_PATH)

    # apply lora only to restored layers 
    model = apply_lora(model, target_modules, TOP_K_LAYERS)
    
    #load perplexity filtered dataset
    dataset = load_training_data(DATASET_PATH, tokenizer)

    #train 
    trainer = train(model, tokenizer, dataset)

    #save merged model
    save_model(trainer, tokenizer)
