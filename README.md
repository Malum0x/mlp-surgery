# mpl-surgery

MLP layer restoration for targeted capability recovery in LLMs.

## Background

In the previous project ([[Perplexity-weighted-selective-finetuning](https://github.com/Malum0x/Perplexity-weighted-selective-finetuning)),
perplexity filtered data was used to improve finetuning efficiency. Despite data quality improvements, finetuning on conversational data degraded both math reasoning (GSM8K) and general reasoning (ARC Challenge).

This project investigates whether the degradation can be reversed by indentifying and restoring the specific MLP layers that were overwritten during finetuning.

## Hypothesis

finetuning on conversational data overwrites specific MLP layers responsible for reasoning capabilities. IOndetifying these layers via gradient norm scoring and restoring them to base model weights will recover performance without full retraining.

## Approach

1. **Identify** - run gradient norm scoring on GSM8K error samples to find which MLP layers are most damaged

2. **Specificity layer** - compute ratio of error gradients to correct answer gradients to separate genuinely broken layers from naturally important ones

3. **Restore** - reset damaged layers to base model weights (Qwen2.5-3B-Instruct checkpoint)

4. **Retrain** - apply LoRA only to restored layers, trained on perplexity-filtered data

5. **Evaluate** - measure GSM8K and ARC Challenge recovery

## Why MLP only?

Research (ROME, 2022) has shown that factual knowledge and reasoning
capabilities are stored primarily in MLP layers. Attention layers
handle structure and token relationships — these are left frozen.
Targeting only MLP layers reduces trainable parameters and minimises
the risk of introducing new damage to working capabilities.

## Results

ALl models evaluated with lm-eval. GSM8K flexible extract, 5 shot. 
Arc Challenge acc_norm 0 shot, no chat template.

| Model | GSM8K | ARC Challenge | GSM8K vs base |
|-------|-------|---------------|---------------|
| Base model | 63.31% | 48.55% | — |
| After fine-tuning | 61.41% | 44.62% | -1.90% |
| Restore top 5 layers | 61.18% | 45.90% | -2.13% |
| Restore top 15 layers | 62.85% | 45.73% | -0.46% |
| **Restore top 30 layers** | **64.75%** | **48.63%** | **+1.44%** |
| Restore specificity top 10 | 60.73% | 44.11% | -2.58% |
| Restore + retrain (surgical) | 62.62% | 44.71% | -0.69% |

## Findings

**Finetuning on conversational data degrades both GSM8K (-1.90%) and ARC Challenge (-3.93%).** This confirms catastrophic forgetting affects multiple capabilities simultaneously.

**Selective MLP restoration recovers both capabilities.** 
Restoring the top 30 most damaged layers (identifieed by gradient norm scoring) brings GSM8K above the base model (+1.44%) and returns ARC Challenge to baseline level (48.63% vs 48.55%)

**More layers restored = more recovery, consistently.**
The trend from top 5 -> top 15 -> top30 shows damage is distributed across middle mlp layers (roughly layers 9-27), not concentrated in specific weights.

**Specificity scoring did not improve targeting.**
Computing gradient ratios (error score / correct score) to isolate math-specific damage performed worse than raw gradient norms. The damage appears uniformly distributed rather than task specific.

**Surgical retraining did not improve on restoration alone.**
Restore + LoRA retrain (62.62%) performed belowe restore only (64.75%). This suggests retraining on the same conversational data type that caused the original damage re-introduces similar overwriting patterns.
The base model weights themselves contain the capability, finetuning simply overwrote them. 

**Key insight** The most effective intervention is the simplest - identify damaged layers, restore from checkpoint, no retraining needed.

## Gradient Scoring - Notable finding

Layer 2 down_proj showed an outlier raw gradient score of 84.1, approximately 7x higher than the next most damaged layer (12.9). Hovewer, specificity scoring revealed this layer is equally active on both error and correct samples - indicating it is a naturally important foundation layer, not specifically broken for math. This demonstrates why raw gradient norms alone are insufficient for precise layer targetting. 

## Pipeline
```
gradinet_scoring.py - score MLP layers on GSM8K error samples
restore_layers.py - restore top-k layers to base_weights
train.py - selective QLoRA retraining (optional)
eval_all.py - benchmark all checkpoints
```

## Related
- Previous project: [Perplexity-weighted-selective-finetuning](https://github.com/Malum0x/Perplexity-weighted-selective-finetuning)
- Dataset: [openhermes2.5-Perplexity_filtered_top30](https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30)
- ROME paper: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
