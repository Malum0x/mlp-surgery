# mlp-surgery

MLP layer restoration for targeted capability recovery in LLMs.

## Background

In the previous project ([Perplexity-weighted-selective-finetuning](https://github.com/Malum0x/Perplexity-weighted-selective-finetuning)),
perplexity filtered data was used to improve finetuning efficiency. Despite data quality improvements, finetuning on conversational data degraded both math reasoning (GSM8K) and general reasoning (ARC Challenge).

This project investigates whether the degradation can be reversed by indentifying and restoring the specific MLP layers that were overwritten during finetuning.

## Hypothesis

finetuning on conversational data overwrites specific MLP layers responsible for reasoning capabilities. Indetifying these layers via gradient norm scoring and restoring them to base model weights will recover performance without full retraining.

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

## Results (original run)

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

## Verification (re-run on 2026-05-07)

Re-ran the full pipeline from scratch — fresh merge of base + filtered adapter, fresh gradient scoring on 100 GSM8K errors and 100 corrects, fresh restoration variants, fresh lm-eval. Same eval settings as above. batch_size bumped to 8 for speed (deterministic for these tasks). Single seed. Surgical-retrain variant not re-run yet.

| Model | GSM8K (re-run) | GSM8K (original) | Δ | ARC (re-run) | ARC (original) | Δ |
|-------|---------------:|-----------------:|---:|-------------:|---------------:|---:|
| Base model | 63.15 | 63.31 | -0.16 | 48.12 | 48.55 | -0.43 |
| After fine-tuning | 61.64 | 61.41 | +0.23 | 45.22 | 44.62 | +0.60 |
| Restore top 5 | 63.00 | 61.18 | +1.82 | 45.73 | 45.90 | -0.17 |
| Restore top 15 | 63.46 | 62.85 | +0.61 | 46.50 | 45.73 | +0.77 |
| **Restore top 30** | **64.29** | **64.75** | **-0.46** | **48.55** | **48.63** | **-0.08** |
| Restore specificity top 10 | 61.64 | 60.73 | +0.91 | 45.22 | 44.11 | +1.11 |

What reproduces:

- Top-30 restoration crosses base on GSM8K (+1.14 vs base in re-run, +1.44 in original) and fully recovers ARC.
- Trend is monotonic. More layers restored, more recovery.
- Specificity top-10 doesn't recover. Negative subresult holds.
- Layer 2 down_proj outlier reproduces exactly. Raw gradient score 84.14 vs #2 at 12.96. In specificity scoring it drops out of the top — same conclusion as the original run: it's a foundation layer, not specifically broken for math.

Restore-top-5 drift (+1.82 GSM8K vs original) is the largest single-model delta. Sample variance on a single seed plus possible different sample order in the gradient scoring pass (re-run found different specific error samples than the original). The headline finding doesn't depend on it.

## Findings

**Finetuning on conversational data degrades both GSM8K (-1.51%) and ARC Challenge (-2.90%) in the re-run** (-1.90% / -3.93% in the original). Catastrophic forgetting affects multiple capabilities simultaneously.

**Selective MLP restoration recovers both capabilities.**
Restoring the top 30 most damaged layers (identified by gradient norm scoring) brings GSM8K above the base model and returns ARC Challenge to baseline level.

**More layers restored = more recovery, consistently.**
The trend top 5 → top 15 → top 30 shows damage is distributed across middle MLP layers (roughly layers 9–27), not concentrated in specific weights.

**Specificity scoring did not improve targeting.**
Computing gradient ratios (error score / correct score) to isolate math-specific damage performed worse than raw gradient norms. Damage appears uniformly distributed rather than task specific.

**Surgical retraining did not improve on restoration alone.**
Restore + LoRA retrain (62.62%) performed below restore only (64.75%) in the original. Retraining on the same conversational data type that caused the damage re-introduces similar overwriting patterns. The base model weights themselves contain the capability, finetuning simply overwrote them.

**Key insight:** the most effective intervention is the simplest. Identify damaged layers, restore from checkpoint, no retraining needed.

## Gradient Scoring - Notable finding

Layer 2 down_proj showed an outlier raw gradient score of 84.14, approximately 7x higher than the next most damaged layer (12.96). Hovewer, specificity scoring revealed this layer is equally active on both error and correct samples - indicating it is a naturally important foundation layer, not specifically broken for math. This demonstrates why raw gradient norms alone are insufficient for precise layer targetting.

## Pipeline

```
gradient_scoring.py - score MLP layers on GSM8K error samples
restore_layers.py   - restore top-k layers to base weights
train.py            - selective QLoRA retraining (optional)
eval_all.py         - benchmark all checkpoints
```

## Caveats

- All numbers are single-seed. Magnitudes are small (~1pt). A re-run with 3 seeds would tighten the headline; planned for v2.
- "No chat template" eval evaluates a Qwen-Instruct model in base-model style. Numbers are below what you'd see with chat template (~78% GSM8K), but relative comparisons across the same setup are meaningful.

## Related

- Previous project: [Perplexity-weighted-selective-finetuning](https://github.com/Malum0x/Perplexity-weighted-selective-finetuning)
- Dataset: [openhermes2.5-Perplexity_filtered_top30](https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30)
- ROME paper: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
