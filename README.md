# MLP-surgery / surgical-finetune

MLP layer restoration for targeted capability recovery.

## Background

In the previous project (link) perplexity filtered data was used to improve finetuning efficiency.
While ARC challenge performance made improvement above the base model, GSM8K (math reasoning) remained degraded after finetuning, dropping from 68% to 61%.

This suggests that finetuning on conversational data overwrites specific layers responsible for mathematical reasoning, regardless of data quality. 

## Hypothesis

Math reasoning degradation is concentrated in specific MLP layers. 
Identifying these layers via gradient norm scoring, restoring them to their base model state, and retraining only those layers on difficulty filtered data will recover GSM8K performance more effectively than standard LoRA finetuning.

## Approach

1. **Identify** - run gradient norm scoring on GSM8K error samples to find which MLP layers are most damaged

2. **Restore** - reset those specific layers to base model weights (Qwen2.5-3B-Instruct checkpoint)

3. **Retrain** - apply LoRA only to restored layers, trained on perplexity-filtered data

4. **Evaluate** - measure GSM8K and ARC Challenge recovery

## Why MLP only? 

Research (ROME, 2022) has shown that factual knowledge and reasoning capabilities are stored primarily in MLP layers.
Attention layers handle structure and token relationships, these are left frozen. Targeting only MLP layers reduces trainable parameters and minimises the risk of introducing new damage to work capabilities.

## Status: 
Gradient norm scoring revealed an unexpected outlier: 

        - layer 2 down_proj scored 84.1, which is approximately 7x higher than the next most damaged layer (12.9),

        - layers 9-27 show consistent moderate damage, consistent with the hypothesis that middle layers store mathematical reasoning capabilities,

        - restoring top 30 layers recovered 53% of the performance drop without any retraining,

        - currently running specificity scoring to separate genuinely broken layers from naturally important ones


## Related 
https://github.com/Malum0x/Perplexity-weighted-selective-finetuning

https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30

