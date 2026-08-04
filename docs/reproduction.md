# Paper reproduction

## Reward-register training

Use the backbone preset under `configs/register/<backbone>/paper.yaml`. The
published data release must be converted to the JSONL schema before launch.
Training uses pairwise Thurstone objectives, frozen generator weights, bf16,
AdamW, EMA 0.999, and the checkpoint-derived layer taps.

## Reward-Guided Sampling

SD3 uses 42 FlowMatch Euler steps and CFG 4.5. FLUX uses 40 FlowMatch Euler
steps and embedded guidance 3.5. Both use per-head unit-RMS gradients followed
by equal-weight summation and step-magnitude matching. Guidance is disabled in
the low-noise tail.

## RG-OPD

Both backbones use ten-step on-policy rollouts and optimize the first nine
steps. A frozen reference step receives the same reward-gradient correction as
RGS; that detached next-state becomes the local regression target. Only the
student LoRA is optimized.

## Evaluation

Pairwise preference accuracy must preserve prompt groups and use the paper's
fixed benchmark split. Generation comparisons must use matched prompts, seeds,
resolution, sampler, and metric implementation. Do not select checkpoints on
the final test prompts.

