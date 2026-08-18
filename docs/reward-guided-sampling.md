# Reward-guided sampling

RGS is exposed as a model-independent loop in
`latent_reward_register.sampling.reward_guided_sample`. The loop delegates two
backbone-specific operations to the adapter:

1. `reference_step` computes the frozen generator transition.
2. `extract_features` supplies register features for score gradients.

Paper presets:

- `configs/rgs/sd3/paper.yaml`: 42 steps, CFG 4.5, bands 0.30/0.05.
- `configs/rgs/flux/paper.yaml`: 40 steps, embedded guidance 3.5, bands 0.30/0.10.

The Z-Image register adapter is supported for training/scoring. Z-Image RGS is
not advertised as a paper-supported release capability until its time/noise
convention and final benchmark protocol are confirmed.

Checkpoint and prompt paths are runtime inputs, not repository contents.
