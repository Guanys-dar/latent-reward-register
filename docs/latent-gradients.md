# Latent gradients

Reward-guided sampling and RG-OPD both consume `d reward / d latent`. Getting
that gradient out of a frozen-trunk register takes one deliberate step, and
skipping it fails loudly rather than silently.

## Why a plain forward gives no gradient

While the register trains, the backbone is frozen and the register reads
detached snapshots of it, so every frozen block runs inside `torch.no_grad()`.
That is correct for training and saves activation memory — but it detaches
latents from the reward score. A plain `forward()` therefore yields a reward
that does not require grad with respect to its input latent, and asking for the
gradient raises:

```
RuntimeError: One of the differentiated Tensors appears to not have been used
in the graph.
```

## Enabling the gradient path

Wrap the scoring call:

```python
from latent_reward_register import latent_gradient_enabled

with latent_gradient_enabled():
    scores = register.score(latents, condition, timesteps)
gradient, = torch.autograd.grad(scores["preference"].sum(), latents)
```

`CheckpointRewardRegister.score_and_grad` does this for you, so guidance and
RG-OPD paths need no extra ceremony:

```python
output = register.score_and_grad(latents, condition, timesteps, heads=("preference",))
direction = output.gradients["preference"]
```

## What it does and does not change

- Trunk **weights stay frozen**. The mode restores the latent path only; it
  never calls `requires_grad_(True)` on a backbone parameter.
- Scores are unchanged: same math, same numbers, only autograd bookkeeping.
- Activation memory grows, because trunk activations must be kept for the
  backward. Expect a real increase at 1024x1024 with the full tap depth.
- The mode is thread-local and restored on exit, including on exception, so a
  scoring call cannot leak it into surrounding work.

## Relation to the research code

The research implementation recovered the same gradient by rebinding the
per-block helper to a grad-enabled clone of its body
(`lrm_reward_backend.py`, `_run_frozen_sd3_block_grad`). The released version
keeps that math but selects it through a documented context manager rather than
patching a method at runtime, so the behaviour is discoverable and applies
uniformly to SD3, FLUX, and Z-Image.
