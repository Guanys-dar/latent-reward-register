# Latent Reward Registers

Research code for **Latent Reward Registers**, **Reward-Guided Sampling (RGS)**,
and **Reward-Gradient On-Policy Distillation (RG-OPD)**.

A reward register is a small set of trainable tokens that ride a frozen
diffusion transformer, reading its hidden states through a side stream and
predicting reward. Because the register is differentiable with respect to the
latent, `d reward / d latent` becomes available during sampling — that gradient
is what RGS steers with and what RG-OPD distills.

| Capability | SD3-Medium | FLUX.1-dev | Z-Image |
| --- | --- | --- | --- |
| Register training | yes (3 heads) | yes (2 heads) | yes (2 heads) |
| Preference scoring | yes | yes | yes |
| Reward-guided sampling | yes | yes | not released |
| RG-OPD | yes | yes | not released |

Z-Image is released for register training and preference scoring only: no
downstream evaluation or distillation consumes the Z-Image register.

## Install

```bash
pip install -e '.[dev]'          # algorithms, configs, tests (CPU)
pip install -e '.[dev,models]'   # add model-backed registers
```

The `models` extra pins an exact diffusers revision. That pin is load-bearing,
not cosmetic: the register implementations read the internal attribute layout of
the SD3/FLUX/Z-Image transformer blocks, and upstream is actively refactoring
attention. A version bump can change numerics silently.

## Check the install

```bash
lrr smoke-release                    # every algorithm path, no weights needed
lrr validate-release --root .        # all seven workflow presets
pytest -q
```

`smoke-release` runs on a synthetic backbone. It proves the equations are wired
correctly; it does not prove a real model runs. For that:

```bash
lrr build-register --config configs/register/sd3/paper.yaml \
    --model-path /path/to/stable-diffusion-3-medium-diffusers \
    --precision fp32 --local-files-only
```

SD3 should report **147,599,619 trainable parameters**, matching the exp11
training log. A different number means the architecture has drifted from the
paper baseline.

## Reward gradients

A frozen-trunk register does not expose `d reward / d latent` by default: the
training forward runs each frozen block under `torch.no_grad()`, which detaches
latents from the score. Enable the gradient path explicitly:

```python
from latent_reward_register import latent_gradient_enabled

with latent_gradient_enabled():
    scores = register.score(latents, condition, timesteps)
```

`score_and_grad` does this internally. Trunk weights stay frozen either way.
See `docs/latent-gradients.md`.

## One teacher, two consumers

RGS and RG-OPD share `RewardGradientTeacher`, which produces
`base_next + reward_delta` where `reward_delta` follows the reward gradient,
unit-RMS normalized per head, magnitude-matched to `scale * RMS(base step)`, and
clipped. `scale` comes from the sigma-banded schedule.

Sharing is deliberate: a distilled student is trained against exactly the
guidance the sampler applies, so the two cannot drift.

```python
step = teacher.guided_step(
    latents=latents, base_next=base_next,
    condition=condition, timesteps=timesteps, sigma=sigma,
)
```

Steps where the schedule resolves to zero skip the register backward entirely,
which is where the reported efficiency comes from. `reward_guided_sample(...,
return_trace=True)` reports the fraction of steps actually guided.

## Configuration

`configs/{register,rgs,rgopd}/<backbone>/paper.yaml` hold the reported settings.
Two contracts matter when editing them:

- `head_names` and `score_keys` are **positional**. The loss aligns
  `targets[:, h]` with `head_names[h]`, so reordering one without the other
  mistrains silently rather than erroring.
- `vis_h`/`vis_w` are **token counts, not pixels**. The pooling layer asserts
  `vis_h * vis_w` tokens, so latent spatial size must match.

Layer taps follow one rule across backbones rather than per-model constants:
1/6, 1/3, and 1/2 of depth, with the register stopping at half depth.

## Data

Register training consumes a group manifest: one record per prompt holding a
ranked list of scored images. Groups are the unit of the loss — all within-group
pairs contribute — so a flat list of preference pairs is not a substitute. See
`docs/data-format.md`.

Training reads cached latents and prompt embeddings, never images: no VAE or
text encoder runs during training.

## Status

Runnable: register construction for all three backbones, group-level training
loss, preference scoring, reward gradients, the guidance schedule and teacher,
the RGS loop, and RG-OPD targets and loss.

Not yet included: the RG-OPD rollout driver, the benchmark generation and
scoring pipeline, published checkpoints, and the paper prompt sets. Model
weights and datasets are not distributed here; see NOTICE.

## Documentation

| Topic | File |
| --- | --- |
| Repository layout | `docs/release-layout.md` |
| Register training | `docs/register-training.md` |
| Reward gradients | `docs/latent-gradients.md` |
| Reward-guided sampling | `docs/reward-guided-sampling.md` |
| RG-OPD | `docs/reward-guided-opd.md` |
| Preference scoring | `docs/preference-scoring.md` |
| Data format | `docs/data-format.md` |
| Checkpoint format | `docs/checkpoint-format.md` |
| Provenance and resolved discrepancies | `docs/source-provenance.md` |
| Reproduction | `docs/reproduction.md` |

`docs/source-provenance.md` is worth reading before comparing checkpoints: the
configuration embedded in a checkpoint is the source of truth, and several
historical experiment names describe incompatible architectures.

## License

Apache-2.0. See LICENSE and NOTICE.
