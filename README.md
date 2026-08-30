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
conda create -n lrr python=3.11 && conda activate lrr
pip install -e '.[dev,models]'
```

Use `pip install -e '.[dev]'` for the algorithms, configs, and CPU tests alone.
The `models` extra pins an exact diffusers revision, and that pin is
load-bearing, not cosmetic: the register implementations read the internal
attribute layout of the SD3/FLUX/Z-Image transformer blocks, and upstream is
actively refactoring attention. A version bump can change numerics silently.

Check the install without downloading any weights:

```bash
bash scripts/smoke_all.sh
```

## Machine paths

Copy the example and fill in where your data and models live:

```bash
cp configs/local.yaml.example configs/local.yaml
```

`configs/local.yaml` is never committed. Keeping machine paths there is what lets
every other file in this repository stay identical across copies of it.

## Run it

Each task has a launch script, and every flag it passes can be given to `lrr`
directly. Add `--dry-run` to any command to print the resolved plan without
loading weights.

**Reward-guided sampling** — the register steers a frozen sampler:

```bash
CHECKPOINT=/path/to/sd3-register.pt PROMPTS=prompts.txt bash scripts/sample_sd3.sh
```

Writes PNGs to `outputs/rgs_sd3` and reports the guidance fraction — how much of
the trajectory was actually steered. FLUX: `scripts/sample_flux.sh`.

**Register training** — needs a prepared group manifest (`docs/data-format.md`):

```bash
MANIFEST=/path/to/groups.jsonl bash scripts/train_register_sd3.sh
```

**RG-OPD** — distill the guided sampler into a LoRA student:

```bash
CHECKPOINT=/path/to/teacher-register.pt PROMPTS=prompts.txt \
    bash scripts/train_rgopd_sd3.sh
```

Add `--max-batches 4` or `--rounds 1` for a reduced-scale run that exercises the
real path in minutes rather than hours.

To confirm a real model builds before committing to a long run:

```bash
lrr build-register --config configs/register/sd3/paper.yaml \
    --model-path /path/to/stable-diffusion-3-medium-diffusers --precision fp32
```

SD3 should report **147,599,619 trainable parameters**, matching the exp11
training log. A different number means the architecture has drifted from the
paper baseline.

## What you need from elsewhere

Model weights, register checkpoints, prepared training manifests, and the paper
prompt sets are not distributed here; see NOTICE. The Table 1 pair file and the
keep-800 evaluation key set are published separately, with a fetch script in
`scripts/fetch_table1_images.py`.

The benchmark generation and scoring pipeline (HPSv3, ImageReward, MUSIQ,
CLIP-IQA) is not included: those scorers are third-party and separately licensed.

## How it works

### One teacher, two consumers

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

### Reward gradients

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

### Two register classes

| | Class | Use |
| --- | --- | --- |
| Real weights | `CheckpointRewardRegister` | `load_legacy_register`, `build_register_from_config` |
| No weights | `ReferenceRewardRegister` | the `smoke-release` scaffold only |

`ReferenceRewardRegister` is not checkpoint-compatible with the paper's weights.
It exists so the training loop, preference scoring, RGS, and RG-OPD can run on a
synthetic backbone with nothing downloaded. Reach for `CheckpointRewardRegister`
for anything real.

RG-OPD likewise has two trainers: `train_rgopd_rollout` is the paper path
(on-policy, the student walks its own trajectory), and `train_rgopd` is an
off-policy single-step trainer kept for ablations.

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

## Documentation

| Document | Contents |
| --- | --- |
| `docs/register-training.md` | The training seam and its configs |
| `docs/data-format.md` | Group manifest schema |
| `docs/latent-gradients.md` | Why the gradient path needs enabling |
| `docs/reward-guided-sampling.md` | The RGS loop and guidance schedule |
| `docs/reward-guided-opd.md` | Rollout driver, targets, presets |
| `docs/preference-scoring.md` | Table 1 evaluation and position bias |
| `docs/checkpoint-format.md` | Portable checkpoint contract |
| `docs/source-provenance.md` | Which experiment each released number came from |
| `docs/release-layout.md` | Directory map |
| `docs/reproduction.md` | What is reproducible from this repository alone |

## Contributing

`pytest -q` runs the full suite on CPU with no weights. Tests cover the algorithm
layer, every config preset, the vendored-implementation checksums, and the
release hygiene guards.
