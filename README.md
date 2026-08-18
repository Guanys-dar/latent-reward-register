# Latent Reward Registers

Research-preview code for **Latent Reward Registers**, **Reward-Gradient On-Policy
Distillation (RG-OPD)**, and **Reward-Guided Sampling (RGS)**.

The repository exposes one reward-register interface across SD3-Medium,
FLUX.1-dev, and Z-Image. RG-OPD and RGS are released for SD3 and FLUX. The
pretrained generator remains frozen while register training and RGS run;
RG-OPD updates a LoRA student.

## Installation

```bash
pip install -e '.[dev]'
lrr list-backbones
lrr validate-release --root .
lrr smoke-release
```

Install `.[models]` only when using the legacy diffusers-backed model ports.

## Release status

The portable checkpoint, register-training, pairwise-evaluation, RGS, and
RG-OPD algorithm seams are runnable and covered by CPU integration tests.
`lrr smoke-release` executes all four paths without external assets.

This revision is **not yet a model-backed reproduction release**. The SD3,
FLUX, and Z-Image workflow commands validate configs and print plans with
`--dry-run`; their diffusers feature-extraction/sampler compatibility layers
have not yet been ported behind the public adapters. Table 1 reproduction and
paper-scale GPU parity therefore remain unverified, independently of the
deferred checkpoint and dataset publication.

The `models` extra pins the diffusers source revision used while consolidating
the research code. Model weights are not included and remain subject to their
upstream licenses and access requirements.

## Repository contracts

- `configs/register/` contains the checkpoint-derived register presets.
- `configs/rgs/` contains the exact SD3 and FLUX guidance schedules reported by
  the paper.
- `configs/rgopd/` contains the ten-step rollout and reward-tilt presets.
- A released checkpoint is a directory containing `register.safetensors`,
  `config.yaml`, and `manifest.json`.
- Dataset manifests use relative paths and the schema in `docs/data-format.md`.
- Original experiment workspaces, logs, absolute paths, and complete upstream
  repositories are deliberately excluded.

## Release layout

See `docs/release-layout.md` for the capability-oriented tree. The package
keeps register training, preference scoring, RGS, and RG-OPD behind shared
Python seams; paper checkpoints, datasets, and exact benchmark prompt files
remain deferred inputs.

Validate every checked-in workflow preset without downloading model weights:

```bash
lrr validate-release --root .
```

The output identifies the canonical FLUX RG-OPD provenance under
`z-image-reward-matrix/node5/src` while keeping that experiment workspace out
of the release package.

## Python interface

```python
output = register.score(latents, condition, sigma)
reward = register.score_and_grad(
    latents, condition, sigma, heads=("preference", "imagereward")
)
direction = guidance.combine(reward.gradients)
guided_latents, diagnostics = guidance.guided_step(
    latents=latents,
    base_next=base_next,
    gradient=direction,
    scale=0.30,
)
```

Existing experiment checkpoints can be loaded without referencing an original
workspace:

```python
from latent_reward_register import load_legacy_register

register = load_legacy_register(
    "checkpoints/reward_token_final_ema.pt",
    model_name_or_path="black-forest-labs/FLUX.1-dev",
)
```

Core training, sampling, and tests cross this same seam. Backbone-specific latent
packing, time conventions, prompt conditioning, and transformer traversal stay
inside adapters; the public diffusers adapters are currently explicit stubs
until those compatibility implementations are ported.

## Reproduction policy

The configuration embedded in the checkpoint that produced a reported result
is the source of truth. This matters because historical paper drafts and
experiment names contain incompatible layer descriptions. See
`docs/source-provenance.md` before comparing or retraining checkpoints.

## Tests

```bash
pytest -q
```

CPU tests cover the shared guidance and RG-OPD equations, data schema, and
checkpoint contract. They do not establish full adapter parity or paper-result
reproduction. Run `lrr smoke-release` for the asset-free integration check.
