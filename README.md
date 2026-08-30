# Latent Reward Registers

Official research code for **Latent Reward Registers (LRR)** and two downstream
uses of their latent reward gradients:

- **Reward-Guided Sampling (RGS)** at inference time.
- **Reward-Gradient On-Policy Distillation (RG-OPD)** at training time.

| Method | SD3-Medium | FLUX.1-dev | Z-Image |
| --- | --- | --- | --- |
| Register training | yes | yes | experimental |
| Reward-guided sampling | yes | yes | no |
| RG-OPD | yes | yes | no |

## Installation

```bash
conda create -n lrr python=3.11
conda activate lrr
pip install -e '.[models]'
```

For CPU-only unit tests, install `pip install -e '.[dev]'` instead. Model
weights, register checkpoints, training data, and paper prompt sets are not
distributed in this repository.

Copy the local path template and point it at your model snapshots:

```bash
cp configs/local.yaml.example configs/local.yaml
```

## Quick start

The scripts contain the paper settings. Override `OUTPUT` or append any CLI
option after the script name when needed.

### Train a register

Register training uses cached latents and prompt embeddings grouped by prompt.
See [docs/TRAIN_REGISTER.md](docs/TRAIN_REGISTER.md) for the manifest format.

```bash
MANIFEST=/path/to/groups.jsonl bash scripts/train_register_sd3.sh
MANIFEST=/path/to/groups.jsonl bash scripts/train_register_flux.sh
```

### Reward-guided sampling

```bash
CHECKPOINT=/path/to/register.pt PROMPTS=/path/to/prompts.txt \
  bash scripts/sample_sd3.sh

CHECKPOINT=/path/to/register.pt PROMPTS=/path/to/prompts.txt \
  bash scripts/sample_flux.sh
```

Generated images are written under `outputs/`. See
[docs/GUIDED_SAMPLING.md](docs/GUIDED_SAMPLING.md) for the guidance schedule.

### Train with RG-OPD

```bash
CHECKPOINT=/path/to/register.pt PROMPTS=/path/to/prompts.txt \
  bash scripts/train_rgopd_sd3.sh

CHECKPOINT=/path/to/register.pt PROMPTS=/path/to/prompts.txt \
  bash scripts/train_rgopd_flux.sh
```

See [docs/REWARD_OPD.md](docs/REWARD_OPD.md) for rollout and student settings.

## Configuration

Paper presets live in:

```text
configs/register/{sd3,flux,zimage}/paper.yaml
configs/rgs/{sd3,flux}/paper.yaml
configs/rgopd/{sd3,flux}/paper.yaml
```

All scripts call the same three commands:

```bash
lrr train-register --config CONFIG --training-manifest MANIFEST --output-directory OUTPUT
lrr sample --config CONFIG --register-checkpoint CHECKPOINT --prompt-file PROMPTS --output-directory OUTPUT
lrr train-rgopd --config CONFIG --register-checkpoint CHECKPOINT --prompt-file PROMPTS --output-directory OUTPUT
```

Use `--model-path` to override the model location in `configs/local.yaml`. Use
`--local-files-only` on offline machines. `--dry-run` validates a configuration
without loading model weights.

## Repository layout

```text
configs/   paper configurations for each method and backbone
scripts/   commands used to launch the main experiments
src/       register, sampling, and RG-OPD implementations
docs/      data and method details
tests/     lightweight algorithm tests
```

## Notes

- The exact Diffusers revision in `pyproject.toml` is intentional: the register
  reads intermediate transformer states whose layout changes across releases.
- Z-Image support is experimental and limited to register training.
- The third-party benchmark scorers used in the paper are not redistributed.

Run the weight-free checks with:

```bash
bash scripts/smoke_all.sh
```
