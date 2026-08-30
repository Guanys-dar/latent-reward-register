<h1 align="center">Latent Reward Registers for Diffusion Preference Alignment</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2608.03929"><img src="https://img.shields.io/badge/arXiv-2608.03929-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Guanys-dar/latent-reward-register"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github" alt="Code"></a>
  <img src="https://img.shields.io/badge/Checkpoints-Coming%20Soon-blue.svg" alt="Checkpoints coming soon">
  <img src="https://img.shields.io/badge/Website-Coming%20Soon-green.svg" alt="Website coming soon">
</p>

<p align="center"><strong>To be submitted to ICLR 2027.</strong></p>

**We propose building visual reward models directly on visual generators,
enabling reward modeling to benefit from generative scaling.**

Diffusion alignment usually scores only the final image, leaving every earlier
denoising step with a difficult temporal credit-assignment problem. **Latent
Reward Registers (LRR)** estimate terminal preference directly from intermediate
noisy latents.

LRR appends learnable, position-free register tokens as an auxiliary read path
to a frozen Diffusion Transformer. It extracts preference signals without
changing the generator's hidden states or velocity field.

The resulting dense, differentiable reward field supports two alignment
strategies:

- **Reward-Gradient On-Policy Distillation (RG-OPD)** creates direct per-step
  targets at states visited by the current generator, avoiding rollout-intensive
  policy-gradient optimization.
- **Reward-Guided Sampling (RGS)** steers the denoising trajectory with
  magnitude-matched reward-gradient corrections and requires no parameter
  updates.

## Highlights

- At high noise (`t = 0.8`), LRR achieves the best pairwise accuracy among the
  evaluated latent reward models.
- RG-OPD outperforms online reinforcement-learning baselines while reducing GPU
  hours by up to **33×**.
- RGS significantly improves reward while maintaining a favorable
  reward–quality balance against training-free baselines.

This repository provides LRR training, RGS, and RG-OPD for SD3-Medium and
FLUX.1-dev. Experimental Z-Image support is included for register training.

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

## Acknowledgements

We thank [DiffusionOPD](https://github.com/ali-vilab/DiffusionOPD),
[Flow-GRPO](https://github.com/yifan123/flow_grpo), and
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) for providing excellent
open-source diffusion reinforcement-learning codebases.
