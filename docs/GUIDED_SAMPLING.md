# Reward-guided sampling

Reward-guided sampling adds a scheduled latent reward-gradient correction to a
frozen FlowMatch Euler sampler:

```text
guided_next = base_next + reward_delta
```

For each configured reward head, the gradient is unit-RMS normalized. The
combined correction is magnitude-matched to the base diffusion step and
clipped. Steps outside the configured sigma bands skip register evaluation.

The paper presets are:

- `configs/rgs/sd3/paper.yaml`: SD3-Medium, 42 steps, CFG 4.5.
- `configs/rgs/flux/paper.yaml`: FLUX.1-dev, 40 steps, embedded guidance 3.5.

Run them with:

```bash
CHECKPOINT=/path/to/register.pt PROMPTS=prompts.txt bash scripts/sample_sd3.sh
CHECKPOINT=/path/to/register.pt PROMPTS=prompts.txt bash scripts/sample_flux.sh
```

The prompt file contains one prompt per line. Images are written to
`outputs/rgs_sd3` or `outputs/rgs_flux` unless `OUTPUT` is set.

The register trunk is frozen during scoring. The implementation temporarily
enables gradients with respect to the input latent only; model parameters remain
frozen. Z-Image guided sampling is not included.
