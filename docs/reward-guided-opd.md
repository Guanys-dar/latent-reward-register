# Reward-guided OPD

The shared RG-OPD math lives in:

- `latent_reward_register.rgopd.build_rgopd_target`
- `latent_reward_register.rgopd.rgopd_loss`
- `latent_reward_register.guidance.RewardGradientGuidance`

Both paper presets use ten rollout steps, nine optimized steps, a frozen
reference anchor, and a rank-32/alpha-64 LoRA student.

## SD3

The package-level RG-OPD seam is configured by
`configs/rgopd/sd3/paper.yaml`. The original SD3 trainer remains provenance
material; its migration should call the shared target/loss functions rather
than duplicate their equations.

## FLUX

The canonical FLUX implementation is the dedicated node5 provenance stack:

- `node5/src/scripts/train_flux_rgopd.py`
- `node5/src/configs/rgopd_flux.py`
- `node5/src/flux_opd/`

The release package records this provenance without copying its checkpoints,
logs, or training data. The unified-v3 HPS `rt0.80 sigma>0.2 e150` run is the
current canonical experiment reference. The node5 source still needs to be
ported behind the release package's RG-OPD seam before asset-backed execution
is enabled.
