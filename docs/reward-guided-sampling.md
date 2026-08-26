# Reward-guided sampling

RGS applies a reward-gradient correction to scheduled steps of an otherwise
frozen sampler. The loop is model-independent:

```python
from latent_reward_register import make_reference_step

sampled, trace = reward_guided_sample(
    register=register,
    latents=noise,
    condition=condition,
    sigmas=sigmas,
    heads=("preference", "imagereward"),
    schedule=GuidanceSchedule(((0.8, 0.30), (0.2, 0.05))),
    reference_step=make_reference_step(cfg_velocity),
    return_trace=True,
)
print(trace.guidance_fraction)
```

## The two pieces a backbone supplies

1. **A velocity model.** `cfg_velocity(latents, condition, timesteps)` returning
   the classifier-free-guided flow velocity.
   `flowmatch.classifier_free_velocity` combines the conditional and
   unconditional predictions.
2. **A register** that exposes `score_and_grad`. Building one from a config is
   `build_register_from_config`.

`make_reference_step` turns the velocity model into the frozen Euler transition:

    z_next = z + (sigma_next - sigma) * v(z, sigma)

Only Euler is released. The research sampler also implemented ab2, midpoint,
heun, and rk4; those cost extra forward passes per step and were not adopted.

## Guidance

Correction runs through the shared `RewardGradientTeacher`, so the guidance here
is identical to what RG-OPD distills. Per-head gradients are unit-RMS
normalized, summed with equal weight, magnitude-matched to
`scale * RMS(base step)`, then clipped.

Paper presets:

- `configs/rgs/sd3/paper.yaml`: 42 steps, CFG 4.5, bands 0.30 / 0.05.
- `configs/rgs/flux/paper.yaml`: 40 steps, embedded guidance 3.5, bands 0.30 / 0.10.

Z-Image is not a released RGS backbone: its register is trained and scored, but
no released sampling path consumes it.

## Cost

Steps below the schedule's lowest threshold skip the register forward and
backward entirely, so they cost exactly a plain sampler step.
`trace.guidance_fraction` measures the fraction actually guided — report it with
any cost claim rather than assuming the schedule.

Guidance needs `d reward / d latent`, which a frozen-trunk register does not
expose by default. See `docs/latent-gradients.md`.

Checkpoint and prompt paths are runtime inputs, not repository contents.
