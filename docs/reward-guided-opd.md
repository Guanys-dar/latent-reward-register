# Reward-guided OPD

RG-OPD distills a reward-guided sampler into a LoRA student. The student learns
to take, in one step, the step that the reward-guided teacher takes.

## The shared teacher

`latent_reward_register.teacher.RewardGradientTeacher` produces the guided
next-state consumed by both RGS and RG-OPD:

    guided_next = base_next + reward_delta

where `reward_delta` points along `d reward / d latent`, unit-RMS normalized per
head, and is magnitude-matched to `scale * RMS(base_next - latents)` before
clipping. `scale` comes from the sigma-banded guidance schedule.

This is one implementation on purpose. If RG-OPD recomputed the correction
itself, the student could be trained against guidance the sampler does not
apply, and the two would drift silently.

`rollout_target` is the RG-OPD entry point:

```python
step = rollout_target(
    teacher=teacher,
    latents=latents,
    reference_next=reference_next,   # frozen reference transition
    condition=condition,
    timesteps=timesteps,
    sigma=sigma,
)
loss = rgopd_loss(student_next, step.guided_next, transition_std)
```

## Cost

Steps where the schedule resolves to zero skip the register forward and
backward entirely: `teacher.is_active(sigma)` reports this, and
`guided_step` returns the base step untouched. For an early-only schedule that
is most of the trajectory, which is where the reported efficiency comes from.

The register backward needs the latent-gradient path; see
`docs/latent-gradients.md`.

## Presets

Both backbones roll out ten steps and optimize the first nine, against a frozen
reference anchor, with a rank-32 / alpha-64 LoRA student.

| Backbone | Config | Reward scale | Teacher register |
| --- | --- | --- | --- |
| SD3 | `configs/rgopd/sd3/paper.yaml` | 0.40 | exp11 EMA |
| FLUX | `configs/rgopd/flux/paper.yaml` | 0.80 | unified-v3 EMA final |

Guidance is active for sigma above 0.2 in both presets. Note the teacher
registers differ per backbone: the FLUX config carries exp11 as a legacy
default and overrides it, so the default alone is misleading. See
`docs/source-provenance.md`.

## What remains

The rollout driver - constructing reference transitions from a real sampler and
stepping a LoRA student across a trajectory - is not yet in the release. The
target, loss, teacher, and schedule are, and are covered by tests.
