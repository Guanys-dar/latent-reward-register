# Reward-Gradient On-Policy Distillation

RG-OPD distills reward-guided transitions into a LoRA student. The student
generates its own trajectory, and the shared reward-gradient teacher constructs
the target at each visited state.

Both presets use ten-step rollouts, optimize the first nine transitions,
and train a rank-32, alpha-64 LoRA student for 300 optimization rounds.
Each round accumulates the configured rollout batches and applies one optimizer
update; it is not a full pass through the prompt file.

| Backbone | Config | Reward scale |
| --- | --- | --- |
| SD3-Medium | `configs/rgopd/sd3/paper.yaml` | 0.40 |
| FLUX.1-dev | `configs/rgopd/flux/paper.yaml` | 0.80 |

Run them with:

```bash
CHECKPOINT=/path/to/register.pt PROMPTS=prompts.txt bash scripts/train_rgopd_sd3.sh
CHECKPOINT=/path/to/register.pt PROMPTS=prompts.txt bash scripts/train_rgopd_flux.sh
```

`CHECKPOINT` is the backbone-matched teacher register. `PROMPTS` contains one
training prompt per line. LoRA weights are written under the selected output
directory. Set `ROUNDS=1` for a short real-model check. A run that ends before
the preset's selected checkpoint round exports its final EMA weights instead;
this short-run output is not the preset's selected checkpoint.
This checks execution only: a short run can finish before the first scheduled
EMA update.

The training target and inference-time sampler share the same
`RewardGradientTeacher`, so the distilled correction is the same one used by
guided sampling. Steps below the active sigma threshold skip register backward.
