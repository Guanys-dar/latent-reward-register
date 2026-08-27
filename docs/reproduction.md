# Paper reproduction

## What can be reproduced from this repository

| Result | Status | Blocker |
| --- | --- | --- |
| Register architecture parity | verified | none — `lrr build-register` reports 147,599,619 trainable for SD3 exp11 |
| Training objective | in repo | training data |
| Reward gradients | verified | none |
| Guidance schedule and teacher | in repo | none |
| RGS trajectories | loop in repo | published register checkpoints |
| RG-OPD targets and loss | in repo | rollout driver, checkpoints |
| Table 1 preference accuracy | evaluator in repo | pair file and images |
| Table 2 / Table 3 generation and scoring | not in repo | generation + scoring pipeline, prompt sets |

Nothing here selects checkpoints on the final test prompts; see the selection
protocol below.

## Register training

Presets are `configs/register/<backbone>/paper.yaml`. Frozen generator, bf16,
AdamW, EMA 0.999, and the checkpoint-derived layer taps. The objective is the
group-level Thurstone loss: all within-group pairs contribute, each group is
normalized by its own pair count, and pair noise scales with the diffusion
sigma.

SD3 (exp11) trains three heads at equal weight; FLUX and Z-Image unified-v3
train two. Head count is per-backbone and must not be unified.

## Reward-Guided Sampling

SD3 uses 42 FlowMatch Euler steps at CFG 4.5; FLUX uses 40 steps at embedded
guidance 3.5. Both use per-head unit-RMS gradients, equal-weight summation, and
step-magnitude matching, with guidance disabled in the low-noise tail.

The released schedule is the sigma-banded one in the configs. The gradient
spectral filters explored during the research (gaussian lowpass, butterworth,
band-match, wiener, and the `lp2zm` combination) were **not adopted** and are
not part of the released method.

## RG-OPD

Ten-step on-policy rollouts, first nine steps optimized, frozen reference
anchor, rank-32 / alpha-64 LoRA student. Reward scale 0.40 for SD3 and 0.80 for
FLUX, guidance active above sigma 0.2.

Teacher registers differ: SD3 uses the exp11 EMA register, FLUX the unified-v3
EMA final. The FLUX config carries exp11 as a legacy default and overrides it,
so the default alone is misleading.

FLUX checkpoint selection is argmax HPSv3 subject to CLIP-IQA >= 0.649, over 100
held-out screen prompts disjoint from the test set. Chosen epochs: HPS 150,
ImageReward 60, TwoHead 150.

## Published evaluation assets

| Asset | Published | Purpose |
| --- | --- | --- |
| Table 1 pair file + fetch script | yes | preference accuracy, end to end |
| keep-800 key set | yes | identifies which samples Table 3 scores |
| 500-prompt generation set | no | — |
| 100 held-out FLUX screen prompts | no | — |

Table 3 is scored on **keep-800**, so that key set is published: it is the
evaluation set, and re-deriving it yields different numbers.

keep-800 holds 800 `[prompt_index, seed]` keys out of 1000 (500 prompts x seeds
42/43). The worst 200 are dropped by an equal-weight z-sum of
{hpsv2, hpsv3, imagereward, pickscore}, computed once on a defining variant and
then applied unchanged to every variant so comparisons stay paired.

The keys carry indices, not prompt text. Since the 500-prompt generation set is
not published, index 400 cannot be resolved to a prompt, so a reader can verify
the filter and re-score images they already have, but cannot regenerate the 1000
images keep-800 selects from. **Table 1 is the table that reproduces end to
end** once its images are fetched.

## Evaluation

Pairwise preference accuracy must preserve prompt groups and use the fixed
benchmark split. The released Table 1 pair file is
`all_table1_pairs_fixed.jsonl` (54170 pairs); verify row count and checksum
before comparing against published accuracies.

Generation comparisons must match prompts, seeds, resolution, sampler, and
metric implementation. Report the guided-step fraction alongside any cost claim:
`reward_guided_sample(..., return_trace=True)` measures it.

## Deferred assets

Checkpoints, training-data manifests, and the paper prompt sets are published
separately. Use `lrr plan <config>` to see the runtime inputs a preset needs.
