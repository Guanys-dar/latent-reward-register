# Paper reproduction

## What can be reproduced from this repository

| Result | Status | Blocker |
| --- | --- | --- |
| Register architecture parity | verified | none — `lrr build-register` reports 147,599,619 trainable for SD3 exp11 |
| Training objective | in repo | training data |
| Reward gradients | verified | none |
| Guidance schedule and teacher | in repo | none |
| RGS trajectories | `lrr sample` runs it | published register checkpoints |
| RG-OPD | `lrr train-rgopd` runs it | teacher registers, prompt sets |
| Table 1 preference accuracy | evaluator in repo | pair file and images |
| Table 2 / Table 3 generation and scoring | not in repo | generation + scoring pipeline, prompt sets |

Every row above the last is executable from this repository once its assets are
in place; the last needs the third-party scorers (HPSv3, ImageReward, MUSIQ,
CLIP-IQA), which are separately licensed and not vendored here.


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
| Table 1 pair file + fetch script | yes | preference accuracy |
| keep-800 sample set | yes | the Table 2 / Table 3 evaluation set |

Both tables reproduce from what is published. Tables 2 and 3 report metrics over
keep-800, and the released `keep800.json` is self-contained: each of its 800
entries carries the prompt text, the seed, and its provenance, so the exact
scored samples can be regenerated. The 200 dropped samples never enter any
reported number, so the full 500-prompt generation set is not needed.

keep-800 was derived once by dropping the worst 200 of 1000 (500 prompts x seeds
42/43) by an equal-weight z-sum of {hpsv2, hpsv3, imagereward, pickscore} on a
defining variant, then applied unchanged to every variant so comparisons stay
paired. Use the published set rather than re-deriving the filter: recomputing it
on new generations selects a different 800 and yields different numbers.

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
