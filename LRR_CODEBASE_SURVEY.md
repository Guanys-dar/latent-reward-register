# Latent Reward Register Open-Source Codebase Survey

Survey date: 2026-08-18

This document records a read-only survey of:

- `/home/guanyuanshen/Reward-Token-Image-0420`
- `/home/guanyuanshen/rt-gradient-opd`
- `/home/guanyuanshen/RT-guided-sampling`
- `/kaimm-distill/ysguan/z-image-reward-matrix`

No source files were changed, moved, or reorganized during the survey.

## 1. Executive Summary

The code needed for the proposed open-source release mostly exists, but it is fragmented across experiment workspaces, copied implementations, generated-output directories, and machine-specific paths.

| Capability | SD3 | FLUX | Z-Image | Current status |
|---|---|---|---|---|
| Reward-register training | Complete | Complete in `node4/src` | Complete in `node4/src` | Needs a canonical implementation and portable data configuration |
| Preference scoring / Table 1 | Present | Present | Present | Table 1 pair data is referenced outside the surveyed repositories |
| Reward-guided sampling | Complete | Complete | Substantial experimental implementation exists | Release scope and canonical schedules need confirmation |
| Reward-guided OPD training | Complete | Several possible implementations | No clearly matching paper pipeline found | Canonical FLUX implementation is ambiguous; Z-Image appears missing from paper scope |
| OPD inference/evaluation | Complete for SD3 | Scattered among several stacks | Partial unrelated self-distillation evaluation | Needs consolidation |
| Table 1 test set | External absolute paths only | Same | Same | Missing documentation/reconstruction recipe |
| Table 2/3 prompts and seeds | Multiple conflicting protocols | Same | Same | Exact requested 400 prompts and two seeds are not currently identified |

There is already a proposed release repository at:

`/kaimm-distill/ysguan/z-image-reward-matrix/latent-reward-register`

It contains a clean package structure, unified model interfaces, paper-style YAML presets, tests, licensing files, and documentation. However, it is not currently an end-to-end reproduction repository. Its CLI only implements:

- `lrr inspect-config`
- `lrr inspect-checkpoint`
- `lrr list-backbones`

It does not expose runnable commands for register training, preference evaluation, RGS generation, RG-OPD training, or paper evaluation.

## 2. Existing Consolidated Release Candidate

### Location

`/kaimm-distill/ysguan/z-image-reward-matrix/latent-reward-register`

### Package entry points and common interfaces

- `src/latent_reward_register/cli.py`
  - Package CLI.
  - Inspection commands only.
- `src/latent_reward_register/register.py`
  - Common register abstraction.
- `src/latent_reward_register/training.py`
  - Shared training-level abstractions.
- `src/latent_reward_register/sampling.py`
  - Shared sampling abstractions.
- `src/latent_reward_register/guidance.py`
  - Reward-gradient combination and guided-step logic.
- `src/latent_reward_register/rgopd.py`
  - Shared RG-OPD math/interface.
- `src/latent_reward_register/checkpoint.py`
  - New release checkpoint contract and legacy checkpoint loading.
- `src/latent_reward_register/data.py`
  - Portable data schemas.
- `src/latent_reward_register/config.py`
  - YAML configuration loading.
- `src/latent_reward_register/types.py`
  - Shared types.

### Backbone abstraction

- `src/latent_reward_register/backbones/base.py`
- `src/latent_reward_register/backbones/diffusers.py`
- `src/latent_reward_register/backbones/registry.py`

### Copied research implementations

- `src/latent_reward_register/implementations/models/backbone.py`
  - SD3 backbone-derived implementation.
- `src/latent_reward_register/implementations/models/latent_reward_grid.py`
  - SD3 latent reward grid.
- `src/latent_reward_register/implementations/models/pooling.py`
- `src/latent_reward_register/implementations/models/flux_backbone.py`
- `src/latent_reward_register/implementations/models/flux_latent_reward_grid.py`
- `src/latent_reward_register/implementations/models/zimage_backbone.py`
- `src/latent_reward_register/implementations/models/zimage_latent_reward_grid.py`
- `src/latent_reward_register/implementations/flux_common.py`
- `src/latent_reward_register/implementations/zimage_common.py`
- `src/latent_reward_register/implementations/loader.py`

### Paper-style configs

- `configs/register/sd3/paper.yaml`
- `configs/register/flux/paper.yaml`
- `configs/register/zimage/paper.yaml`
- `configs/rgs/sd3/paper.yaml`
- `configs/rgs/flux/paper.yaml`
- `configs/rgopd/sd3/paper.yaml`
- `configs/rgopd/flux/paper.yaml`

The register configurations declare:

| Backbone | Model | Feature layers | Heads | Register tokens |
|---|---|---:|---|---:|
| SD3 | `stabilityai/stable-diffusion-3-medium-diffusers` | 4, 8, 12 | preference, imagereward | 32 |
| FLUX | `black-forest-labs/FLUX.1-dev` | 9, 19, 28 | preference, imagereward | 32 |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | 5, 10, 15 | preference, imagereward | 32 |

The RGS configurations declare:

| Backbone | Steps | Resolution | Text/base guidance | Reward schedule |
|---|---:|---|---|---|
| SD3 | 42 | 1024×1024 | CFG 4.5 | `(u≥0.8, 0.30)`, `(u≥0.2, 0.05)` |
| FLUX | 40 | 1024×1024 | embedded guidance 3.5 | `(u≥0.8, 0.30)`, `(u≥0.2, 0.10)` |

The RG-OPD configurations declare ten rollout steps, nine optimized steps, LoRA rank 32/alpha 64, a frozen reference anchor, and reward scales 0.40 for SD3 and 0.80 for FLUX.

### Existing documentation

- `README.md`
- `docs/checkpoint-format.md`
- `docs/data-format.md`
- `docs/reproduction.md`
- `docs/source-provenance.md`
- `docs/IMPLEMENTATION_SHA256SUMS`

### Existing tests

- `tests/test_checkpoint.py`
- `tests/test_data.py`
- `tests/test_guidance.py`
- `tests/test_losses.py`
- `tests/test_release_hygiene.py`
- `tests/test_rgopd.py`

### Release-candidate limitations

- No executable training command.
- No executable preference-scoring command.
- No Table 1 evaluator command.
- No executable RGS generation command.
- No executable RG-OPD training command.
- No RG-OPD student inference/evaluation command.
- No benchmark prompt lists.
- No released checkpoints or checkpoint download manifest.
- CPU-level common-math tests exist, but full GPU adapter parity is not demonstrated.

## 3. Capability 1: Reward-Register Training

### 3.1 SD3 training

Primary workspace:

`/home/guanyuanshen/Reward-Token-Image-0420`

#### Entry points

- `scripts/train/train_reward_token.py`
  - Principal reward-token/register training entry point.
- `scripts/train/train_diffusion_probe.py`
  - Diffusion-probe baseline.
- `scripts/train/train_probe.py`
  - Probe baseline.
- `scripts/train/train_pavrm.py`
  - PAVRM baseline.

#### Main paper-era launch scripts

- `exp11_resume_e3_cmd.sh`
- `exp12_L4_cmd.sh`
- `exp13_inputkv_cmd.sh`
- `exp14_fulltoken_cmd.sh`
- `exp15_L4snap_fulltoken_cmd.sh`
- `exp16_L4stop4_fulltoken_cmd.sh`
- `scripts/run/train_reward_token_dina_pair_thurstone_8gpu.sh`
- `scripts/run/train_reward_token_multihead_dina_pair_8gpu.sh`
- `scripts/run/train_latent_reward_grid_mlp_8gpu.sh`

#### Candidate final/paper configs

- `configs/train/exp14_fulltoken_multihead_noattn2_sd3.yaml`
- `configs/train/exp15_L4snap_fulltoken_multihead_noattn2_sd3.yaml`
- `configs/train/exp16_L4stop4_fulltoken_multihead_noattn2_sd3.yaml`
- `configs/train/latent_reward_grid_pool_nope_multihead_noattn2_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_pool_nope_multihead_noattn2_sd3_8gpu_resume_e3.yaml`

#### Other architecture/config variants

- `configs/train/reward_token_sd3.yaml`
- `configs/train/reward_token_sd3_4gpu.yaml`
- `configs/train/reward_token_sd3_not_skip_ffn.yaml`
- `configs/train/reward_token_sd3_4gpu_not_skip_ffn.yaml`
- `configs/train/reward_token_sd3_multihead.yaml`
- `configs/train/reward_token_sd3_4gpu_dina_thurstone.yaml`
- `configs/train/reward_token_sd3_4gpu_dina_pair_thurstone.yaml`
- `configs/train/reward_token_sd3_8gpu_dina_pair_thurstone.yaml`
- `configs/train/reward_token_sd3_8gpu_multihead_dina_pair.yaml`
- `configs/train/reward_token_sd3_8gpu_rt_dina_pp.yaml`
- `configs/train/latent_reward_grid_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_pool_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_pool_nope_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_pool_nope_multihead_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_mlp_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_set_sd3_8gpu.yaml`
- `configs/train/latent_reward_grid_xattn_sd3_8gpu.yaml`

#### SD3 model definitions

- `src/sd3_reward/models/backbone.py`
  - Loads and traverses the SD3 transformer.
  - Supports activation extraction and gradient checkpointing.
- `src/sd3_reward/models/latent_reward_grid.py`
  - Latent reward grid/register implementation.
- `src/sd3_reward/models/reward_token.py`
  - Earlier single-head reward-token architecture.
- `src/sd3_reward/models/reward_token_multihead.py`
  - Multi-head reward-token implementation.
- `src/sd3_reward/models/reward_token_dina_head.py`
  - DINA/pairwise head.
- `src/sd3_reward/models/pooling.py`
  - Feature pooling.
- `src/sd3_reward/models/diffusion_probe.py`
- `src/sd3_reward/models/diffusion_probe_components.py`
- `src/sd3_reward/models/probe.py`
- `src/sd3_reward/models/pavrm.py`

#### Training engine and checkpointing

- `src/sd3_reward/train/engine.py`
- `src/sd3_reward/train/checkpoint.py`
- `src/sd3_reward/train/eval.py`
- `src/sd3_reward/train/bench_eval.py`

#### Loss implementations

- `src/sd3_reward/losses/dina_thurstone.py`
- `src/sd3_reward/losses/hybrid.py`
- `src/sd3_reward/losses/multihead_loss.py`
- `src/sd3_reward/losses/regression_rank.py`

#### Dataset implementations

- `src/sd3_reward/data/group_dataset.py`
- `src/sd3_reward/data/multihead_group_dataset.py`
- `src/sd3_reward/data/dina_pair_dataset.py`
- `src/sd3_reward/data/multihead_dina_pair_dataset.py`
- `src/sd3_reward/data/collate.py`
- `src/sd3_reward/data/schemas.py`

#### Data preparation

- `scripts/reward_data/prepare_sd3_group_manifest.py`
- `scripts/reward_data/cache_sd3_latents.py`
- `scripts/reward_data/cache_sd3_prompts.py`
- `scripts/reward_data/add_normalized_score_targets.py`
- `scripts/reward_data/run_prepare_hpdv3_sd3_cache.sh`
- `scripts/reward_data/run_prepare_hpdv3_sd3_cache_with_images.sh`

#### Teacher-score preparation and external reward models

- `scripts/score_hpsv3.py`
- `scripts/score_pickscore_imagereward.py`
- `scripts/run_hpsv3_scoring.sh`
- `scripts/run_pickscore_imagereward_scoring.sh`
- Vendored HPSv3 checkout: `HPSv3/`

#### External model and checkpoint references

- SD3 model: `/kaimm-distill/ysguan/stable-diffusion-3-medium-diffusers`
- One resume config references:
  `/kaimm-distill/ysguan/Image-Reward-Token-Exps/Reward-Token-Image-0420/outputs/exp11_multihead_noattn2_sd3_8gpu/checkpoints/reward_token_step_0015000.pt`
- Historical outputs use files such as:
  - `reward_token_step_*.pt`
  - `reward_token_epoch_*.pt`
  - `reward_token_final_ema.pt`

#### SD3 ambiguity

The experiment naming and consolidated paper config do not describe exactly the same architecture. Names such as `L4snap`, `fulltoken`, `noattn2`, and `L4stop4` need to be reconciled with the release config's feature layers `[4, 8, 12]`. The checkpoint-embedded config should be used to identify the actual paper model.

### 3.2 FLUX training

Primary workspace:

`/kaimm-distill/ysguan/z-image-reward-matrix/node4/src`

#### Entry point

- `scripts/train/train_reward_token.py`
  - Extended copy of the SD3 trainer supporting FLUX and Z-Image.

#### Principal configs

- `configs/train/flux_exp18_reward_register_2head.yaml`
- `configs/train/flux_exp18_reward_register_2head_smoke.yaml`
- `configs/train/flux_reward_register_singlehead_imagereward.yaml`
- `configs/train/flux_reward_register_singlehead_imagereward_smoke.yaml`

#### Launcher

- `scripts/run/train_reward_register_flux_exp11.sh`

#### Model definitions

- `sd3_reward/models/flux_backbone.py`
- `sd3_reward/models/flux_latent_reward_grid.py`
- `flux_common.py`
- Shared pooling/loss/data/trainer files copied from the SD3 workspace.

#### Data preparation

- `scripts/reward_data/cache_flux_prompts.py`
- `scripts/reward_data/make_flux_manifests.py`

#### External references

- Base model path: `/kaimm-distill/ysguan/ckpt_dev`
- Intended public model: `black-forest-labs/FLUX.1-dev`
- Training manifests under `shared/data/` contain absolute paths to cached embeddings, latents, and source images.

### 3.3 Z-Image training

Primary workspace:

`/kaimm-distill/ysguan/z-image-reward-matrix/node4/src`

#### Entry point

- `scripts/train/train_reward_token.py`

#### Candidate configs

- `configs/train/zimage_exp13_reward_register_2head_dinapair.yaml`
- `configs/train/zimage_exp13_reward_register_3head_dinapair.yaml`
- `configs/train/zimage_exp14_reward_register_2head_fixv2.yaml`
- `configs/train/zimage_exp16_reward_register_3head64_v3.yaml`
- `configs/train/zimage_exp17_selfgen_partial.yaml`
- Corresponding smoke configs are present for several runs.

#### Launchers

- `scripts/run/train_reward_register_fixv2.sh`
- `scripts/run/train_reward_register_v3.sh`
- `scripts/run/train_reward_register_selfgen_partial.sh`

#### Model definitions

- `sd3_reward/models/zimage_backbone.py`
- `sd3_reward/models/zimage_latent_reward_grid.py`
- `zimage_common.py`
- Shared pooling/loss/data/trainer files copied from the SD3 workspace.

#### Data preparation

- `scripts/reward_data/cache_zimage_latents.py`
- `scripts/reward_data/cache_zimage_prompts.py`
- `scripts/reward_data/prepare_zimage_group_manifest.py`
- `scripts/reward_data/build_zimage_dina_pairs.py`
- `scripts/reward_data/generate_zimage_selfgen.py`

#### Shared manifests

Relevant large manifests exist under:

`/kaimm-distill/ysguan/z-image-reward-matrix/shared/data`

Examples include:

- `multireward_manifest_unified_val_v3.jsonl`
- Other `multireward_manifest_*.jsonl` train/validation variants.

These records include absolute paths to:

- `/kaimm-distill/ysguan/reward-token-image-dataset/HPDv3/images/...`
- `shared/data/zimage_latents/*.latent_x0.pt`
- `shared/data/zimage_qwen3_embeds/*.prompt_embeds.pt`
- `shared/data/zimage_qwen3_embeds/*.pooled_prompt_embeds.pt`

#### External model references

- Internal base model path: `/kaimm-distill/ysguan/Z-Image`
- Consolidated public identifier: `Tongyi-MAI/Z-Image-Turbo`

#### Z-Image-specific correctness constraint

The Z-Image code explicitly uses a native time convention described as `t = 1 - sigma`. The FLUX evaluator states that FLUX uses scheduler timestep normalization directly. These adapters must not be merged by treating their time/noise conventions as identical.

## 4. Capability 2: Register Preference Inference and Table 1

### 4.1 SD3

Relevant implementation files:

- `/home/guanyuanshen/Reward-Token-Image-0420/src/sd3_reward/train/bench_eval.py`
- `/home/guanyuanshen/Reward-Token-Image-0420/src/sd3_reward/train/eval.py`
- `/home/guanyuanshen/Reward-Token-Image-0420/scripts/sweep/run_sweep.sh`

`run_sweep.sh` locates a trained `reward_token_step_*.pt` checkpoint and invokes a Table 1 evaluator. Some historical evaluator references are machine-specific or live outside the core workspace, so the exact final SD3 Table 1 command still needs to be recovered from the paper run record/checkpoint metadata.

### 4.2 FLUX

Primary evaluator:

- `/kaimm-distill/ysguan/z-image-reward-matrix/node4/src/scripts/eval/evaluate_flux_table1.py`

Supporting loader/common code:

- `node4/src/scripts/eval/flux_eval_common.py`
- `node4/src/flux_common.py`

Launch/report evidence:

- `node4/NODE4_FLUX_EXP11_PORT_EXECUTION_REPORT.md`
- `node4/NODE4_V3_TABLE1_EXECUTION_REPORT.md`

The evaluator loads a register checkpoint, encodes both candidate images with the FLUX-compatible VAE path, scores the requested heads, and compares predicted preference direction against labeled pairs.

### 4.3 Z-Image

Primary evaluator:

- `/kaimm-distill/ysguan/z-image-reward-matrix/node4/src/scripts/eval/evaluate_zimage_table1.py`

Supporting code:

- `node4/src/scripts/eval/zimage_eval_common.py`
- `node4/src/zimage_common.py`

Launchers and aggregation:

- `node4/src/scripts/eval/run_table1_v3.sh`
- `node4/src/scripts/eval/run_table1_fixv2.sh`
- `node4/src/scripts/eval/run_table1_fixv2_aggregate.sh`
- `node4/src/scripts/eval/run_table1_aggregate.sh`
- `node4/src/scripts/eval/run_table1_full.sh`

### 4.4 Table 1 pair data

The pair dataset is not self-contained in the surveyed roots. At least two external paths are referenced:

- `/home/guanyuanshen/iclr-exp-matrix/evaluate-lrm/data/all_table1_pairs_fixed.jsonl`
- `/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/canonical/all_table1_pairs.jsonl`

Sweep configs also refer to:

- `/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/canonical/all_table1_pairs.jsonl`

The difference between `all_table1_pairs_fixed.jsonl` and `all_table1_pairs.jsonl` is not documented in the surveyed code. The release needs either:

1. a reconstruction/download recipe plus checksums and expected row counts, or
2. permission to include the pair metadata directly.

### 4.5 Table 1 missing pieces

- Authoritative pair-list version.
- Public source/reconstruction instructions.
- Exact checkpoints used for each paper row.
- Exact head selection and noise/time parameter for each row.
- One cross-backbone command producing the final Table 1 table.

## 5. Capability 3: Reward-Guided Sampling

### 5.1 SD3 RGS

Primary workspace:

`/home/guanyuanshen/RT-guided-sampling`

#### Main experiment launchers

- `run_exp14_table2_gen.sh`
- `run_exp14_table2_score.sh`
- `run_exp15_table2_gen.sh`
- `run_exp15_table2_score.sh`
- `run_exp15_allheads_2node.sh`
- `run_exp15_pref_lp2zm_gen.sh`
- `run_exp15_pref_lp2zm_score.sh`
- `run_exp15_control_gen.sh`
- `run_exp15_visual_grid.sh`
- `recover_score_exp15_allheads.sh`

#### Table 3/single-head launchers

- `run_singlehead_ir_table3_driver.sh`
- `run_singlehead_ir_table3_sd3_gen.sh`
- `run_singlehead_ir_table3_score.sh`

#### Main package

`flow_grpo/`

Relevant areas include:

- `flow_grpo/flow_grpo/`
  - Sampling, rewards, prompt loading, patched diffusion logic, scoring, and logging.
- `flow_grpo/config/`
  - Runtime configurations.
- `flow_grpo/scripts/`
  - Training/generation/evaluation entry points and Accelerate configs.
- `flow_grpo/dataset/`
  - Benchmark prompt sources and metadata.

#### Evaluation protocol

- `docs/rt-guided-sampling-evaluation-report.md`

The repository-level instructions identify `rt_balanced500_v1` as the formal configuration-comparison prompt set and require prompt/seed/model pairing against CFG, retained manifests, non-circular metrics, and human review for final quality claims.

#### External references

- SD3 base model: `/kaimm-distill/ysguan/stable-diffusion-3-medium-diffusers`
- Single-head ImageReward checkpoint:
  `/kaimm-distill/ysguan/z-image-reward-matrix/node4/outputs/reward_register_sd3_singlehead_ir/checkpoints/reward_token_final_ema.pt`
- Other scripts refer to checkpoint trees under:
  `/kaimm-distill/ysguan/Image-Reward-Token-Exps/Reward-Token-Image-0420/outputs`

### 5.2 FLUX RGS

Primary implementation:

- `/kaimm-distill/ysguan/z-image-reward-matrix/node4/src/scripts/eval/flux_rt_guided_sampling.py`

Supporting code:

- `node4/src/scripts/eval/flux_eval_common.py`
- `node4/src/flux_common.py`
- `node4/src/sd3_reward/models/flux_backbone.py`
- `node4/src/sd3_reward/models/flux_latent_reward_grid.py`

Generation/scoring:

- `node4/src/scripts/eval/run_table2_flux_gen.sh`
- `node4/src/scripts/eval/run_table2_flux_score.sh`
- `node4/src/scripts/eval/run_table2_flux_fid.sh`
- `node4/src/scripts/eval/aggregate_table2_flux.py`

Table 3 launcher:

- `/home/guanyuanshen/RT-guided-sampling/run_singlehead_ir_table3_flux_gen.sh`

Diagnostics and correctness tests:

- `node4/src/scripts/eval/flux_parity_check.py`
- `node4/src/scripts/eval/diag_gamma_eff_flux.py`
- `node4/src/scripts/eval/diag_pulse_retention_flux.py`

### 5.3 Z-Image RGS

Primary implementation:

- `/kaimm-distill/ysguan/z-image-reward-matrix/node4/src/scripts/eval/zimage_rt_guided_sampling.py`

Supporting code:

- `node4/src/scripts/eval/zimage_eval_common.py`
- `node4/src/zimage_common.py`
- `node4/src/sd3_reward/models/zimage_backbone.py`
- `node4/src/sd3_reward/models/zimage_latent_reward_grid.py`

Generation/scoring/FID:

- `node4/src/scripts/eval/run_table2_gen.sh`
- `node4/src/scripts/eval/run_table2_score.sh`
- `node4/src/scripts/eval/run_table2_fid.sh`
- `node4/src/scripts/eval/run_table2_fixv2_gen.sh`
- `node4/src/scripts/eval/run_table2_fixv2_score.sh`
- `node4/src/scripts/eval/run_table2_fixv2_fid.sh`
- `node4/src/scripts/eval/aggregate_table2.py`

The consolidated release README says RGS is released only for SD3 and FLUX, although this Z-Image implementation exists. Its release status is therefore a policy/product decision rather than a code-availability issue.

## 6. Capability 4: Reward-Guided OPD

### 6.1 SD3 RG-OPD

Primary workspace:

`/home/guanyuanshen/rt-gradient-opd/RG-OPD`

#### Training entry point

- `scripts/train_sd3_rgopd.py`

#### Config

- `config/rgopd.py`

#### Core reward-guidance implementation

- `rg_opd/reward_grad_teacher.py`
  - Loads/uses the reward register as the gradient teacher.
- `rg_opd/guidance_math.py`
  - Reward-gradient guidance equations.
- `rg_opd/__init__.py`

#### Training launchers

- `scripts/single_node/rgopd_hps.sh`
- `scripts/single_node/rgopd_hps_exp11.sh`
- `scripts/single_node/rgopd_hps_exp11_cfgdistill_rt020.sh`
- `scripts/single_node/rgopd_hps_exp11_cfgdistill_rt040.sh`
- `scripts/single_node/rgopd_hps_exp11_emaanchor.sh`
- `scripts/single_node/rgopd_hps_exp11_rt040.sh`
- `scripts/single_node/rgopd_hps_exp11_sched.sh`
- `scripts/single_node/rgopd_hps_selfanchor.sh`
- `scripts/single_node/rgopd_matrix.sh`
- `scripts/single_node/rgopd_matrix_hps.sh`
- `scripts/single_node/rgopd_matrix_ir.sh`
- `scripts/single_node/rgopd_matrix_twohead.sh`
- `scripts/single_node/rgopd_mh.sh`
- `scripts/single_node/smoke.sh`

#### Supporting scripts

- `scripts/smoke_teacher.py`
- `scripts/select_matrix_checkpoint.py`
- `scripts/scrub_prompts.py`
- `scripts/run_efficiency_chain.sh`

#### Dataset files

- `dataset/pickscore/train.txt`
- `dataset/pickscore/test.txt`
- `dataset/pickscore_clean/train.txt`
- `dataset/pickscore_clean/test.txt`
- `dataset/pickscore_clean/scrub_report.json`
- `dataset/smoke/train.txt`
- `dataset/smoke/test.txt`

#### Inference and Table 3 evaluation

- `eval_table3/gen_student_table3.py`
  - Generates from a trained LoRA student.
- `eval_table3/run_gen_8gpu.sh`
  - Multi-GPU generation orchestration.
- `eval_table3/run_score.sh`
  - Scoring orchestration.
- `eval_table3/build_table3_comparison.py`
  - Table comparison/aggregation.
- `eval_table3/preflight.py`
  - Dependency/path checks.
- `eval_table3/student_paths.py`
  - Student checkpoint resolution.
- `eval_table3/README_RUNBOOK.md`
  - Reproduction runbook.
- `eval_table3/test_gen_student_table3.py`

#### Trajectory/efficiency evaluation

- `eval_table3/gen_trajectory.py`
- `eval_table3/trajectory_config.py`
- `eval_table3/build_trajectory_report.py`
- `eval_table3/run_trajectory_eval.sh`

Additional paper scripts live under:

`/kaimm-distill/ysguan/z-image-reward-matrix/final-paper-record/scripts/eval_trajectory`

including:

- `merge_and_plot.py`
- `EFFICIENCY_FIX_RUNBOOK.md`

These records describe a 256-prompt, seed-42, 40-step, 1024×1024 trajectory protocol, which conflicts with the requested 400 prompts × two seeds.

### 6.2 Baseline DiffusionOPD

Separate workspace:

`/home/guanyuanshen/rt-gradient-opd/DiffusionOPD`

Relevant files:

- `scripts/train_sd3_opd.py`
- `config/base.py`
- `config/opd.py`
- `scripts/evaluation.py`
- `scripts/single_node/mopd.sh`
- `scripts/single_node/sopd.sh`
- `scripts/single_node/eval.sh`
- `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`
- `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- `flow_grpo/diffusers_patch/pipeline_with_logprob.py`
- `flow_grpo/diffusers_patch/solver.py`
- `flow_grpo/diffusers_patch/train_dreambooth_lora_sd3.py`
- `flow_grpo/rewards.py`
- `flow_grpo/reward_ckpt_path.py`

This appears to be the OPD baseline rather than the final reward-gradient method. It should only be included if baseline reproduction is part of the public scope.

### 6.3 FLUX RG-OPD candidates

No single FLUX pipeline was found that clearly mirrors the SD3 `reward_grad_teacher.py` implementation and is demonstrably responsible for the paper result.

Candidate stack A: consolidated package config

- `latent-reward-register/configs/rgopd/flux/paper.yaml`
- `latent-reward-register/src/latent_reward_register/rgopd.py`

This supplies configuration and common equations, but not a full runnable trainer.

Candidate stack B: Flow-Factory

`/kaimm-distill/ysguan/z-image-reward-matrix/Flow-Factory`

Core files:

- `src/flow_factory/train.py`
- `src/flow_factory/cli.py`
- `src/flow_factory/trainers/opd/trainer.py`
- `src/flow_factory/trainers/opd/common.py`
- `src/flow_factory/hparams/training_args/opd.py`
- `src/flow_factory/models/flux/flux1.py`
- `src/flow_factory/models/stable_diffusion/sd3_5.py`
- `src/flow_factory/models/z_image/z_image.py`
- `src/flow_factory/rewards/imagereward_reward.py`
- `src/flow_factory/rewards/hpsv2_reward.py`
- `src/flow_factory/rewards/pick_score.py`
- `examples/opd/lora/sd3_5/DiffusionOPD_aligned.yaml`
- `examples/opd/lora/sd3_5/geneval_pickscore_ocr.yaml`
- `inference/example_lora.py`
- `inference/example_full.py`

Flow-Factory supports OPD-like optimization and several backbones, but the surveyed configs do not establish that it is the final reward-register-guided FLUX pipeline.

Candidate stack C: node-specific FLUX experiments

- `node1-flux/`
- `node2-flux/`
- `node3/FLUX_PORT.md`
- `node4/NODE4_FLUX_EXP11_PORT_EXECUTION_REPORT.md`
- `node5/FLUX_OPD_UNIFIED_V3_EXPERIMENT_PLAN.md`

These contain ports, reports, tests, and outputs, but no single canonical final pipeline is identified by the directory structure alone.

### 6.4 D-OPSD and Z-Image self-distillation

Workspace:

`/home/guanyuanshen/rt-gradient-opd/D-OPSD`

Subprojects:

- `z-image-turbo_self-distill-vlm/`
- `flux2-klein_self-distill-edit/`
- `flux2-klein-edit-self-distill-gt-ref/`

Each includes variants of:

- `train_dopsd.py`
- `arguments.py`
- `dataset.py`
- `dataset_validate.py`
- `ema_utils.py`
- `utils.py`
- `configs/default.yaml`
- `configs/z2.json`
- `scripts/train_lora*.sh`

The Z-Image project additionally contains:

- `vlm_utils.py`
- `eval/eval_data_gen.py`
- `eval/run_z.py`
- `eval/run_score.py`
- `eval/README.md`

These are self-distillation/VLM methods and are not obviously the same method as reward-register-guided OPD. They should not be treated as interchangeable without confirmation.

## 7. Capability 5: Test Sets

### 7.1 Table 1

Status: not self-contained.

Referenced pair files:

- `/home/guanyuanshen/iclr-exp-matrix/evaluate-lrm/data/all_table1_pairs_fixed.jsonl`
- `/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/canonical/all_table1_pairs.jsonl`

Required release action:

- Describe the constituent preference datasets.
- Describe image acquisition/reconstruction.
- Record pair ordering, labels, filtering, and any corrected rows.
- Publish row count and checksum for the canonical pair metadata.
- Explain the difference between `fixed` and non-`fixed` files.

### 7.2 Table 2 prompt candidates

#### Formal RGS stress set

`/home/guanyuanshen/RT-guided-sampling/flow_grpo/dataset/rt_stress/ella_dpg_prompts_rt_balanced500_v1`

This is a directory of 500 prompt text files assembled from ELLA/DPG-related sources. It is identified by repository instructions as the official configuration-comparison set.

#### Other evidence

`node1/NODE1_TABLE2_EVAL_REPORT.md` states:

- source: `prompts500.jsonl`
- seeds: `{42, 43}`
- 500 prompts × two seeds = 1,000 images per variant
- a keep-800 subset is used for reported/FID comparisons

Node4 aggregators refer to the retained set as `trim20_exp11` or “keep-800.” This strongly suggests 20% of the 500 prompts were removed, leaving 400 prompts × two seeds, but the exact retained prompt-index file was not identified as a clean source asset during this survey.

#### Other prompt files in RT-guided-sampling

- `exp15_visual_prompts.txt`
- `hf_artifact_prompts.txt`
- `flow_grpo/dataset/drawbench/train.txt`
- `flow_grpo/dataset/drawbench/test.txt`
- `flow_grpo/dataset/geneval/train_metadata.jsonl`
- `flow_grpo/dataset/geneval/test_metadata.jsonl`
- `flow_grpo/dataset/ocr/train.txt`
- `flow_grpo/dataset/ocr/test.txt`
- `flow_grpo/dataset/pickscore/train.txt`
- `flow_grpo/dataset/pickscore/test.txt`

The vendored `ELLA/dpg_bench/prompts/` directory contains a much larger collection and should not be confused with the curated 500-prompt set.

### 7.3 Table 3 prompt candidates

The current SD3 RG-OPD evaluation uses:

- `/home/guanyuanshen/rt-gradient-opd/RG-OPD/dataset/pickscore/test.txt`

Existing generated outputs and the trajectory runbook indicate 256 prompts and seed 42. This does not satisfy the requested 400 prompts and two seeds.

The RGS Table 3 launchers in `/home/guanyuanshen/RT-guided-sampling` may instead use the Table 2-derived prompt protocol. The paper's final Table 3 definition must be checked before selecting a source.

### 7.4 Seed evidence

- Table 2 reports explicitly use seeds 42 and 43.
- SD3/FLUX/Z-Image register training configs generally use seed 42.
- Existing RG-OPD Table 3/trajectory evaluation commonly uses only seed 42.
- Z-Image self-generated training data uses unrelated seeds such as `10007 + k`, explicitly disjoint from evaluation seeds `{42,43}`.

The requested two evaluation seeds are probably 42 and 43, but Table 3 currently contradicts that assumption.

## 8. Checkpoints and External Assets

### Base models

- SD3:
  - Internal: `/kaimm-distill/ysguan/stable-diffusion-3-medium-diffusers`
  - Public: `stabilityai/stable-diffusion-3-medium-diffusers`
- FLUX:
  - Internal: `/kaimm-distill/ysguan/ckpt_dev`
  - Public: `black-forest-labs/FLUX.1-dev`
- Z-Image:
  - Internal: `/kaimm-distill/ysguan/Z-Image`
  - Public config: `Tongyi-MAI/Z-Image-Turbo`

### Reward-register checkpoints

Observed conventions include:

- `reward_token_step_*.pt`
- `reward_token_epoch_*.pt`
- `reward_token_final_ema.pt`
- Raw and EMA variants.

Known explicit reference:

- `/kaimm-distill/ysguan/z-image-reward-matrix/node4/outputs/reward_register_sd3_singlehead_ir/checkpoints/reward_token_final_ema.pt`

Other checkpoint families are under node output directories such as:

- `node4/outputs/reward_register_fixv2/`
- `node4/outputs/reward_register_v3/`
- `node4/outputs/reward_register_flux_exp11/`
- `node4/outputs/reward_register_sd3_singlehead_ir/`
- `node4/outputs/reward_register_flux_singlehead_ir/`

The proposed release format instead requires a checkpoint directory containing:

- `register.safetensors`
- `config.yaml`
- `manifest.json`

Conversion and parity verification are therefore required.

### Training data and caches

The training manifests point to internal assets such as:

- HPDv3 source images under `/kaimm-distill/ysguan/reward-token-image-dataset/HPDv3/images`
- SD3 cached latents and prompt embeddings.
- FLUX cached prompt embeddings.
- Z-Image cached latents and Qwen3 prompt embeddings.
- Teacher scores from HPSv3, ImageReward, and PickScore.

Portable release manifests cannot retain these absolute paths.

## 9. External Dependencies

### Core runtime

- Python 3.10 or later in the newer workspaces.
- PyTorch.
- torchvision.
- Diffusers.
- Transformers.
- Accelerate.
- PEFT.
- Safetensors.
- NumPy.
- Pillow.
- PyYAML.

### Configuration/training utilities

- OmegaConf/Hydra-style YAML in register training.
- `ml_collections` in Flow-GRPO/RG-OPD-style configs.
- DeepSpeed or FSDP launch configurations in several workspaces.
- WandB.
- SwanLab.

### Reward/evaluation dependencies

- HPSv2.
- HPSv3.
- ImageReward.
- PickScore.
- CLIP.
- MUSIQ.
- CLIP-IQA.
- FID tooling.
- Geneval and OCR-specific dependencies in OPD baselines.

### Vendored dependencies

- Complete Diffusers trees appear in multiple workspaces.
- HPSv3 is vendored under `Reward-Token-Image-0420/HPSv3`.
- `RT-guided-sampling` includes several unrelated third-party projects and benchmarks.
- `Flow-Factory` contains its own model and reward framework.

The release should pin a single Diffusers revision. The consolidated configs mention revision `c8c84018e`, but historical code should be checked against that exact source before declaring parity.

## 10. Duplicated Implementations

### Reward-register model copies

At least three copies/derivatives exist:

1. `/home/guanyuanshen/Reward-Token-Image-0420/src/sd3_reward`
2. `/kaimm-distill/ysguan/z-image-reward-matrix/node4/src/sd3_reward`
3. `/kaimm-distill/ysguan/z-image-reward-matrix/latent-reward-register/src/latent_reward_register/implementations`

The first is SD3-centric. The second extends it for FLUX and Z-Image. The third is a cleaned release-oriented copy.

### OPD implementations

- `rt-gradient-opd/DiffusionOPD`: baseline OPD stack.
- `rt-gradient-opd/RG-OPD`: reward-gradient OPD stack.
- `z-image-reward-matrix/Flow-Factory`: generalized framework with OPD trainer.
- `rt-gradient-opd/D-OPSD`: self-distillation variants.

These are algorithmically different despite overlapping names.

### RGS implementations

- SD3-oriented experiment stack in `RT-guided-sampling/flow_grpo`.
- FLUX and Z-Image direct samplers in `node4/src/scripts/eval`.
- Unified common guidance abstraction in `latent-reward-register`.

### Evaluation scripts

Table 1 and Table 2 generation/scoring scripts are copied or adapted across `node1`, `node1-flux`, `node4`, and top-level `RT-guided-sampling` launchers.

### Vendored Diffusers

Full Diffusers source trees appear under multiple workspaces. Only the exact custom patches or a pinned upstream revision should survive release consolidation.

## 11. Conflicting Implementations or Protocols

### Register architecture naming

- Historical SD3 experiments use names such as `L4snap`, `fulltoken`, `noattn2`, and `stop4`.
- Consolidated SD3 paper config specifies feature layers `[4, 8, 12]`.
- FLUX uses `[9, 19, 28]`.
- Z-Image uses `[5, 10, 15]`.
- Some checkpoints have two heads; others have three heads or single-head ImageReward.

The exact paper checkpoint, not directory naming, must determine the release architecture.

### RGS schedules

- Consolidated configs declare 42 steps for SD3 and 40 for FLUX.
- Historical scripts contain multiple pulse schedules, head combinations, and strength sweeps.
- Z-Image has several `fixv2`, `v3`, and diagnostic schedules.

### Table 2 size

- Formal stress source: 500 prompts.
- Generation: 500 prompts × two seeds = 1,000 samples.
- Reported subset: keep 800 samples, apparently corresponding to 400 prompts × two seeds.
- Requested release: exact 400 prompts × two seeds.

The exact retention/index rule is missing from the clean release candidate.

### Table 3 size

- Existing RG-OPD trajectory protocol: 256 prompts × seed 42.
- Requested release: 400 prompts × two seeds.

### Checkpoint formats

- Historical: Python `.pt` payloads with embedded configuration and raw/EMA variants.
- Proposed release: Safetensors plus YAML and JSON manifest.

### Backbone time conventions

- Z-Image: native `t = 1 - sigma` convention.
- FLUX: scheduler timestep normalization.
- SD3: separate scheduler/CFG handling.

These are deliberate differences, not refactoring noise.

### Z-Image scope

- Training and preference scoring definitely exist.
- RGS implementation exists.
- Consolidated README says RGS/RG-OPD release covers only SD3 and FLUX.
- No clearly final reward-register-guided Z-Image OPD pipeline was identified.

## 12. Missing Components

### Capability 1: register training

- No single public training CLI spanning all three backbones.
- No portable public training manifest or data acquisition recipe.
- No definitive mapping from paper rows to checkpoint/config combinations.
- No released checkpoints/checksums.
- No verified GPU parity test for the consolidated adapters.

### Capability 2: preference scoring/Table 1

- Table 1 pair metadata absent from the target release.
- Dataset reconstruction documentation absent.
- Conflicting pair-list filenames.
- No unified command producing the final Table 1 results.

### Capability 3: RGS

- Consolidated package has math but no complete generation command.
- SD3 implementation is embedded in a large unrelated repository.
- FLUX/Z-Image implementation is embedded in an experiment node.
- Canonical checkpoint and schedule per paper result remain unconfirmed.
- Z-Image inclusion is unresolved.

### Capability 4: RG-OPD

- No confirmed canonical FLUX trainer matching the paper.
- No confirmed Z-Image RG-OPD method, if Z-Image is intended in this capability.
- No unified student inference/evaluation command.
- Existing SD3 Table 3 protocol conflicts with requested prompt/seed count.
- Baseline OPD, RG-OPD, Flow-Factory OPD, and D-OPSD need explicit separation.

### Capability 5: test sets

- Exact Table 2 400-prompt retained list is not identified.
- Exact Table 3 400-prompt list is not identified.
- Second Table 3 seed is not established.
- Table 1 reconstruction recipe is missing.

### Documentation/environment

- Per-module READMEs do not yet exist in one repository.
- Historical commands use internal absolute paths.
- Scoring environment is not covered by one lockfile.
- Model access/license requirements need explicit documentation.
- Benchmark data licenses need review.

## 13. Questions Requiring Clarification

1. Which exact checkpoint produced each final Table 1 row for SD3, FLUX, and Z-Image?
2. Is `all_table1_pairs_fixed.jsonl` or `all_table1_pairs.jsonl` authoritative?
3. What changed in the `fixed` Table 1 pair list?
4. Are Tables 2 and 3 based on the same 400 prompts?
5. Is the final 400-prompt set the retained subset of `rt_balanced500_v1`?
6. Where is the authoritative retention/index list that turns 500 prompts into 400?
7. Are the required two seeds exactly 42 and 43 for both tables?
8. Why does the current RG-OPD Table 3 trajectory protocol use 256 prompts and seed 42?
9. Which SD3 register architecture is final: exp14, exp15, exp16, fixv2, v3, or a single-head model?
10. Which FLUX register checkpoint/config is final?
11. Which Z-Image register checkpoint/config is final?
12. Should Z-Image RGS be released despite the consolidated README excluding it?
13. Is Z-Image expected for RG-OPD, or only register training/scoring?
14. Is the canonical FLUX RG-OPD pipeline based on `RG-OPD`, `Flow-Factory`, or a node-specific port?
15. Should baseline `DiffusionOPD` be included for paper comparison?
16. Should D-OPSD be excluded as a different method?
17. Are HPSv3, ImageReward, PickScore, MUSIQ, CLIP-IQA, and FID all required for official reproduction?
18. Can the Table 1 metadata and exact Table 2/3 prompts be redistributed under their source licenses?
19. Where will released register and LoRA checkpoints be hosted?
20. Is the existing `latent-reward-register` directory the intended destination repository or only a consolidation prototype?

## 14. Recommended Canonical Source Choices Pending Clarification

These are inventory conclusions, not proposed code changes:

- Use `Reward-Token-Image-0420` as the provenance source for SD3 register training.
- Use `node4/src` as the provenance source for FLUX and Z-Image register adapters and evaluators.
- Use `RT-guided-sampling/flow_grpo` plus its top-level paper launchers as the provenance source for SD3 RGS.
- Use `node4/src/scripts/eval/flux_rt_guided_sampling.py` as the provenance source for FLUX RGS.
- Use `node4/src/scripts/eval/zimage_rt_guided_sampling.py` only if Z-Image RGS is in scope.
- Use `rt-gradient-opd/RG-OPD` as the provenance source for SD3 RG-OPD.
- Treat `DiffusionOPD` as a baseline, not the final RG-OPD implementation.
- Do not select a FLUX RG-OPD source until the final paper run/checkpoint is identified.
- Treat the existing `latent-reward-register` package as a useful interface/design prototype, not yet as proof of end-to-end reproducibility.

## 15. Final Readiness Assessment

The project has enough implementation material to build the release, but the release design should not begin until four provenance questions are answered:

1. the exact final checkpoints and their embedded configurations;
2. the authoritative Table 1 pair list;
3. the exact retained 400 prompts and two seeds for Tables 2 and 3;
4. the canonical FLUX RG-OPD implementation.

Those choices determine which of the duplicated implementations is authoritative and prevent the release from documenting a clean but historically incorrect reproduction path.
