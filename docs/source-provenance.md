# Source provenance and release decisions

Sources are named by repository, never by machine path, so the table stays valid
outside the machine the research ran on.

| Released subsystem | Authoritative source | Notes |
| --- | --- | --- |
| SD3 register training | `Reward-Token-Image-0420` → `src/sd3_reward` | Shared data, losses, EMA, training engine |
| FLUX and Z-Image registers | `z-image-reward-matrix` → `node5/src/sd3_reward` | Adds FLUX/Z-Image backbones; package name is historical |
| SD3 RGS | `RT-guided-sampling` → `flow_grpo/flow_grpo/diffusers_patch` | `sd3_pipeline_with_rt_guidance.py`, `lrm_reward_backend.py` |
| SD3 RG-OPD | `rt-gradient-opd` → `RG-OPD` | `scripts/train_sd3_rgopd.py`, `rg_opd/` |
| FLUX RG-OPD | `z-image-reward-matrix` → `node5/src` | `scripts/train_flux_rgopd.py`, `flux_opd/` |

## What changed in the ported implementations

The files under `src/latent_reward_register/implementations/` are ports, not
copies. Three changes were made and nothing else; the architecture and numerics
are untouched, which is what keeps the published checkpoints loadable.

| Change | Why |
| --- | --- |
| Added a module docstring to each file | Names the source and warns against refactoring checkpoint-bearing code |
| `torch.no_grad()` → `frozen_trunk_context()` in the frozen-trunk block loops | The trunk must stay frozen in training but be differentiable for RGS and RG-OPD; see `docs/latent-gradients.md` |
| `import zimage_common` → relative import | The research tree relied on a flat `sys.path`; the release is a package |

The pos-embed helper in `latent_reward_grid.py` keeps its literal
`torch.no_grad()`: it slices a constant table and is not on the latent path.

`docs/IMPLEMENTATION_SHA256SUMS` pins the released bytes of all ten vendored
modules, and `tests/test_implementation_checksums.py` verifies both the digests
and that the manifest covers every vendored file. It does not prove the port was
faithful to the originals — those live only in the research workspace — it proves
nothing has drifted since release.

## Layer tap rule

All three backbones follow one rule rather than per-model magic numbers: taps at
1/6, 1/3, and 1/2 of transformer depth, with the register stopping at half depth.

| Backbone | Depth | Taps | Heads |
| --- | --- | --- | --- |
| SD3 | 24 | 4, 8, 12 | preference, pickscore, imagereward |
| FLUX | 57 (19 double + 38 single) | 9, 19, 28 | preference, imagereward |
| Z-Image | 30 | 5, 10, 15 | preference, imagereward |

Tap 19 on FLUX is the last double-stream block.

## Resolved discrepancies

- Release configurations follow the configuration embedded in the checkpoint
  that produced the result, not a historical experiment nickname.
- The release baseline is **exp11** for SD3: `[4, 8, 12]` taps, 12 register
  layers, `skip_attn2=true`, three heads at equal weight. Later exp15/exp16
  variants (single `[4]` tap, truncated trunk) are exploratory, not the release.
- The paper's register-training table lists `{2, 5, 8}` in the **FLUX** column,
  not the SD3 one. It does not match the shipped FLUX config either, which taps
  `[9, 19, 28]` of 57 blocks. The configs in this repository are authoritative;
  see `RELEASE_TODO.md` section 3.8 for the exact paper lines to correct.
- SD3 trains three heads; FLUX and Z-Image unified-v3 train two. Head count is
  per-backbone and must not be unified.
- RG-OPD teacher registers differ per backbone: SD3 uses the **exp11** EMA
  register; FLUX uses the **unified-v3** EMA final. The FLUX RG-OPD config
  carries exp11 as a legacy default and overrides it to unified-v3 for the
  shipped runs, so the config default alone is misleading.
- Z-Image is released for register training and preference scoring only. No
  downstream evaluation or RG-OPD consumes the Z-Image register.
- FLUX RG-OPD checkpoint selection is argmax HPSv3 subject to
  CLIP-IQA >= 0.649, over 100 held-out screen prompts disjoint from the test
  set. Chosen epochs: HPS 150, ImageReward 60, TwoHead 150. An older
  selection file from the exp11-teacher generation records different epochs and
  must not be cited.
- Reward-gradient guidance ships the sigma-banded schedule only. The gradient
  spectral filters (including `lp2zm`) were explored but not adopted.

## Excluded material

Training logs, generated images, local model paths, cached credentials, vendored
HPSv3, complete diffusers clones, baseline implementations, and failed or
superseded experiments are excluded. See `release/EXCLUDE.txt`.
