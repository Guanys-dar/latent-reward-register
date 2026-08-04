# Source provenance and release decisions

No source workspace is modified during consolidation. Before a public release,
the copied implementation files and checkpoints must be accompanied by their
final SHA-256 hashes.

| Released subsystem | Authoritative source | Notes |
| --- | --- | --- |
| SD3 register training | `/home/guanyuanshen/Reward-Token-Image-0420/src/sd3_reward` | Shared data, losses, EMA, training engine; exp11/exp15 research variants |
| FLUX and Z-Image registers | `z-image-reward-matrix/node4/src/sd3_reward` | Adds backbone and register ports; package name is historical |
| SD3 RGS | `/home/guanyuanshen/RT-guided-sampling/flow_grpo` | Contains the original register backend and SD3 sampler patch |
| FLUX RGS | `z-image-reward-matrix/node4/src/scripts/eval` | Final FLUX Table-2 schedule and checkpoint loader |
| SD3 RG-OPD | `/home/guanyuanshen/rt-gradient-opd/RG-OPD` | On-policy trainer and reward-gradient teacher |
| FLUX RG-OPD | `z-image-reward-matrix/node5/src` | FLUX trainer port and final configs |

## Resolved discrepancies

- Release configurations follow the configuration embedded in the checkpoint
  that produced the result, not a historical experiment nickname.
- The deployed FLUX checkpoint uses feature layers `[9, 19, 28]`; an existing
  paper draft instead states `{2, 5, 8}`.
- SD3 experimental records refer to both exp11 and exp15. Published checkpoint
  manifests must identify the concrete backend used by every RGS and RG-OPD
  result rather than aliasing both through a single method name.
- Z-Image is released for register training and evaluation only. Its exploratory
  guided-sampling scripts are not part of the paper-supported RGS interface.

## Excluded material

Training logs, generated images, local model paths, cached credentials, vendored
HPSv3, complete diffusers clones, baseline implementations, and failed or
superseded experiments are excluded.

