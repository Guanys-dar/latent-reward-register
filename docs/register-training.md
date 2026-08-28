# Register training

The register-training seam is shared by SD3, FLUX, and Z-Image. Backbone-specific
feature extraction stays inside the model implementation; the training loop
consumes group batches and writes the release checkpoint contract.

## Configurations

- `configs/register/sd3/paper.yaml`
- `configs/register/flux/paper.yaml`
- `configs/register/zimage/paper.yaml`

Validate a configuration without loading a model, checkpoint, or dataset:

```bash
lrr plan configs/register/sd3/paper.yaml
lrr plan configs/register/flux/paper.yaml
lrr plan configs/register/zimage/paper.yaml
```

The current release deliberately leaves the dataset manifest and output
directory as command inputs. No internal cache, image store, or checkpoint is
required to install or validate the package.

## Building a register against real weights

Config validation and `--dry-run` never construct a model, so they cannot catch
a config key the model does not accept or a shape error in the group plumbing.
This command does, and it is the cheapest check that the model path works:

```bash
lrr build-register --config configs/register/sd3/paper.yaml \
    --model-path /path/to/stable-diffusion-3-medium-diffusers \
    --precision fp32 --local-files-only
```

It reports the model class, heads, and trainable/total parameter counts. For
SD3 the trainable count is 147,599,619, matching the exp11 training log; a
different number means the architecture has drifted from the paper baseline.

`vis_h`/`vis_w` are token counts, not pixels. The pooling layer asserts
`vis_h * vis_w` tokens, so the latent spatial size must match the configured
grid (SD3: 1024x1024 pixels gives a 128x128 latent and 64x64 tokens). A smaller
input raises rather than silently rescaling.

## Python seam

- `latent_reward_register.training.GroupBatch`
- `latent_reward_register.training.TrainConfig`
- `latent_reward_register.training.train_register`
- `latent_reward_register.data.read_group_manifest`
- `latent_reward_register.checkpoint.save_register_checkpoint`

`train_register` needs only `score_groups`, so it drives either register class.
For a real run that is `CheckpointRewardRegister`, whose model owns latent
packing, conditioning, and transformer traversal. It also takes
`register_config`: the architecture record written into the checkpoint's
`config.yaml`, without which the checkpoint cannot be rebuilt.
