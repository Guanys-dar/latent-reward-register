# Register training

The register-training seam is shared by SD3, FLUX, and Z-Image. Backbone-specific
feature extraction stays behind the adapter/model implementation; the training
loop consumes pair batches and writes the release checkpoint contract.

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

## Python seam

- `latent_reward_register.training.PairBatch`
- `latent_reward_register.training.TrainConfig`
- `latent_reward_register.training.train_register`
- `latent_reward_register.data.read_manifest`
- `latent_reward_register.checkpoint.save_register_checkpoint`

The trainable model is expected to implement `RewardRegister.score`; the
backbone adapter owns latent packing, conditioning, and transformer traversal.
