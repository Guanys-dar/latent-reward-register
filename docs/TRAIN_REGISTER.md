# Register training

The same training loop supports SD3-Medium, FLUX.1-dev, and experimental
Z-Image registers. Backbone-specific feature extraction stays in the model
implementation.

## Input data

Training consumes JSON Lines with one group per prompt. Each group contains
cached prompt embeddings and a ranked list of cached image latents:

```json
{
  "group_id": "example",
  "prompt": "a prompt",
  "prompt_embeds_path": "embeddings/example.pt",
  "pooled_prompt_embeds_path": "embeddings/example_pooled.pt",
  "image_records": [
    {
      "sample_id": "sample-1",
      "latent_x0_path": "latents/sample-1.pt",
      "teacher_score_zscore": 1.2,
      "pickscore": 0.8,
      "imagereward_score": 0.4
    }
  ]
}
```

Paths are resolved relative to `data_root` in `configs/local.yaml`. Training
does not run the VAE or text encoder. Groups shorter than `--group-size` are
skipped rather than padded.

The paper checkpoints use DiNa pairs, so the release scripts set
`--group-size 2`.

`head_names` and `score_keys` in the register config are parallel lists; keep
their ordering aligned. `vis_h` and `vis_w` are transformer token dimensions,
not image pixels.

## Commands

```bash
MANIFEST=/path/to/groups.jsonl bash scripts/train_register_sd3.sh
MANIFEST=/path/to/groups.jsonl bash scripts/train_register_flux.sh
```

Use `--max-batches 4` for a short real-model check. Z-Image uses
`configs/register/zimage/paper.yaml` directly and is provided as experimental
register support only.

Checkpoints include the register architecture config needed to reconstruct the
model. Published or pretrained checkpoints are external assets.
