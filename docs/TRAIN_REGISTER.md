# Register training

The same training loop supports SD3-Medium and FLUX.1-dev. Backbone-specific
feature extraction stays in the model implementation.

## Input data

Training consumes JSON Lines with one group per prompt. Each group contains
cached prompt embeddings and at least two cached image latents with teacher
scores. The example below is expanded for readability; store each complete
group object on one line in the JSONL file:

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
    },
    {
      "sample_id": "sample-2",
      "latent_x0_path": "latents/sample-2.pt",
      "teacher_score_zscore": -0.3,
      "pickscore": 0.2,
      "imagereward_score": 0.1
    }
  ]
}
```

Paths are resolved relative to `data_root` in `configs/local.yaml`. Training
does not run the VAE or text encoder. Groups shorter than `--group-size` are
skipped rather than padded.

The paper checkpoints use DiNa pairs, so the release scripts set
`--training-parquet` and `--multihead-manifest`. The scripts read these from
`PARQUET` and `MULTIHEAD_MANIFEST`; neither path is stored in the paper preset.

`head_names` and `score_keys` in the register config are parallel lists; keep
their ordering aligned. `vis_h` and `vis_w` are transformer token dimensions,
not image pixels.

## Commands

```bash
MANIFEST=/path/to/groups.jsonl PARQUET='/path/to/pairs/*.parquet' \
  MULTIHEAD_MANIFEST=/path/to/scored-groups.jsonl bash scripts/train_register_sd3.sh
MANIFEST=/path/to/groups.jsonl PARQUET=/path/to/pairs.parquet \
  MULTIHEAD_MANIFEST=/path/to/scored-groups.jsonl bash scripts/train_register_flux.sh
```

Use `--max-batches 4` for a short real-model check. Both presets set
`drop_last: true`, so the input must contain at least one full batch: eight
pairs for SD3 or four for FLUX. For a smaller check, pass `--batch-size 1`;
each manifest group still needs at least two image records.

Checkpoints include the register architecture config needed to reconstruct the
model. Published or pretrained checkpoints are external assets.
