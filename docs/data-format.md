# Data format

Register training consumes a **group** manifest in JSON Lines: one record per
prompt, each holding a ranked list of images. Groups are the unit of the loss —
`dina_thurstone` compares all within-group pairs — so a flat list of preference
pairs is not an accepted substitute.

One record (two of its `image_records` shown):

```json
{
  "group_id": "5b441007c3bdea45",
  "prompt": "The image depicts bubbles with an abstract and surreal style.",
  "prompt_embeds_path": "sd3_cache/prompt_embeds/5b441007c3bdea45.prompt_embeds.pt",
  "pooled_prompt_embeds_path": "sd3_cache/pooled_prompt_embeds/5b441007c3bdea45.pooled_prompt_embeds.pt",
  "image_records": [
    {
      "sample_id": "09bd383a202a055e5548",
      "image_path": "images/c3de51d9.jpg",
      "latent_x0_path": "latents/09bd383a202a055e5548.latent_x0.pt",
      "teacher_score_raw": -10.301783561706543,
      "teacher_score_zscore": -2.3107109160184747,
      "teacher_score_zscore_clipped": -2.3107109160184747,
      "pickscore": 0.7274289131164551,
      "imagereward_score": -1.658400297164917
    },
    {
      "sample_id": "4a2116cce34e87904a18",
      "image_path": "images/7d4cc31d.jpg",
      "latent_x0_path": "latents/4a2116cce34e87904a18.latent_x0.pt",
      "teacher_score_raw": 0.4218,
      "teacher_score_zscore": 0.01055365014334843,
      "teacher_score_zscore_clipped": 0.01055365014334843,
      "pickscore": 0.8063517212867737,
      "imagereward_score": 0.26207250356674194
    }
  ]
}
```

All paths are relative to the data root supplied at launch. No machine-specific
absolute path is valid in a released manifest.

## Cached tensors, not images

Training reads `latent_x0_path` and the two prompt-embedding paths. It never
loads `image_path`, and never runs a VAE or text encoder — those are done once
during preparation. `image_path` is retained for provenance and for re-scoring.

## Score keys map to heads by position

`score_keys` and `head_names` are parallel lists; the loss aligns
`targets[:, h]` with `head_names[h]`. Reordering one without the other
mistrains silently rather than erroring.

| Head | Score key | Teacher |
| --- | --- | --- |
| `preference` | `teacher_score_zscore` | HPSv3 |
| `pickscore` | `pickscore` | PickScore v1 |
| `imagereward` | `imagereward_score` | ImageReward |

SD3 (exp11) trains all three. FLUX and Z-Image unified-v3 train `preference`
and `imagereward` only.

`teacher_score_raw` is the teacher's native scale; `teacher_score_zscore` is
normalized within each prompt group and is what the `preference` head fits.

## Backbone-specific conditioning

SD3 uses CLIP/T5 conditioning and therefore both `prompt_embeds_path` and
`pooled_prompt_embeds_path`. FLUX and Z-Image use their native prompt encoders.
Although the studied FLUX and Z-Image VAEs were found to be byte-identical, each
manifest still records its owning backbone so this is never assumed.

## Table 1 preference test set

The released pair file is `all_table1_pairs_fixed.jsonl` (54170 pairs). A
second file, `all_table1_pairs.jsonl`, carries identical `preferred` labels but
differs in the image paths of 6399 pairs; it is superseded and must not be used
for reported numbers. Verify the released file by row count and checksum before
comparing against published accuracies.
