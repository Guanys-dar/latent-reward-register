# Data format

Register training consumes JSON Lines records. Paths are relative to the
manifest directory or supplied data root.

```json
{
  "sample_id": "group-0001-image-0",
  "prompt": "a lighthouse during a storm",
  "latent_path": "latents/group-0001-image-0.safetensors",
  "prompt_embed_path": "prompt_embeds/group-0001.safetensors",
  "pooled_prompt_embed_path": "pooled_prompt_embeds/group-0001.safetensors",
  "rewards": {"preference": 1.42, "imagereward": 0.87}
}
```

Pair construction is kept separate from cached tensors. Pair indices identify
two `sample_id` values from the same prompt group and which item is preferred
for each reward head. No machine-specific absolute path is valid in a release
manifest.

SD3 uses its native VAE and CLIP/T5 conditioning. FLUX and Z-Image use their
native prompt encoders; although the studied FLUX and Z-Image VAEs were found
to be byte-identical, the manifest still records the owning backbone so this
is never assumed for an arbitrary checkpoint.

