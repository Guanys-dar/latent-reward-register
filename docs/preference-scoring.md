# Preference scoring

Preference scoring uses the same register object as training and guidance. It
must never load a separate reward implementation merely because a benchmark
uses a different metric name: the configured register heads are the source of
truth.

The public package currently provides the model-level API:

```python
scores = register.score(latents, condition, sigma)
```

and gradient scoring:

```python
output = register.score_and_grad(
    latents, condition, sigma, heads=("preference", "imagereward")
)
```

Pairwise evaluation is executable through
`latent_reward_register.evaluate_preference_pairs`. It accepts iterable
`PreferencePairBatch` objects, treats label `0` as the first sample and label
`1` as the second, counts exact score ties separately, and returns aggregate
accuracy metrics.

## Table 1

The released pair file holds 54170 pairs across four benchmarks:

| Dataset | Pairs | Subset |
| --- | --- | --- |
| GenAI-Bench | 18099 | `train1600_rating_pairs` |
| HPDv2 | 15300 | `test_pairs` |
| HPDv3 | 14372 | `test_clean` |
| ImageReward | 6399 | `test_rank_pairs` |

Each record carries `dataset`, `image1`, `image2`, `preferred` (0 or 1),
`prompt`, and provenance under `source`. Image paths are relative to a
per-dataset root: the images belong to those four projects and are not
redistributed here.

```bash
python scripts/fetch_table1_images.py --pairs all_table1_pairs.jsonl --out ./table1_images
python scripts/fetch_table1_images.py --pairs all_table1_pairs.jsonl --out ./table1_images --verify
```

Run `--verify` before trusting a reported accuracy: it exits non-zero if any
referenced image is absent, so a partially downloaded root cannot silently
produce a number covering only part of the benchmark.

Two pair files exist in the research history with identical labels but different
ImageReward image paths on 6399 rows. The released file derives from
`all_table1_pairs_fixed.jsonl`; verify the checksum in `TABLE1_MANIFEST.json`.

The evaluator itself is covered by CPU tests and by `lrr smoke-release`.
