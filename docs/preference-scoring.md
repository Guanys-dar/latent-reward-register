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

Table 1 pair metadata, image assets, and released register checkpoints are
intentionally deferred. Once supplied, the evaluator should preserve pair
ordering, prompt groups, and the checkpoint-embedded configuration.

Until those assets are published, an end-to-end Table 1 file loader and runner
remain unavailable. The evaluator itself is covered by CPU tests and the
asset-free release smoke command.
