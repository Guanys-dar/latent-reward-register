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

Table 1 pair metadata, image assets, and released register checkpoints are
intentionally deferred. Once supplied, the evaluator should preserve pair
ordering, prompt groups, and the checkpoint-embedded configuration.

Until those assets are published, the asset-backed Table 1 runner remains
disabled. The model-level API remains available for integration once the
checkpoint and Table 1 pair manifest are published.
