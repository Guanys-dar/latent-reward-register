# Checkpoint format

Each checkpoint is a directory:

```text
checkpoint/
├── register.safetensors
├── config.yaml
└── manifest.json
```

`manifest.json` is the compatibility interface. It records format version,
backbone and revision, adapter, head order, feature layers, and noise
convention. `config.yaml` records training and architecture settings.

Legacy experiment checkpoints are PyTorch dictionaries containing `config`
and `model`. `read_legacy_checkpoint` accepts only that self-describing shape;
anonymous state dictionaries are intentionally rejected because their layer,
head-order, and timestep semantics cannot be recovered safely.

