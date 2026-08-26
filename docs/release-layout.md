# Release layout

```text
configs/
  register/{sd3,flux,zimage}/       register presets
  rgs/{sd3,flux}/                    sampling presets
  rgopd/{sd3,flux}/                  OPD presets
src/latent_reward_register/
  backbones/                        adapter interfaces and registry
  implementations/                  consolidated research model ports
  training.py                       shared register training loop
  sampling.py                       shared RGS loop
  rgopd.py                          shared RG-OPD target/loss/trainer
  preference.py                     pairwise preference evaluator
  teacher.py                        reward-gradient teacher (RGS + RG-OPD)
  gradmode via implementations/     latent-gradient scoring mode
  smoke.py                          asset-free integration smoke
  workflows.py                      config validation and release plans
  checkpoint.py                     portable checkpoint contract
  data.py                           portable manifest schema
tests/                               CPU contract tests
docs/                                module and reproduction documentation
```

Run the repository-wide validation command:

```bash
pip install -e '.[dev]'
lrr validate-release --root .
```

The validator checks all seven paper workflow presets and reports checkpoints,
training data, and paper test sets as deferred inputs.

The validator confirms configuration completeness, not model execution:
`valid` refers only to checked-in configuration validity.

Three checks, in increasing strength:

1. `lrr validate-release --root .` — configs parse and satisfy their contracts.
2. `lrr smoke-release` — every algorithm path executes on a synthetic backbone.
3. `lrr build-register --config ...` — a real model is constructed from a real
   config against real weights, and its trainable parameter count is reported.

Only the third can catch a config key the model does not accept or a shape
error in the group plumbing.
