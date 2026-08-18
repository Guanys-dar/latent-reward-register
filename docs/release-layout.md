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
  rgopd.py                          shared RG-OPD target/loss
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
