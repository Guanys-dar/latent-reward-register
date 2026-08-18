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

The validator confirms configuration completeness, not model execution. Run
`lrr smoke-release` to execute all shared algorithm paths. Model-backed paper
workflows remain blocked on the unported diffusers compatibility adapters.
Its `valid` field refers only to checked-in configuration validity;
`release_ready` and `model_execution_ready` remain false until those blockers
are implemented and tested.
