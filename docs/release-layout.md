# Release layout

```text
configs/
  register/{sd3,flux,zimage}/       register presets
  rgs/{sd3,flux}/                    sampling presets
  rgopd/{sd3,flux}/                  OPD presets
  local.yaml.example                 machine paths; copy, never commit the copy
scripts/                             one launch script per task
src/latent_reward_register/
  cli.py                            the lrr command
  runtime.py                        real pipelines and the three task runners
  local_config.py                   machine paths, kept out of every config
  dataset.py                        group manifest -> training batches
  backbones/                        model-backed register builders and registry
  implementations/                  consolidated research model ports
  training.py                       shared register training loop
  sampling.py                       shared RGS loop
  rgopd.py                          RG-OPD target/loss + off-policy trainer
  rollout.py                        on-policy RG-OPD driver (the paper path)
  flowmatch.py                      Euler transitions and the sigma schedule
  velocity.py                       SD3/FLUX velocity models and LoRA student
  table1.py                         Table 1 preference-accuracy evaluator
  preference.py                     pairwise preference evaluator
  teacher.py                        reward-gradient teacher (RGS + RG-OPD)
  gradmode via implementations/     latent-gradient scoring mode
  register.py                       weight-free reference register (smoke only)
  smoke.py                          asset-free integration smoke
  workflows.py                      config validation and release plans
  checkpoint.py                     portable checkpoint contract
  data.py                           portable manifest schema
tests/                               CPU contract tests
docs/                                module and reproduction documentation
```

Only `runtime.py` imports diffusers pipelines. Everything above it takes its
backbone as a callable, which is why the algorithm layer is testable on CPU.


Run the repository-wide validation command:

```bash
pip install -e '.[dev]'
lrr validate-release --root .
```

The validator checks all seven paper workflow presets and lists the assets a run
still needs from outside the repository.

`valid` refers to checked-in configuration validity: it confirms every preset
parses and satisfies its contract, not that a model ran.

Three checks, in increasing strength:

1. `lrr validate-release --root .` — configs parse and satisfy their contracts.
2. `lrr smoke-release` — every algorithm path executes on a synthetic backbone.
3. `lrr build-register --config ...` — a real model is constructed from a real
   config against real weights, and its trainable parameter count is reported.
4. `lrr sample --config ... --register-checkpoint ...` — a full guided
   trajectory, which is the only check that exercises the sampler and the
   register together.

The first two need nothing downloaded. Only the third can catch a config key the
model does not accept or a shape error in the group plumbing, and only the fourth
proves guidance actually fires on the scheduled steps.
