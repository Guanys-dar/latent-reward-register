# Release TODO

Working document. Delete before the public export (it is listed in
`release/EXCLUDE.txt`).

Repository roles: this tree is the **private working repo and the single source
of truth**. The public repo is produced from it by `release/export.py`, never
maintained alongside it. Machine-specific values live only in
`configs/local.yaml`, which is excluded from export, so every other file stays
byte-identical between the two.

---

## 1. Commands for you to run

### 1.1 Rotate the two exposed credentials — do this first

The **HuggingFace token** was pasted into a chat transcript. Treat it as
compromised regardless of what else happens; it is not stored in this repo.
Rotate at <https://huggingface.co/settings/tokens>.

The **SwanLab key** has been removed from all 9 files that carried it (see
§2.4), but it existed in plaintext on disk and may persist in git history,
shell history, and training logs. Rotate it too.

```bash
# Verify no key values remain in the research repo
cd ~/Reward-Token-Image-0420
grep -rn 'SWANLAB_API_KEY' main.sh exp*.sh scripts/run/*.sh

# Expected on every line:
#   export SWANLAB_API_KEY="${SWANLAB_API_KEY:?set SWANLAB_API_KEY in your environment}"

# Set it in your shell instead (add to ~/.bashrc, do not commit)
export SWANLAB_API_KEY=<new-key>
```

Check whether the old key reached that repo's git history:

```bash
cd ~/Reward-Token-Image-0420
git log --all -S 'SWANLAB_API_KEY' --oneline | head
```

Any hit means rotation is mandatory, not optional — removing it from the
working tree does not remove it from history.

### 1.2 Drop the author-rewrite backups

Commit authors on `main` are now `Yuanshen Guan <guanys@mail.ustc.edu.cn>`.
The pre-rewrite commits, which carried an internal cluster hostname, survive
only in `refs/original/` and in the reflog. This is irreversible, so it was
left for you:

```bash
cd ~/latent-reward-register-release-work

# Confirm the backups are all that still carry the old identity
git log --all --format='%an <%ae>' | sort -u

git for-each-ref --format='%(refname)' refs/original | xargs -n1 git update-ref -d
git reflog expire --expire=now --all
git gc --prune=now

# Should now print exactly one identity
git log --all --format='%an <%ae>' | sort -u
```

### 1.3 Decide what happens to `origin`

`origin` points at `z-image-reward-matrix/latent-reward-register`, whose
history still has the old author. Nothing was pushed. Either overwrite it:

```bash
cd ~/latent-reward-register-release-work
git push --force origin main    # destructive: rewrites published history
```

or detach and treat this tree as the new origin:

```bash
git remote remove origin
```

### 1.4 Export and inspect the public tree

```bash
cd ~/latent-reward-register-release-work

# Scan only; exits non-zero on any machine path, internal identifier, or credential
python release/export.py --out /tmp/lrr-public --check

# Write it
python release/export.py --out /tmp/lrr-public

# The export must stand on its own
cd /tmp/lrr-public && pip install -e '.[dev]' && pytest -q
```

### 1.5 Publish the dataset as a private HF dataset

Decide first whether HPDv3 images may be redistributed (§3.4). Until then,
publish manifests and scores only — not images.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # paste the ROTATED token; never inline it in a command

cd <dataset staging dir>
hf upload guanys/latent-reward-register . --repo-type=dataset --private
```

Stage manifests with relative paths only. Verify before uploading:

```bash
grep -rl '/home/\|/kaimm-distill/' . | head    # must print nothing
```

---

## 2. Done

### 2.1 Import breakage fixed
`implementations/models/reward_token_dina_head.py` (534 lines) was missing;
four modules imported it, so the package's largest subtree raised
`ModuleNotFoundError` — including the `load_legacy_register` path the README
advertises. Copied from `Reward-Token-Image-0420` (identical in all three
source repos; scanned clean).

`tests/test_import_integrity.py` now imports every module. Verified by
removing the file (suite fails) and restoring it (suite passes). The old suite
reported 34 passing tests while that subtree was unimportable.

### 2.2 Configs aligned to exp11
- SD3: three heads at equal weight (`preference`, `pickscore`, `imagereward`);
  `[4, 8, 12]` taps, 12 register layers, `skip_attn2=true`.
- FLUX: two heads, `[9, 19, 28]` of 57. Z-Image: two heads, `[5, 10, 15]` of 30.
- `score_keys` recorded next to `head_names`, since the loss pairs them
  positionally and a reorder mistrains silently.
- Tap rule documented (1/6, 1/3, 1/2 of depth, register stops at half) instead
  of per-model magic numbers.
- RG-OPD teachers recorded per backbone: SD3 exp11 EMA, FLUX unified-v3 EMA
  final. The FLUX config carries exp11 as a legacy default and overrides it,
  so the default alone is misleading.

### 2.3 Data schema corrected
The documented flat `{latent_path, rewards{}}` schema matched no producer.
Replaced with the real group manifest (`group_id`, `prompt`,
`prompt_embeds_path`, `image_records[]`). Verified against a 300-group
production manifest: all parsed, all three heads resolved.

Table 1: `all_table1_pairs_fixed.jsonl` (54170 pairs) is authoritative. The
other file has identical labels but different image paths on 6399 pairs.

### 2.5 Backbones wired to the real implementations
`backbones/diffusers.py` declared three adapters whose `extract_features` and
`reference_step` looked up `feature_extractor` / `sampler_step` attributes that
were never assigned anywhere, so every call raised. The real models sat in
`implementations/` with nothing referencing them.

Those models are complete registers, not feature extractors, and already return
`dict[head -> (batch, group_size)]`. They are now reached through
`build_register` / `build_register_from_config` instead of being forced behind
`BackboneAdapter`.

Building from `configs/register/sd3/paper.yaml` against real weights surfaced a
config/model mismatch that no dry-run could catch: the release configs had
invented `num_register_tokens` and `num_attention_heads`, while the models take
`num_reward_tokens` and `num_attn_heads`. All three configs now use the exp11
key names verbatim and pass straight into the model constructors.

Architecture parity check: the register built from
`configs/register/sd3/paper.yaml` reports **147,599,619 trainable parameters**,
which is exactly the `trainable_parameter_count` recorded in the exp11 training
log (`logs/exp11_resume_e3.log`). The wired architecture therefore matches the
one that produced the paper numbers, not merely something that constructs.
Backbone stays frozen: 147.6M trainable of 2175.9M total.

A real forward pass then exposed a second defect that no CPU test could reach:
`CheckpointRewardRegister.score` added a group axis to `prompt_embeds`, and the
research model's `_flatten_inputs` added another, so any real call raised. The
models own group flattening and repeat conditioning internally. Fixed, with
regression tests that fail if the extra axis returns.

Verified against SD3 weights: `score_groups` returns
`{preference, pickscore, imagereward}` each shaped `(batch, group_size)`, and
the restored group loss computes over those real scores.

### 2.6 Group loss restored
See the commit body; verified bit-exact against the research implementation.

### 2.4 Hygiene and export
- Scan widened from `src/` + `configs/` to the whole tree, plus internal
  identifier and credential checks. The old scan could not see the leaks in
  `docs/` and the root.
- `docs/source-provenance.md` rewritten to name repositories, not machine paths.
- `release/export.py` refuses to export on any violation (verified with a
  planted leak), `release/EXCLUDE.txt` is the authority, and
  `configs/local.yaml.example` documents the private-only values.
- SwanLab key replaced with `${SWANLAB_API_KEY:?...}` in 9 files; all 9 still
  pass `bash -n`. Two remaining matches are comments, not keys.
- Commit authors unified.

---

## 3. Blocked on your decisions

### 3.1 SD3 RGS register — RESOLVED
`lrm_exp11_e3_ema` (`exp11_multihead_noattn2_sd3_8gpu/checkpoints/
reward_token_epoch_003_ema.pt`). exp12/exp14/exp15 are exploratory variants that
differ in `pool_factor`, `head_input_tokens`, and tap count, so their registry
entries must not be used for release numbers. Recorded in
`docs/source-provenance.md`.

### 3.2 SD3 RG-OPD run — RESOLVED
`reward_scale: 0.40` with a frozen reference anchor, on the exp11 EMA teacher.
That is what `configs/rgopd/sd3/paper.yaml` already carries. The `emaanchor` /
`selfanchor` / `cfgdistill_*` variants are exploratory.

### 3.3 FLUX RGS scope — RESOLVED
FLUX RGS is in scope: Table 3 compares both backbones, so
`configs/rgs/flux/paper.yaml` stays in the export and the README lists RGS as
supported for SD3 and FLUX.

### 3.4 Data publication — RESOLVED
Training set stays private, documented as a filtered HPDv3 subset. Dataset
artifacts go to the private HF dataset `guanys/latent-reward-register`.

Table 1 ships as a pair file plus a fetch script, not as images:
`release/export_table1_pairs.py` produces the portable file (54170 pairs, no
machine paths, labels verified row-by-row against the source) and
`scripts/fetch_table1_images.py` reconstructs the four image roots. Its
`--verify` exits non-zero on any missing image.

### 3.5 Anonymity window — NOT A CONSTRAINT
Confirmed not a concern. Author is `Yuanshen Guan <guanys@mail.ustc.edu.cn>`.

### 3.6 Checkpoint publication — DECIDED, packaging tool ready
Publish SD3 + FLUX registers and SD3 + FLUX OPD LoRAs. **Z-Image is not
published**: it has no downstream experiment, so its register would be a
14.2 GB artifact nothing consumes.

`release/package_checkpoints.py --plan` lists exactly what ships; `--out <dir>`
stages it with a `CHECKSUMS.json`. Nothing is uploaded by the script.

| Artifact | Raw | Packaged | Note |
| --- | --- | --- | --- |
| SD3 register (exp11 e3 EMA) | 2.29 GB | ~0.6 GB | 147,599,619 params — matches the exp11 log exactly |
| FLUX register (unified-v3 final EMA) | 9.22 GB | ~2.4 GB | 594.8M params |
| FLUX OPD LoRA x3 | 0.52 GB each | unchanged | hps e150, imagereward e60, twohead e150 |
| SD3 OPD LoRA x2 | 0.08 GB each | unchanged | exp11ema HPS rt0.40, epochs 60 and 90 |

Registers shrink because research checkpoints carry optimizer and LR-scheduler
state (and, for FLUX, a duplicate `extra_state`) that is useless for inference:
FLUX is 594.8M params but 1114.1M more in optimizer alone.

**FLUX epochs: full sweep.** You asked for 60 and 90; 90 is not the selected
epoch for any FLUX run (the protocol chose hps 150, imagereward 60,
twohead 150), so publishing 60/90 there would hand readers checkpoints that do
not reproduce the paper. Per your "upload them all if unsure", FLUX now ships
every epoch (30/60/90/120/150) and each manifest entry carries
`selected_by_protocol` so the citable one is unambiguous. SD3 ships 60 and 90 as
you asked.

Raw total 19.5 GB; registers repack from 11.5 GB to about 3.0 GB, so roughly
9.3 GB is published.

### 3.7 Stale FLUX selection file — MITIGATED
The correct epochs (HPS 150, ImageReward 60, TwoHead 150) and an explicit "do
not cite the older selection file" note are in `docs/source-provenance.md`, so
the release cannot mislead. The stale
`flux_opd/eval_flux/outputs/_select/selection.json` still exists in the research
workspace; deleting or renaming it there is optional cleanup.

### 3.8 Paper table disagrees with the shipped FLUX config — FIX THE PAPER
The repository configs are verified against the runs that produced the release
checkpoints: SD3 `[4, 8, 12]` of 24, FLUX `[9, 19, 28]` of 57, Z-Image
`[5, 10, 15]` of 30, all on the 1/6-1/3-1/2 depth rule with the register
stopping at half depth. The FLUX values were confirmed against the actual
training config,
`node5/outputs/reward_register_flux_unified_v3/flux_exp20_unified_v3.yaml`,
whose `reward_token` block matches `configs/register/flux/paper.yaml`.

`rt_guided_sampling_iclr2027.tex` Table `tab:register-training-config`
disagrees in the FLUX column on both rows:

| Paper line | Row | Paper says | Should say |
| --- | --- | --- | --- |
| 767 | Register-equipped blocks $L$ | 12 (of 38) / 8 (of 19 double blocks) | 12 (of 24) / 28 (of 57) |
| 768 | Feature-snapshot blocks $\mathcal{B}$ | $\{4,8,12\}$ / $\{2,5,8\}$ | $\{4,8,12\}$ / $\{9,19,28\}$ |

The SD3 taps `{4,8,12}` are correct; only the FLUX column and SD3's block total
are wrong. Note the register spans 28 FLUX blocks, not 8, and reaches into the
single-stream stack — tap 19 is the *last* double-stream block, not an interior
one, so "of 19 double blocks" also mis-describes the architecture. No repository
change is needed; this is a paper edit.

---

## 4. Not yet implemented

Ordered by impact.

### 4.1 Wire the backbone adapters — DONE
`backbones/diffusers.py` reads `feature_extractor` and `sampler_step` via
`getattr`, and **nothing ever assigns them**, so `extract_features` and
`reference_step` raise for all three backbones. `implementations/` (the real
ported code) and `backbones/` (the interface) do not reference each other, so
no code path reaches the working model. This is the whole distance between
"imports cleanly" and "runs a model". Depends on nothing — ready to start.

### 4.1b Recover the input-latent gradient — DONE
`score_and_grad` against real SD3 weights fails with:

    RuntimeError: One of the differentiated Tensors appears to not have been
    used in the graph.

This is architectural, not a wiring slip. The training forward runs each SD3
joint block inside `torch.no_grad()` (`latent_reward_grid.py:670`), because the
backbone is frozen while the register trains. That hard-detaches
latent -> visual-features, so `d reward / d latent` does not exist on a plain
`forward()` — and that gradient is exactly what RGS and RG-OPD consume.

The research code solves it explicitly, and the comment at
`lrm_reward_backend.py:861-868` spells out why: it rebinds
`_run_frozen_sd3_block` to a grad-enabled clone
(`_run_frozen_sd3_block_grad`, bound via `types.MethodType` at
`lrm_reward_backend.py:922`, and again at 1122 and 1455 for the other loaders).
Weights stay frozen through `requires_grad_(False)`; only the input-latent
gradient is recovered. Nothing about the checkpoint or training code changes.

**Done.** Implemented as `latent_gradient_enabled()` in
`implementations/gradmode.py`: the block bodies consult a thread-local mode
instead of hardcoding `torch.no_grad()`, and `score_and_grad` enables it
internally. Same math as the research monkey-patch, but discoverable, uniform
across SD3/FLUX/Z-Image, and reversible. Documented in
`docs/latent-gradients.md`.

Verified against real SD3 weights at 1024x1024: gradient shape
`(1, 16, 128, 128)`, finite, nonzero, RMS 2.7e+02, and the trunk still fully
frozen afterwards.

The positional-embedding helper keeps its `no_grad`: it slices a constant table
and is not on the latent path.

### 4.2 SD3 RG-OPD — teacher and rollout driver DONE
The reward-gradient teacher is ported: `teacher.py` provides
`RewardGradientTeacher`, and `rgopd.rollout_target` is the RG-OPD entry point.
Verified that `rollout_target` and `build_rgopd_target` produce identical
targets, so the student trains against exactly the guidance the sampler applies.

Deliberately not ported verbatim: the research teacher loaded its register
backend by absolute file path from a sibling repo
(`DEFAULT_LRM_BACKEND_PATH`), and carried a second copy of the guidance
correction. Both are gone.

The rollout driver is in `rollout.py`: `train_rgopd_rollout` walks the
student's own trajectory, labels each visited state via the teacher, and
regresses. Two properties are pinned by tests — the trajectory is on-policy
(changing the student changes visited states), and schedule-off steps skip the
register backward (the register is called exactly as often as the schedule
allows).

`flowmatch.py` supplies the transitions: `make_reference_step` for the frozen
anchor, `make_student_policy` for the differentiable student. An end-to-end test
composes flowmatch + teacher + rollout on CPU.

Remaining for an actual paper run: a LoRA-wrapped backbone velocity model (the
release takes the velocity model as a callable, so this is integration rather
than new algorithm work), and the SD3 run choice in §3.2.

Not ported: `eval_table3/` (belongs with §4.7).

### 4.3 RGS — loop and sampler step DONE
`reward_guided_sample` now runs through the shared teacher, takes an explicit
`reference_step`, and reports a `SamplingTrace` with the guided-step fraction
(the cost claim in the efficiency table, now measured rather than asserted).

The Euler transition is extracted into `flowmatch.py` as arithmetic over a
caller-supplied velocity model, verified bit-exact against the research update.
Nothing from the 19 GB `flow_grpo` fork is shipped, and none of its ~30
absolute-path loaders came along.

Remaining for an actual paper run: the backbone CFG velocity model for SD3 and
FLUX — one function returning the guided flow velocity, which
`classifier_free_velocity` composes. Per §3.3 both backbones are in scope since
Table 3 compares them.

### 4.4 Runnable commands — DONE
`lrr build-register` constructs a register from a config against real weights and
reports the trainable parameter count (147,599,619 for SD3, matching the exp11
log). That check caught both the invented config keys and the doubled group axis.

`velocity.py` supplies the backbone piece: `SD3VelocityModel`,
`FluxVelocityModel`, and `attach_lora_student`. The algorithm layer takes the
velocity model as a callable, so nothing backbone-specific leaks into
`sampling.py` or `rollout.py`. A CPU test drives
velocity -> flowmatch -> teacher -> rollout end to end.

All three workflow subcommands now execute. They previously raised `RuntimeError`
for any non-`--dry-run` invocation — including `train-register`, which earlier
notes here missed. Three modules were added to close that gap:

- `local_config.py` resolves machine paths from the uncommitted
  `configs/local.yaml`, which is what keeps every committed file identical
  between the private and public trees.
- `dataset.py` turns a group manifest into `GroupBatch` objects, reading cached
  latents. Short groups are skipped rather than padded.
- `runtime.py` loads real pipelines, encodes prompts once outside the loop, and
  drives the three tasks. It is the only module importing diffusers pipelines.

`flowmatch.sigma_schedule` was also missing and is now verified against
`FlowMatchEulerDiscreteScheduler` at 4/10/28/42 steps to float32 precision. An
approximate grid would have moved which steps the sigma-banded schedule guides.

`scripts/` carries one launch script per task plus `smoke_all.sh`. Reduced-scale
flags (`--max-batches`, `--rounds 1`) give reviewers a real-path run in minutes.

### 4.5 Restore the group loss — DONE
Ported and verified bit-exact against the research implementation across batch
sizes, group sizes, masking, and `min_target_gap`.

### 4.6 Third-party licensing
`OneIG-Benchmark`, `self-refine-video`, and `T2I-CompBench` carry no license
grant. `self-refine-video` and `classifier-free-guidance-pytorch` are never
imported — delete them. The rest are used only as prompt sources (plus
CompBench's CLIPScore), so ship the derived prompt lists and a setup script
instead of the clones. Ported files also need origin/license headers.

### 4.7 Benchmark runners and prompt lists — Table 1 done
`release/export_keep800.py` exports the keep-800 evaluation set joined against
`prompts500.jsonl`, so each of the 800 entries carries prompt text, seed, and
provenance. That is sufficient for Tables 2 and 3: they report metrics only over
keep-800, and the dropped 200 never enter a number, so `balanced500` is not
needed and is not published.

`table1.py` is the Table 1 runner: it reads the released pair file, resolves
images against per-dataset roots, and reports accuracy overall and per dataset.
Verified against the real 54170-pair file.

It also guards a hazard found while parsing that file: `image1` is always the
human-preferred image, so every label is 0 and a constant "first" answer scores
100%. Use `shuffled()` or `position_bias()` before reporting a number.

Remaining: the Table 2/3 generate -> score -> aggregate driver. Generation is now
executable via `lrr sample`, so what is left is the metric harness wiring (hpsv3,
imagereward, musiq, clipiqa) and the aggregation over keep-800. Those scorers are
third-party and separately licensed, which is why this is last.


### 4.8 README rewrite — DONE
Restructured so a reader reaches a runnable command before any design rationale:
install, machine paths, then one command per task, with the mechanism (shared
teacher, gradient mode, the two register classes) moved below under "How it
works". The earlier version opened with rationale and buried the commands.

`docs/reproduction.md` leads with a per-result status table, and
`docs/reward-guided-opd.md` and `docs/latent-gradients.md` are new.

### 4.9 Hygiene fixes found during the release pass — DONE
- `docs/IMPLEMENTATION_SHA256SUMS` had 7 of 9 digests wrong and had never
  matched, because nothing verified it. It also omitted
  `reward_token_dina_head.py`. Corrected, extended to all ten vendored modules,
  and enforced by `tests/test_implementation_checksums.py` (verified by planting
  drift). Its header no longer claims byte-identity with the research originals,
  which was never true: the three intentional deltas are recorded in
  `docs/source-provenance.md`.
- Dependencies had lower bounds only, leaving the load-bearing diffusers pin
  doing half a job. `torch`, `transformers`, `peft`, `numpy`, `pillow` now carry
  upper bounds; `peft`'s floor was above the version actually in use.
- `.github/workflows/ci.yml` runs the suite through an editable install (the path
  the README documents), lints, checks all seven presets resolve, and installs the
  exported public tree from scratch to prove it is self-contained.
- `release/export.py` no longer `rmtree`s `--out` unconditionally: a directory
  that is neither empty nor a previous export needs `--force`.
- ruff excludes the checksum-pinned vendored files, since satisfying it there
  would break the manifest. The five remaining findings are ignored with reasons.

