# Third-party harnesses: disposition for the release

Working document — excluded from export. Records what to do with the six
vendored clones in `RT-guided-sampling`, since three carry no license grant and
redistributing them would be the riskiest part of this release.

None of the six has local source modifications; the only dirty entries are
`__pycache__` and generated outputs. So nothing of ours is lost by dropping
them.

| Directory | Upstream license | Used for | Disposition |
| --- | --- | --- | --- |
| `T2I-CompBench` | **none** | prompts + `CLIPScore_eval` | drop; ship derived prompt list, document the CLIPScore dependency |
| `OneIG-Benchmark` | **none** | prompts only | drop; ship derived prompt list |
| `self-refine-video` | **none** | nothing | **delete** — never imported |
| `classifier-free-guidance-pytorch` | MIT | nothing | **delete** — never imported |
| `geneval` | MIT | prompts only | drop; ship derived prompt list |
| `ELLA` | Apache-2.0 | DPG prompts only | drop; ship derived prompt list |

Verified by grep: `self-refine-video` and `classifier-free-guidance-pytorch` are
referenced by zero first-party Python or shell files. `self-refine-video` is the
conceptual origin of the predict-and-perturb idea, but the SD3 sampler
reimplements it independently and imports nothing from it.

## Why prompts rather than clones

The four referenced harnesses are used **only as prompt sources**, plus
CompBench's `CLIPScore_eval` on `PYTHONPATH`. Shipping the derived prompt lists
plus a setup script that fetches a harness when a user wants to re-score:

- removes the three no-license redistributions entirely,
- keeps the exact prompts that produced the reported numbers, which a fresh
  clone at a moving HEAD would not guarantee,
- and drops a large amount of unrelated code from the release.

Key sets to export, with checksums and counts:

- **`keep800`** — the Table 3 evaluation key set. This is the only one published.
  Exported by `release/export_keep800.py`.

Tables 2 and 3 are scored on keep800, and the released file is self-contained:
every one of its 800 entries carries prompt text plus seed, so those exact
samples can be regenerated. The 200 dropped samples never enter a reported
number, so `balanced500` (the full 500-prompt generation set) is not needed and
is not published.

## One more redistribution note

The pinned `T2I-CompBench` HEAD includes an upstream `MLLM_eval/` subtree that
POSTs base64 images to an external API. RT never invokes it, but vendoring the
clone would redistribute it. Another reason to ship prompts instead.
