# Tutorial 06 — Ship an experiment another person can inspect

Publication is the last milestone, after training, validation selection and held-out
evaluation. A model upload alone is insufficient: the artifact must load, its claims
must match measured games, and the demo must distinguish recorded model decisions
from live human play. This tutorial describes the release process; publication links
and verified results are recorded in the final release report when available.

## Separate the four deliverables

The source repository contains the simulator, native model adapter, training and
evaluation scripts, tests, locked environment and tutorials. The model repository
contains the selected checkpoint's LoRA weights and separately trained decision
head, plus source and evidence needed to reproduce the study. The dataset repository
contains exact train/validation JSONL files and the generator manifest. The demo
repository runs the game and replays cached model decisions without downloading a
9B model or renting GPU hardware.

The source bundle also contains `source-manifest.json` with its source commit and
file hashes. Scripts use an exact Git root when available, otherwise they verify
that manifest. Changed, missing or newly added source files mark the run as modified;
an unrelated parent repository is never mistaken for Stackcraft's source identity.
See Tutorial03 for hash-checked reproduction of the original dataset bytes without
access to the original Git repository.

These are different visibility decisions. A private GitHub repository does not
make files inside a public model repository or public Space private. In particular,
the reproducibility source in the model release and the Space's game source are
public if those repositories are public. Review the concrete local bundle, proposed
owners and visibility before creating external repositories.

The proposed destinations are `kkarimi/stackcraft` on GitHub and
`nima1/stackcraft-clef-flash-lora`, `nima1/stackcraft-data`, and Space
`nima1/stackcraft` on Hugging Face. They are proposals until the owner approves this
new project's publication settings. Do not infer approval from a previous project.

## Preserve evidence while preparing checkpoint metadata

Raw epoch checkpoints remain under the training run directory. PEFT can generate
an `adapter/README.md` with placeholders and a local cache path. Before validation,
copy each epoch checkpoint into a separate candidate directory and replace only
that documentation with accurate upstream attribution and loading instructions.
Compare every other file's SHA256 with the raw checkpoint and record the mapping.
The preparation script performs these checks and writes a sibling provenance file:

```bash
uv run --locked --extra ml python scripts/prepare_checkpoint.py \
  --source runs/study-v1/epoch-01 --output checkpoints/candidates/epoch-01
uv run --locked --extra ml python scripts/prepare_checkpoint.py \
  --source runs/study-v1/epoch-02 --output checkpoints/candidates/epoch-02
```
This edits no learned weights or training configuration.

Validate those exact candidate directories, then freeze the selected directory's
complete file/hash mapping. Do not revise it after selection. Model cards, reports
and release links belong outside `model/checkpoint/`, so adding release documentation
does not silently change the checkpoint used in validation and test evaluation.

A final model repository must contain both `joint_head.safetensors` and
`adapter/adapter_model.safetensors`. Loading only the adapter discards the learned
joint head. Loading the selected checkpoint requires the pinned upstream
`Cloudflare/clef-flash` snapshot and Stackcraft's strict loader, not a generic
text-generation pipeline. See Tutorial04 for the reason and serialization checks.

## Assemble and audit a local release

After final evaluation and the fixed-seed demo export, run the release builder
from the source repository. Use its `--help` for the exact implemented arguments;
it takes the selected checkpoint, frozen selection, complete evaluation directory,
audited dataset and a new output directory. It must reject incomplete evaluation,
mismatched hashes or a missing selected checkpoint reference.

The builder stages separate model, dataset and demo directories and a release
manifest. It includes only explicit source/build inputs. Environments, credentials,
Git history, local service data, raw upstream backbone weights and arbitrary files
from the working directory are excluded. This is safer and more reproducible than
uploading the repository directory recursively and hoping ignore patterns cover it.

Audit the concrete staged files:

- Model and dataset cards describe synthetic data, teacher bias, simplified rules,
  the selected training configuration and actual complete-game outcomes.
- Negative results, model errors, the FP32-head ablation and the cheap heuristic
  remain visible. There are no unmeasured claims of superiority or calibrated confidence.
- Every planned test seed is represented for every preregistered player, including
  failures; aggregate uncertainty agrees with the stored episode outcomes.
- The fixed demo seed is30000, chosen before results, with actual action probabilities.
- Dataset bytes and checkpoint bytes match their frozen hashes. The trained
  checkpoint's probability references identify the same four training positions.
- LICENSE and NOTICE retain the Apache-2.0 attribution for Clef-flash/Qwen and
  identify Stackcraft as an independent project.
- Cards have no placeholder metrics or paths that require the author's machine.

A release manifest identifies bytes, not scientific correctness. Independent replay
checks, selection review and honest documentation are still needed. Keep the
manifest hash and source commit with the final release report.

## Validate the game-only deployment

Follow [the deployment guide](../demo-hosting.md) to build the pinned Docker image.
It installs through uv's lockfile with no ML extra, runs as UID1000 and listens on
port7860 for a Docker Space. Use the same image configuration locally first.

Test a human game, a downloaded/reloaded replay, the recorded comparison, probability
displays, scrubbing to final outcomes and returning to the preserved human session.
Check desktop and phone layouts. The recorded race must not imply that changing
playback speed changes inference speed. It illustrates one sequence; the report
summarizes all200 paired sequences.

The public demonstration needs CPU hosting because neural inference already happened
during evaluation. This removes ongoing GPU expense and makes the portfolio usable
when the training machine is offline. An interactive live-model endpoint would be a
separate deployment with admission, latency and concurrency requirements.

## Publish only the approved bundle

Confirm concrete destinations and visibility after the local bundle is reviewable.
Use existing CLI authentication; do not put access tokens in code, Git, model cards,
commands or chat. Create the approved GitHub repository and push the reviewed local
commits. Create the approved model, dataset and Docker Space repositories, uploading
the corresponding staged directories only. The Space can use the default free CPU
hardware; this study does not authorize a paid GPU upgrade.

Save immutable Hugging Face commit revisions returned by each upload, together with
the Git commit and model/dataset/demo URLs. Repository names alone are mutable;
revisions identify which bytes were checked. A successful upload does not establish
that the Space builds or the model loads.

## Verify the actual release, not the local staging copy

Download the uploaded model and dataset into a new directory using those exact
revisions. Compare every released file against the release manifest. Run the dataset
audit on downloaded JSONL files and its manifest. Use a fresh Python process to load
the downloaded checkpoint onto the pinned upstream backbone and compare its four
raw reference probability distributions at maximum absolute tolerance1e-4.

Use `huggingface_hub.snapshot_download` with each recorded immutable revision and
an empty `local_dir` for each repository. Use `repo_type="dataset"` for the dataset
and `repo_type="space"` for the demo. `local_dir` materializes regular files; passing
a shared cache snapshot directory directly can expose symlinks, which the verifier
rejects. Keep the original local `release-manifest.json` separately as the trusted
comparison, rather than trusting a manifest fetched with the same download.

From the reviewed source checkout, verify all three downloaded repositories:

```bash
uv run --locked --extra ml python scripts/verify_release.py \
  --trusted-manifest LOCAL_RELEASE/release-manifest.json \
  --model DOWNLOADED_MODEL \
  --dataset DOWNLOADED_DATASET \
  --demo DOWNLOADED_SPACE \
  --output runs/released-files-verification.json
```

This verifies file sizes and SHA256 hashes, checks each uploaded per-repository
manifest, and reruns the dataset schema/split audit. It rejects missing or extra
payloads and links, except narrowly defined Hugging Face download bookkeeping,
which it lists in the report. It does not establish publisher authenticity,
prove that a download was fresh, or load the neural model. Save the download
revisions and command separately, then perform the GPU parity check below.

The existing probe script can perform that final serialization check:

```bash
uv run --locked --extra ml python scripts/probe_clef_training.py \
  --dataset DOWNLOADED_DATASET \
  --reload DOWNLOADED_MODEL/checkpoint \
  --output runs/released-model-reload
```

Run this only with sufficient free GPU memory. On the original workstation, use the
approved GLM stop/restore wrapper and verify restored health afterward. Another
machine must arrange its own workloads; the workstation-specific wrapper is not a
portable permission to stop services.

Check the published Space's `/health`, static assets, human moves and recorded race
in a browser. Verify model/dataset links and any private GitHub access expectations.
Record failures and correct the actual published bundle before declaring release
complete. Then update the plan's M6 status and final outcomes with verified links,
revisions, hashes, measured limitations and fresh-download evidence.
