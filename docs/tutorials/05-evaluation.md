# Tutorial 05 — Select on validation, then measure complete games

This tutorial describes the frozen study procedure. It does not claim completed
training, selection, held-out performance, or an improvement. Actual evidence and
milestone status belong in `record.md` and the generated reports.

## Separate learning, selection and final testing

The two candidate checkpoints are epochs 01 and 02 of the same seed-42 LoRA run.
Their training configuration and dataset are fixed before comparison. Evaluate
both on **all 215 validation positions** and select the lowest finite mean target
negative log likelihood (NLL). An exact tie chooses the earlier epoch. A candidate
with inference errors, missing probability distributions, or nonfinite NLL is
ineligible. If both are ineligible, stop before opening the test set.

NLL measures how much probability the model assigns to the teacher's recommended
placement. Selecting on a smooth probability score can distinguish checkpoints
even when their top choices tie. Teacher agreement and Brier score remain useful
diagnostics, but cannot override the preregistered rule after results are visible.

Copying the teacher is not the same as playing well. Each small mistake changes
the next board; the model eventually encounters states different from its training
examples. The teacher itself uses a limited search horizon and a heuristic value
function. Final evaluation must therefore play complete games on unseen sequences.

## Generate validation evidence

These commands load real GPU models. Use the memory admission and service-restoration
procedure from Tutorial 04. The evaluator never downloads weights implicitly or
stops another workload. Replace paths if your run directory differs.

```sh
uv run --locked --extra ml python scripts/evaluate_clef.py positions \
  --dataset data/study-v1 --checkpoint runs/study-v1/epoch-01 \
  --players base base-fp32 trained --output runs/validation-epoch-01

uv run --locked --extra ml python scripts/evaluate_clef.py positions \
  --dataset data/study-v1 --checkpoint runs/study-v1/epoch-02 \
  --players trained --output runs/validation-epoch-02
```

Every validation row gets an unrounded probability record or an explicit error.
The position report binds those records to checkpoint file hashes and the dataset
manifest. Aggregate NLL and Brier are unavailable if any probability distribution
is missing; a successful subset cannot masquerade as a complete evaluation.

The first command also measures two unchanged-model references. `base` retains
the original native BF16 head. `base-fp32` keeps those same learned weights but
uses our FP32 head wrapper. Training uses that wrapper, so the extra comparison
shows whether a change comes from numerical precision rather than learning.
The primary comparison remains trained versus native base. All three share the
same complete observation encoding and BF16 backbone.

## Freeze the choice before running test episodes

```sh
uv run --locked python scripts/select_checkpoint.py \
  --dataset data/study-v1 \
  --candidate01 runs/study-v1/epoch-01 runs/validation-epoch-01/report.json \
  --candidate02 runs/study-v1/epoch-02 runs/validation-epoch-02/report.json \
  --output runs/selection-v1
```

The selector independently recomputes metrics from every raw prediction against
the audited validation rows. It verifies actual checkpoint bytes, training seed
and configuration, epoch identity, common dataset and evaluation provenance. It
accepts exactly the two preregistered candidates. It then copies both validation
reports, prediction files and checkpoint metadata into a self-contained evidence
directory. The large model weights remain in their checkpoint directories and
are referenced by hashes.

`selection.json` records the selected epoch, decision rule, both candidates,
their eligibility, validation metrics, and the frozen test protocol. The runner
rechecks that the selected epoch actually wins the declared rule. Manual editing
of a label such as "best model" is not sufficient to unlock final evaluation.
Use a new output directory for a new selection attempt; existing evidence is not
silently overwritten.

## Run the paired test once

Use the selected checkpoint path from the evidence, not whichever epoch you hoped
would win. The reserved pool is all **200 seeds, 30000–30199**, with a **200-piece
cap**. Each player gets the same piece sequence for a given seed. There are five
players: native base, FP32-head base ablation, selected trained model, random and
the fixed cheap heuristic.

```sh
uv run --locked --extra ml python scripts/evaluate_clef.py tournament \
  --checkpoint SELECTED_CHECKPOINT \
  --seeds 30000:30200 --max-pieces 200 --final-test \
  --selection-file runs/selection-v1/selection.json \
  --output runs/final-evaluation-v1
```

`--final-test` explicitly acknowledges that these held-out seeds are being consumed.
Without it, any overlap with the reserved pool is rejected before model loading.
The cap and full seed pool cannot silently shrink to finish sooner. The initial
development timing estimate permits hours in the worst case; short early deaths
can make actual evaluation much faster.

Each completed episode is saved atomically, with actions, probabilities, runtime
configuration, errors and a replay. To continue an interrupted run, repeat the same
command with `--resume`. The runner checks source/dependency hashes, checkpoint
identity, selection, and saved replay outcomes. It skips completed players without
loading their models again. Recorded errors remain part of the experiment, rather
than being retried until a nicer result appears.

## Read the report without hiding failures

The primary outcome is lines cleared. Score and pieces survived are secondary.
The report includes means, medians, cap-hit rate, failure counts, latency and input
lengths. A game reaching 200 pieces survived **at least** 200; its eventual death
was not observed. Always report cap-hit rates with survival statistics.

For the primary reliability-adjusted analysis, any errored episode receives zero
lines, score and pieces. Its measured partial outcome remains visible separately.
Dropping failures would bias the result toward easier successful games. Both base
and trained must have zero episode errors for the default positive-signal flag.

For each seed, subtract base lines from trained lines, then bootstrap these paired
differences 10,000 times with RNG seed 2026. Report the mean and percentile 95%
interval. Resampling complete episode pairs respects the fact that moves within
one game are correlated. Treating every move as independent would exaggerate the
amount of evidence.

A lower interval bound above zero supports improved mean lines for this checkpoint
and seed distribution. It does not establish competitive Tetris ability, training
seed stability, or superiority over the heuristic. Report the trained-versus-
heuristic comparison even if it is embarrassing. A reproducible negative result
with clear limitations is a stronger portfolio artifact than a selective win.

After test results are opened, do not use them to choose between epochs or change
the observation format. A subsequent experiment needs a new preregistered study
and fresh held-out sequences. The current report should preserve failures and
limitations alongside the playable recorded comparison.
