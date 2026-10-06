# Tutorial 05 — Select on validation, then measure complete games

Training, validation selection, selected-checkpoint reload and the held-out game
evaluation are complete. Epoch 02 cleared **16.81 lines per game**, versus **0.07**
for native Clef and **76.53** for the cheap heuristic on all 200 reserved seeds.
This tutorial gives the frozen procedure and explains the measured outcome. See
[the final study](../../reports/final-study.md) for full results and limitations,
and [the compact JSON](../../reports/final-summary.json) for exact metrics and hashes.
The experiment results are complete; release verification is tracked separately.

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

## Train exactly the preregistered candidates

Pass the real GPU feasibility and fresh-process reload gates in Tutorial 04 first.
Use the exact 827/215-row dataset from Tutorial 03. The training runner rejects a
substituted dataset even when its labels look similar: hashes bind the experiment
to specific bytes and provenance. Use a fresh run directory when reproducing.

```bash
uv sync --locked --extra ml --group dev
uv run --locked --extra ml python scripts/train_clef.py \
  --dataset data/study-v1 --output runs/study-v1 \
  --mode lora --epochs 2 --learning-rate 1e-5 \
  --accumulation 8 --max-length 4096
```

This command needs a free GPU. On the original workstation only, run it through
the authorized stop/restore wrapper from Tutorial 04 with a 3,600-second timeout.
The script uses seed 42, batch size one, rank-4/alpha-8 LoRA with zero dropout,
AdamW weight decay 0.01, gradient norm clipping at 1.0, and the fixed smoothed
cross-entropy plus Brier loss from Tutorial 04. The backbone stays BF16 and frozen;
LoRA matrices and the native FP32 head learn. Each epoch visits all training rows
in a recorded deterministic shuffled order. Validation rows never train the model.

Accumulating eight single-example gradients fits the memory budget while averaging
over several boards per optimizer step. The final group has only three rows and
is divided by three, not eight. Two complete epochs give two declared candidates
without an open-ended search. Neither falling loss nor a promising demo authorizes
extra epochs after the test set has been opened. The script defaults to one epoch,
so the explicit `--epochs 2` above is essential for reproducing this study.

The completed run processed 827 examples and 104 optimizer steps in each epoch:

| Epoch | Mean training loss | Time |
| --- | ---: | ---: |
| 01 | 2.577967 | 742.48 s |
| 02 | 1.966718 | 744.66 s |

Total training time including hashes and checkpoint work was 1,511.25 seconds
(about 25.2 minutes). Peak CUDA allocation was 23.23 GB, or about 21.64 GiB.
The head and LoRA parameters changed, while all frozen parameter hashes remained
unchanged. These are training measurements, not held-out performance. See
[the training report](../../reports/study-training.json),
[the original launch source](../../reports/training-launch-source.json), and
[the archived preregistration](../../reports/study-preregistration.md).

## Prepare checkpoint metadata before freezing it

PEFT can write a generic adapter README with placeholders and a local cache path.
Keep raw training outputs immutable. Prepare candidate copies before validation:

```bash
uv run --locked --extra ml python scripts/prepare_checkpoint.py \
  --source runs/study-v1/epoch-01 --output checkpoints/candidates/epoch-01
uv run --locked --extra ml python scripts/prepare_checkpoint.py \
  --source runs/study-v1/epoch-02 --output checkpoints/candidates/epoch-02
```

The script replaces only `adapter/README.md`, compares every other file byte for
byte, and saves a sibling preparation record. It does not alter learned weights
or training settings. Validate and select these exact copies. Once selected,
even a README edit would break the frozen checkpoint file mapping. See Tutorial 06
for how release documentation stays outside the checkpoint directory.

## Generate validation evidence

These commands load real GPU models. Use the memory admission and service-restoration
procedure from Tutorial 04. The evaluator never downloads weights implicitly or
stops another workload. Replace paths if your run directory differs.

```sh
uv run --locked --extra ml python scripts/evaluate_clef.py positions \
  --dataset data/study-v1 --checkpoint checkpoints/candidates/epoch-01 \
  --players base base-fp32 trained --output runs/validation-epoch-01

uv run --locked --extra ml python scripts/evaluate_clef.py positions \
  --dataset data/study-v1 --checkpoint checkpoints/candidates/epoch-02 \
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
uv run --locked --extra ml python scripts/select_checkpoint.py \
  --dataset data/study-v1 \
  --candidate01 checkpoints/candidates/epoch-01 runs/validation-epoch-01/report.json \
  --candidate02 checkpoints/candidates/epoch-02 runs/validation-epoch-02/report.json \
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

## Inspect selection and reload the selected artifact

The measured validation results used all 215 positions for every policy, with
zero errors and complete probability distributions:

| Policy | Teacher agreement | Mean target NLL | Mean Brier score |
| --- | ---: | ---: | ---: |
| Native base | 21/215 (9.77%) | 2.952702 | 0.923173 |
| Unchanged FP32 head | 22/215 (10.23%) | 2.952393 | 0.923102 |
| Epoch 01 | 89/215 (41.40%) | 1.860870 | 0.722190 |
| Epoch 02 | 92/215 (42.79%) | 1.775923 | 0.708980 |

The declared NLL rule selected epoch 02. The saved
[validation summary](../../reports/validation-summary.json) identifies its complete
file hashes and the selection hash. Teacher agreement improved on these positions;
that still does not establish better complete-game play.

Before consuming test seeds, reload the selected artifact in another process:

```bash
uv run --locked --extra ml python scripts/probe_clef_training.py \
  --dataset data/study-v1 --reload checkpoints/candidates/epoch-02 \
  --output runs/selected-reload-v1
```

This command also needs the GPU admission/restoration procedure. In this study it
reproduced all four training-position reference distributions with **0.0 maximum
absolute difference**, below the predeclared `1e-4` tolerance. See
[the reload report](../../reports/selected-checkpoint-reload.json).
The references use training positions, so this check does not consume test games.

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
  --output runs/final-evaluation
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
lengths. The [hardware snapshot](../../reports/evaluation-hardware.json) records
the RTX 5090, driver, CPU, Python and inference settings observed during this run.
Compare latency only with that environment in mind; the snapshot does not measure
energy consumption or continuous GPU clock behavior. A game reaching 200 pieces survived **at least** 200; its eventual death
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


## Completed outcome and what it teaches

All five players completed the 200 paired seeds at the declared 200-piece cap:

| Player | Mean lines | Median lines | Mean pieces | Cap hits |
| --- | ---: | ---: | ---: | ---: |
| Native base | 0.070 | 0 | 26.020 | 0/200 |
| Unchanged FP32 head | 0.105 | 0 | 26.000 | 0/200 |
| Selected epoch 02 | 16.810 | 16 | 83.565 | 0/200 |
| Random | 0.140 | 0 | 26.645 | 0/200 |
| Fixed heuristic | 76.530 | 77 | 200.000 | 200/200 |

There were zero inference errors and zero invalid decisions. The paired mean line
gain over native base was **16.74 [15.74, 17.77]** (95% percentile bootstrap,
10,000 replicates, seed 2026). Against the unchanged FP32 head, the gain was
**16.705 [15.705, 17.74]**, so the observed improvement cannot be attributed to head
precision alone. Against the heuristic the difference was
**−59.72 [−60.805, −58.619875]**. Both the improvement and this large shortfall belong
in the result. The heuristic's eventual survival remains unknown because every
one of its games reached the cap.

The selected model's mean decision latency was **206.8 ms**, compared with the
heuristic's **0.1985 ms**, about 1,042 times slower on these measured workloads.
Policies visited different states, and neural inference used the GPU while the
heuristic used the CPU. This is an operational comparison, not an isolated
same-input speed benchmark. The final wrapper took 85.1 minutes, including model
loading, report writing and service restoration. Energy and monetary costs were
not measured. The simple heuristic remains the practical choice for this game.

An independent CPU audit replayed all 1,000 saved episodes and independently
recomputed every paired interval. See [its results](../../reports/final-audit.json).
The learned model passes the narrow improvement test over its unchanged base;
that does not turn its 42.79% teacher agreement into a general measure of game
quality. The final report includes score/survival intervals, all latency summaries,
hardware, source identities and portable release evidence paths.

To replicate the analysis, first verify the report/checkpoint/dataset hashes in
[final-summary.json](../../reports/final-summary.json), then use the frozen source
and commands above in fresh output directories. Reproducing inference needs the
GPU runtime; inspecting metrics and replaying saved actions do not. Reproduction
uses the already disclosed sequences, so it is not a new blind test. Any follow-up
training or design change needs a separate declared experiment and fresh test
seeds. No post-test tuning was used to produce the results reported here.
