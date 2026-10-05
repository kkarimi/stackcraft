# Stackcraft evaluation protocol — completed study, frozen procedure

The final test used all **200 episode seeds, 30000–30199**, at the frozen 200-piece
cap, with all five players and no errors. Training, validation selection, reload
verification, final evaluation and the independent CPU audit are complete. The
[final study](../reports/final-study.md) reports the measured outcomes and
[compact JSON](../reports/final-summary.json) preserves exact statistics and hashes.
Publication is pending.

The procedure below records decisions frozen before test results were opened;
completion does not authorize tuning against these now-disclosed sequences.
`evaluation.py` consumes existing episode artifacts and position predictions;
it does not invoke models, select checkpoints, or start games.

## Decisions that must be frozen before opening the test results

The final episode cap is now **frozen at 200 pieces**, using all 200 reserved seeds.
This was chosen before inspecting any test results. The unchanged-model development
pilot measured approximately 144 ms per decision. If all three neural players reach
every cap, 120,000 decisions would take roughly 4.8 hours at that rate, excluding
loading and reporting. Earlier base development games died much sooner; runtime
must still be measured and reported rather than promised. Each episode is saved
atomically so interrupted evaluation can resume without discarding difficult games.

Freeze the selected trained checkpoint and original pinned Clef revision, input
encoding, precision, hardware/runtime configuration, heuristic revision, all
seeds, cap, inference failure behavior, bootstrap parameters and compute budget.
The same legal placements and one-piece preview must reach every player. Primary
base is the pinned native model with its original BF16 head. Training uses an FP32
head; therefore an explicitly named `base-fp32` ablation keeps the original weights
but uses the same FP32 wrapper as training. Report trained-versus-native-base as
the primary comparison and trained-versus-base-fp32 to separate training from head
precision effects. Backbone BF16, input encoding, context limit, device and source
revision must agree. A changed base quantization would be a separate comparison.

Checkpoint selection uses validation only. Teacher-action agreement and target
cross-entropy on validation positions are diagnostic selection metrics, separate
from complete-game outcomes. A model that imitates the expert more closely may
still play worse. Preregister any training configuration search and tie rule;
`paired_report` never makes those choices from test results.

## Paired episode outcomes

Primary outcome is lines cleared. Secondary outcomes are game score and pieces
placed. Retain every planned seed for each player, including failed episodes.
Duplicate player/seed pairs, mismatched seed sets, caps or rules, and changing
player identity/runtime within a comparison are rejected by the report function.
Its schema checks are not a replacement for verifying replay integrity when
episodes are collected.

An inference exception or illegal decision ends that player's episode. For the
primary analysis, assign **zero lines, score and pieces to any errored episode**.
Keep its original partial achievements in `observed_before_error`, retain its seed,
and report its error. This explicit reliability penalty prevents a model from
benefiting statistically from dropping difficult failures. The same policy applies
to base, trained and heuristic. Never describe a failed episode as a successful
zero-line game; status and error counts remain separate.

Report mean and median metrics, cap-hit rate, failed episodes, error rate and
invalid-decision count. Survival is censored at the cap: a cap hit means at least
that many pieces, not a measured death at that turn. This protocol cannot establish
uncapped longevity or full competitive Tetris performance.

For each seed, subtract base outcome from trained outcome. Resample these paired
episode differences with replacement, **10,000 bootstrap replicates, RNG seed
2026**, and report their mean plus the percentile 95% interval. Resampling turns
would falsely treat correlated moves within a game as independent observations.
Pieces and score get descriptive intervals; lines are the single primary endpoint.
The intervals are conditional on the tested checkpoint and episode distribution;
one trained seed does not characterize training-seed variability.

`positive_mean_lines_signal` is true only if there are at least two pairs, the
primary interval's lower bound exceeds zero, and both base and trained episode
error rates are at or below the declared threshold (**default 0%**). This flag is
evidence for that measured comparison, not an automatic claim of general
superiority, approval to publish, or a way to select a checkpoint on the test set.
The heuristic remains a separate required comparator. Run the same paired analysis
against it and disclose whether the model actually beats that cheap baseline.

## Timing and input lengths

Summarize decision latency mean, median and nearest-rank p95 from all attempts,
including failed attempts. These are timings recorded around `Player.choose`;
they include tokenization and policy overhead, exclude game rendering/replay, and
should not be advertised as pure GPU kernel time. CUDA probability transfer must
complete before `choose` returns. Record cold loading separately and identify any
warmup excluded before measured episodes. Never mix cached-replay speed with live
model latency. Input-token summaries use only events containing `input_tokens`
and report their count, so missing token metadata remains visible.

## Position predictions

`summarize_positions` reports agreement with the fixed teacher label, target NLL
and multiclass Brier score (sum over choices, averaged over positions). The fixed
label uses engine-order tie-breaking; agreement therefore penalizes choosing a
different equally valued action. Consider a separately named optimal-value tie
agreement metric later if preregistered; do not silently change the definition.

Missing or invalid predictions stay in the agreement denominator as failures.
Aggregate NLL/Brier are available only with a complete legal probability
distribution for every position. Partial diagnostic means have explicit subset
denominators and cannot be presented as full-set results. A zero probability on
the teacher label means infinite NLL; JSON stores a null mean plus an explicit
`nll_is_infinite` flag and zero-target count, rather than silently clipping it.

```python
from stackcraft.evaluation import paired_report, summarize_positions

# Load complete, previously collected artifacts; this runs no model or game.
report = paired_report(episodes, trained_id="trained", base_id="base")
position_report = summarize_positions(validation_rows, predictions_by_record_id)
```

Toy tests verify known constant paired improvements, mismatched-pair rejection,
failure retention, metadata consistency and proper probability-score arithmetic.
Passing them verifies report mechanics, not any model's game performance.

## Execute and resume a frozen evaluation

`scripts/evaluate_clef.py positions --dataset data/study-v1 --checkpoint PATH
--output NEWDIR` evaluates the entire validation split and writes unrounded
predictions plus NLL/agreement/Brier evidence. Its default players are native base,
base-fp32 and trained. Model loading is sequential to fit one GPU; this is real
inference and must follow memory admission. `--players trained` evaluates another
candidate without rerunning the unchanged baselines.

After validation, run `scripts/select_checkpoint.py` with both explicit epoch
checkpoint/report pairs and a new output directory, as shown in Tutorial 05.
It recomputes metrics from the raw predictions, verifies both checkpoint trees,
selects by the preregistered rule, and copies self-contained validation evidence.
Its `selection.json` contains schema_version 1,
selection_split `validation`, selected_key, decision_rule, checkpoint_sha256 mapping,
validation_metrics, validation_evidence `{path,sha256}`, test_seeds, their canonical
JSON SHA256 as test_seeds_sha256, max_pieces 200, encoding_version, max_length,
players, head_dtypes, bootstrap_samples 10000, bootstrap_seed 2026, max_error_rate
0.0. Validation evidence must be the script's position report for the same checkpoint;
its path is relative to selection.json. Checkpoint hashes include every checkpoint
file. Set head_dtypes to `{base: bfloat16, base-fp32: float32, trained: float32}`
when all three neural players are included.

Both epoch candidates and their eligibility reasons are mandatory in the selection
artifact. Before opening test seeds, the evaluator checks all 215 positions,
complete finite probability metrics, zero inference errors, checkpoint/dataset
binding, and the lowest-NLL winner with an earlier-epoch tie rule. All five players,
including the FP32-head ablation, are required by this study.

```sh
uv run --locked --extra ml python scripts/evaluate_clef.py tournament \
  --checkpoint PATH --seeds 30000:30200 --max-pieces 200 \
  --final-test --selection-file selection.json --output runs/final-evaluation
```

The default includes all five players. The final-test switch is an explicit
acknowledgment that these held-out seeds will now be consumed. Without it, the
script rejects any overlap with the reserved pool before loading a player.

Every completed episode is stored as `PLAYER/seed-N.json`; `progress.json` records
progress, and the final `report.json` includes all episodes and paired comparisons.
Repeat exactly the same command with `--resume` after an interruption. The script
checks request, source hashes, selected checkpoint, frozen protocol and saved replay
outcomes, skips completed players without loading them, and resumes missing seeds.
It never retries or deletes a recorded error episode. Loading is timed separately;
there are no silently excluded warmup decisions.
