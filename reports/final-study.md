# Stackcraft: completed Clef fine-tuning study

Fine-tuning improved Clef's play in this small, turn-based block game, but a fixed
arithmetic heuristic remained much stronger and cheaper. The selected model
cleared **16.81 lines per game**, compared with **0.07** for the unchanged native
model and **76.53** for the heuristic. The paired trained-minus-native improvement
was **16.74 lines, 95% bootstrap interval [15.74, 17.77]**.

These are results from all **200 held-out piece sequences**, seeds 30000–30199,
with a 200-piece cap for each of five players: 1,000 completed episodes. There
were **zero inference errors and zero invalid decisions**. Training, selection,
reload verification, final evaluation and an independent computational audit are
complete. Publication and fresh-download verification are tracked separately in
the [source repository](https://github.com/kkarimi/stackcraft).

The [compact machine-readable report](final-summary.json) contains full-precision
summaries, all paired intervals, runtime configuration and evidence hashes.

## What was tested

Stackcraft uses a 10×20 board, seven tetrominoes and deterministic seven-bag piece
sequences. A turn selects a legal rotation-and-column placement that drops
vertically. There is no hold, wall kick, tuck, T-spin scoring or real-time movement
deadline. Scores are 100/300/500/800 for clearing one/two/three/four lines in a move.
Each player receives the board, current piece, one next-piece preview and the same
legal options. Seed, sequence index and future random state are excluded from the
policy observation. This tests decisions over structured state, not visual play.

The neural players use `Cloudflare/clef-flash`, pinned to revision
`17f0b0ad64efb65d273590632833508766b2aae6`, with the same full observation encoding,
option order, BF16 backbone and 4,096-token limit. The three neural conditions are:

- **Native base:** unchanged weights and original BF16 decision head.
- **FP32 base:** unchanged weights, using the FP32 head wrapper also used in training.
- **Trained:** selected rank-4 LoRA adapters plus the learned FP32 native decision head.

**Random** chooses uniformly among legal options using its own recorded RNG.
**Heuristic** greedily scores each resulting board with weights frozen before final evaluation:
`−aggregate height − 4 × holes − bumpiness + 8 × cleared lines`. It does not use
the available preview. The lookahead teacher used for labeling is distinct from
this greedy heuristic and was not one of the five final tournament players.

## Held-out game outcomes

Entries show **mean / median** across all 200 episodes for each player. Survival
means pieces placed. A cap hit means survival of *at least* 200 pieces, not death
at piece 200.

| Player | Lines | Score | Pieces survived | Cap hits | Errors / invalid decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native base | 0.070 / 0 | 7.0 / 0 | 26.020 / 26 | 0/200 | 0 / 0 |
| FP32 base | 0.105 / 0 | 10.5 / 0 | 26.000 / 26 | 0/200 | 0 / 0 |
| Trained | 16.810 / 16 | 1745.5 / 1650 | 83.565 / 82 | 0/200 | 0 / 0 |
| Random | 0.140 / 0 | 14.0 / 0 | 26.645 / 27 | 0/200 | 0 / 0 |
| Heuristic | 76.530 / 77 | 8306.5 / 8300 | 200.000 / 200 | 200/200 | 0 / 0 |

No failures were dropped. The frozen analysis assigns zero lines, score and pieces
to any errored episode and preserves the observed partial outcome separately.
Because none failed, failure-adjusted and observed summaries coincide here.

For each comparison, subtract the second player's outcome from the first on the
same seed. Resample the 200 paired differences with replacement 10,000 times,
using bootstrap seed 2026, and take the percentile 95% interval. Resampling whole
episodes preserves dependence among moves within a game.

| Comparison | Mean line difference | Paired 95% interval |
| --- | ---: | ---: |
| Trained − native base (primary) | +16.740 | [15.740, 17.770] |
| Trained − FP32 base | +16.705 | [15.705, 17.740] |
| Trained − heuristic | −59.720 | [−60.805, −58.619875] |
| FP32 base − native base | +0.035 | [−0.020, 0.090] |

The trained gain persists against the unchanged FP32 head. Head precision alone
does not explain that gain in this experiment. The FP32-versus-native interval
includes zero. This is not an ablation separating the effects of learned LoRA
from learned head weights: both were trained together.

The primary trained-minus-native score difference was +1738.5 [1633.0, 1848.5],
and survival difference was +57.545 pieces [54.930, 60.210]. All twelve intervals
(lines, score and pieces for all four comparisons) are preserved in
[final-summary.json](final-summary.json). Lines are the primary endpoint; score
and survival intervals are descriptive secondary results, without a multiplicity
correction. These intervals describe episode variability for one selected
checkpoint, not variability across training runs.

## Data, training and checkpoint selection

The disclosed synthetic dataset contains **827 training positions and 215
validation positions**. Separate episode seed pools supplied mixed random,
heuristic and lookahead-expert trajectories, sampled for at most 40 moves per
episode. Occupancy-normalized deduplication removed six within-training duplicates
and two cross-split duplicates. The teacher enumerates placements of the current
piece and the single visible preview, scoring line clears and the resulting board;
it cannot inspect unseen pieces. Its labels are bounded-search recommendations,
not proven optimal moves. See the [dataset manifest](dataset-manifest.json) and
[Tutorial 03](../docs/tutorials/03-data.md).

One seed-42 training run produced exactly two preregistered epoch candidates.
Training used BF16 frozen backbone weights without quantization, rank-4/alpha-8
LoRA with zero dropout, and the full FP32 native joint head: 132,582,404 trainable
parameters. The fixed loss was cross-entropy with 0.05 label smoothing plus Brier
loss weighted 0.1. AdamW used learning rate 1e-5 and weight decay 0.01, with batch
size one, accumulation eight and gradient clipping at 1.0. The final accumulation
group was normalized by its actual three examples. Each epoch processed all 827
training positions in 104 optimizer steps. Frozen parameter hashes stayed unchanged.

Selection used the lowest finite mean target negative log likelihood (NLL) on
**all 215 validation positions**, with exact ties assigned to the earlier epoch.
Candidates with errors or incomplete probabilities were ineligible. Both were
eligible; epoch 02 won. Test games did not choose the checkpoint.

| Validation condition | Teacher agreement | Mean target NLL | Mean Brier |
| --- | ---: | ---: | ---: |
| Native base | 21/215 (9.77%) | 2.952702 | 0.923173 |
| FP32 base | 22/215 (10.23%) | 2.952393 | 0.923102 |
| Epoch 01 | 89/215 (41.40%) | 1.860870 | 0.722190 |
| Epoch 02, selected | 92/215 (42.79%) | 1.775923 | 0.708980 |

**42.79% teacher agreement is not 42.79% game quality.** It measures agreement
with one chosen teacher action, including its deterministic tie rule. Small errors
change subsequent states, and imitation can fail on boards absent from training.
Complete games provide a separate behavioral measurement.

The selected checkpoint reloaded in a fresh process with **0.0 maximum absolute
probability difference** on four fixed training-position references, below the
1e-4 tolerance. This checks those reference distributions, not all possible boards
or downloaded publication artifacts. Full details are in the
[training](study-training.json), [validation](validation-summary.json) and
[reload](selected-checkpoint-reload.json) reports. The training report's status
reflects its creation before external validation; later evidence completes that
stage. The [archived preregistration](study-preregistration.md) preserves the
pre-test choices; it is a local source archive, not a signed external registration.

## Runtime and practical cost

The workstation used an NVIDIA RTX 5090 (reported 32,607 MiB), AMD Ryzen 9 9950X3D,
eight Torch CPU threads and about 123 GiB host RAM. Runtime was Python 3.13.16,
Torch 2.14.1+cu130, CUDA 13.0, Transformers 5.18.0 and PEFT 0.21.2. TF32 was disabled.
The [hardware snapshot](evaluation-hardware.json) records the remaining details.

| Player | Mean decision ms | Median ms | p95 ms | Decisions | Local model initialization s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native base | 149.110 | 139.381 | 244.916 | 5,204 | 3.866 |
| FP32 base | 150.188 | 140.968 | 247.880 | 5,200 | 2.176 |
| Trained | 206.814 | 171.844 | 309.000 | 16,713 | 2.528 |
| Random | 0.001497 | 0.001250 | 0.002879 | 5,329 | 0.000035 |
| Heuristic | 0.198510 | 0.139489 | 0.280528 | 40,000 | 0.000002 |

Latencies cover all `Player.choose` calls, including encoding and probability
transfer, with no excluded warmup. They exclude rendering, replay serialization
and model loading. Initialization was measured separately with weights already
cached locally; these are not download or reliably disk-cold timings. The neural
players' mean input lengths were 1,354.30, 1,349.73 and 1,532.11 tokens respectively
(native, FP32, trained); the trained p95 was 2,259 tokens.

The trained model took about **1,042 times** the heuristic's mean decision time
while clearing far fewer lines. This is an observed workload comparison: neural
players use the GPU, the heuristic uses the CPU, and policies visit different
boards. It is not a controlled same-state kernel benchmark. For a practical game
bot under these rules, the heuristic is the clear choice. The fine-tuning result
is useful as a reproducible learning experiment, not a reason to replace the
heuristic in production.

Training took **1,511.25 seconds (25.2 minutes)** including checkpoint and hash
work; peak CUDA allocation was **23.23 GB (21.64 GiB)** and reservation 24.18 GB.
The final evaluation wrapper took **5,103.11 seconds (85.1 minutes)** including
loading, reporting and the local service stop/restore health checks. This was
within the pre-test multi-hour budget. These durations exclude earlier development,
validation and dataset generation. No energy consumption or monetary cost was
measured; the GPU's configured 575 W ceiling is not an energy-use measurement.

## A fixed replay illustration

Seed **30000**, the first reserved seed and the declared demo sequence, provides
an exploratory illustration. These values come from saved replays; no additional
games were generated for this description.

| Player | Lines | Pieces | Final maximum height | Final holes |
| --- | ---: | ---: | ---: | ---: |
| Native base | 0 | 23 | 20 | 86 |
| FP32 base | 0 | 26 | 20 | 79 |
| Trained | 16 | 82 | 20 | 28 |
| Random | 0 | 24 | 20 | 88 |
| Heuristic | 78 | 200 (cap) | 3 | 1 |

A hole is an empty cell below an occupied cell in its column. These are different
endpoints: the heuristic is still alive at the cap, whereas the other players
have topped out. The trained player clears lines but eventually reaches the top.
The table describes this replay; it does not establish why the model fails or
prove a causal relationship between final holes and the aggregate result.

## Reproducibility and remaining limits

The independently implemented CPU audit checked all 1,000 persisted episodes:
replayed states, observations, choices, outcomes, source/checkpoint/selection
bindings and aggregate statistics. It independently recomputed all twelve paired
intervals and matched the report. The [audit results](final-audit.json) and
[archived audit source](final-audit-source.md) preserve that check; its script hash
and compact outcome are also included in the summary. This is an independent code-path check by another agent,
not an external scientific replication.

The full evaluation JSON is identified by SHA-256
`5585574b77e81c8d7dd44464c4dfb09f9511ccec0d9150c3ea44793760416857`.
The frozen selection SHA-256 is
`025b62587a26bd8cc78c80f362524b8c178190df5188b317299121afa6b1203b`.
Dataset manifest, checkpoint files, evaluation source files and supporting report
hashes are all included in [final-summary.json](final-summary.json).

The prepared model-release layout contains `code/` with these reports, source,
locked environment, tests and tutorials; `evidence.zip` losslessly preserves
`evidence/evaluation/` with the full
report, request and per-player episode records; and `evidence/selection/` with
the frozen choice and both candidates' validation evidence. Extract the ZIP from
the model repository root to restore these exact paths; `evidence-files.json`
records their unchanged uncompressed hashes and sizes. These portable paths
are relative to the model repository root and do not require access to private Git
history. See the [model release](https://huggingface.co/nima1/stackcraft-clef-flash-lora),
[dataset](https://huggingface.co/datasets/nima1/stackcraft-data), and
[local demo instructions](https://github.com/kkarimi/stackcraft/blob/main/docs/demo-hosting.md).
The owner deferred hosted deployment; no live Space is claimed.
[Tutorial 05](../docs/tutorials/05-evaluation.md) gives the reproduction procedure.

The principal limitations are one training seed, a small synthetic dataset, an
approximate teacher, a single structured encoding and simplified rules. Training
positions cover shorter trajectories than the 200-piece test horizon. All 200
heuristic games reached the cap, so their uncapped longevity is unknown. The weak
unchanged-model results characterize this pinned model and encoding, not every
possible way to use Clef. No model, prompt, dataset, policy or checkpoint was tuned
after opening this final test. Future improvements require a new declared study
and fresh held-out sequences; these results remain the record of this experiment.
