---
license: apache-2.0
language:
- en
tags:
- synthetic
- imitation-learning
- games
- stackcraft
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# Stackcraft search-teacher positions

A synthetic dataset for a deterministic, simplified falling-block placement game.
**827 training positions and215 validation positions**, generated entirely by
code. No human demonstrations, personal data, web scraping or human row-by-row
label review are claimed. Independent agent review recomputed16 pilot labels;
that sample audit is not a review of every row.

Each row contains a visible board, current piece, one next piece, all legal actions,
a teacher action, teacher action values and provenance. The board uses integer
piece colors; the model encoding reduces them to occupancy because color does not
affect these rules. Metadata includes seed and turn for reproduction; **do not
feed seed or hidden sequence metadata to the policy**. Stackcraft's encoder feeds
only the visible observation. Read the dataset schema and tutorial03 before use.

## Generation

The teacher searches the current placement and exactly one preview placement.
It rewards current line clears and the best next afterboard, using fixed weights
for aggregate height, holes, bumpiness and cleared lines. No later pieces are
available. Labels are heuristic recommendations, not proofs of optimal play.

Collection cycles random, heuristic and search-expert policies, at a40-piece cap,
to expose different board conditions. Training seeds10000–10023 and validation
seeds20000–20005 are disjoint. Six within-training and two cross-split duplicate
observations were excluded. Final test seeds30000–30199 are reserved for complete
game evaluation; there is no test-position split for training or selection.

Source commit: `70d84bd8dc5d4a60f3b96455a57d9e6f9416d109`.
The bundle includes `manifest.json` with source/configuration and split hashes.

- train SHA256: `edd682761db95a4f25bb30a284489c54d9336a36da0a9b19d2cda860b428baa8`
- validation SHA256: `eff9cdc5932e935959ac4d26dce6470d335090f7428a91930001954266d133bc`

The model release includes public source under `code/` and the exact regeneration
recipe in `code/docs/tutorials/03-data.md`. Use its **Reproduce from the public
source bundle without private Git access** procedure: first verify all six generator
source hashes against this dataset's `manifest.json`, then generate with the
original source-commit provenance. Access to the original Git repository is not
required. The procedure reproduced both JSONL hashes exactly on the study machine.
Ordinary `stackcraft generate-data --output NEWDIR` records the current source
identity; a new source commit deliberately changes provenance fields and therefore
file hashes even if labels are identical. A regeneration and an independent
sample-label audit preceded study training.

## Limits

The study is small and entirely synthetic. Teacher bias, narrow collection seeds,
short collection episodes and one-step preview limit generalization. Duplicate
removal prevents exact overlap, not all structural similarity. It is not full
real-time Tetris: no hold, kicks, tucks or gravity timing. Colors and piece IDs
are program-generated; Stackcraft uses original game visuals and branding.

Use teacher agreement as a diagnostic and complete held-out games for outcome
measurement. Do not call a model better solely because it copies these labels.
Apache-2.0; see LICENSE and accompanying generation/reproduction tutorials.
