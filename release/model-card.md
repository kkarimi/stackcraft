---
license: apache-2.0
base_model: Cloudflare/clef-flash
base_model_relation: adapter
library_name: stackcraft
language:
- en
tags:
- clef
- decision-model
- lora
- imitation-learning
- stackcraft
---

# Stackcraft Clef-Flash

[Read the technical report (PDF)](https://huggingface.co/nima1/stackcraft-clef-flash-lora/resolve/main/paper/stackcraft-technical-report.pdf) ·
[Paper source and build instructions](https://github.com/kkarimi/stackcraft/tree/main/paper)

The report presents the completed synthetic study, with methods, paired results,
limitations and AI-assistance disclosure. It is not peer reviewed.

**Release preparation: full-study results and selected checkpoint are pending.**
This is a small, synthetic imitation-learning study of a falling-block placement
policy. It is not a generally capable game agent, reinforcement-learning result,
or competitive Tetris implementation.

The artifact contains a rank-4 LoRA adapter **and a separately trained FP32 native
joint decision head** for `Cloudflare/clef-flash`, pinned revision
`17f0b0ad64efb65d273590632833508766b2aae6`. Load both. The frozen BF16 backbone
is downloaded separately; these files are not a standalone model. Clef-flash is
based on Qwen3.5-9B. The native schema head scores every supplied legal placement
in one forward pass, with no generated text or command parsing.

Related releases: [source and tutorials](https://github.com/kkarimi/stackcraft),
[model](https://huggingface.co/nima1/stackcraft-clef-flash-lora),
[dataset](https://huggingface.co/datasets/nima1/stackcraft-data), and
[local playable demo](https://github.com/kkarimi/stackcraft/blob/main/docs/demo-hosting.md).
The owner deferred hosted deployment. The game and Docker image need no GPU
or Hugging Face subscription when run locally.

## Intended use

Learn how to build deterministic environments, generate search-teacher labels,
train a custom decision model, preserve its native head, and evaluate complete
games with paired uncertainty. The playable demo uses recorded model replays;
it does not provide public live GPU inference.

The observation is a 10×20 occupied-cell grid, current tetromino, exactly one next
piece, rules and every legal placement with its coordinates. The action selects
an orientation and column, then drops vertically. There is no hold, wall kick,
tuck, gravity timer, T-spin bonus or changing speed. Hidden future pieces and
sequence seeds are excluded from neural input. Probabilities reflect the model's
choice distribution; they are not calibrated chances of clearing a line or winning.

## Training and data

The study uses 827 training and 215 validation positions generated from fixed
seven-bag streams. Random, heuristic and search policies collect states. Every
label comes from a fixed two-placement search using the current piece and one
preview, with a board-height/holes/bumpiness/line-clear heuristic. These are
synthetic teacher preferences, not human demonstrations or globally optimal moves.
Exact duplicate observations are removed across splits. Six source hashes and
exact file hashes accompany the dataset manifest.

Preregistered training: seed 42, two epochs, batch 1, accumulation 8, rank 4/alpha 8,
no LoRA dropout, AdamW lr 1e-5/weight_decay 0.01, gradient norm cap 1.0, label smoothing
0.05 and Brier sum weight0.1. BF16 backbone, FP32 head, text-layer LoRA including
both hybrid attention types and MLPs. Vision, output and other original parameters
remain frozen. There are 132,582,404 trainable parameters: 121,762,820 in the
joint head and 10,819,584 in LoRA. This is why the head is a substantial part of
the saved artifact, even though most backbone weights stay frozen. Select only between epoch 01/02 using lowest finite target NLL on
all 215 validation positions; exact ties choose the earlier epoch.

## Evaluation and limitations

Final preregistered comparison: 200 untouched seeds 30000–30199, 200-piece cap,
selected model versus unchanged native BF16-head Clef, unchanged FP32-head
ablation, random legal placement and a cheap heuristic. Primary metric is lines
cleared. Report score, survival, cap hits, failures and latency. Paired 95% bootstrap
intervals use 10000 episode resamples with seed 2026. Failed episodes remain in the
analysis, assigned zero primary outcomes with partial outcomes reported separately.

**Insert verified held-out results here before publication.**

Fine-tuning improved complete-game play over both unchanged Clef references in
this study. The fixed heuristic still cleared substantially more lines and used
about one thousandth of the trained model's decision time on this workstation.
All five policies had zero errors and invalid decisions. Every heuristic game hit
the 200-piece cap; no neural or random game did. This is evidence of task adaptation,
not a reason to prefer this 9B model over the heuristic for the simplified game.
See [the readable study report](code/reports/final-study.md) and
[the hardware/runtime snapshot](code/reports/evaluation-hardware.json).


Only one training seed, one small synthetic dataset, one simplified game ruleset
and one hardware/software configuration are studied. The finite cap censors
survival. Validation teacher agreement does not establish game quality. Test
uncertainty does not capture variation across training seeds.

## Loading

Use the released Stackcraft source and locked ML dependencies. Stackcraft uses
PEFT internally but requires its custom native-head loader. Generic
`AutoModel` or adapter-only loading omits the custom decision head and is incorrect.
From the downloaded model repository's `code/` directory, install the locked ML
environment with `uv sync --locked --extra ml` and cache the pinned upstream
snapshot as described in [Tutorial 02](code/docs/tutorials/02-baselines.md).
Run this example with `uv run --locked --extra ml python`; the sibling
`../checkpoint` directory contains the released adapter and head:

```python
from stackcraft.clef import ClefPlayer
from stackcraft.training import load_checkpoint
from stackcraft.engine import new_game
from stackcraft.players import observe

player = ClefPlayer.from_pretrained(trust_pinned_code=True, local_files_only=True)
load_checkpoint(player.model, "../checkpoint")
print(player.choose(observe(new_game(42))))
```

`trust_pinned_code=True` executes reviewed, hash-checked upstream Python. It is
not a sandbox. The measured RTX 5090 inference baseline allocated 19.7 GB; bounded
LoRA training allocated 23.2 GB. Load only with sufficient free VRAM. Context is
limited to 4096 tokens, with explicit rejection before any state truncation.

See the released tutorials, training configuration, artifact hashes and evaluation
report for exact reproduction. Apache-2.0; see LICENSE and NOTICE for upstream
attribution. This project is independent of Cloudflare, Qwen and Tetris.
