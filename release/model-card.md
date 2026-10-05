---
license: apache-2.0
base_model: Cloudflare/clef-flash
base_model_relation: adapter
library_name: peft
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

Preregistered training: seed42, two epochs, batch1, accumulation8, rank4/alpha8,
no LoRA dropout, AdamW lr1e-5/weight_decay0.01, gradient norm cap1.0, label smoothing
0.05 and Brier sum weight0.1. BF16 backbone, FP32 head, text-layer LoRA including
both hybrid attention types and MLPs. Vision, output and other original parameters
remain frozen. There are132,582,404 trainable parameters:121,762,820 in the
joint head and10,819,584 in LoRA. This is why the head is a substantial part of
the saved artifact, even though most backbone weights stay frozen. Select only between epoch01/02 using lowest finite target NLL on
all215 validation positions; exact ties choose the earlier epoch.

## Evaluation and limitations

Final preregistered comparison: 200 untouched seeds30000–30199, 200-piece cap,
selected model versus unchanged native BF16-head Clef, unchanged FP32-head
ablation, random legal placement and a cheap heuristic. Primary metric is lines
cleared. Report score, survival, cap hits, failures and latency. Paired95% bootstrap
intervals use10000 episode resamples with seed2026. Failed episodes remain in the
analysis, assigned zero primary outcomes with partial outcomes reported separately.

**Insert verified held-out results here before publication.** A successful GPU
training/checkpoint test is not evidence of better play. The initial unchanged
model cleared zero lines in two short development games; those games are not the
test result. The heuristic may outperform the neural model substantially.

Only one training seed, one small synthetic dataset, one simplified game ruleset
and one hardware/software configuration are studied. The finite cap censors
survival. Validation teacher agreement does not establish game quality. Test
uncertainty does not capture variation across training seeds.

## Loading

Use the released Stackcraft source and locked ML dependencies. Generic
`AutoModel` or adapter-only loading omits the custom decision head and is incorrect.
Cache the pinned upstream snapshot, then:

```python
from stackcraft.clef import ClefPlayer
from stackcraft.training import load_checkpoint
from stackcraft.engine import new_game
from stackcraft.players import observe

player = ClefPlayer.from_pretrained(trust_pinned_code=True, local_files_only=True)
load_checkpoint(player.model, "downloaded-stackcraft-checkpoint")
print(player.choose(observe(new_game(42))))
```

`trust_pinned_code=True` executes reviewed, hash-checked upstream Python. It is
not a sandbox. The measured RTX5090 inference baseline allocated19.7GB; bounded
LoRA training allocated23.2GB. Load only with sufficient free VRAM. Context is
limited to4096tokens, with explicit rejection before any state truncation.

See the released tutorials, training configuration, artifact hashes and evaluation
report for exact reproduction. Apache-2.0; see LICENSE and NOTICE for upstream
attribution. This project is independent of Cloudflare, Qwen and Tetris.
