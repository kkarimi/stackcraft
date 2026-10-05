# Native Clef adapter

Status: native encoding, real-weight BF16 inference and bounded head/LoRA GPU
training have passed. Fresh-process checkpoint parity and full held-out quality
are separate gates; see the plan and reports for their current status. Fake
components in unit tests exercise mapping and failure handling only.

## Pinned upstream contract

- Model: `Cloudflare/clef-flash`.
- Revision: `17f0b0ad64efb65d273590632833508766b2aae6`.
- Native file: `joint_schema_model.py`.
- Source SHA256: `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
- Observation encoding: `stackcraft-clef-v1`.

Inspected source: [pinned native implementation](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_schema_model.py).
The loader pins a complete snapshot and supplies its local directory to upstream
`load_release_model`; passing only the repository name to upstream would download
an unpinned snapshot. `import_pinned_source` checks exact file bytes before
executing them. Callers must explicitly set `trust_pinned_code=True`: this executes
reviewed upstream Python with the process's permissions, not in a sandbox.

The unchanged loader constructs `Qwen3_5ForConditionalGeneration`, loads a separate
`JointSchemaHead`, then constructs `AutoProcessor`. Its processor configuration is
multimodal even though this application passes only text. Optional ML dependencies
must therefore support that native construction; importing `stackcraft.clef` or
playing the game itself requires none of them.

## Input and outputs

`observation_record(observation)` includes the 20 board rows, current piece, exactly
one preview piece, rules and coordinate definitions. Occupied cells are `#` and
empty cells are `.`; colors carry no decision-relevant information in these rules.
The native choice question contains every legal action ID with its rotation,
column, landing row and four absolute cell coordinates. It does not include a seed,
hidden stream, score-derived helper features, or heuristic candidate pruning.

Native choice encoding sorts option IDs lexicographically. Probabilities must be
mapped using `EncodedQuestion.option_ids`, rather than the input dictionary order.
The adapter reads `model(collate_records([encoded], pad_id, device))[0][0]` and takes
FP32 softmax. It retains unrounded float probabilities. Upstream `systemone()`
rounds probabilities to four decimals, which is unsuitable for precise validation
and probability metrics. On exact probability ties, Stackcraft uses the shared
engine action order.

## Context integrity

The native encoder reserves schema tokens and silently truncates state tokens to
fit `max_length`. Losing some board rows would change the game state presented to
the model. Before encoding, Stackcraft counts every separately tokenized segment
used by the pinned encoder. It rejects the entire decision if complete input does
not fit the configured limit, then checks the resulting length and option IDs.
It never removes choices or board rows to make a record fit.

The default limit is 4,096 tokens, a memory-conscious initial setting, not a claim
about the backbone's advertised context. If observed positions exceed it, measure
the required size and preregister a consistently larger limit for both base and
trained models. Any different text representation needs a new encoding version.

## Integration example

After ML dependencies and the pinned snapshot are available, and available GPU
memory has been checked:

```python
from stackcraft.clef import ClefPlayer
from stackcraft.engine import new_game
from stackcraft.players import observe

player = ClefPlayer.from_pretrained(
    trust_pinned_code=True,
    local_files_only=True,
    device="cuda",
    max_length=4096,
)
decision = player.choose(observe(new_game(42)))
print(decision.action_id, decision.probabilities, player.last_input_tokens)
```

This example loads real 9B weights. Real inference passed on an RTX 5090.
The loader does not stop other workloads, allocate rented compute, or implicitly
download missing weights by default. A caller can deliberately set
`local_files_only=False` to obtain this pinned snapshot. Model identity, source
hash and encoding version must accompany measurements.

`player.runtime_config` records model ID, player/checkpoint revision, pinned base
revision, encoding version, native source SHA256 and configured context limit.
After each successful decision it also records the actual first model parameter's
dtype and device. These are initially null until inference succeeds; construction
does not consume model parameters or import Torch for metadata. The native release
uses one device and dtype; this field is not a full precision inventory for future
mixed-precision training models. `last_input_tokens` records the successful
decision's complete encoded length. Tournament artifacts should snapshot runtime
metadata and retain token counts per decision, rather than infer them from defaults.

Run the offline adapter checks with:

```sh
uv run --locked pytest tests/test_clef.py
```

CPU verification used the actual `Qwen3VLProcessor` and native `encode_record`,
without constructing a backbone or loading weights. Fourteen fixtures covered
all seven pieces on an empty board and on a dense 15-row patterned stack, including
34-choice positions. Exact token counts matched in every case: 848–2,248 tokens,
all within 4,096. Rejecting a limit one token below each required size also passes
the dedicated integration check. These fixtures do not establish a universal
maximum; every inference request still runs preflight.

Reproduce that optional check after installing the ML extra and caching pinned
tokenizer/processor/source files:

```sh
STACKCRAFT_TEST_NATIVE_ENCODING=1 uv run --locked --extra ml pytest tests/test_clef.py
```

The two-seed real-weight development baseline used19.69GB peak allocated memory
and averaged144ms per decision; reports/clef-development.json records the result.
These development measurements do not establish trained-model quality. Training must call native
`ClefModel.forward` with gradients enabled; `choose()` intentionally uses inference
mode and must not be used as a training path.
