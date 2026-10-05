# Tutorial 04 — Prove training and reload before running an experiment

This milestone asks a narrow engineering question: can the real pinned Clef-flash
model take gradient steps on this GPU, save all learned parameters, and reproduce
its probabilities in a fresh process? It does not ask whether the model plays
better. The latter requires complete games on the held-out sequences in Milestone 5.

**The bounded real-weight training and fresh-process reload gate passed.** Both
head-only and LoRA-plus-head artifacts reproduced their reference probabilities
exactly. The first LoRA reload exposed a saved-configuration compatibility issue;
the corrected loader and successful retry are documented below. This establishes
local training feasibility, not improved game performance.

## Start from the native decision model

The base is `Cloudflare/clef-flash`, revision
`17f0b0ad64efb65d273590632833508766b2aae6`. The local loader verifies the pinned
`joint_schema_model.py` source hash and loads an explicitly resolved snapshot.
It requires `trust_pinned_code=True`: pinning and inspecting Python source controls
which code executes; it does not turn that code into a sandbox.

An observation becomes one native choice question containing every legal
placement. The answer is a distribution over action IDs, not a generated command
or natural-language explanation. `encode_observation` checks the complete token
budget before native encoding, because silently truncating a board would change
the task. Training uses only the visible board, current piece, one preview and
legal placements. Dataset seed and episode metadata remain outside the input.

The native encoder sorts option IDs lexicographically. `decision_loss` locates the
teacher's action in `encoded.questions[0].option_ids`; it must not reuse the
engine's rotation/column index as a target index. Both orders contain the same
legal moves, but their order can differ.

Do not train through `systemone()` or `ClefPlayer.choose()`: those are inference
paths. The probe calls `native.collate_records` and then `model(batch)` with
normal autograd enabled. `choose()` is appropriate for the before/after reference
probabilities, where gradients are unnecessary.

## Keep the backbone small in memory and the head stable

`src/stackcraft/training.py` implements two bounded configurations:

| Configuration | Updated parameters | Purpose |
| --- | --- | --- |
| Head only | Native joint decision head | Isolate head training and checkpoint handling before a backbone backward pass. |
| Rank-4 LoRA plus head | Small added text-layer matrices and the native head | Check whether the task can adapt both text representations and decision scoring. |

The backbone remains BF16. The native decision head is converted to FP32, and
`FP32DecisionHead` disables autocast inside the head. Its floating inputs are
converted to FP32 while token IDs and masks retain their original types. This
keeps the trainable scoring operations in full precision without converting the
whole multi-billion-parameter backbone.

The head reads selected rows of the output vocabulary embedding. Converting the
entire vocabulary matrix to FP32 first creates a large temporary allocation
(roughly 3.79 GiB for this release). `GatheredFloat32Embedding` instead implements
`weight[indices].float()`: select the needed rows first, then cast. This preserves
the actual native head computation and autograd connections. A tiny actual-native-
head test compares outputs and gradients with the full-cast reference exactly.
That test checks the optimization, not full-model training feasibility.

LoRA targets are full module paths under the text transformer layers. They include
both ordinary attention projections and Qwen3.5's linear-attention projections,
as well as the MLP projections. Matching only suffixes such as `q_proj` could
accidentally select the vision encoder; matching only ordinary attention would
miss the hybrid backbone. The vision tower, vocabulary output layer and unrelated
modules are excluded. Original backbone parameters are frozen. The probe uses
rank 4, alpha 8, no LoRA dropout and no trained backbone biases.

Non-reentrant gradient checkpointing recomputes intermediate activations during
backward instead of retaining all of them. This trades time for memory while
preserving gradients through frozen parts of the backbone into LoRA matrices.
A tiny actual Qwen3.5 hybrid-backbone test checks nonzero LoRA gradients in ordinary
attention, linear attention and MLP layers. These are stronger plumbing checks
than testing a stand-in linear model alone, but still do not establish the real
9B GPU memory requirement.

We start with BF16 rather than immediately adding 4-bit quantized training. The
native model has a custom head and loader; another quantization layer would add
an unverified compatibility dependency. If BF16 fails, record the failure and
make a bounded next decision rather than silently changing the experiment.

## Prepare before borrowing the GPU

Run these commands from the repository root or the public bundle's `code/`
directory. Install the ML tools and cache the pinned release before testing:

```bash
uv sync --locked --extra ml --group dev
uv run --locked --extra ml hf download Cloudflare/clef-flash \
  --revision 17f0b0ad64efb65d273590632833508766b2aae6
```

Downloading only populates the cache; it does not load GPU weights. It can run
while other GPU services remain available. The actual native-head tests need
this cached source. Explicitly enable the native tokenizer/encoder check too:

```bash
STACKCRAFT_TEST_NATIVE_ENCODING=1 uv run --locked --extra ml \
  pytest tests/test_training.py tests/test_clef.py
```

These checks use CPU fixtures and the real tokenizer/source; they do not load
the full backbone or establish GPU feasibility. Inspect the test summary: a
skipped test is not a passed native-model test.

The training probe expects the audited `data/study-v1` bundle established by
Tutorial03's exact-study reconstruction or a verified dataset download. If you
used a different directory, pass that same path with `--dataset` below.
It validates the train/validation bundle, then uses only the first four training
positions. It does not sample held-out test seeds. Record the current source
commit and any uncommitted probe changes in `record.md` before running.

This workstation normally runs a local GLM Docker service. The user explicitly
authorized temporarily stopping that service and restoring it after these
Stackcraft experiments. `scripts/gpu_session.py` is specific to that approved
service; it is not a generic instruction to stop another machine's workloads.
It first requires the configured container to be running, then stops it, runs a
bounded child job, and attempts restoration during cleanup. The training script
itself never stops services. It refuses to load unless CUDA is available and at
least **25 GiB is free**. Free-memory admission is a precondition, not a guarantee
that backward will fit.

## Run the bounded real-weight probe

The initial budget is two configurations and at most 30 minutes of child-job
time. The commands below allocate 20 minutes to the training probe and 10 minutes
to the separate reload check. Choose fresh output and session-record paths on a
retry; neither existing evidence nor checkpoints should be overwritten.

```bash
uv run --locked --extra ml python scripts/gpu_session.py \
  --timeout 1200 --record runs/gpu-training-probe-v1.json -- \
  uv run --locked --extra ml python scripts/probe_clef_training.py \
  --dataset data/study-v1 --output runs/clef-training-probe-v1
```

The probe first records unchanged native probabilities. It measures the initial
probability drift introduced by moving the head to FP32, before training; a
precision change must not be mistaken for a learned improvement. It then runs
three head-only steps. After saving that checkpoint, it releases the model and
loads a fresh unchanged base for five LoRA-plus-head steps. The LoRA experiment
therefore does not inherit the head-only warm-up.

Both configurations use batch size one, AdamW with learning rate `1e-5`, norm
clipping at 1.0, and this loss:

    smoothed_cross_entropy(label_smoothing=0.05)
      + 0.1 * sum_over_options((probability - one_hot_teacher_label)**2)

The cross-entropy term learns the expert action. Smoothing avoids assigning the
entire training target to one action. The Brier term also penalizes the predicted
probability distribution's distance from the teacher target. These fixed choices
are a probe configuration, not evidence of calibrated confidence or optimal
hyperparameters.

For every step, the probe checks finite loss and gradients and records nonzero
head/LoRA gradient sums, tokens, elapsed time, and CUDA peak allocated/reserved
bytes. Before/after parameter hashes check that intended parameters changed and
frozen parameters did not. Hashing the entire frozen backbone costs time but
checks something that `requires_grad=False` alone cannot prove: what bytes actually
changed during this run.

Inspect `runs/clef-training-probe-v1/report.json` and the session record. A timeout,
CUDA error, absent gradient or failed hash check is a failed gate. A report left at
`running` after forced termination is incomplete evidence, even if some checkpoint
files exist. Do not start long training merely because loss fell on four examples.

## Save the head as well as the adapter

A LoRA checkpoint has this structure:

```text
checkpoint/
  joint_head.safetensors
  adapter/
    adapter_config.json
    adapter_model.safetensors
  training_config.json
  reference.json
```

The head is an independently trained part of Clef; saving only the PEFT adapter
would omit learned parameters. `joint_head.safetensors` stores FP32 native head
keys without wrapper-specific prefixes. The adapter stores only the added LoRA
weights, not another copy of the frozen base. Metadata identifies the base
revision, native source hash, input encoding, head format, LoRA settings and
teacher rows. The loader checks this contract, head shapes and dtypes, and actual
saved adapter configuration before restoring weights.

The head-only probe writes `head-checkpoint/` without an adapter. It is diagnostic
output, not a selected release model.

## Require a separate fresh-process reload

Only after the first probe succeeds, run:

```bash
uv run --locked --extra ml python scripts/gpu_session.py \
  --timeout 600 --record runs/gpu-training-reload-v1.json -- \
  uv run --locked --extra ml python scripts/probe_clef_training.py \
  --dataset data/study-v1 --output runs/clef-training-reload-v1 \
  --reload runs/clef-training-probe-v1/checkpoint
```

This starts another Python process, loads the unchanged pinned base, restores both
adapter and head, and evaluates the same four training observations. Require the
same option IDs and a **maximum absolute probability difference of at most
`1e-4`** across all recorded options. The tolerance was declared before the run.
Compare unrounded probabilities; rounded display values can conceal a mismatch.

A same-process test could accidentally reuse trained parameters that were never
saved. The fresh process tests the actual inference artifact. A passing reload
establishes serialization fidelity on these references, not general game skill
or correctness on every possible input.

## Restore service and record the limits

Inspect both `restored_running` and `restored_healthy` after success and failure.
The wrapper checks `/health` inside the restored container and allows up to 120
seconds for readiness, retrying transient timeouts and malformed startup responses.
It does not query the unrelated service on host port 8080. A running process alone
would not establish readiness. Nested cleanup ensures a child-termination error
still reaches restoration; repeated ordinary interrupts are ignored during that
cleanup. Preserve the wrapper record, probe reports and console logs, including
errors. Mocked tests exercise these recovery paths without touching Docker.

Process cleanup cannot guarantee recovery after `SIGKILL`, host power loss or a
Docker daemon failure. If the wrapper was forcibly terminated, inspect the exact
approved container and any remaining job before acting. Restore the existing
service only after the experiment has released GPU memory; do not start another
probe simply because the last terminal stopped printing output.

The earlier M2 real-weight development pilot is useful context: on this RTX 5090,
it recorded 55 unchanged-Clef decisions at a mean of about **0.144 seconds per
decision**, with **19,686,539,776 bytes** peak CUDA allocation (about 19.7 GB,
18.3 GiB). The benchmark script took 12.60 seconds; the stop/job/restore wrapper
interval was about 14.20 seconds and recorded successful restoration. These are
inference measurements on two short development games, not training memory or
held-out quality results. Sources are `runs/clef-development-v1.json` and
`runs/gpu-baseline-session.json`.

## Measured training probe and the first reload failure

The real-weight probe passed on 2026-10-05 using the RTX 5090 and Torch
`2.14.1+cu130`. Evidence is `runs/probe-v1/report.json`; the actual run directory
differs from the fresh reproduction paths shown above. It used training positions
`seed-10000-turn-0` through `seed-10000-turn-3`, with 1,302–2,252 encoded tokens.

| Measurement | Head-only probe | LoRA-plus-head probe |
| --- | ---: | ---: |
| Optimization steps | 3 | 5 |
| Observed step times | 0.154–0.328 s | 0.670–1.322 s |
| Peak allocated CUDA bytes | 21,323,760,640 | 23,180,389,888 |
| Peak reserved CUDA bytes | 21,541,945,344 | 23,595,057,152 |
| Intended parameter changes | Verified | Verified |
| Frozen parameter hashes | Unchanged | Unchanged |

All reported losses and gradients were finite, with nonzero gradients in the
intended groups. Total probe time was **51.81 seconds**, including model reload,
parameter hashing and checkpoint work. This is not an estimate of a full epoch:
only four short training positions were exercised. Peak reserved memory is the
PyTorch allocator's reservation, not the same quantity as all memory reported by
`nvidia-smi`.

The FP32 head changed initial probabilities by at most **0.00074412** before any
training. That is a precision effect relative to the original BF16 head, not a
failed serialization comparison. The `1e-4` reload threshold compares the saved
trained FP32-head model with its own fresh reload.

The first fresh reload, `runs/probe-reload-v1/report.json`, failed after 4.15
seconds because the saved adapter configuration differed from our metadata. PEFT
automatically shortened **248 full target paths to 12 module suffixes**. The strict
loader correctly refused a mismatch; the failure does not by itself mean the
weight tensors are corrupt. Accepting suffixes without checking what they match
could attach an adapter to unintended modules. The correction must expand saved
target selectors against the actual pinned backbone and require exactly the
intended full target set, rejecting any extra destinations. This is why fresh
reload is a milestone gate rather than a final packaging chore.

The corrected loader resolves saved selectors against the pinned backbone and
requires the exact intended destinations. The retry preserved the original
checkpoint and failed report; it changed validation of equivalent target selectors,
not the trained weights. A regression test covers target shortening and rejects
selectors that also match unintended modules.

Both independent fresh-process checks then passed:

| Artifact | Report | Maximum probability difference | Process time |
| --- | --- | ---: | ---: |
| Head only | `runs/probe-head-reload-v1/report.json` | 0.0 | 5.44 s |
| LoRA plus head | `runs/probe-reload-v2/report.json` | 0.0 | 5.65 s |

The LoRA retry used source commit
`f7be7ceee9c38566dfebad0ebf5f1c04958faee3`. The dataset manifest SHA-256 was
`aeddcfc1ea5390f12122d2e786c1e2030dcc97d3790b6d7523432548198b1f94` throughout.
The approved GLM service was restored and passed its health check after these
probes. **M4 is complete.** Milestone 5 can now train a selected checkpoint and
measure complete games; the four-position probe does not demonstrate that the
trained policy wins or generalizes.
