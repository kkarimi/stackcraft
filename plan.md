# Complete Stackcraft


Status: M0–M7 complete for the agreed scope. Hosted CPU deployment remains deferred by user on 2026-10-06. This living plan owns the Stackcraft goal. Update Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective as work proceeds.

## Purpose


Build a playable falling-block game that shows whether fine-tuning improves a decision model. A visitor can play and watch a heuristic bot, unchanged Clef-flash, and our trained model receive the same piece sequence. Publish reproducible training and evaluation artifacts with a tutorial for every milestone. Winning against the heuristic is an empirical question, not a promised result.

## Context and Orientation


Working directory: `/home/nima/Work/stackcraft`. This is a new project; only planning files exist at goal creation. Previous projects in sibling directories are Bashcraft and Decisioncraft. Do not modify them. The user prefers uv, modern Python tooling, local commits, repository planning documents, and safe parallel work. Continue those preferences without asking again.

Clef returns probabilities over supplied choices rather than generating a move as prose. Clef-flash has approximately 9 billion parameters and a custom decision head, the component that converts model representations into choice probabilities. LoRA trains small added parameter matrices rather than all backbone weights. Quantized training uses compressed backbone weights but must be tested with this custom head; memory fit is not established. The machine previously reported an RTX 5090 with 32 GB VRAM. Recheck free memory before any model load and do not terminate other workloads.

## Plan Layout and Workflow


`plan.md` is the source of truth for milestones. `record.md` stores compact evidence and experiment intentions. Tutorials belong in `docs/tutorials/00-foundation.md` through `07-technical-report.md`. The M7 paper source, isolated uv environment and generated PDF live in `paper/`. Keep plans, concise records, source, tests and tutorials in Git. Ignore environments, credentials, downloaded weights, raw datasets, checkpoints, and bulky run outputs; retain artifact hashes and release links in reports.

Use local commits on main initially; no pull requests are required. Review each diff before committing. Remote repository and artifact names will use Stackcraft where available. Confirm new-project destinations and visibility before external publication; previous private-GitHub/public-Hugging-Face preferences are context, not a new publication decision. Prepare concrete release artifacts before requesting that final decision. No paid cloud compute is authorized by this plan.

Use one plan while milestones share a single game and evaluation contract. Delegate bounded work such as UI implementation, independent engine review, or training feasibility after interfaces are written. Give each worker exclusive file ownership; the primary agent integrates and updates the plan. Split child plans only when independent work requires it.

## Work Boundaries


Version one is a documented turn-based placement game on a 10-column, 20-row board with seven tetromino shapes and deterministic seven-bag sequences (each shuffled bag contains every shape once). Each turn chooses an orientation and horizontal position, then drops the piece vertically. Exclude hold, wall kicks, tucks, real-time gravity deadlines, and T-spin bonuses initially. Explain this simplified ruleset in the UI and reports rather than claiming full competitive Tetris fidelity. Use original visual assets and Stackcraft branding.

Python owns the authoritative simulator and legal placements. The web client displays states and sends user actions; it must not implement a second conflicting rules engine. The same placement rules apply to humans and all bots. Every player sees the current piece and exactly one next piece. A training expert may search that preview but may not inspect the rest of the seeded stream. Hidden future pieces must never enter model input. Finite episode limits must be disclosed because survival can be censored by the limit.

## Definition of Done


A clean checkout installs through a committed uv lockfile, runs the game and replays without model credentials, passes engine and integration checks, and reproduces documented evaluation procedures. The comparison demo visibly identifies real inference, cached replay, and unavailable players. A saved trained model reloads and reproduces reference probabilities within a declared tolerance. Published results compare the trained model to unchanged Clef-flash and a heuristic on untouched sequences, including uncertainty and failures. Dataset provenance, model revisions, rules, splits, training configuration, and reproduction tutorials accompany the release. Release links and fresh-download validation for GitHub, model and dataset must exist before completing the goal. The user deferred hosted CPU deployment on 2026-10-06; retain the verified local game/Docker demo and do not require a public Space for completion.

An honest negative training result is acceptable scientific evidence; silently replacing Clef with another model or calling an untrained adapter a fine-tune is not. If Clef training is infeasible after bounded probes, report the evidence and obtain a scope decision for a smaller student or rented compute.

## Progress


- [x] (2026-10-05) User selected Tetris-style Stackcraft and requested an active goal. Goal and durable plan created.
- [x] (2026-10-05) M0: Isolated repository, locked Python 3.13.16 environment, rules/interfaces and tutorial00 validated, including clean wheel installation.
- [x] (2026-10-05) M1: Deterministic engine, playable browser UI, replay and tutorial01 verified by behavioral tests, independent landing review and desktop/mobile interaction.
- [x] (2026-10-05) M2: Offline tournaments, replayable race, native encoding and actual unchanged Clef GPU baseline verified. Development seeds0–1 yielded0mean lines for Clef versus9heuristic at30pieces, zero errors;144ms mean Clef decision latency.
- [x] (2026-10-05) M3: Search expert, frozen disjoint splits, duplicate audit and tutorial03 complete.827train/215validation regenerated from70d84bd and matched the independently audited pilot; manifest retained under reports/.
- [x] (2026-10-05) M4: Real head-only and rank4LoRA GPU steps passed finite-gradient/intended-change/frozen-hash checks. Both fresh-process reloads reproduce probabilities exactly. Tutorial04 and reports/gpu-feasibility.json retain evidence and caught PEFT serialization issue.
- [x] (2026-10-05) M5: Selected epoch 02 passed exact fresh-process reload and all 1,000 final games. Independent replay/statistics and narrative audits passed. Trained mean lines 16.81 versus native 0.07 and heuristic 76.53. Fixed-seed three-bot recordings with live human play passed desktop/mobile checks; CPU Docker smoke passed. Final report and Tutorial 05 complete.
- [x] (2026-10-06) M6: Public GitHub/model/dataset released; anonymous pinned downloads passed byte/schema and all 1,017 archived-record checks. Downloaded-code checkpoint reload matched reference probabilities exactly; GLM restored healthy. reports/publication.md and JSON record revisions and evidence. User deferred hosted CPU deployment; the verified local game/Docker demo remains included.

- [x] (2026-10-06) M7: Seven-page technical report by Nima Karimi, figures, locked CPU-only PDF build and Tutorial 07 published on GitHub and HF. Independent claim review, all-page visual inspection, repeat-build parity and anonymous pinned byte checks passed. See reports/technical-report-publication.json.

## Milestones


### M0 — Reproducible foundation


Initialize a new Git repository only after checking that the target is not inside another repository. Create `pyproject.toml`, `uv.lock`, `src/stackcraft/`, `tests/`, `.gitignore`, README, and local agent instructions. Use Python 3.13 if compatible with the selected ML dependencies; record deviations. Use Ruff for formatting/linting, pytest for behavioral checks, and a type checker. Keep ML dependencies optional so the game is usable without Torch or downloaded weights. Pin actual installed versions in the lockfile rather than copying guessed version constraints.

Write `docs/rules.md`, define the observation/action/replay schema, and freeze scoring: a line-clear table of 100/300/500/800 for one/two/three/four lines, with no changing level multiplier in the research ruleset. Primary research outcome is lines cleared per episode; score is secondary. Tutorial 00 explains the tools, simplified rules, deterministic experiments, and why a large model must compete with a cheap heuristic. Acceptance: a clean locked install and runnable CLI help with offline checks.

### M1 — Playable deterministic game


Implement pure state transitions in `src/stackcraft/engine.py`, sequence generation in `pieces.py`, and replay serialization in `replay.py`. Provide an HTTP service in `server.py` using FastAPI and a small web UI in `web/`; begin with plain TypeScript or JavaScript and avoid a frontend build system unless it adds clear value. The player can select, rotate, drop, restart, and replay a completed game. The authoritative service rejects illegal actions without changing state.

Acceptance: tests cover all piece rotations, board boundaries, collisions, single/multiple line clears, top-out, bag determinism, illegal actions, and replay equality. A browser inspection confirms keyboard and button controls, board readability, score, terminal state, and restart. Tutorial 01 explains pure transitions and why deterministic replay makes model comparisons fair.

### M2 — Baseline tournament


Implement random legal placement and a heuristic based on height, holes and unevenness under `src/stackcraft/players/`. Freeze heuristic weights after development evaluation. Add a Clef-flash player with a pinned model revision and the native joint decision-head interface. Prototype either locally or with an explicitly available hosted credential; never fabricate scores if inference is unavailable. A hosted baseline is distinguished from the pinned local baseline used to compare training.

Represent the board compactly and include legal placement IDs with descriptions. Pass the same observed state to every player. Implement deterministic tie-breaking and explicit error outcomes. If an action count exceeds native schema limits, revise the representation or use documented candidate scoring consistently; do not silently prune only one player's choices. Benchmark individual decision latency separately from rendering and replay speed.

Acceptance: all offline players complete repeatable paired tournaments; actual unchanged Clef runs produce persisted decisions and probabilities. Tutorial 02 explains why a heuristic and untouched model are both required, why rules differ from real-time Tetris, and why a smaller game-specific policy may ultimately be cheaper.

### M3 — Search-expert dataset


Implement bounded search in `expert.py` using only current and preview pieces. Record the horizon, search budget, state value function and action values. Search labels are expert recommendations, not guaranteed globally optimal moves. Collect states from multiple policies and varying stack difficulty so the model learns recovery as well as near-empty-board play.

Assign disjoint seed pools to training, validation, calibration if used, and test before sampling. Keep every state from an episode in one partition; deduplicate exact observations across splits and report exclusions. Freeze test seeds without running model-selection experiments on them. Store schema version, rules version, seed, episode ID, generator revision, expert settings and labels. Start with a small audited dataset before scaling; use a learning curve rather than assuming 100,000 examples are necessary.

Acceptance: regenerating the pilot produces matching hashes; a split audit detects overlaps; manually inspected difficult examples match simulator transitions and expert information limits. Tutorial 03 explains imitation learning (learning to copy expert choices), expert bias, state diversity, and leakage.

### M4 — GPU feasibility


This can begin after the M2 input contract using a small M3 pilot, before expensive data generation. Inspect native Clef code and candidate community training code. Record available GPU memory, library versions, pinned weights, LoRA targets, decision-head treatment, precision, sequence length and batch size. First test inference, then a handful of optimization steps with finite loss, nonzero gradients and changed intended parameter bytes.

Require saved adapter plus decision head to reload into a fresh process and reproduce reference probabilities within a predeclared tolerance. Measure peak GPU memory and elapsed time. Test quantized training only if its implementation supports the native head. Bound initial exploration to a documented small probe budget; record each failed configuration before trying another. Do not launch a long training job before this milestone passes.

Acceptance: real-weight training and fresh-process parity succeed on available hardware. Tutorial 04 explains which parameters are trained, how memory is measured, and why an adapter alone may omit learned decision-head parameters.

### M5 — Training, evaluation and comparison demo


Preregister training configurations, a bounded validation search, episode cap, and final evaluation procedure in `record.md`. Select checkpoints using validation only. Use a pilot to estimate evaluation cost, then freeze a practical test size (target at least 200 paired episodes, subject to measured feasibility) before inspecting test results. Report mean/median lines, score, pieces survived, cap-hit rate, invalid/error rate and latency. Include paired uncertainty intervals over episode-level results and separate expert-action agreement from actual game outcomes.

Run the selected model and unchanged pinned base with identical observation representation and rules. Evaluate against the heuristic as a separate comparison. Never promote a model solely because loss decreases. If it loses, publish that finding and explain observed failure categories. Record training seeds and compute budget; if only one training seed is feasible, disclose that limit.

Extend the UI to show three boards and probability summaries with identical piece streams. Allow a human board alongside them. Cached replays must be clearly marked and cannot be presented as live GPU inference. Tutorial 05 explains selection, paired evaluation, uncertainty, failure analysis, and reproducibility.

### M6 — Review and release


Prepare model and dataset cards, dependency/license attribution, reproduction commands, small demo replays, and an honest report. Keep model binaries out of Git. Validate offline replay usability and build a deployable comparison demo without assuming permanent paid GPU hosting. Review all release files for secrets, private data, unsupported claims, and reproducibility. Ask the user for concrete new-project publication destinations and visibility once the release is reviewable.

After authorization, publish the agreed code, model, dataset and demo. Download released artifacts into a separate directory, validate schema/hashes and model output parity, and test published links. Tutorial 06 explains what is shipped, how another person reproduces inference/training, deployment tradeoffs, and measured limitations. Only then mark all milestones and this goal complete.

### M7 — Technical report and PDF publication

User approved the recommendation to turn the existing study into a technical
report, publish source on GitHub, and attach a PDF to the existing public HF model
repository. This authorizes these additive publications. No arXiv submission,
blog, Space, subscription, retraining, or paid compute is in scope.

Use the frozen final study/summary and publication verification as the evidence.
Keep experimental code, weights, data, original reports and ML lockfile unchanged.
Create `paper/report.md`, figures derived from recorded metrics, an isolated locked
uv build under `paper/`, and `paper/stackcraft-technical-report.pdf`. Include methods,
paired uncertainty, the stronger heuristic, single-training-seed limits, synthetic
data provenance, references, artifact revisions and AI assistance disclosure.
Use the user-selected author display name Nima Karimi, with no affiliation. Clearly label the work a non-peer-reviewed technical report.

Delegate manuscript preparation separately from build/figure work. Independently
review numerical and scientific claims, inspect every rendered PDF page, validate
links and text extraction, and reproduce the build. Add tutorial 07 explaining
why PDF + repository assets were chosen and how to rebuild. Publish via reviewed
local commits on main and additive HF commits; preserve historical release
manifests and verify the new asset through an anonymous pinned download.
Record final hashes, revisions, validation and any review findings compactly in
`record.md` and a new report-publication record. M7 is complete only after the
public PDF and source links work and downloaded bytes match the reviewed artifact.

## Interfaces and Dependencies


The planned engine API is `legal_actions(state) -> tuple[Placement, ...]` and `step(state, placement) -> Transition`. `Observation` contains board, current piece, one-piece preview, rules version and legal actions, never hidden random generator state. `Player.choose(observation) -> Decision` returns a legal action ID, optional per-action probabilities, runtime metadata and explicit errors. An episode artifact stores initial seed, rules version, observations/actions, outcome and player revision. Exact types belong in `src/stackcraft/schema.py` during M0.

Use uv for environments, FastAPI for serving, pytest for tests, Ruff for formatting/linting, and Hugging Face Transformers/PEFT only after native Clef compatibility is verified. Proposed modules and commands in this plan are implementation targets, not existing working interfaces.

## Concrete Steps and Validation


Work in `/home/nima/Work/stackcraft`. Establish these commands during M0 and keep them accurate in each tutorial:

    uv sync --locked --group dev
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked pytest
    uv run --locked stackcraft --help

During M1 implement `uv run --locked stackcraft serve --host 127.0.0.1 --port 8080`; the expected observation is a playable board at `http://127.0.0.1:8080`. Document exact tournament, generate-data, train, evaluate and replay commands when those interfaces exist. Fast iteration runs focused engine tests; final validation adds browser play/replay, frozen tournament evaluation, fresh model reload and clean artifact download. Do not label planned commands as passed checks.

## Idempotence and Recovery


Use unique run directories with immutable configs and artifact hashes. Resume checkpoints only when model, data and configuration match. Keep test results separate from model selection. Never overwrite previous measurements or user files silently. If a long run fails, retain its error and last valid checkpoint. Use ignored local credentials and never print tokens. Recheck GPU use before retrying memory failures.

## Surprises and Discoveries


Training feasibility passed on the RTX5090 with BF16 backbone and FP32 decision head. Community `MersivMedia/clef-finetune` documents only a tiny random-model CPU test; it is a source to inspect, not evidence of successful Clef training. RouterPlus documents a Tetris comparison application, demonstrating interest in the format but not fine-tuning gains. Read-only hardware reconnaissance found the local GLM llama.cpp Docker service occupies about 23 GB of the 32 GB card. The user authorized temporarily stopping and restoring that service. The real-weight baseline passed, and the service was restored and its in-container health endpoint returned status ok.

## Decision Log

- 2026-10-06: User approved a polished technical report/PDF on GitHub and as an asset in the existing public HF model repository. Keep the original study and release verification immutable; document the additive publication separately.


- 2026-10-05: User selected a Tetris-style project over a dungeon game because it looks fun. Use Stackcraft as the working name.
- 2026-10-05: Preserve the user's uv, local-commit, in-repository plan and per-milestone tutorial preferences. Parallel work is welcome when file ownership and interfaces make it safe.
- 2026-10-05: Start with deterministic turn-based placement. This isolates decision quality from keyboard control and service latency.
- 2026-10-05: Treat improved performance as a hypothesis. Completion requires a valid experiment and honest release, not a manufactured winning metric.
- 2026-10-05: Create a single project plan with a compact record because the game, data and evaluation share interfaces; split it if actual delegation warrants independent child plans.

## Outcomes and Retrospective


M0–M6 are complete for the agreed scope. Fine-tuning improved the selected model over unchanged Clef,
with mean lines 16.81 versus 0.07 and a paired gain interval [15.74, 17.77]. The
heuristic remains substantially stronger (76.53 lines) and faster. All 1,000
held-out episodes and all paired intervals passed independent review; no errors
were excluded. The final comparison demo includes live human play alongside
actual base/trained/heuristic recordings and passed desktop/mobile and CPU Docker
checks. Full report, compact metrics, source hashes and all seven tutorials exist.
Public GitHub/model/dataset publication and fresh-download verification are complete.
The actual downloaded checkpoint passed exact reference-probability parity, and
GLM was restored healthy. See reports/publication.md for immutable revisions and
verification evidence. Hosted CPU deployment was explicitly deferred by the user.

## Next Action

No required implementation or experiment remains within the agreed scope.
The completion commit publishes final verification records on GitHub. Public
artifacts and immutable revisions are listed in reports/publication.md and
reports/publication.json. Hosted CPU deployment is deferred, not silently claimed
as delivered. Any later hosted demo or new training study is separate follow-up
work requiring the user's request; preserve this study's frozen evidence.


### Technical report outcome

M7 is complete. The seven-page technical report by Nima Karimi presents the
frozen study with two vector figures, three data-derived tables, references,
limitations and AI-assistance disclosure. Its isolated uv build reproduces the
PDF and figures without a GPU or model download. Tutorial 07 explains each step
and the alternatives. Independent scientific review and all-page visual checks
passed. GitHub source/PDF and the additive HF report release were downloaded
anonymously and matched the reviewed bytes. The model card links the report;
checkpoint, dataset and original evidence remain unchanged. The only original HF
metadata changes are the report link and an automatic PDF LFS rule. See
reports/technical-report-publication.json for the exact revisions and checks.
