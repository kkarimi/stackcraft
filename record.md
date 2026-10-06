# Stackcraft evidence


## Record format


Before experiments, record question, baseline, method, metric, budget/stop condition, starting commit, expected artifacts and decision rule. Afterward append actual commands, concise results, artifact paths/hashes, limitations and next action. Keep raw logs and model files in ignored run directories.

## 2026-10-05 — Goal setup


The user requested a goal to achieve the proposed Tetris-style project. The active goal covers game, baselines, expert data, real GPU fine-tuning, evaluation, tutorials and release. `plan.md` is the durable execution document. Existing preferences retained: uv, local commits, plans inside the repository, and safe delegation. No benchmark or training claim is made.

Research sources inspected in the preceding discussion:

- https://huggingface.co/Cloudflare/clef-flash — base decision model and native interface.
- https://github.com/MersivMedia/clef-finetune — community training code; real Clef GPU training explicitly unverified at inspection.
- https://app.routerplus.com/docs/playground#tetris-royale — existing model comparison game; not evidence of fine-tuning improvements.
- https://developers.cloudflare.com/workers-ai/models/clef-flash/ — optional hosted baseline; pricing and limits require rechecking before use.

Next action: M0 foundation. No paid job, credential use or remote publication has begun.

## 2026-10-05 — Foundation and GPU reconnaissance


Initial contract/plan commit: `53db6bd`. Resolved Python 3.13.16 through uv 0.12.23, FastAPI 0.142.2, uvicorn 0.54.0, pytest 9.1.1, Ruff 0.16.10 and ty 0.0.84. Actual versions are in `uv.lock`. `stackcraft --help` and focused root-code Ruff/ty checks passed. Engine and web UI are delegated with exclusive files defined by `docs/interfaces.md`; an independent worker is implementing offline baselines next.

Read-only inspection found 32,607 MiB total GPU memory, about 22,976 MiB used by root-owned llama-server in Docker container `omarchy-local-ai-glm-4.7-flash.llamacpp.q5xl.32k.rtx-5090-engine`, leaving about 9 GB free. No workload was stopped. User authorization to temporarily stop/restore it was requested asynchronously; no answer recorded yet.

Clef-flash source revision inspected: `17f0b0ad64efb65d273590632833508766b2aae6`. No local Clef cache found. Backbone shards total 18,819,627,488 bytes, before its separate decision head. Native `systemone()` uses inference mode; training must use encoded/collated records and `ClefModel.forward`. Targets must follow lexicographically sorted `EncodedQuestion.option_ids`. Encoding silently truncates the state after reserving question tokens, so complete board/schema length must be checked first. The native loader pins one device and downloads an unpinned snapshot when passed an ID; resolve the pinned local snapshot explicitly. Do not copy the community wrapper's full FP32 vocabulary embedding cast, which allocates roughly 3.79 GiB; gather option-token rows first, then verify equivalence before adopting that optimization.

Future M4 probe decision: start with BF16 inference and head-only steps, then rank-4 LoRA at batch one with checkpointing if sufficient free memory. Record finite gradients and parameter changes, then test unrounded probability parity after a fresh-process reload at predeclared absolute tolerance 1e-4. Limit initial probe execution to two configurations and 30 minutes. This is a proposed probe, not a measured fit or training result.

Sources: official pinned `joint_schema_model.py` and `model.safetensors.index.json` under https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6 ; community code https://github.com/MersivMedia/clef-finetune/blob/main/src/clef_finetune/model.py .

## 2026-10-05 — M0/M1 verified; offline portion of M2


Validation:83 pytest tests passed; Ruff check/format and ty passed; JavaScript `node --check` passed. Built a wheel and installed it into an isolated environment: HTML, CSS, JS, health, game creation and a validated move all passed. Switched development test client to httpx2 because the installed Starlette emits a deprecation warning for httpx.

Independent engine review compared legal landings to a separate nearest-obstacle calculation over1,067 states/40seeds and checked cell conservation and replay equality. Engine golden fixture pins seed42 first28 pieces. Independent service/UI review found two concrete issues now fixed: pointer buttons retained focus and disabled game shortcuts; a lost move response could cause a second placement on retry. Pointer controls now restore board focus. Move API requires expected_pieces and returns409 on stale state; UI refreshes after uncertain responses and disables moves if it cannot refresh. A server regression test proves repeated stale requests cannot advance again.

Browser at http://localhost:8087 verified button move/rotate/drop, keyboard controls after pointer rotation, uploaded replay stepping to2/2 and return to preserved live state. Simulated fetch failure after a committed move refreshed the browser to the correct2-piece state; the next deliberate drop reached3. Desktop1280×800 shows full board and controls; mobile390×844 has no horizontal overflow and functional controls. Screenshot: `/home/nima/.t3/userdata/browser-artifacts/browser-screenshot-localhost-muvnlig0-1a4cfb1d.png`. A local server remains running on8087 (8080 was occupied by another service); no existing service was stopped.

Offline pilot preregistration and complete reproduction are in tutorial02. Seeds0–19,100-piececap, fixed heuristic weights,20paired episodes. Random meanlines0.10, score10, pieces26.9; heuristic meanlines35.95, score3900, pieces100. Both error rates0; heuristic cap-hit100%, so unrestricted survival is unknown. Decision-only mean latency was0.00162ms random and0.17928ms heuristic on this machine. All40 replay outcomes were independently recomputed by the root. `runs/baselines-development.json` SHA256 `ed4973c69496cdc0ebb5720e3beaae621406988a6d298233c10ab962e1c499af`. CLI integration smoke used development seed1000 and10pieces in `runs/cli-smoke.json`. Exclude these development seeds from later held-out test pools.

Next: complete real Clef baseline and build a one-preview search expert/dataset while GPU permission remains pending. No model results, publication or goal completion claimed.

## 2026-10-05 — Native encoder, search data and recorded race


Added a recorded Random/Heuristic race with authoritative replay reconstruction and human-session preservation. Root browser verified seed0 final outcomes: Random26pieces/top-out/0lines; Heuristic100pieces/capped/37lines/4100score. Playback is visibly recorded, with Clef pending and no live-model speed claim. Recorded boards use the same piece index; the shorter run freezes at its final state.

Optional ML stack resolves Torch2.14.1+cu130, torchvision0.29.1+cu130, Transformers5.18.0, PEFT0.21.2, HuggingFaceHub1.33.0 and Pillow12.3.0. Imports and native processor construction pass. A pinned release download is live via exec session78859; no weights were loaded into GPU. Independent reviewer confirmed native source SHA256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`, correct sorted-choice mapping, and exact segment token preflight. Actual CPU encoder tests on14fixtures used848–2248tokens and rejected too-small limits before truncation. `ClefPlayer` records source/revision/encoding/context/dtype/device and unrounded probabilities; tournament artifacts retain runtime metadata and input-token counts when available. Full inference remains unverified.

M3 preregistration and pilot evidence are in tutorial03. Two complete generations matched exactly:827train,215validation, with6within-training and2cross-split duplicates excluded. Seed pools10000–10023train,20000–20005validation,30000–30199reservedtest; no test trajectories generated. Source hashes accompany every manifest. Root inspected collector/expert/audit code and the independent16-position teacher recomputation report. The separate five-seed development search tournament averaged38.6lines versus37.8heuristic; both hit100pieces, so this is not robust superiority evidence. Pilot artifacts predate the last provenance-validator edit and must be regenerated from committed source before release.

Before final M3 generation: commit reviewed implementation; generate once using recorded clean commit; compare observations/action values/labels to the historical reproducible pilot ignoring only source_commit; audit all hashes and split exclusions. Stop on any difference or failed audit. Expected destination `data/study-v1`. Final test seeds stay untouched.


## 2026-10-05 — Committed M3 artifact accepted


Implementation commit `70d84bd8dc5d4a60f3b96455a57d9e6f9416d109`. Root ran `uv run --locked --extra ml stackcraft generate-data --output data/study-v1` from the clean commit. Structural/provenance audit returned827train/215validation; all six source hashes matched current files. Removing only source_commit metadata produced exact row equality with the independently checked historical pilot, including every observation, teacher value and label. Final JSONL hashes: train `edd682761db95a4f25bb30a284489c54d9336a36da0a9b19d2cda860b428baa8`, validation `eff9cdc5932e935959ac4d26dce6470d335090f7428a91930001954266d133bc`. Compact manifest committed in `reports/dataset-manifest.json`; rawdata ignored. Reserved test seeds untouched.

Latest checks:117tests passed with actual pinned native CPU encoding enabled, plus Ruff and ty. Root verified race scrubbing to100, final scores and human-game preservation. Final race screenshot `/home/nima/.t3/userdata/browser-artifacts/browser-screenshot-localhost-muvnwvs6-60bf91fb.png`; controls fit1280×800 (bottom786px).

At checkpoint, download session78859 was polled and confirmed still running; cache about12GB. GPU remains22,976MiB used/9,175MiB free; no user response to service-stop question, no workload stopped and no real-weight inference attempted. This is progress with a live download, not a blocked/completed goal. Next: implement bounded training probe, finish download and run real inference/training when memory is available.


## 2026-10-05 — GPU interruption authorized

The user explicitly approved temporarily stopping and restoring the local GLM service for Stackcraft GPU training. Read-only inspection confirms Docker container `omarchy-local-ai-glm-4.7-flash.llamacpp.q5xl.32k.rtx-5090-engine` is running with restart policy `unless-stopped`. Stop only when pinned model files and executable probes are ready. Restore with `docker start` after GPU experiments (including failure paths), verify running/health and record the result. No ongoing GLM request has been inspected. Previous turn is classified as progress: committed game, teacher data, and native encoding checks. Download session78859/PID2602150 was re-polled and remains live, about13GB cached at continuation.

Preregistered M4 execution: use development positions only, record unchanged native probabilities before modifications, three head-only steps then five rank4LoRA steps at batch1, label smoothing0.05 and Brier weight0.1. Assert finite nonzero gradients and intended parameter changes, save head plus adapter, and compare fresh-process raw probabilities at absolute tolerance1e-4. Limit initial realGPU exploration to two configurations/30minutes; preserve logs and report failures. Full training and test evaluation remain later gates.


M2 real-weight development pilot preregistered: pinned unchanged Clef-flash, development seeds0–1,30-piececap, Random/Heuristic/Clef on identical streams. One warmup atseed42 is excluded from tournament timing. Persist decisions, rawprobabilities, per-decisiontokenlengths, runtimeconfiguration, peakVRAM and elapsedtime. Output runs/clef-development-v1.json. Wrapper timeout900seconds; restoreGLM in finally including failures. This pilot measures native feasibility and informs later budgets, not final test quality.


## 2026-10-05 — Real unchanged Clef baseline passed

Pinned weights loaded successfully. runs/clef-development-v1.json contains2paired development seeds0–1/cap30; Clef meanlines0,pieces27.5,55decisions144.26msmean/240.66msp95; heuristic meanlines9,pieces30; allerrorrates0. Warmup+load4.53s. GPU session exited0 and restored GLM; docker exec curl localhost:8080/health returned status ok. Host8080 is unrelated SearXNG; health checks must run inside the exact GLM container. Correct reference attention kernels in use. M2complete. No held-out test was inspected.

M4 code review: CPUtests exercise actual tiny Qwen3.5 hybrid attention with realPEFT, nativehead, checkpointing and gradients. Probe will save each candidate separately and bind reload reference to datasetmanifesthash and fourtrainingrowIDs. Head-only and LoRA use fresh base instances. Keep head-only frozenbackbone in eval mode.


## 2026-10-05 — M4 training and M5 preregistration

Real-weight probe from e38ac53 passed3head-only and5rank4LoRA steps. Both trainable groups changed; full frozen parameter hashes were unchanged. All recorded losses and gradients finite, head and LoRA gradients nonzero. LoRA peak allocated23,180,389,888bytes/reserved23,595,057,152bytes; step time0.67–1.32s on1302–2252tokens. Wrapper restored GLM and verified its in-container health. Fresh-process parity is the remaining M4 gate; artifacts runs/probe-v1 and runs/probe-reload-v1.

Conditional on M4 reload passing, preregister M5: one seed42, BF16 native backbone, FP32 native head, rank4/alpha8/dropout0 text-only LoRA, no quantization, max4096tokens, AdamW lr1e-5/weight_decay0.01, batch1, gradient accumulation8 (last partial group normalized by actual size), norm clip1, CE label smoothing0.05 plus Brier sum weight0.1. Train exactly two epochs over the827 fixed training positions shuffled deterministically per epoch; save both epoch checkpoints. Bounded search is these two checkpoints only; exclude feasibility-probe checkpoints. Use all215 validation positions to select the epoch with lowest finite mean target NLL; exact tie chooses earlier epoch. Report teacher agreement and Brier as diagnostics, not selection overrides. If either candidate has inference errors or nonfinite NLL it is ineligible. If both fail, preserve evidence and reconsider before tests. Initial fulltraining timeout3600s; no test access during training/selection.

Freeze final evaluation at200paired seeds30000–30199 and200piececap, bootstrap10000replicates/seed2026, error threshold0. Primary comparison selectedtrained versus unchanged nativeBF16headClef. Include an unchangedFP32head ablation to separate precision change from learning, plus random and heuristic. Same rules, complete observation encoding, pinnedbackbone, legalchoices and preview. Three neural policies at200×200moves and144ms imply4.8h worst-case, but unchangedbase dies around28pieces in development; expect much less. Allow bounded per-policy execution with incremental artifacts; never drop failedseeds or quietly reducepool. Record exact checkpoint and validation evidence hashes before any testepisode. Publish negative outcomes honestly; no tuning after testresults.

Fresh LoRA reload v1 correctly rejected PEFT's shortened target list. Saved496adapter tensors independently map exactly248intended text layers; weights were sound. Loader now expands saved targets over every backbone module and requires exact target equality, rejecting accidental vision/MTP/output matches. A32-layer tiny-backbone regression reproduces PEFT shortening. New saves use explicit full targets and canonical pinned model ID. Original failed artifact and probe weights preserved; retry uses same checkpoint, not a new training configuration. Head-only fresh-process reload passed independently; GLM restored healthy on both success and rejectedreload.

M4 complete: head-only fresh reload maxabsoluteprobabilitydifference0.0 (tolerance1e-4); unchanged saved LoRA checkpoint retry after semantic-target validation also0.0. Both fresh processes loaded pinned weights; no retraining needed for reload fix. Compact reports/gpu-feasibility.json retains success and original rejectedreload with sourcehashes. Fulltraining preregistration above is now enabled; output runs/study-v1 with two epoch checkpoints,3600second wrapper timeout and restored GLM health.


M6 preparation while training runs: GitHub CLI is authenticated as kkarimi and HuggingFace as nima1. Read-only destination checks found no existing kkarimi/stackcraft, nima1/stackcraft-clef-flash-lora, dataset nima1/stackcraft-data or Space nima1/stackcraft. No repositories created or uploads performed. Root prepared cards, Apache-2.0 license/NOTICE and tutorial06; final metrics remain explicitly pending. Raw epoch checkpoints will remain immutable. Before validation, candidate copies may replace only autogenerated adapter/README.md to remove placeholders/localcachepaths; compare all other bytes, then selection/test bind these candidate hashes. This is documentation-only preparation, not weight or configuration tuning.

CPU demo packaging reviewed: pinned Python3.13.16/uv image digests, nonroot UID1000, game-only install and noTorch. Worker verified actualimage12bd960f10c2d03bf85d3cec836c25880b09cfff6dad84cfc0aa33966af12308, health, configurableport, human move/replay equality and packaged raceassets; temporarycontainer removed. Root browser on8087 verified olddevelopmentrace stillscrubs toactual26/100piece outcomes and returning preserves humanonepiece game. New probability UI awaitsactual finalmodelrecordings. Export fixedseed30000 beforetest; fullresultsURL optional/HTTPSonly and remainsunset.

Independent evaluation review led to stricter complete215prediction eligibility, rawmetric recomputation, bothcandidate evidence/lowestNLLselection, allfiveplayersrequired, fullresume source/dependency identity and completedepisode/player consistency. Root set8CPUthreads and disabledTF32 explicitly, matching measuredconfiguration. Focusedselection/evaluation and exporttests passed; one broader run caught an in-progress testedit, fixed by owner before integration. No held-outtest generated.

Public-source Tutorial03 exact regeneration recipe executed while epoch02 trained. Verified all six generator-source hashes, generated data/original-study-reproduction with original70d84bd provenance, and matched both original train/validation fileSHA256 exactly in9.8s. No test trajectories generated. This validates the documented route without privateGit history.


## 2026-10-05 — Full training complete; validation started

Both epochs completed all827positions and104optimizersteps each. Meantrainingloss2.577967 then1.966718; epoch durations742.48s/744.66s, total1511.25s including hashing/checkpointwork. Intendedhead/LoRA parameters changed; frozenbackbone hashes unchanged. Peakallocated23.23GB/reserved24.18GB. Wrapper exited0 and restored GLM healthy. Fullreport retained as reports/study-training.json; launchsource/supplement available withoutprivateGit. Candidate01/02 copies replaceonly adapterREADME with siblingpreparationhashrecords. No learnedbytes changed.

Validation now runs in session96574 / runs/validation-v1.log under900s GLMwrapper: all215positions for nativebase, FP32base and epoch01; all215 for epoch02. Outputs runs/validation-epoch-01 and -02. Afterward verify restoredhealth and completeness, selectpreregisteredlowestfiniteNLL, freshreloadselected, thenstartall200pairedtestseeds onlyafter selectionhashes recorded.


## 2026-10-05 — Validation-selected checkpoint frozen

All four215-position evaluations completed with0errors/fullrawprobabilitycoverage; GLM restored healthy. Nativebase teacheragreement21/215(9.767%), NLL2.952702; unchangedFP32head22/215(10.233%), NLL2.952393. Epoch01agreement89/215(41.395%), NLL1.860870; epoch02agreement92/215(42.791%), NLL1.775923. PreregisteredlowestfiniteNLL selects epoch02. scripts/select_checkpoint.py independently recomputedrawmetrics and bound bothcandidate/data/config/sourcehashes; output runs/selection-v1. Frozen selectionSHA256025b62587a26bd8cc78c80f362524b8c178190df5188b317299121afa6b1203b. Compact evidence reports/validation-summary.json; fullrawpredictions remainruns and willshipwithrelease. No held-outtest yet. Selectedcheckpoint freshreload now in session77580 / runs/selected-reload-v1.


Selectedepoch02 fresh-process probabilityparity passed exactly0.0 withrestoredGLMhealthy; report retained reports/selected-checkpoint-reload.json. Finalteststarted afterfrozenvalidationselection in session49782, childPID2685785, wrapperruns/gpu-final-evaluation-session.json / log runs/final-evaluation.log, timeout21600s. Exactcommand: evaluate_clef.py tournament --checkpoint checkpoints/candidates/epoch-02 --seeds30000:30200 --max-pieces200 --final-test --selection-file runs/selection-v1/selection.json --output runs/final-evaluation (CLIarguments separatednormally in wrapperrecord). First13nativebaseepisodes persisted throughseed30012; jobalive. This isprogress, notcompletedM5. No test-basedtuningallowed. Keep protectedevaluation/sourcefiles unchanged; use --resume onlyafterverifyingwrapper/jobfinished and request/sourcehashesstillmatch.

UIworker owns web-only finalacceptance improvement: anoptional livehuman board alongside the three recordedbots onthemanifestseed, with visiblelive/recorded distinction. This completes the plan's explicit human-alongside requirement without touching frozenneural/game evaluationcode. Current shippedmanifestremainsdevelopment untilfinaltestfinishes.

## 2026-10-05 — Restore experiment record from Git history

Release-preparation commit `363400c` accidentally replaced the earlier experiment
record with a copy of Tutorial 06. Restored the complete record from `55dcaab`,
including the training and final-test preregistration, and retained every subsequent
experiment entry. The original preregistration remains independently available in
Git history before training and test execution. This restoration changes no study
configuration, source code, data, checkpoint or outcome. Tutorial 06 remains at
`docs/tutorials/06-release.md`.

The public release will include `reports/study-preregistration.md`, preserving the
preregistration excerpt from actual training-launch commit `d9fcd05` and identifying
the full original record hash. This makes the historical text inspectable without
private GitHub access; it is not a signed or externally timestamped registration.

While final evaluation runs, `stack_engine` independently reviews release tooling
and tutorials (read-only). `stack_training_recon` owns only new
`scripts/verify_release.py` and `tests/test_verify_release.py`, implementing download
byte verification against a trusted local bundle manifest plus downloaded dataset
audit. This check does not replace GPU probability parity or published browser
verification. The UI worker retains exclusive ownership of web changes.

## 2026-10-05 — Human board beside recorded opponents

Integrated the UI worker's web-only comparison mode. Starting it explicitly creates
a human game on the recording's seed and cap; opening tabs alone does not reset a
game. The original board and authoritative API session are reused. Root browser
checks confirmed that a human drop synchronizes recordings to piece 1, moving the
recorded timeline to piece 2 leaves the human at piece 1, and switching to Play
preserves the same comparison state. Worker checks covered desktop/mobile controls,
pending-response view guards, and 100 actual API placements through the development
cap with no extra requests. At an exact-cap top-out, the UI reports top-out rather
than censored survival. JavaScript syntax and diff whitespace checks pass.

The current recording is still the disclosed development seed 0 comparison.
Final seed 30000 export and browser review remain required after evaluation.
Tutorial 01 and the deployment guide now explain the human comparison workflow.

## 2026-10-05 — Release review and download verification tooling

Independent release review found public packaging gaps: the Space README linked
to excluded documentation, the bundled source README linked to excluded plan.md,
and the dataset card assumed access to a private historical commit. The builder
now generates a self-contained Space README, redirects the one bundled planning
link to Tutorial 06, and includes the download verifier. Cards point readers to
the already verified public-source regeneration procedure. Final repository links
will be added after destinations and visibility are confirmed.

`scripts/verify_release.py` checks separately downloaded model, dataset and demo
payloads against the retained local release manifest, including every expected
size/hash and uploaded manifest. It rejects missing, extra, linked or traversing
payloads, permits only explicit HF bookkeeping, and audits downloaded dataset
records. Its report explicitly excludes GPU parity, publisher authenticity and
download freshness. Tests exercise corruption, unexpected files, inconsistent
metadata, schema-invalid but rehashed datasets, and the actual installed HF 1.33
metadata writer. A temporary-copy CPU smoke used the real 827/215 dataset and an
explicit toy checkpoint; this is not a released-model verification claim.

Combined checks: 261 tests passed with one optional native-encoding test skipped;
that native CPU encoding test was then enabled separately and passed (262 total).
Ruff lint/format and ty passed; the human-API focused suite passed all five tests.
No protected evaluation source or selected-checkpoint bytes changed. The frozen
GPU job remains live, with 157 native-base episodes saved at this check.

## 2026-10-05 — Complete result summaries before final evidence arrives

The model-card renderer currently exposes only the primary comparison interval.
Extend its presentation to include all four preregistered comparisons, outcome
medians, and latency distribution/invalid-decision counts already calculated by
the frozen evaluator. This changes reporting only, not metrics or study selection.
Check with synthetic development fixtures before consuming the final report.

Renderer changes passed all 17 release tests plus focused Ruff/format/ty checks.
The card now shows all four paired line differences and intervals, mean/median
outcomes, cap/error rates, latency mean/median/p95 and invalid counts. It identifies
the primary comparison and discloses that intervals are not adjusted for multiple
comparisons. A read-only integrity audit confirmed all 12 protected source hashes,
all six selected-checkpoint files and the frozen selection still match the running
request. No final outcome claim is made from incomplete games.

## 2026-10-05 — Fill the full-training tutorial gap

Tutorial 05 described validation and testing but omitted the full training command,
used raw checkpoint paths despite preparation before selection, and retained an
outdated status. Add the exact two-epoch configuration, preparation and selected
fresh-reload steps; document completed training/validation evidence separately from
still-running game evaluation. An independent reviewer checks the other milestone
tutorials against actual CLI contracts. Documentation only; no experiments restart.

Tutorials 02–06 now form a continuous reproduction path: cache the pinned model,
run enabled native CPU checks, generate or download exact `data/study-v1` bytes,
train explicitly for two epochs, prepare candidate copies, validate/select,
fresh-reload, evaluate, export seed 30000, and stage a release. Tutorial 06 includes
the actual export/build commands and requires a matching optional report URL.
Tutorial 03 supports bundled source identity without private Git and checks full
manifest equality before writing exact study bytes. Root checked all 27 tutorial
shell/Python blocks for syntax, verified local Markdown links, and matched the
training table against the real report. No tutorial GPU command was rerun.

Historical Tutorial 01 development-race descriptions remain accurate for the
currently shipped development manifest and must be revised at the final export.

## 2026-10-05 — Source publication preflight while evaluation runs

At source commit `ac10793`, scanned all 85 tracked files for recognizable Hugging
Face, GitHub and AWS credential formats and private-key headers; no matches were
found. No weight binaries, private-key files or .env files are tracked. This is a
limited source preflight, not a guarantee of secret absence or a substitute for
reviewing the actual final staged bundle.

The pinned upstream model card identifies Apache-2.0 and Qwen/Qwen3.5-9B. The local
LICENSE text exactly matches that snapshot's LICENSE after CRLF/LF normalization,
including its upstream copyright notice; NOTICE records Stackcraft's modifications
and upstream revision. Upstream LICENSE raw SHA256:
`bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`.
The full evaluation is still live in session 49782; no experiment settings changed.

## 2026-10-05 — Record the inference hardware context

Saved `reports/evaluation-hardware.json` while the same final-evaluation process
was live. It identifies RTX 5090, driver 610.57.04, Ryzen 9 9950X3D, Python 3.13.16,
32 logical CPUs, host/GPU memory, kernel, configured power limit, and the actual
trained-player inference settings. It is bound to the frozen evaluation request
hash. The power limit is not measured consumption; this snapshot is not an energy
or continuous clock trace. Tutorial 05 links it for interpreting latency. No
hardware settings, dependency versions or evaluation source changed.

## 2026-10-05 — Final evaluation and independent audit complete

All five players completed all 200 frozen seeds (1,000 episodes), with no errors
or invalid decisions. Mean lines: native base 0.070, FP32-head base 0.105, trained
16.810, random 0.140, heuristic 76.530. Trained minus native base is +16.74 lines
(paired 95% interval [15.74, 17.77]); trained minus heuristic is -59.72
([-60.805, -58.619875]). Only the heuristic hit the 200-piece cap, in every game.
These results support task adaptation, not superiority over a cheap heuristic.

Independent CPU audit replayed all saved observations/actions and recomputed
all summaries and all twelve paired metric intervals; all matched. Exact audit
result and source are archived in reports/final-audit.json and
reports/final-audit-source.md. Report SHA256:
5585574b77e81c8d7dd44464c4dfb09f9511ccec0d9150c3ea44793760416857.

The GPU wrapper exited zero and restored the authorized GLM container running
and healthy. Root independently checked its in-container /health: status ok.
Exported fixed seed 30000 (base, trained, heuristic) to runs/demo-final-v1.json
and deliberately replaced the packaged development manifest. Final browser,
Docker, report and release bundle checks follow; no new training or test tuning.

## 2026-10-05 — M5 accepted and concrete release staged

Final narrative review matched every compact-summary metric, all twelve intervals,
all evidence hashes and fixed-seed endpoint descriptions. No blocking findings.
The full test run passed 260 tests and found one obsolete expectation for two
recorded players; changed it to require base/trained/heuristic. All five focused
server tests then passed. One optional native encoding test was skipped in this
run; it passed separately before these reporting/UI edits. Ruff lint/format, ty,
JavaScript syntax and local Markdown links passed. Exact archived audit source
still matches the executed script hash.

Actual final-recording browser checks passed desktop and 390×844 mobile controls,
probabilities, synchronized human moves, scrubbing, preserved sessions and the
200-piece human comparison cap. Fixed a real-name heading wrap alignment issue.
The CPU Docker image stackcraft-demo:final-local passed health/assets, exact
manifest, all three replay outcomes and human move/replay roundtrip. It runs as
UID1000 without Torch. Image SHA256:
200b46e282d2b09b73cd597fe2150ab6039139489c7fbdd35d9d50f425dda3b1.
The isolated smoke container was removed. No GPU workload was changed.

Local commit 3c98736 contains the study, report and final demo. The builder
validated all final evidence and staged 1,149 files at runs/release-v1 from that
clean source. Actual bundle hash/schema, card and publication-scope review follow.
Nothing has been uploaded.

The actual staged bundle passed byte/schema verification and independent card,
path, source-manifest, checkpoint and tutorial review. It contains 1,149 files,
2,757,587,696 bytes. Release manifest SHA256:
0fe02dadf3a793bfce936e1c54a762d8dd2108a35de30293279adeabe6a8c0c1.
A limited recognizable-token/private-key scan covered all 1,147 nonbinary files
and found no matches or forbidden runtime files. This is not a guarantee of
secret absence. Local verification is not a fresh-download or GPU parity claim.

Presented concrete Stackcraft publication destinations and visibility options
after review, explicitly disclosing public source inside public HF artifacts.
Await the user's answer before creating/uploading repositories. M6 remains open;
no publication is claimed and the goal remains active.

## 2026-10-05 — Pending publication decision, loading example clarified

First automatic continuation after the concrete publication question: no
visibility/destination answer has arrived. Rechecked clean repository, plan and
local verification evidence. The prior turn made concrete progress by completing
M5 and staging/reviewing the actual release; no remote publication is authorized
until the required new-project scope is confirmed.

Addressed the packaging review's minor loading-example issue: the model-card
template now names the real ../checkpoint path from downloaded code/, the uv
invocation and the pinned-cache tutorial. No model/evaluation changes. The staged
release-v1 remains immutable; the final bundle must be rebuilt after approved
publication cross-links are added, including this documentation correction.

## 2026-10-05 — Publication blocker confirmed across three goal turns

The same missing new-project publication settings remain after the original
concrete release-review turn and two automatic continuations. The preceding
continuation made a small loading-documentation correction (13 release tests
passed); it did not resolve publication authorization. Current worktree is clean,
local bundle verification passed, and published-artifact/download/GPU-parity
evidence is still absent, as expected before upload.

No independent milestone work remains that would unblock publication. The
explicit goal and plan require confirming destinations and visibility first.
Mark the goal blocked pending that existing question, not complete or paused.
On answer, rebuild with approved cross-links and the loading correction, publish
the agreed repositories, then verify actual downloads and the hosted demo.

## 2026-10-06 — All four public repositories approved

User explicitly selected all four public for GitHub kkarimi/stackcraft, HF model
nima1/stackcraft-clef-flash-lora, dataset nima1/stackcraft-data, and CPU Space
nima1/stackcraft after disclosure that public artifacts contain source code.
The publication blocker is resolved. Existing CLI identities are kkarimi and
nima1; all three proposed HF names are currently absent. Add approved links,
build a fresh release bundle, verify/upload, download exact published revisions,
check bytes/schema and GPU reload parity, and exercise the hosted demo.
Only the already authorized GLM service may be temporarily stopped/restored.

## 2026-10-06 — GitHub/dataset published; Docker Space account gate

GitHub kkarimi/stackcraft was created PUBLIC and source commit 898c958 pushed.
Dataset upload succeeded at 4fe704eeaeaf3ea140544052c510abde2b2aa804.
Creating Docker Space nima1/stackcraft with cpu-basic returned HTTP 402: personal
Docker CPU Spaces now require PRO. Official Spaces overview confirms no hourly
cpu-basic charge but a paid-plan creation requirement. No account subscription
or paid hardware was purchased. Asked the user to enable PRO or supply another
CPU host; model upload continues independently. Updated current source hosting
instructions to disclose the actual prerequisite. Rebuild release source metadata
after hosting is resolved; no frozen model/dataset/evaluation bytes will change.

Fresh Hugging Face 1.33 downloads include a revision-keyed tree cache that the
original verifier did not allow. The verifier now checks exact cache schema,
complete expected payload set, canonical paths, per-file revision/etag, size and
Git/LFS hashes; it still rejects unknown files and malformed metadata. The actual
published dataset passed this check without deleting any cache files. Forty-eight
focused tests cover the installed writer and adversarial tree-cache variants.
The next source bundle includes this fix and accurate Docker hosting status.

A fresh anonymous clone of public GitHub commit ef972f1 installed only the 25
game/dev packages through the lockfile. Offline checks: 261 passed, six optional
checks skipped; Ruff/format, scoped game type check and CLI/server/engine smoke
passed. Full ty needs optional ML imports, so README/Tutorial00 now distinguish
the scoped no-ML check from `uv run --locked --extra ml ty check`. They also
explain retaining --extra ml when working in an ML environment. No source
diagnostics were suppressed and no runtime dependency was added to the game.

## 2026-10-06 — Lossless evidence packaging for the slow upload

The initial upload is live and transferring the FP32 head; observed upstream
throughput is slow relative to the 2.75 GB raw bundle. Most remaining bytes are
repeated JSON evaluation evidence. Before another upload, package evidence/ as
a deterministic lossless evidence.zip with entries rooted at evidence/, plus a
raw evidence-files.json hash/size manifest. Preserve all raw records and all
frozen hashes after extraction. Verify each archive entry against staged source
before removing only the newly staged uncompressed evidence directory. Keep the
selected checkpoint bytes unchanged and test extraction equality. This is a
release transport change only, not a new analysis or reduced study deliverable.

Compression measurement: 1,017 evidence files, 2,222,025,418 raw bytes, became
74,541,451 ZIP bytes with deflate level 6 (96.65% reduction). Archive/read-back
tests passed, including exact extraction paths/bytes and malformed archives.
The strict downloaded-payload verifier must run before extraction into its root;
extraction restores the original evidence/ layout for offline inspection.

## 2026-10-06 — Hosted CPU deployment explicitly deferred

User answered the hosting question: "Forget the cpu hosting for now". This changes
the remaining release scope to public GitHub/model/dataset with the verified
local game and Docker demo retained. Do not create a Space, buy PRO, or require
external hosting as a completion gate. Model publication and fresh-download/GPU
parity checks continue. Remove promised live-demo links and clarify this deferral
in release cards/tutorials before the final documentation revision.

The full suite passed 304 tests, including the opt-in native CPU encoding check;
Ruff lint/format and ty passed. Public model page inspection found Hugging Face's
generic PEFT use-model snippet reporting an invalid task type for this custom
decision head. Set explicit library_name: stackcraft, as supported by the Hub's
custom-library metadata, while documenting PEFT as an internal dependency. This
points users to the actual bundled loader instead of an inapplicable generated
snippet. No checkpoint or native loader implementation changed.

## 2026-10-06 — Public release and completion audit passed

Final model revision c4272310bd6c63ee97a941255abf8f36ff162229 and dataset
revision 211708dd56f1a8af711c060c2ab1ecab35e7166d are PUBLIC. Both were freshly
downloaded anonymously into absent directories with force_download=True. All
bytes/schema/splits passed against the retained v6 manifest; the skipped demo
is explicit. All 1,017 raw ZIP entries also passed after download.

The downloaded source manifest is clean at 82a8854258f228a214f0964bbbd3d2b52e2be1e5.
Its actual checkpoint reload passed four reference rows with max difference0.0
(tolerance1e-4), in5.67seconds. The bounded authorized wrapper exited0 and restored
GLM healthy; root separately confirmed /health status ok. No training occurred.

Independent completion audit verified M0-M6 against evidence and anonymous public
URLs. The final model page renders the actual results and custom loader without
the invalid generic snippet. Full suite304passed; Ruff/format/typassed. Final
publication reports preserve links, revisions, trusted manifest, download evidence,
archive validation and GPU parity/restoration. Hosting is deferred by explicit
user instruction; local gameplay/Docker verification remains delivered. Push this
completion record, verify remote main, then complete the goal.
