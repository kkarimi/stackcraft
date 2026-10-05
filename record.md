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
