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
