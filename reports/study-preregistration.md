# Archived training and final-test preregistration

The excerpt below is preserved verbatim from `record.md` at training launch commit
`d9fcd05c02e9327d4d791f2338725092cede8bdd`. The original complete record's SHA256 is
`bb18a6fc7c21d3d186afcf682e724015263340f3c269500951f8f159054453e1`. This archive lets readers inspect the
recorded decisions without access to the original Git repository. It documents
what was recorded; it is not a signed timestamp or independent registration service.

The final checkpoint selection and evaluation request provide separate hashes of
the actual artifacts used. This archive is historical: later outcomes belong in
the training, validation, reload and final evaluation reports.

---

## 2026-10-05 — M4 training and M5 preregistration

Real-weight probe from e38ac53 passed3head-only and5rank4LoRA steps. Both trainable groups changed; full frozen parameter hashes were unchanged. All recorded losses and gradients finite, head and LoRA gradients nonzero. LoRA peak allocated23,180,389,888bytes/reserved23,595,057,152bytes; step time0.67–1.32s on1302–2252tokens. Wrapper restored GLM and verified its in-container health. Fresh-process parity is the remaining M4 gate; artifacts runs/probe-v1 and runs/probe-reload-v1.

Conditional on M4 reload passing, preregister M5: one seed42, BF16 native backbone, FP32 native head, rank4/alpha8/dropout0 text-only LoRA, no quantization, max4096tokens, AdamW lr1e-5/weight_decay0.01, batch1, gradient accumulation8 (last partial group normalized by actual size), norm clip1, CE label smoothing0.05 plus Brier sum weight0.1. Train exactly two epochs over the827 fixed training positions shuffled deterministically per epoch; save both epoch checkpoints. Bounded search is these two checkpoints only; exclude feasibility-probe checkpoints. Use all215 validation positions to select the epoch with lowest finite mean target NLL; exact tie chooses earlier epoch. Report teacher agreement and Brier as diagnostics, not selection overrides. If either candidate has inference errors or nonfinite NLL it is ineligible. If both fail, preserve evidence and reconsider before tests. Initial fulltraining timeout3600s; no test access during training/selection.

Freeze final evaluation at200paired seeds30000–30199 and200piececap, bootstrap10000replicates/seed2026, error threshold0. Primary comparison selectedtrained versus unchanged nativeBF16headClef. Include an unchangedFP32head ablation to separate precision change from learning, plus random and heuristic. Same rules, complete observation encoding, pinnedbackbone, legalchoices and preview. Three neural policies at200×200moves and144ms imply4.8h worst-case, but unchangedbase dies around28pieces in development; expect much less. Allow bounded per-policy execution with incremental artifacts; never drop failedseeds or quietly reducepool. Record exact checkpoint and validation evidence hashes before any testepisode. Publish negative outcomes honestly; no tuning after testresults.

Fresh LoRA reload v1 correctly rejected PEFT's shortened target list. Saved496adapter tensors independently map exactly248intended text layers; weights were sound. Loader now expands saved targets over every backbone module and requires exact target equality, rejecting accidental vision/MTP/output matches. A32-layer tiny-backbone regression reproduces PEFT shortening. New saves use explicit full targets and canonical pinned model ID. Original failed artifact and probe weights preserved; retry uses same checkpoint, not a new training configuration. Head-only fresh-process reload passed independently; GLM restored healthy on both success and rejectedreload.

M4 complete: head-only fresh reload maxabsoluteprobabilitydifference0.0 (tolerance1e-4); unchanged saved LoRA checkpoint retry after semantic-target validation also0.0. Both fresh processes loaded pinned weights; no retraining needed for reload fix. Compact reports/gpu-feasibility.json retains success and original rejectedreload with sourcehashes. Fulltraining preregistration above is now enabled; output runs/study-v1 with two epoch checkpoints,3600second wrapper timeout and restored GLM health.
