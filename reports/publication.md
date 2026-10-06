# Stackcraft public release verification

Released on 2026-10-06. GitHub, the model and dataset are public. The owner
explicitly deferred CPU hosting; the playable game and Docker demo remain local.
No Space, subscription purchase or paid compute is claimed.

| Artifact | Public release | Verified revision |
| --- | --- | --- |
| Source and tutorials | [kkarimi/stackcraft](https://github.com/kkarimi/stackcraft) | Artifact source `82a8854258f228a214f0964bbbd3d2b52e2be1e5`; these completion records are in the containing commit on main |
| Adapter and decision head | [nima1/stackcraft-clef-flash-lora](https://huggingface.co/nima1/stackcraft-clef-flash-lora) | `c4272310bd6c63ee97a941255abf8f36ff162229` |
| Synthetic dataset | [nima1/stackcraft-data](https://huggingface.co/datasets/nima1/stackcraft-data) | `211708dd56f1a8af711c060c2ab1ecab35e7166d` |

## What was verified

Both Hugging Face repositories were downloaded anonymously at those exact
revisions into previously absent directories, with `force_download=True`.
Every model/dataset payload matches the independently retained release manifest;
the dataset audit confirms 827 training and 215 validation positions. The
verifier explicitly reports the hosted demo as skipped, per the owner's decision.

The actual downloaded archive was streamed and checked against all 1,017 raw-file
hashes. Its 74,541,451 bytes preserve 2,222,025,418 bytes of original evidence.
Compression changes no record, model weight, checkpoint or measured result.
Verify download integrity before extracting the ZIP into its repository root.

A fresh GPU process used the downloaded source, adapter, decision head and dataset.
All four fixed reference distributions matched exactly: maximum absolute
probability difference **0.0**, tolerance **1e-4**. Source-manifest verification
reported a clean source tree. This checks those four distributions, not every
possible board. The authorized GLM service was restored running and healthy;
a separate in-container health request also returned `status: ok`.

The complete local suite passed **304 tests**, including native CPU encoding.
Ruff lint/format and ty passed. An anonymous clean public GitHub checkout installed
and ran the game without ML packages; its game-only checks passed. Full type
checking requires the ML extra, as documented. The final model page renders its
results, custom-loader instructions and hosting deferral without the inapplicable
generic PEFT snippet. Public repository links and visibility were checked.

The existing desktop/mobile and CPU Docker checks verify live human gameplay,
recorded base/trained/heuristic comparisons, probability displays, synchronized
moves, replay, and the 200-piece comparison cap. The image runs as UID 1000 without
Torch. Follow [the local Docker guide](../docs/demo-hosting.md) to play.

## Evidence and reproduction

- [Machine-readable publication report](publication.json)
- [Retained release manifest](published-release-manifest.json)
- [Actual downloaded file and dataset checks](published-files-verification.json)
- [Downloaded archive checks](published-evidence-verification.json)
- [Downloaded-checkpoint GPU parity](published-checkpoint-reload.json)
- [GLM restoration record](published-gpu-restoration.json)
- [Anonymous download origins and revisions](published-download-provenance.json)
- [Release tutorial](../docs/tutorials/06-release.md)
- [Full scientific report](final-study.md)

The retained manifest's `published: false` field describes its original local
staging step. Actual publication revisions are recorded above and in
`publication.json`; the original manifest bytes were not rewritten after upload.
The source bundled on Hugging Face is the immutable artifact source revision.
These later verification records live on GitHub, avoiding circular self-hashes.

The scientific finding remains bounded: training improved mean lines from 0.07
to 16.81 on 200 held-out sequences, while the heuristic achieved 76.53 and was
much faster. One training seed, a small synthetic dataset and simplified rules
limit generalization. See the study report for all paired intervals and failures.
