# Host the playable game and recorded comparison

The demo serves the actual Python game and validated recorded model decisions. It
requires no GPU, Hugging Face token, Torch installation or model download. Human
moves are live simulator calls; comparison boards replay decisions measured during
the evaluation. Playback timing is deliberately separate from inference latency.

This is a deployment package, not evidence of publication. Create or upload a new
Space only after the project owner confirms its destination and visibility.

## Export the fixed example after evaluation

The demonstration seed is **30000**, chosen before inspecting results. Do not pick
a better-looking seed after the evaluation. A single game illustrates behavior;
the aggregate evaluation report establishes the quantitative result.

After the final tournament has completed, export to a new file:

```bash
uv run --locked python scripts/export_demo.py \
  --input runs/final-evaluation/report.json \
  --players base trained heuristic \
  --seed 30000 \
  --output runs/release-demo/baseline-demo.json
```

The input path above is an example; use the actual complete evaluation report.
`--input` can be repeated for separate tournament or individual episode artifacts.
The exporter requires two to four distinct players, an identical piece cap, and
exactly one recording per selected player at the fixed seed. The optional
`base-fp32` player is a precision ablation, not the primary unchanged model.

The exporter replays every selected action with the authoritative engine and
checks final outcomes, stop reasons, observations, probability distributions and
decision/action alignment. It preserves errors as errors. It keeps complete
replays, compact decision probabilities, measured decision times, actual player
revisions, runtime metadata, source filenames and SHA-256 hashes. It strips the
large repeated observation payloads because the game can reconstruct them.
Neither missing players nor invalid records are replaced with simulated model
results. Existing outputs are never overwritten.

After the results page's publication destination is confirmed, optionally add
`--report-url` with its absolute HTTPS URL. This adds a "Full study results" link
to the race without changing source hashes or recorded decisions. Omit it until
the actual destination is known; the UI hides the link when absent or invalid.

The JSON is compatible with the race manifest loaded from
`src/stackcraft/web/baseline-demo.json`. Before a release, review the exported
example and replace that packaged asset explicitly. The checked-in development
asset must remain labelled as development until actual evaluation replays replace
it. Do not rename a heuristic recording to make a missing model board appear.

## Build the game-only container

```bash
docker build -t stackcraft-demo:local .
docker run --rm --name stackcraft-demo-local \
  -p 127.0.0.1:7860:7860 stackcraft-demo:local
```

Open `http://127.0.0.1:7860`. The container runs as UID 1000 and listens on all
interfaces inside the container; this local command exposes it only on loopback.
The Dockerfile uses Python 3.13 and a versioned uv executable. Its build stage runs
`uv sync --locked --no-dev --no-editable` with no ML extra. Installing the project
rather than an editable source link checks that the wheel contains HTML, CSS,
JavaScript and the replay JSON. The runtime stage contains that environment and
uses the CLI directly, with no dependency resolution at startup. This follows the
[uv container installation guidance](https://docs.astral.sh/uv/guides/integration/docker/).

`.dockerignore` allows only the source and locked build inputs into the build
context. Local credentials, model weights, training runs, `.venv`, raw datasets
and Git history are excluded. The image does not need them to reconstruct games.
The dependency lock freezes Python packages. Both the Python base and uv image
are pinned by digest; the verified Python base supplies 3.13.16. Update these
explicitly when upgrading the deployment, and record the final application image
identity for the published build.

The container defaults to port 7860. To change it locally:

```bash
docker run --rm --name stackcraft-demo-local \
  -e PORT=8090 -p 127.0.0.1:8090:8090 stackcraft-demo:local
```

This container setting does not change the normal CLI's default local port 8080.
Use a free host port; do not stop another application merely to reclaim 8080.

## Configure a Hugging Face Docker Space

Once publication is authorized, the Space repository needs the Dockerfile,
`.dockerignore`, `pyproject.toml`, `uv.lock`, `src/` including the reviewed replay
asset, and a README. Put this configuration at the top of the Space README:

```yaml
---
title: Stackcraft
emoji: 🧩
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
---
```

The README body should link the model, dataset, code, aggregate report and
reproduction tutorials, and explain which activity is live. The Space configuration
uses `sdk: docker` and the same exposed port as the application. UID 1000 matches
Hugging Face's documented container permissions. No GPU hardware upgrade or model
credential is needed for this recorded demonstration. See the official
[Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).

Hugging Face builds from the Space repository. Upload only the explicit release
files, not the entire working directory. The source repository's visibility and
the Space's visibility are separate publication choices.

## Check the actual deployment

Run these checks against the built image and repeat them against the published
Space URL after release:

1. `/health` returns `{"status":"ok"}` and `/` loads its packaged assets.
2. A human can rotate, place a piece, restart and download a replay.
3. Opening that replay reproduces the same score, lines and board.
4. The race clearly says recorded; scrub to both the first and last frame.
5. The shorter run stays at its actual final state. An error is labelled as an
   error, and reaching the episode cap is distinct from top-out.
6. The selected action and any probability alternatives come from the recorded
   decision at that position. They are not an explanation generated after the run.
7. "Start live board · seed N" explicitly starts a new human game beside the
   recorded players using the exact recording seed. Check the same episode cap,
   keyboard controls, per-move recording synchronization and the live/recorded labels.
   Scrubbing recorded moves must leave the human board unchanged. "Match my move"
   restores the shared piece index. Switching Play/race preserves the human session;
   "Leave comparison" returns it to uncapped ordinary play without a reset. On narrow
   screens boards stack so controls remain readable; check for horizontal overflow.
8. The running image imports the game without Torch. No requests download model
   weights, and no inference service is contacted.

Game sessions live in process memory. A container restart discards them, and the
service retains a bounded number of sessions. Save a replay for durable evidence.
Use one application worker with the current in-memory session store; independent
worker processes would not share game IDs. A larger deployment would need a
shared session store and request limits, rather than silently adding workers.

Keep the final image identity, released manifest hash and successful URL checks in
`record.md`. A local Dockerfile or a successful upload alone does not establish
that the public comparison works.
