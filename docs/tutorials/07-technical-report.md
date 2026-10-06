# Milestone 07 — Turn an experiment into a reproducible technical report

The deliverable is a readable [PDF](../../paper/stackcraft-technical-report.pdf),
its [Markdown edition](../../paper/stackcraft-technical-report.md), two vector
figures, and the source needed to rebuild them. The paper explains the completed
study; it does not start a new experiment or change the checkpoint selection.

## 1. Freeze the evidence before writing

The manuscript draws from `reports/final-study.md` and
`reports/final-summary.json`. Publication and downloaded-checkpoint claims come
from `reports/publication.json` and `reports/published-checkpoint-reload.json`.
The paper's `evidence.json` records SHA-256 hashes for these three JSON inputs.
The build refuses to proceed if any of them changes.

This separation matters because presentation work must not quietly change the
experiment. The original model revision, dataset revision, selection rule and
test set remain fixed. Improvements suggested in the discussion are future work,
not additional results. The model's performance must be compared with both the
unchanged model and the stronger heuristic.

## 2. Write methods and limitations before polishing the layout

Edit `paper/report.md`. Its sections explain the task, observation and action
contract, synthetic labels, training, validation selection, paired evaluation,
runtime, limitations, available artifacts and AI assistance. Update author and
version information in `paper/metadata.json`.

The paper states that only one training seed was used, the teacher is approximate,
the game rules are simplified, and the heuristic reaches the episode cap.
Bootstrap intervals describe differences across piece sequences for the selected
checkpoint; they do not measure variation across training runs. Validation teacher
agreement is distinct from complete-game quality. The computational audit was
performed by another project agent, not an independent laboratory.

This is labeled a technical report rather than a peer-reviewed paper. That gives
readers the actual publication status without understating the engineering work.

## 3. Generate figures and tables from the saved numbers

`paper/build.py` fills three explicit placeholders from the frozen JSON:
`{{OUTCOME_TABLE}}`, `{{VALIDATION_TABLE}}` and `{{RUNTIME_TABLE}}`. Do not manually
edit the generated tables in `stackcraft-technical-report.md`; edit the manuscript
template or builder and rebuild instead.

The outcomes figure uses a zero-based linear axis for mean lines and a separate
panel for paired differences with the recorded 95% intervals. It shows the
heuristic alongside the neural policies. The latency figure uses a logarithmic
axis because the measured times span several orders of magnitude. Its caption
states that policies visit different boards and use different hardware paths.
These are observed workloads, not a controlled same-state microbenchmark.

Matplotlib writes SVG, keeping labels and curves sharp in the PDF. Generated
illustrations are unnecessary here: every mark must correspond to a measured
quantity. No new confidence intervals or evaluations are computed by this build.

## 4. Build using an isolated uv environment

From the repository root:

```bash
uv run --project paper --locked python paper/build.py
```

`paper/pyproject.toml` and `paper/uv.lock` define a small CPU-only environment.
This preserves the original ML lockfile and avoids installing Torch or downloading
the base model just to typeset a document. Markdown is converted to HTML, then
WeasyPrint applies the print stylesheet. The builder uses the fonts shipped with
the locked Matplotlib package and embeds them in the PDF.

WeasyPrint requires native libraries such as Pango. If the import fails on a new
machine, follow its [platform installation instructions](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).
Use the committed PDF if you only want to read the report. This project chooses
Markdown and CSS over a separate LaTeX toolchain because the existing report is
Markdown and the required layout is straightforward.

The build writes the PDF, readable Markdown, SVG figures and a build manifest
with input/output hashes, Python/package versions and page count. It checks for
unresolved table placeholders, expected text, local filesystem links in the PDF,
and renderer warnings. A lockfile plus embedded fonts reduces variation; native
library differences can still alter PDF bytes across machines.

## 5. Review both the science and the rendered pages

Have a separate reviewer compare the manuscript, tables and figures with the
saved evidence. Then inspect every PDF page for clipped text, split captions,
tiny labels, empty pages and misleading axes. Automated text extraction is useful
but cannot establish that the visual layout is readable.

Optional system tools for inspection:

```bash
mkdir -p runs/paper-review
pdftotext -layout paper/stackcraft-technical-report.pdf runs/paper-review/report.txt
pdftoppm -scale-to 1400 -png paper/stackcraft-technical-report.pdf runs/paper-review/page
```

Rebuild a second time and compare the PDF/figure hashes on the same environment.
Check that the repository diff contains only the intended paper, documentation
and planning updates. No training or full game evaluation needs to be repeated
for a presentation-only addition.

## 6. Publish an additive artifact and verify it anonymously

Commit and push the reviewed source, figures and PDF to the existing GitHub
repository. On Hugging Face, add the PDF, readable Markdown, figures and build
manifest under `paper/`, then link the PDF prominently from the existing model
card. Include a link to the exact GitHub source commit. Review the current remote
card and retain its metadata, loader instructions and result tables.

Use the current model revision as `parent_commit` when making the additive Hub
commit, so concurrent edits cause a conflict rather than being silently lost.
Do not replace the checkpoint, dataset, original `code/`, evidence archive, or
historical manifests. Hugging Face documents atomic file operations in its
[upload guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload).

Download the new report files anonymously at the returned immutable revision.
Compare their SHA-256 hashes with the reviewed files and check the public GitHub
PDF and model-card links. Record the new revision and checks in
`reports/technical-report-publication.json`. The older release verification
continues to describe the original model/data revisions; the new record covers
this additive document publication.

This path uses repository files and requires no running Space or paid inference.
Uploading a PDF does not create an indexed Hugging Face Paper Page. The documented
[Paper Pages route](https://huggingface.co/docs/hub/paper-pages) uses an arXiv ID.
An arXiv submission or separate blog publication would be a later, explicit task.
