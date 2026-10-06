# Stackcraft technical report

[Read the PDF](stackcraft-technical-report.pdf) ·
[Read the Markdown edition](stackcraft-technical-report.md)

**Stackcraft: Adapting Clef-flash to a Falling-Block Decision Task**

This is a technical report, not a peer-reviewed publication. It presents the
completed experiment: the trained policy improved over unchanged Clef-flash,
while the fixed heuristic remained much stronger and faster. It includes the
study's synthetic-data provenance, limitations and AI-assistance disclosure.

From the repository root:

```bash
uv run --project paper --locked python paper/build.py
```

The isolated paper environment installs no Torch, model weights or GPU packages.
It requires Python 3.13 and the system libraries needed by
[WeasyPrint](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html),
including Pango. On the original Linux workstation these libraries were already
present. The build embeds DejaVu fonts supplied by the locked Matplotlib package.
Native rendering libraries can still affect PDF bytes on another platform.

`report.md` is the manuscript template; its three table placeholders are filled
directly from the frozen study summary. `metadata.json` contains the author,
title, date and version. `style.css` controls typesetting. `build.py` generates
both vector figures, the readable Markdown edition, the PDF and
`build-manifest.json`. `evidence.json` pins the source reports by SHA-256 and the
build refuses changed inputs. It neither trains a model nor reruns evaluation.

The committed PDF and figures are intentional publication artifacts. Inspect the
rendered pages after any change, then update the public assets as a new additive
publication. Historical model/data revisions and release verification records
remain valid. See [Tutorial 07](../docs/tutorials/07-technical-report.md) for the
complete workflow and the reasons for these choices.

The repository's Apache-2.0 license applies. This document does not claim a DOI,
arXiv record, or an indexed Hugging Face Paper Page.
