"""Typeset the existing study; no model execution or new statistical estimates."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import logging
import platform
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
from pypdf import PdfReader
from weasyprint import CSS, HTML
from weasyprint.text.fonts import FontConfiguration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAYERS = ["base", "base-fp32", "trained", "random", "heuristic"]
LABELS = ["Native base", "FP32 base", "Trained", "Random", "Heuristic"]
COLORS = ["#758693", "#a0aeb7", "#087e9b", "#b6bbc2", "#ba6820"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]
    )


def tables(summary: dict) -> dict[str, str]:
    outcomes, runtime = [], []
    for key, label in zip(PLAYERS, LABELS, strict=True):
        player = summary["players"][key]
        result = player["failure_adjusted"]
        outcomes.append(
            [
                label,
                f"{result['lines']['mean']:.3f}",
                f"{result['lines']['median']:.0f}",
                f"{result['score']['mean']:.1f}",
                f"{result['pieces']['mean']:.3f}",
                f"{player['cap_hits']}/{player['episodes']}",
            ]
        )
        latency = player["latency_seconds"]
        runtime.append(
            [label]
            + [f"{latency[k] * 1000:.6f}" for k in ("mean", "median", "p95")]
            + [f"{latency['count']:,}"]
        )
    validation = summary["selection"]["validation"]
    candidates = [
        ("Native base", validation["epoch-01"]["players"]["base"]),
        ("FP32 base", validation["epoch-01"]["players"]["base-fp32"]),
        ("Epoch 01", validation["epoch-01"]["players"]["trained"]),
        ("Epoch 02 (selected)", validation["epoch-02"]["players"]["trained"]),
    ]
    val_rows = [
        [
            name,
            f"{p['correct']}/{p['positions']} ({p['teacher_agreement']:.2%})",
            f"{p['mean_nll']:.6f}",
            f"{p['mean_brier']:.6f}",
        ]
        for name, p in candidates
    ]
    return {
        "{{OUTCOME_TABLE}}": table(
            ["Player", "Mean lines", "Median lines", "Mean score", "Mean pieces", "Cap hits"],
            outcomes,
        ),
        "{{VALIDATION_TABLE}}": table(
            ["Condition", "Teacher agreement", "Mean NLL", "Mean Brier"], val_rows
        ),
        "{{RUNTIME_TABLE}}": table(
            ["Player", "Mean ms", "Median ms", "p95 ms", "Decisions"], runtime
        ),
    }


def figures(summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.labelcolor": "#344653",
            "text.color": "#202b38",
            "svg.hashsalt": "stackcraft-technical-report-v1",
            "svg.fonttype": "path",
        }
    )
    destination = HERE / "figures"
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.6), layout="constrained")
    means = [summary["players"][p]["failure_adjusted"]["lines"]["mean"] for p in PLAYERS]
    ax = axes[0]
    ax.barh(LABELS, means, color=COLORS, height=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 88)
    ax.set_xlabel("Mean lines cleared · 200 sequences per player")
    ax.set_title(
        "A  Fine-tuning improves play; the heuristic remains stronger", loc="left", size=10
    )
    for y, value in enumerate(means):
        ax.text(value + 1.2, y, f"{value:.3f}", va="center", size=9)
    ax.tick_params(axis="y", length=0)
    ax = axes[1]
    names = ["Trained − native", "Trained − FP32 base", "Trained − heuristic"]
    keys = ["trained_vs_base", "trained_vs_base_fp32", "trained_vs_heuristic"]
    for y, key in enumerate(keys):
        data = summary["comparisons"][key]["paired_trained_minus_base"]["lines"]
        mean, low, high = (data[k] for k in ("mean_difference", "ci95_lower", "ci95_upper"))
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color="#087e9b" if mean > 0 else "#ba6820",
            capsize=4,
        )
        ax.text(high + 2, y, f"{mean:+.2f} [{low:.2f}, {high:.2f}]", va="center", size=8.5)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_ylim(2.5, -0.5)
    ax.set_xlim(-70, 64)
    ax.axvline(0, color="#9aa7b1", lw=0.8, linestyle="--")
    ax.set_xlabel("Paired mean difference in lines · 95% bootstrap interval")
    ax.set_title(
        "B  Uncertainty across sequences, conditional on this trained checkpoint",
        loc="left",
        size=10,
    )
    ax.tick_params(axis="y", length=0)
    fig.savefig(destination / "outcomes.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 2.65), layout="constrained")
    latencies = [summary["players"][p]["latency_seconds"]["mean"] * 1000 for p in PLAYERS]
    ax.scatter(latencies, range(5), c=COLORS, s=55, zorder=3)
    ax.set_yticks(range(5), LABELS)
    ax.invert_yaxis()
    ax.set_ylim(4.6, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.0006, 2500)
    ax.grid(axis="x", alpha=0.18)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean decision time (milliseconds, logarithmic scale)")
    ax.set_title("Observed workloads: neural players on GPU, others on CPU", loc="left", size=10)
    for y, value in enumerate(latencies):
        ax.text(value * 1.3, y, f"{value:.4g} ms", va="center", size=9)
    fig.savefig(destination / "latency.svg", metadata={"Date": None})
    plt.close(fig)


class RenderWarnings(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            raise RuntimeError(f"PDF rendering warning: {record.getMessage()}")


def main() -> None:
    evidence = json.loads((HERE / "evidence.json").read_text())
    for name, expected in evidence.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    summary = json.loads((ROOT / "reports/final-summary.json").read_text())
    meta = json.loads((HERE / "metadata.json").read_text())
    figures(summary)
    manuscript = (HERE / "report.md").read_text()
    if not manuscript.startswith("# "):
        raise ValueError("Manuscript must start with one H1 title")
    body = manuscript.split("\n", 1)[1]
    for placeholder, replacement in tables(summary).items():
        if body.count(placeholder) != 1:
            raise ValueError(f"Expected exactly one {placeholder}")
        body = body.replace(placeholder, replacement)
    if "{{" in body:
        raise ValueError("Unresolved manuscript placeholder")
    markdown = (
        f"# {meta['title']}\n\n{meta['author']} · {meta['date']} · Version {meta['version']}\n\n"
        f"**{meta['status']}**\n\n" + body
    )
    (HERE / "stackcraft-technical-report.md").write_text(markdown)
    content = MarkdownIt("commonmark", {"html": True}).enable("table").render(body)
    content = content.replace("<!-- pagebreak -->", '<div class="pagebreak"></div>')
    # Keep a following Figure caption attached to its image across page breaks.
    content = re.sub(
        r"<p>(<img [^>]+>)</p>\s*<p>((?:<(?:strong|em)>)?Figure\s.*?</p>)",
        lambda m: "<figure>" + m[1] + "<figcaption>" + m[2][:-4] + "</figcaption></figure>",
        content,
        flags=re.DOTALL,
    )
    escape = html.escape
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(meta['title'])}</title>"
        f'<meta name="author" content="{escape(meta["author"])}">'
        f'<meta name="dcterms.created" content="{meta["date"]}T00:00:00Z">'
        "</head><body>"
        f'<div class="status">{escape(meta["status"])}</div>'
        f"<h1>{escape(meta['title'])}</h1>"
        f'<p class="byline"><a href="{meta["profile"]}">{escape(meta["author"])}</a>'
        f" · {meta['date']} · Version {meta['version']}</p>{content}</body></html>"
    )
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    faces = []
    for family, file in [("ReportSerif", "DejaVuSerif"), ("ReportSans", "DejaVuSans")]:
        for style, weight, suffix in [
            ("normal", "normal", ""),
            ("normal", "bold", "-Bold"),
            ("italic", "normal", "-Italic" if "Serif" in file else "-Oblique"),
        ]:
            uri = (fonts / f"{file}{suffix}.ttf").as_uri()
            faces.append(
                f'@font-face {{font-family:{family}; src:url("{uri}"); '
                f"font-style:{style}; font-weight:{weight};}}"
            )
    uri = (fonts / "DejaVuSansMono.ttf").as_uri()
    faces.append(f'@font-face {{font-family:ReportMono; src:url("{uri}");}}')
    css = "\n".join(faces) + "\n" + (HERE / "style.css").read_text()
    font_config = FontConfiguration()
    logging.getLogger("weasyprint").addHandler(RenderWarnings())
    rendered = HTML(string=document, base_url=str(HERE)).render(
        stylesheets=[CSS(string=css, font_config=font_config)], font_config=font_config
    )
    pdf = HERE / "stackcraft-technical-report.pdf"
    rendered.write_pdf(pdf)
    reader = PdfReader(pdf)
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ("16.810", "76.530", "0.070", "Not peer reviewed", meta["author"]):
        if expected not in text:
            raise ValueError(f"Missing PDF text: {expected}")
    for page in reader.pages:
        for item in page.get("/Annots", []):
            action = item.get_object().get("/A", {})
            if str(action.get("/URI", "")).startswith("file:"):
                raise ValueError("Local filesystem link leaked into PDF")
    outputs = [
        pdf,
        HERE / "stackcraft-technical-report.md",
        *sorted((HERE / "figures").glob("*.svg")),
    ]
    inputs = [
        HERE / name
        for name in (
            "report.md",
            "metadata.json",
            "style.css",
            "build.py",
            "pyproject.toml",
            "uv.lock",
            "evidence.json",
        )
    ]
    manifest = {
        "schema_version": 1,
        "scope": "Presentation of frozen evidence; no new experiments or statistical estimates",
        "frozen_evidence": evidence,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "outputs": {
            str(p.relative_to(HERE)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in outputs
        },
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("matplotlib", "markdown-it-py", "weasyprint", "pypdf")
        },
        "pages": len(reader.pages),
        "pdf_text_checks": "passed",
        "fonts": "DejaVu fonts from the locked Matplotlib distribution; embedded in PDF",
    }
    (HERE / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "pages": len(reader.pages), "sha256": sha256(pdf)}))


if __name__ == "__main__":
    main()
