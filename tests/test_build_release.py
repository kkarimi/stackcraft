"""Public packaging stays navigable without the private repository."""

import importlib.util
import re
from pathlib import Path

import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/build_release.py"
    spec = importlib.util.spec_from_file_location("stackcraft_test_packaging", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_source_includes_download_verifier(script):
    paths = script.code_files(Path(__file__).resolve().parents[1], demo=False)
    assert Path("scripts/verify_release.py") in paths


def test_source_readme_replaces_only_known_private_plan_link(script):
    before = (
        "See [the project plan](plan.md) for milestones.\n"
        "Literal plan.md and [another link](elsewhere.md) remain untouched.\n"
    )
    assert script.bundled_source_readme(before) == (
        "See [the release tutorial](docs/tutorials/06-release.md) for milestones.\n"
        "Literal plan.md and [another link](elsewhere.md) remain untouched.\n"
    )
    assert script.bundled_source_readme("No planning link here.\n") == "No planning link here.\n"


@pytest.mark.parametrize("report_url", [None, "https://example.org/released-study"])
def test_space_readme_links_only_shipped_files_or_explicit_report(script, report_url):
    readme = script.space_readme(report_url)
    root = Path(__file__).resolve().parents[1]
    files = {str(path) for path in script.code_files(root, demo=True)}
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert links
    for link in links:
        if link.startswith("<"):
            assert link == f"<{report_url}>"
        elif link.endswith("/"):
            assert any(path.startswith(link) for path in files), link
        else:
            assert link in files, link
    assert readme.startswith("---\ntitle: Stackcraft\nsdk: docker\napp_port: 7860\n")
    assert "**Human play is live. Bot comparisons are recorded.**" in readme
    assert "runs on CPU" in readme
    assert "Playback speed changes the animation, not measured inference latency." in readme
    assert "plan.md" not in readme
    assert "huggingface.co/" not in readme
    assert ("Complete study report" in readme) == (report_url is not None)
