"""Exercise restoration failures without touching Docker, processes, or signals."""

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "gpu_session.py"
    spec = importlib.util.spec_from_file_location("stackcraft_gpu_session_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = tmp_path / "session.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--record", str(record), "--", "fake-job"])
    calls = []
    handlers = []
    failures = {}
    child = SimpleNamespace(pid=12345, code=0)

    def wait(timeout):
        calls.append(("wait", timeout))
        if "wait" in failures:
            raise failures["wait"]
        return child.code

    child.wait = wait
    child.poll = lambda: None if "wait" in failures else child.code

    def run(command, **kwargs):
        calls.append(tuple(command))
        if command[1] in failures:
            raise failures[command[1]]
        if command[1] == "exec" and failures.get("health_responses"):
            response = failures["health_responses"].pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return SimpleNamespace(returncode=0, stdout='{"status":"ok"}')

    def popen(command, **kwargs):
        calls.append(("Popen", *command))
        assert kwargs == {"start_new_session": True}
        if "popen" in failures:
            raise failures["popen"]
        return child

    def killpg(pid, sig):
        calls.append(("killpg", pid, sig))
        if "killpg" in failures:
            raise failures["killpg"]

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "true\n")
    monkeypatch.setattr(module.os, "killpg", killpg)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(
        module.signal, "signal", lambda sig, handler: handlers.append((sig, handler))
    )
    return module, record, calls, handlers, failures, child


def restored(record, calls):
    assert any(call[:2] == ("docker", "start") for call in calls)
    report = json.loads(record.read_text())
    assert report["restored_running"] is True
    assert report["restored_healthy"] is True
    return report


def test_success_restores_service_and_checks_its_health(runner):
    module, record, calls, handlers, _, _ = runner
    module.main()
    report = restored(record, calls)
    assert report["exit_code"] == 0
    health = next(call for call in calls if call[:2] == ("docker", "exec"))
    assert health[-1] == "http://localhost:8080/health"
    assert (signal.SIGTERM, signal.SIG_IGN) in handlers
    assert (signal.SIGINT, signal.SIG_IGN) in handlers


def test_failed_child_restores_and_records_original_exit(runner):
    module, record, calls, _, _, child = runner
    child.code = 7
    with pytest.raises(RuntimeError, match="code7"):
        module.main()
    assert restored(record, calls)["exit_code"] == 7


@pytest.mark.parametrize("stage", ["stop", "popen"])
def test_partial_stop_or_spawn_failure_still_restores(runner, stage):
    module, record, calls, _, failures, _ = runner
    failures[stage] = OSError(f"{stage} failed")
    with pytest.raises(OSError, match=f"{stage} failed"):
        module.main()
    restored(record, calls)


@pytest.mark.parametrize("cleanup_error", [ProcessLookupError("gone"), PermissionError("denied")])
def test_cleanup_exception_cannot_skip_restoration(runner, cleanup_error):
    module, record, calls, _, failures, _ = runner
    failures["wait"] = subprocess.TimeoutExpired("fake-job", 10)
    failures["killpg"] = cleanup_error
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        module.main()
    report = restored(record, calls)
    assert "restoration_error" in report
    assert any(call[0] == "killpg" for call in calls)


def test_docker_restart_failure_is_recorded_even_when_it_raises(runner):
    module, record, calls, _, failures, _ = runner
    failures["start"] = subprocess.CalledProcessError(1, ["docker", "start"])
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    report = json.loads(record.read_text())
    assert "CalledProcessError" in report["restoration_error"]
    assert any(call[:2] == ("docker", "start") for call in calls)
    assert report.get("restored_healthy") is not True


def test_transient_invalid_json_and_timeout_retry_before_health_success(runner):
    module, record, calls, _, failures, _ = runner
    failures["health_responses"] = [
        SimpleNamespace(returncode=0, stdout="not ready yet"),
        subprocess.TimeoutExpired("docker exec health", 10),
        SimpleNamespace(returncode=0, stdout='{"status":"ok"}'),
    ]
    module.main()
    restored(record, calls)
    assert sum(call[:2] == ("docker", "exec") for call in calls) == 3
    assert sum(call == ("sleep", 2) for call in calls) == 2


def test_unexpected_initial_service_state_is_not_altered(runner, monkeypatch):
    module, record, calls, _, _, _ = runner
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "false\n")
    with pytest.raises(SystemExit):
        module.main()
    assert not record.exists()
    assert calls == []
