"""Run a bounded Stackcraft job while temporarily borrowing the GLM GPU.

Use only under the user's recorded authorization to stop and restore GLM.
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

CONTAINER = "omarchy-local-ai-glm-4.7-flash.llamacpp.q5xl.32k.rtx-5090-engine"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.record.exists():
        parser.error("supply a command and a new record path")
    state = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER], text=True
    ).strip()
    if state != "true":
        parser.error("GLM is not currently running; do not alter an unexpected service state")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    report = {"container": CONTAINER, "command": command, "original_running": True}
    child = None
    stopped = False

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal{signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        # Set before stop so a partial/interrupted Docker call still attempts restoration.
        stopped = True
        subprocess.run(["docker", "stop", "--timeout", "30", CONTAINER], check=True, timeout=45)
        report["stopped_at"] = time.time()
        args.record.write_text(json.dumps(report, indent=2) + "\n")
        child = subprocess.Popen(command, start_new_session=True)
        report["job_pid"] = child.pid
        args.record.write_text(json.dumps(report, indent=2) + "\n")
        code = child.wait(timeout=args.timeout)
        report["exit_code"] = code
        if code:
            raise RuntimeError(f"Stackcraft GPU job exited with code{code}")
    finally:
        # A second ordinary interrupt must not interrupt service restoration.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                if child is not None and child.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
            finally:
                if stopped:
                    subprocess.run(["docker", "start", CONTAINER], check=True, timeout=60)
                    restored = subprocess.check_output(
                        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER],
                        text=True,
                    ).strip()
                    report["restored_running"] = restored == "true"
                    if not report["restored_running"]:
                        raise RuntimeError("GLM did not restart")
                    # Probe inside the container: host port8080 belongs to another service.
                    deadline = time.monotonic() + 120
                    while True:
                        health = subprocess.run(
                            [
                                "docker",
                                "exec",
                                CONTAINER,
                                "curl",
                                "-fsS",
                                "--max-time",
                                "5",
                                "http://localhost:8080/health",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if (
                            health.returncode == 0
                            and json.loads(health.stdout).get("status") == "ok"
                        ):
                            report["restored_healthy"] = True
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "GLM restarted but health did not recover within120s"
                            )
                        time.sleep(2)
                    report["restored_at"] = time.time()
        except BaseException as error:
            report["restoration_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            args.record.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
