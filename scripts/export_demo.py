"""Export a fixed-seed, validated recorded comparison without loading any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from stackcraft.players import Decision, observe, validate_decision
from stackcraft.replay import replay_states
from stackcraft.schema import RULES_VERSION

DISPLAY_NAMES = {
    "base": "Clef-flash base",
    "base-fp32": "Base · FP32 head",
    "trained": "Stackcraft fine-tune",
    "heuristic": "Heuristic",
    "random": "Random",
    "clef-base": "Clef-flash base",
}
DEMO_SEED = 30000


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validated_player(episode: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Re-simulate moves and check probabilities against their exact observations."""
    if (
        type(episode.get("schema_version")) is not int
        or episode["schema_version"] != 1
        or episode.get("rules_version") != RULES_VERSION
    ):
        raise ValueError("episode uses an incompatible schema or rules version")
    cap = episode.get("max_pieces")
    if type(cap) is not int or not 1 <= cap <= 2000:
        raise ValueError("demo episode cap must be between 1 and 2000")
    replay = episode["replay"]
    if replay.get("seed") != episode.get("seed"):
        raise ValueError("episode seed differs from replay seed")
    states = replay_states(replay)
    final = states[-1]
    if final.piece_index > cap:
        raise ValueError("episode exceeds its declared cap")
    outcome = episode["outcome"]
    expected = {
        "score": final.score,
        "lines": final.lines,
        "pieces": final.piece_index,
        "terminal": final.terminal,
    }
    if any(type(outcome.get(k)) is not type(v) or outcome[k] != v for k, v in expected.items()):
        raise ValueError("episode outcome differs from replay")
    errors = episode.get("errors")
    if not isinstance(errors, list):
        raise ValueError("episode errors must be an explicit list")
    status = "error" if errors else "top_out" if final.terminal else "cap"
    if outcome.get("status") != status or outcome.get("cap_hit") is not (
        not errors and not final.terminal and final.piece_index == cap
    ):
        raise ValueError("episode stop reason or cap flag is inconsistent")
    if not errors and not final.terminal and final.piece_index != cap:
        raise ValueError("successful recording ended before its declared cap")
    events = episode["decisions"]
    if not isinstance(events, list) or len(events) != final.piece_index + bool(errors):
        raise ValueError("decision count differs from replay and error outcome")
    compact = []
    for index, event in enumerate(events):
        if type(event.get("turn")) is not int or event["turn"] != index:
            raise ValueError("recorded decision indices must be consecutive")
        observation = observe(states[index])
        if _json_value(event.get("observation")) != _json_value(asdict(observation)):
            raise ValueError("recorded decision observation differs from replay")
        seconds = event.get("decision_seconds")
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("recorded decision time must be finite and nonnegative")
        item = {"turn": index, "decision_seconds": seconds}
        if index == final.piece_index:
            if event.get("error") not in errors or "action_id" in event:
                raise ValueError("terminal error event differs from episode errors")
            item["error"] = event["error"]
        else:
            action = event.get("action_id")
            if action != replay["actions"][index]:
                raise ValueError("recorded decision differs from replay action")
            distribution = event.get("probabilities")
            validate_decision(Decision(action, distribution), observation)
            item.update(action_id=action, probabilities=distribution)
        if "input_tokens" in event:
            if type(event["input_tokens"]) is not int or event["input_tokens"] < 1:
                raise ValueError("input token counts must be positive integers")
            item["input_tokens"] = event["input_tokens"]
        compact.append(item)
    player = episode["player"]
    player_id = episode["player_id"]
    if not isinstance(player.get("revision"), str) or not player["revision"]:
        raise ValueError("player revision must identify the actual recorded player")
    return {
        "id": player_id,
        "name": DISPLAY_NAMES.get(player_id, player.get("name", player_id)),
        "revision": player["revision"],
        "runtime_config": player.get("runtime_config", {}),
        "replay": replay,
        "outcome": outcome,
        "errors": errors,
        "decisions": compact,
        "source_index": source_index,
    }


def export_manifest(
    inputs: list[Path], players: list[str], seed: int = DEMO_SEED, *, report_url: str | None = None
) -> dict[str, Any]:
    if report_url is not None:
        parsed = urlsplit(report_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(char.isspace() or ord(char) < 32 for char in report_url)
            or "\\" in report_url
        ):
            raise ValueError("report URL must be an absolute HTTPS URL without credentials")
        _ = parsed.port  # Also reject malformed or out-of-range explicit ports.
    if seed != DEMO_SEED or type(seed) is not int:
        raise ValueError("the release demo seed is fixed before results at 30000")
    if len(players) != len(set(players)) or not 2 <= len(players) <= 4:
        raise ValueError("choose two to four distinct players")
    sources = []
    selected = {}
    caps = set()
    for path in inputs:
        raw = path.read_bytes()
        source = json.loads(raw)
        if not isinstance(source, dict):
            raise ValueError("source artifact must be a JSON object")
        episodes = source.get("episodes", [source] if "replay" in source else None)
        if not isinstance(episodes, list):
            raise ValueError(f"{path.name} is not a tournament or episode artifact")
        source_index = len(sources)
        provenance = {
            key: source[key]
            for key in ("provenance", "checkpoint_sha256", "final_test", "development_only")
            if key in source
        }
        sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), **provenance})
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("episodes must be JSON objects")
            identity = episode.get("player_id")
            if episode.get("seed") != seed or identity not in players:
                continue
            if identity in selected:
                raise ValueError(f"duplicate recording for {identity} at seed {seed}")
            selected[identity] = validated_player(episode, source_index)
            caps.add(episode["max_pieces"])
    missing = set(players) - set(selected)
    if missing:
        raise ValueError(f"missing fixed-seed recordings for: {', '.join(sorted(missing))}")
    if len(caps) != 1:
        raise ValueError("all recorded players must have the same episode cap")
    return {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "kind": "recorded-evaluation",
        "label": "Recorded player comparison",
        "seed": seed,
        "max_pieces": caps.pop(),
        "selection": {"seed": seed, "method": "fixed-before-results"},
        "disclaimer": (
            "Recorded decisions, no live model inference. Seed 30000 was fixed before results. "
            "One replay illustrates behavior; it is not the aggregate evaluation. "
            "Playback speed does not represent decision latency."
        ),
        "sources": sources,
        **({"report_url": report_url} if report_url is not None else {}),
        "players": [selected[name] for name in players],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--players", nargs="+", default=["base", "trained", "heuristic"])
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-url", help="Confirmed public HTTPS location of full study results"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve earlier evidence")
    try:
        manifest = export_manifest(args.input, args.players, args.seed, report_url=args.report_url)
        serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Exported recorded seed {args.seed} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
