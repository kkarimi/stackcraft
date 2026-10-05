"""Actual pinned unchanged Clef development tournament; no held-out test seeds."""

import argparse
import json
import time
from pathlib import Path

import torch

from stackcraft.clef import ClefPlayer
from stackcraft.engine import new_game
from stackcraft.players import HeuristicPlayer, RandomPlayer, observe
from stackcraft.tournament import run_tournament


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-pieces", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.episodes <= 20:
        parser.error("use a new output and1–20 development episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    free, total = torch.cuda.mem_get_info()
    if free < 25 * 1024**3:
        raise RuntimeError(f"insufficient free GPU memory: {free}")
    started = time.monotonic()
    player = ClefPlayer.from_pretrained(trust_pinned_code=True)
    warmup = player.choose(observe(new_game(42)))
    print(
        json.dumps({"warmup": warmup.action_id, "load_seconds": time.monotonic() - started}),
        flush=True,
    )
    result = run_tournament(
        {"random": RandomPlayer, "heuristic": HeuristicPlayer, "clef-base": lambda: player},
        list(range(args.episodes)),
        args.max_pieces,
    )
    result["development_only"] = True
    result["gpu"] = torch.cuda.get_device_name()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
