"""Small CLI that keeps optional model dependencies out of the game path."""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stackcraft: play, replay, and measure decisions.")
    parser.add_argument("--version", action="version", version="stackcraft 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local game and replay service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    tournament = commands.add_parser("tournament", help="Run paired offline development baselines")
    tournament.add_argument("--seed-start", type=int, default=1000)
    tournament.add_argument("--episodes", type=int, default=20)
    tournament.add_argument("--max-pieces", type=int, default=100)
    tournament.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate-data", help="Generate and audit expert-labelled data")
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("stackcraft.server:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "tournament":
        from stackcraft.players import HeuristicPlayer, RandomPlayer
        from stackcraft.tournament import run_tournament

        if args.episodes < 1 or args.max_pieces < 1:
            parser.error("--episodes and --max-pieces must be positive")
        if args.output.exists():
            parser.error(f"output already exists: {args.output}; choose a new path")
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        result = run_tournament(
            {"random": RandomPlayer, "heuristic": HeuristicPlayer}, seeds, args.max_pieces
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Saved {args.episodes} paired development episodes to {args.output}")
    elif args.command == "generate-data":
        from stackcraft.data import DatasetConfig, generate_dataset, write_dataset

        if args.output.exists():
            parser.error(f"output already exists: {args.output}; choose a new path")
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            if dirty:
                commit += "+working-tree"
        except (OSError, subprocess.CalledProcessError):
            commit = "unversioned"
        bundle = generate_dataset(DatasetConfig(), source_commit=commit)
        write_dataset(bundle, args.output)
        print(f"Saved expert dataset and manifest to {args.output}")


if __name__ == "__main__":
    main()
