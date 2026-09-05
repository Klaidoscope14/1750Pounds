"""Phase B, step 1 — build a training set: sample positions and label them with Stockfish.

Runs OFFLINE on your machine. Stockfish is only a teacher here; it never ships in the agent.
Output is JSONL lines: {"fen": "...", "cp": <int, side-to-move relative centipawns>}.

Usage (from repo root, with the uv env):
    uv run python tools/label.py --pgn games.pgn --out data/train.jsonl --depth 12 --max 300000
    uv run python tools/label.py --selfplay 2000 --out data/selfplay.jsonl --depth 12

Get varied positions from a PGN (recommended): download a Lichess or CCRL PGN, e.g.
    https://database.lichess.org/  (a standard-rated month, then unzip a slice)
or pass --selfplay N to generate positions by self-play from the current agent (no download).

Tips for the M2 Pro: set --threads to ~ (cores-1) and --hash 512. Depth 12 is a good
speed/quality point; ~10-40 ms/position, so ~1M positions is an overnight run.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn

MATE_CP = 3000  # clamp mate scores to a large finite centipawn value


def positions_from_pgn(pgn_path: Path, every: int, skip_plies: int):
    """Yield sampled board positions from every game in a PGN."""
    with open(pgn_path) as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                return
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply >= skip_plies and ply % every == 0 and not board.is_game_over():
                    yield board.fen()


def positions_from_selfplay(agent_module, games: int, every: int, skip_plies: int):
    """Yield positions from lightly-randomised self-play of the current agent."""
    for _ in range(games):
        board = chess.Board()
        # a few random opening plies for variety, then let the agent play
        for _ in range(random.randint(2, 8)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        ply = 0
        while not board.is_game_over() and ply < 160:
            uci = agent_module.get_move(board.fen(), 300)  # 300ms/move, fast
            try:
                board.push(chess.Move.from_uci(uci))
            except (ValueError, AssertionError):
                break
            ply += 1
            if ply >= skip_plies and ply % every == 0 and not board.is_game_over():
                yield board.fen()


def label(fens, engine, limit, out_path: Path, max_positions: int) -> int:
    seen: set[str] = set()
    written = 0
    with open(out_path, "w") as out:
        for fen in fens:
            key = fen.rsplit(" ", 2)[0]  # ignore clocks/move number for dedup
            if key in seen:
                continue
            seen.add(key)
            board = chess.Board(fen)
            info = engine.analyse(board, limit)
            score = info["score"].pov(board.turn)  # side-to-move perspective
            cp = score.score(mate_score=MATE_CP)
            if cp is None:
                continue
            cp = max(-MATE_CP, min(MATE_CP, cp))
            out.write(json.dumps({"fen": fen, "cp": int(cp)}) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  labeled {written}...", flush=True)
            if written >= max_positions:
                break
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Label positions with Stockfish for NNUE training.")
    p.add_argument("--pgn", type=Path, help="source PGN of games")
    p.add_argument("--selfplay", type=int, default=0, help="instead: generate N self-play games")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--engine", default="stockfish", help="path/name of the Stockfish binary")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--hash", type=int, default=512)
    p.add_argument("--every", type=int, default=4, help="sample every Nth ply")
    p.add_argument("--skip-plies", type=int, default=8, help="skip the first N plies of each game")
    p.add_argument("--max", type=int, default=300_000, help="stop after this many labeled")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    if args.pgn:
        fens = positions_from_pgn(args.pgn, args.every, args.skip_plies)
    elif args.selfplay:
        spec_path = Path(__file__).resolve().parent.parent / "agent.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent", spec_path)
        agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent)
        fens = positions_from_selfplay(agent, args.selfplay, args.every, args.skip_plies)
    else:
        raise SystemExit("pass --pgn <file> or --selfplay <N>")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads, "Hash": args.hash})
    try:
        n = label(fens, engine, chess.engine.Limit(depth=args.depth), args.out, args.max)
    finally:
        engine.quit()
    print(f"done: {n} labeled positions -> {args.out}")


if __name__ == "__main__":
    main()
