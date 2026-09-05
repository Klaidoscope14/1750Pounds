"""Phase B, step 1 (better) — turn the Lichess eval dump into our training set.

The Lichess eval DB (https://database.lichess.org/#evals) is ~395M positions already
evaluated by Stockfish at high depth. We stream it, take each position's deepest eval and
its first PV's score, and write our training format: {"fen","cp"} with cp in SIDE-TO-MOVE
centipawns (what the net predicts).

Stream without downloading the whole file (stops early after --max):
    curl -sL https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc | uv run python tools/parse_lichess.py --out data/train.jsonl --max 3000000

First, CONFIRM the perspective convention against your local Stockfish (important!):
    curl -sL https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc | uv run python tools/parse_lichess.py --verify 300
It prints how well cp matches a local Stockfish eval under each assumption and tells you
which --source-pov to use (default: white).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

MATE_CP = 3000


def full_fen(fen: str) -> str:
    """Lichess FENs omit the move clocks; python-chess needs all six fields."""
    parts = fen.split()
    while len(parts) < 6:
        parts.append("0" if len(parts) == 4 else "1")
    return " ".join(parts)


def best_eval(rec: dict) -> dict | None:
    """Pick the evaluation with the greatest depth."""
    evals = rec.get("evals") or []
    if not evals:
        return None
    return max(evals, key=lambda e: e.get("depth", 0))


def score_cp(ev: dict) -> int | None:
    """First-PV score of an eval, in centipawns (mate clamped). White-relative as stored."""
    pvs = ev.get("pvs") or []
    if not pvs:
        return None
    pv = pvs[0]
    if "cp" in pv:
        return max(-MATE_CP, min(MATE_CP, int(pv["cp"])))
    if "mate" in pv:
        m = int(pv["mate"])
        return MATE_CP if m > 0 else -MATE_CP
    return None


def to_stm(cp_stored: int, turn: bool, source_pov: str) -> int:
    """Convert the stored score to side-to-move perspective."""
    if source_pov == "stm":
        return cp_stored
    # source is white-relative: negate when black is to move
    return cp_stored if turn == chess.WHITE else -cp_stored


def run_parse(out_path: Path, max_positions: int, source_pov: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as out:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = best_eval(rec)
            if ev is None:
                continue
            cp = score_cp(ev)
            if cp is None:
                continue
            fen = full_fen(rec["fen"])
            turn = chess.WHITE if " w " in fen else chess.BLACK
            out.write(json.dumps({"fen": fen, "cp": to_stm(cp, turn, source_pov)}) + "\n")
            written += 1
            if written % 100000 == 0:
                print(f"  {written} positions...", file=sys.stderr, flush=True)
            if written >= max_positions:
                break
    return written


def run_verify(sample: int) -> None:
    """Cross-check the stored cp against a local Stockfish eval to fix the perspective."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Threads": 4, "Hash": 512})
    agree_white = agree_stm = n = 0
    try:
        for line in sys.stdin:
            if n >= sample:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = best_eval(rec)
            if ev is None:
                continue
            cp = score_cp(ev)
            if cp is None or abs(cp) < 40:  # skip near-equal (sign is noise)
                continue
            fen = full_fen(rec["fen"])
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=10))
            sf_stm = info["score"].pov(board.turn).score(mate_score=MATE_CP)
            if sf_stm is None:
                continue
            n += 1
            # white-assumption -> stm value:
            white_as_stm = cp if board.turn == chess.WHITE else -cp
            agree_white += (white_as_stm > 0) == (sf_stm > 0)
            agree_stm += (cp > 0) == (sf_stm > 0)
    finally:
        engine.quit()
    if n == 0:
        print("no decisive positions sampled", file=sys.stderr)
        return
    pw, ps = agree_white / n, agree_stm / n
    print(f"\nsampled {n} decisive positions:", file=sys.stderr)
    print(f"  assume source=WHITE : sign matches local SF {pw:.1%}", file=sys.stderr)
    print(f"  assume source=STM   : sign matches local SF {ps:.1%}", file=sys.stderr)
    print(f"\n=> use --source-pov {'white' if pw >= ps else 'stm'}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Parse the Lichess eval dump into training data.")
    p.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--max", type=int, default=3_000_000)
    p.add_argument("--source-pov", choices=["white", "stm"], default="white")
    p.add_argument("--verify", type=int, metavar="N", help="cross-check POV on N positions")
    args = p.parse_args()

    if args.verify:
        run_verify(args.verify)
        return
    n = run_parse(args.out, args.max, args.source_pov)
    print(f"done: {n} positions -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
