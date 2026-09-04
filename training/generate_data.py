import csv
import chess
import chess.pgn
import chess.engine

STOCKFISH = r"C:\Users\Rishabh\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
PGN_FILE = "training/data/games.pgn"
OUTPUT_FILE = "training/data/dataset.csv"

TARGET_POSITIONS = 30000
STOCKFISH_DEPTH = 10

positions = []

with open(PGN_FILE, "r", encoding="utf-8", errors="ignore") as f:
    while len(positions) < TARGET_POSITIONS:
        game = chess.pgn.read_game(f)

        if game is None:
            break

        board = game.board()
        ply = 0

        for move in game.mainline_moves():
            board.push(move)
            ply += 1

            if ply >= 6 and ply % 4 == 0:
                positions.append(board.copy())

                if len(positions) >= TARGET_POSITIONS:
                    break

print(f"Collected {len(positions)} positions.")

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["fen", "evaluation"])

    for i, board in enumerate(positions, 1):
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=STOCKFISH_DEPTH)
        )

        score = result["score"].pov(board.turn)

        if score.is_mate():
            evaluation = 1.0 if score.mate() > 0 else -1.0
        else:
            cp = score.score()
            evaluation = cp / (abs(cp) + 400.0)

        writer.writerow([board.fen(), evaluation])

        if i % 500 == 0:
            print(f"Evaluated {i}/{len(positions)}")

engine.quit()

print(f"Dataset saved to {OUTPUT_FILE}")