import time
import chess
import numpy as np
import torch
import torch.nn as nn

MODEL = "training/value_net.pt"


def encode_board(board):
    x = np.zeros(772, dtype=np.float32)

    for square, piece in board.piece_map().items():
        piece_index = (piece.piece_type - 1) + (0 if piece.color else 6)
        x[piece_index * 64 + square] = 1.0

    x[768] = 1.0 if board.turn == chess.WHITE else 0.0
    x[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0

    return x


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(772, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


model = ValueNet()
model.load_state_dict(torch.load(MODEL, map_location="cpu"))
model.eval()

board = chess.Board()

with torch.no_grad():
    for _ in range(1000):
        x = torch.from_numpy(encode_board(board)).unsqueeze(0)
        model(x)

start = time.perf_counter()

with torch.no_grad():
    for _ in range(10000):
        x = torch.from_numpy(encode_board(board)).unsqueeze(0)
        model(x)

elapsed = time.perf_counter() - start

print("Evaluations: 10000")
print(f"Total time: {elapsed:.4f} seconds")
print(f"Evaluations/sec: {10000 / elapsed:.0f}")
print(f"Time/evaluation: {elapsed / 10000 * 1_000_000:.2f} us")