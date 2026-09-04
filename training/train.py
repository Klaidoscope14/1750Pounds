import csv
import random
import chess
import numpy as np
import torch
import torch.nn as nn

DATASET = "training/data/dataset.csv"
MODEL = "training/value_net.pt"

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001


def encode_board(fen):
    board = chess.Board(fen)
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


data = []

with open(DATASET, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        data.append(
            (
                encode_board(row["fen"]),
                float(row["evaluation"])
            )
        )

random.shuffle(data)

split = int(len(data) * 0.9)

train_data = data[:split]
val_data = data[split:]

X_train = torch.tensor(
    np.array([x for x, _ in train_data]),
    dtype=torch.float32
)

y_train = torch.tensor(
    np.array([y for _, y in train_data]),
    dtype=torch.float32
).unsqueeze(1)

X_val = torch.tensor(
    np.array([x for x, _ in val_data]),
    dtype=torch.float32
)

y_val = torch.tensor(
    np.array([y for _, y in val_data]),
    dtype=torch.float32
).unsqueeze(1)

model = ValueNet()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()

    indices = torch.randperm(len(X_train))
    total_loss = 0.0

    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]

        xb = X_train[batch_indices]
        yb = y_train[batch_indices]

        optimizer.zero_grad()

        prediction = model(xb)

        loss = loss_fn(prediction, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_indices)

    model.eval()

    with torch.no_grad():
        val_prediction = model(X_val)
        val_loss = loss_fn(val_prediction, y_val).item()

    train_loss = total_loss / len(X_train)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f}"
    )

torch.save(model.state_dict(), MODEL)

print(f"Model saved to {MODEL}")