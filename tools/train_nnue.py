"""Phase B, step 2 — train the NNUE and export int16 weights for the agent.

Runs on Colab (GPU) or locally (CPU). Reads data/train.jsonl ({"fen","cp"} with cp in
side-to-move centipawns) and writes weights/nnue.npz.

Architecture (MUST match the numba runtime in agent.py exactly):
  HalfKP features (king-relative), NFEAT = 40960
    feature index = king_sq*640 + piece_sq*10 + piece_idx
    piece_idx = (piece_type-1)*2 + (0 if piece belongs to the perspective side else 1)
    black perspective flips squares and king by ^56
  accumulator: EmbeddingBag(NFEAT, L1, sum) + bias, clipped to [0,1]      (per perspective)
  input to the net: concat(acc[side-to-move], acc[other]) -> 2*L1
  body: clamp01( Linear(2L1, 32) ) -> Linear(32, 1)  = centipawn prediction

Quantization (documented so the runtime can reproduce it with integers):
  QA = 127  (activation scale: float [0,1] <-> int [0,127])
  QW = 64   (layer-weight scale; layer2 divides by 64 == >>6)
  W1_q = round(W1*QA) int16 ; b1_q = round(b1*QA) int32
  W2_q = round(W2*QW) int16 ; b2_q = round(b2*QA*QW) int32
  W3_q = round(W3*QW) int16 ; b3_q = round(b3*QA*QW) int32
  int forward: acc = clip(sum(W1_q)+b1_q, 0,127); out2 = clip((acc@W2_q + b2_q)>>6, 0,127)
               cp  = (out2@W3_q + b3_q) // (QA*QW)          # == // 8128

Usage:
  uv run python tools/train_nnue.py --data data/train.jsonl --out weights/nnue.npz --epochs 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

NFEAT = 40960
L1 = 256
QA = 127
QW = 64
OUT_DIV = QA * QW    # 8128
EVAL_SCALE = 400     # net predicts cp/EVAL_SCALE (small range); cp = EVAL_SCALE * net_output
MATE_CP = 3000


def piece_features(board: chess.Board) -> tuple[list[int], list[int]]:
    """Return (white_perspective_features, black_perspective_features) for all non-king pieces."""
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    wf: list[int] = []
    bf: list[int] = []
    for sq, pc in board.piece_map().items():
        if pc.piece_type == chess.KING:
            continue
        base = (pc.piece_type - 1) * 2
        # white perspective
        pidx = base + (0 if pc.color == chess.WHITE else 1)
        wf.append(wk * 640 + sq * 10 + pidx)
        # black perspective (flip squares + king)
        pidx = base + (0 if pc.color == chess.BLACK else 1)
        bf.append((bk ^ 56) * 640 + (sq ^ 56) * 10 + pidx)
    return wf, bf


class Positions(Dataset):
    def __init__(self, path: Path, limit: int | None) -> None:
        self.stm: list[np.ndarray] = []
        self.opp: list[np.ndarray] = []
        self.cp: list[float] = []
        with open(path) as handle:
            for i, line in enumerate(handle):
                if limit and i >= limit:
                    break
                rec = json.loads(line)
                board = chess.Board(rec["fen"])
                wf, bf = piece_features(board)
                if board.turn == chess.WHITE:
                    self.stm.append(np.array(wf, dtype=np.int64))
                    self.opp.append(np.array(bf, dtype=np.int64))
                else:
                    self.stm.append(np.array(bf, dtype=np.int64))
                    self.opp.append(np.array(wf, dtype=np.int64))
                self.cp.append(float(max(-MATE_CP, min(MATE_CP, rec["cp"]))))

    def __len__(self) -> int:
        return len(self.cp)

    def __getitem__(self, i: int):
        return self.stm[i], self.opp[i], self.cp[i]


def collate(batch):
    stm, opp, cp = zip(*batch, strict=True)

    def bag(arrs):
        offsets = np.zeros(len(arrs), dtype=np.int64)
        offsets[1:] = np.cumsum([len(a) for a in arrs])[:-1]
        flat = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.int64)
        return torch.from_numpy(flat), torch.from_numpy(offsets)

    fs, os_ = bag(stm)
    fo, oo = bag(opp)
    return fs, os_, fo, oo, torch.tensor(cp, dtype=torch.float32).unsqueeze(1)


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(NFEAT, L1, mode="sum")
        # bias starts at 0.5 so the summed accumulator lands mid-range and the [0,1] clip
        # does not zero out every activation (which would kill gradients early in training)
        self.b1 = nn.Parameter(torch.full((L1,), 0.5))
        self.fc2 = nn.Linear(2 * L1, 32)
        self.fc3 = nn.Linear(32, 1)
        nn.init.uniform_(self.ft.weight, -0.1, 0.1)

    def forward(self, fs, os_, fo, oo):
        acc_s = torch.clamp(self.ft(fs, os_) + self.b1, 0.0, 1.0)
        acc_o = torch.clamp(self.ft(fo, oo) + self.b1, 0.0, 1.0)
        h = torch.cat([acc_s, acc_o], dim=1)
        out2 = torch.clamp(self.fc2(h), 0.0, 1.0)
        return self.fc3(out2)  # ~ cp / EVAL_SCALE (small range; scaled to centipawns at inference)


def quantize(model: NNUE) -> dict[str, np.ndarray]:
    w1 = model.ft.weight.detach().cpu().numpy()
    b1 = model.b1.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy().T          # (2L1, 32)
    b2 = model.fc2.bias.detach().cpu().numpy()
    w3 = model.fc3.weight.detach().cpu().numpy().reshape(-1)  # (32,)
    b3 = float(model.fc3.bias.detach().cpu().numpy()[0])
    return {
        "W1": np.round(w1 * QA).astype(np.int16),
        "b1": np.round(b1 * QA).astype(np.int32),
        "W2": np.round(w2 * QW).astype(np.int16),
        "b2": np.round(b2 * QA * QW).astype(np.int32),
        "W3": np.round(w3 * QW).astype(np.int16),
        "b3": np.array(round(b3 * QA * QW), dtype=np.int32),
        "meta": np.array([NFEAT, L1, QA, QW, OUT_DIV, EVAL_SCALE], dtype=np.int32),
    }


def int_forward(q, stm_feats, opp_feats) -> int:
    """Reference numpy int forward — must match the numba runtime exactly."""
    acc_s = q["b1"] + q["W1"][stm_feats].sum(axis=0)
    acc_o = q["b1"] + q["W1"][opp_feats].sum(axis=0)
    h = np.clip(np.concatenate([acc_s, acc_o]), 0, 127)
    out2 = np.clip((h @ q["W2"].astype(np.int64) + q["b2"]) >> 6, 0, 127)
    out_pre = int(out2 @ q["W3"].astype(np.int64) + q["b3"])
    return (out_pre * EVAL_SCALE) // OUT_DIV  # -> centipawns, side-to-move relative


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("weights/nnue.npz"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    ds = Positions(args.data, args.limit)
    print(f"positions: {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    model = NNUE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total = 0.0
        for fs, os_, fo, oo, cp in dl:
            fs, os_, fo, oo, cp = (t.to(dev) for t in (fs, os_, fo, oo, cp))
            pred = model(fs, os_, fo, oo)  # pred ~ cp / EVAL_SCALE
            loss = ((torch.sigmoid(pred) - torch.sigmoid(cp / EVAL_SCALE)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(dl):.5f}")

    q = quantize(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **q)
    print(f"saved {args.out}  (W1 {q['W1'].nbytes/1e6:.1f} MB)")

    # self-check: quantized int forward vs float net, on a validation slice
    model.eval()
    errs, floats, ints = [], [], []
    with torch.no_grad():
        for i in range(min(400, len(ds))):
            s, o, _cp = ds[i]
            fs = torch.from_numpy(s).to(dev)
            fo = torch.from_numpy(o).to(dev)
            off = torch.zeros(1, dtype=torch.int64, device=dev)
            yf = EVAL_SCALE * float(model(fs, off, fo, off)[0, 0])  # -> centipawns
            yi = int_forward(q, s, o)
            errs.append(abs(yf - yi))
            floats.append(yf)
            ints.append(yi)
    r = np.corrcoef(floats, ints)[0, 1]
    print(f"quant check: mean|float-int| = {np.mean(errs):.1f} cp, corr = {r:.4f}")


if __name__ == "__main__":
    main()
