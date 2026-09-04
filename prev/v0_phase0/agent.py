"""AI Chessathon submission.

A classical chess engine: iterative-deepening negamax with alpha-beta (PVS), a
transposition table, quiescence search, and a tapered piece-square-table evaluation.
Move generation and legality come from python-chess, so the agent can never emit an
illegal move. Everything the platform needs ships in this one file; no weights, no
network, no third-party engine.

Contract: get_move(fen, time_left_ms) -> UCI string. Your colour is the side to move.
The process is reused between your own moves within a game, so the transposition table
and history heuristic survive from one move to the next. It is discarded between games.
"""

import time
from collections.abc import Hashable

import chess

# --- Scores -----------------------------------------------------------------------

INF = 32_000
MATE_SCORE = 30_000          # checkmate, adjusted by distance so shorter mates score higher
MATE_THRESHOLD = 29_000      # scores beyond this magnitude are "a mate is involved"
MAX_PLY = 128
MAX_DEPTH = 64
TT_LIMIT = 3_000_000         # cap the table so a long game cannot exhaust 2 GB

# --- Evaluation tables ------------------------------------------------------------
# Material and piece-square tables are the well-known public-domain "PeSTO" set, a
# hand-crafted tapered evaluation (midgame + endgame). Indexed by piece_type 1..6
# (pawn..king). Tables below are written as a board diagram: first row rank 8, last
# row rank 1, files a..h left to right.

MG_PIECE = [0, 82, 337, 365, 477, 1025, 0]
EG_PIECE = [0, 94, 281, 297, 512, 936, 0]
PHASE_INC = [0, 0, 1, 1, 2, 4, 0]
PHASE_MAX = 24

# fmt: off
MG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
MG_KNIGHT = [
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
]
EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
EG_BISHOP = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]
MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
EG_ROOK = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]
MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]
# fmt: on

_ZERO = [0] * 64  # index 0 is unused; piece_type runs 1..6 (pawn..king)
_MG_PST = [_ZERO, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING]
_EG_PST = [_ZERO, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING]

# Precompute per-colour lookup: value + piece-square bonus, indexed [piece_type][square].
# A white piece on `square` reads the diagram at square ^ 56 (vertical flip); a black piece
# reads it directly, which mirrors the board so its own back rank matches white's.
MG_W: list[list[int]] = [[0] * 64 for _ in range(7)]
MG_B: list[list[int]] = [[0] * 64 for _ in range(7)]
EG_W: list[list[int]] = [[0] * 64 for _ in range(7)]
EG_B: list[list[int]] = [[0] * 64 for _ in range(7)]
for _pt in range(1, 7):
    for _sq in range(64):
        MG_W[_pt][_sq] = MG_PIECE[_pt] + _MG_PST[_pt][_sq ^ 56]
        EG_W[_pt][_sq] = EG_PIECE[_pt] + _EG_PST[_pt][_sq ^ 56]
        MG_B[_pt][_sq] = MG_PIECE[_pt] + _MG_PST[_pt][_sq]
        EG_B[_pt][_sq] = EG_PIECE[_pt] + _EG_PST[_pt][_sq]

BISHOP_PAIR = 30
TEMPO = 12

# Move-ordering score bands, kept far apart so categories never overlap.
TT_MOVE_BONUS = 10_000_000
CAPTURE_BONUS = 1_000_000
PROMO_BONUS = 900_000
KILLER_BONUS = 800_000

# MVV-LVA victim values by piece_type.
_VICTIM = [0, 100, 320, 330, 500, 900, 0]

# Transposition-table bound flags.
FLAG_EXACT = 0
FLAG_LOWER = 1  # score is a lower bound (fail-high / beta cutoff)
FLAG_UPPER = 2  # score is an upper bound (fail-low, never raised alpha)


class TimeUp(Exception):
    """Raised deep in the search when the move budget is spent, to unwind cleanly."""


class Searcher:
    def __init__(self) -> None:
        # transposition table: key -> (depth, score, flag, best_move).
        # Keys are board._transposition_key() values, typed Hashable by python-chess.
        self.tt: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}
        # history heuristic: (from_square, to_square) -> score, kept across the game
        self.history: dict[tuple[int, int], int] = {}
        # killer moves per ply, reset each search
        self.killers: list[list[chess.Move | None]] = []
        # keys of positions that actually occurred in the game (for repetition draws)
        self.seen: dict[Hashable, int] = {}
        # keys currently on the search stack, for repetition detection within a line
        self.path: dict[Hashable, int] = {}
        self.nodes = 0
        self.start = 0.0
        self.hard_ms = 0.0

    # -- evaluation ----------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> int:
        """Static evaluation from the side-to-move's perspective, in centipawns."""
        mg = 0
        eg = 0
        phase = 0
        white = board.occupied_co[chess.WHITE]
        for square, piece in board.piece_map().items():
            pt = piece.piece_type
            phase += PHASE_INC[pt]
            if (1 << square) & white:
                mg += MG_W[pt][square]
                eg += EG_W[pt][square]
            else:
                mg -= MG_B[pt][square]
                eg -= EG_B[pt][square]

        # bishop pair
        bishops = board.bishops
        if (bishops & white).bit_count() >= 2:
            mg += BISHOP_PAIR
            eg += BISHOP_PAIR
        if (bishops & board.occupied_co[chess.BLACK]).bit_count() >= 2:
            mg -= BISHOP_PAIR
            eg -= BISHOP_PAIR

        if phase > PHASE_MAX:
            phase = PHASE_MAX
        score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX  # white-relative
        score = score if board.turn == chess.WHITE else -score
        return score + TEMPO

    # -- move ordering -------------------------------------------------------------

    def mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        if victim is None:  # en passant, or a non-capture in check evasions
            victim = chess.PAWN
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return _VICTIM[victim] * 16 - attacker

    def order(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        killers = self.killers[ply] if ply < MAX_PLY else (None, None)
        history = self.history

        def score(move: chess.Move) -> int:
            if move == tt_move:
                return TT_MOVE_BONUS
            if board.is_capture(move):
                return CAPTURE_BONUS + self.mvv_lva(board, move)
            if move.promotion:
                return PROMO_BONUS + move.promotion
            if move in killers:
                return KILLER_BONUS
            return history.get((move.from_square, move.to_square), 0)

        moves.sort(key=score, reverse=True)
        return moves

    # -- transposition table -------------------------------------------------------

    def tt_store(
        self, key: Hashable, depth: int, score: int, flag: int, move: chess.Move | None, ply: int
    ) -> None:
        if len(self.tt) >= TT_LIMIT:
            self.tt.clear()
        if score >= MATE_THRESHOLD:
            score += ply
        elif score <= -MATE_THRESHOLD:
            score -= ply
        self.tt[key] = (depth, score, flag, move)

    # -- quiescence ----------------------------------------------------------------

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 1023 == 0 and (time.monotonic() - self.start) * 1000.0 >= self.hard_ms:
            raise TimeUp
        if ply >= MAX_PLY:
            return self.evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
        else:
            stand = self.evaluate(board)
            if stand >= beta:
                return beta
            if stand > alpha:
                alpha = stand
            moves = list(board.generate_legal_captures())

        moves.sort(key=lambda m: self.mvv_lva(board, m), reverse=True)
        for move in moves:
            board.push(move)
            score = -self.qsearch(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # -- main search ---------------------------------------------------------------

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 1023 == 0 and (time.monotonic() - self.start) * 1000.0 >= self.hard_ms:
            raise TimeUp

        key = board._transposition_key()

        # Draw detection: repetition (within this line or the real game) and fifty-move.
        if self.seen.get(key, 0) + self.path.get(key, 0) >= 1:
            return 0
        if board.halfmove_clock >= 100:
            return 0

        alpha_orig = alpha

        entry = self.tt.get(key)
        tt_move: chess.Move | None = None
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_score >= MATE_THRESHOLD:
                    e_score -= ply
                elif e_score <= -MATE_THRESHOLD:
                    e_score += ply
                if e_flag == FLAG_EXACT:
                    return e_score
                if e_flag == FLAG_LOWER and e_score > alpha:
                    alpha = e_score
                elif e_flag == FLAG_UPPER and e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score

        in_check = board.is_check()
        if in_check:  # check extension: never evaluate statically while in check
            depth += 1

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply)

        # Null-move pruning: give the opponent a free move; if we are still above beta,
        # this node is almost certainly a cutoff. Skip in check, in likely-zugzwang
        # endgames (no non-pawn material), and near mate scores.
        if (
            depth >= 3
            and not in_check
            and abs(beta) < MATE_THRESHOLD
            and board.occupied_co[board.turn]
            & (board.knights | board.bishops | board.rooks | board.queens)
        ):
            reduction = 2 + depth // 6
            board.push(chess.Move.null())
            self.path[key] = self.path.get(key, 0) + 1
            try:
                score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
            finally:
                self._path_pop(key)
                board.pop()
            if score >= beta:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        self.order(board, moves, tt_move, ply)

        best = -INF
        best_move: chess.Move | None = None
        self.path[key] = self.path.get(key, 0) + 1
        try:
            # gives_check() is last in the guard so it runs only when the cheap
            # late-move-reduction conditions already hold (Python short-circuits).
            for move_count, move in enumerate(moves):
                is_capture = board.is_capture(move)
                is_quiet = not is_capture and move.promotion is None

                reduction = 0
                if (
                    is_quiet
                    and depth >= 3
                    and move_count >= 4
                    and not in_check
                    and not board.gives_check(move)
                ):
                    reduction = 2 if (move_count >= 8 and depth >= 5) else 1

                board.push(move)
                if move_count == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and reduction:
                        score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                board.pop()

                if score > best:
                    best = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if is_quiet:
                        self._on_quiet_cutoff(move, depth, ply)
                    break
        finally:
            self._path_pop(key)

        flag = FLAG_UPPER if best <= alpha_orig else (FLAG_LOWER if best >= beta else FLAG_EXACT)
        self.tt_store(key, depth, best, flag, best_move, ply)
        return best

    def _path_pop(self, key: Hashable) -> None:
        count = self.path.get(key, 0) - 1
        if count > 0:
            self.path[key] = count
        else:
            self.path.pop(key, None)

    def _on_quiet_cutoff(self, move: chess.Move, depth: int, ply: int) -> None:
        if ply < MAX_PLY:
            slot = self.killers[ply]
            if slot[0] != move:
                slot[1] = slot[0]
                slot[0] = move
        k = (move.from_square, move.to_square)
        self.history[k] = self.history.get(k, 0) + depth * depth

    # -- root / iterative deepening ------------------------------------------------

    def search_root(self, board: chess.Board, depth: int) -> tuple[int, chess.Move]:
        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = entry[3] if entry is not None else None
        moves = self.order(board, list(board.legal_moves), tt_move, 0)

        alpha, beta = -INF, INF
        best = -INF
        best_move = moves[0]
        self.path[key] = self.path.get(key, 0) + 1
        try:
            first = True
            for move in moves:
                board.push(move)
                if first:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if score > alpha:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                board.pop()
                first = False
                if score > best:
                    best = score
                    best_move = move
                if score > alpha:
                    alpha = score
        finally:
            self._path_pop(key)

        self.tt_store(key, depth, best, FLAG_EXACT, best_move, 0)
        return best, best_move

    def budget(self, time_left_ms: int) -> tuple[float, float]:
        """Return (soft_ms, hard_ms): stop starting new depths past soft, abort past hard."""
        t = float(time_left_ms)
        if t <= 0:
            return 0.0, 0.0
        soft = t / 30.0
        hard = t / 10.0
        # never risk the clock: leave a margin for our own return overhead
        hard = min(hard, t - 300.0) if t > 600.0 else t * 0.4
        if hard < soft:
            hard = soft
        return soft, hard

    def search(self, board: chess.Board, time_left_ms: int) -> str:
        # record the real position for repetition-draw detection across moves
        root_key = board._transposition_key()
        self.seen[root_key] = self.seen.get(root_key, 0) + 1

        legal = list(board.legal_moves)
        if not legal:
            return "0000"
        if len(legal) == 1:
            return legal[0].uci()

        self.start = time.monotonic()
        soft_ms, self.hard_ms = self.budget(time_left_ms)
        self.nodes = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]

        best_move = self.order(board, list(legal), None, 0)[0]
        for depth in range(1, MAX_DEPTH + 1):
            self.path = {}
            try:
                score, move = self.search_root(board, depth)
            except TimeUp:
                break
            best_move = move
            if abs(score) >= MATE_THRESHOLD:  # forced mate found, no need to search deeper
                break
            if (time.monotonic() - self.start) * 1000.0 >= soft_ms:
                break
        return best_move.uci()


SEARCHER = Searcher()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    board = chess.Board(fen)
    try:
        return SEARCHER.search(board, time_left_ms)
    except Exception:
        # A search bug must never forfeit the game: fall back to any legal move.
        for move in board.legal_moves:
            return move.uci()
        return "0000"
