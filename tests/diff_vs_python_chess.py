# Differential test: pgx chess legal move generation vs python-chess (ground truth).
# Standalone dev tool, intentionally not named test_* (pgx's pytest suite is heavy).
#
# Usage (keep it to ONE process, the platform allocator returns memory to the OS):
#   XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
#   python tests/diff_vs_python_chess.py
import random

import jax
import jax.numpy as jnp
import numpy as np

import chess as pychess

from pgx._src.games import chess as C

FROM_PLANE = np.asarray(C.FROM_PLANE)
TO_PLANE = np.asarray(C.TO_PLANE)
UNDER = [pychess.ROOK, pychess.BISHOP, pychess.KNIGHT]

game = C.Game()
jit_mask = jax.jit(C._legal_action_mask)
jit_step = jax.jit(game.step)
jit_terminal = jax.jit(game.is_terminal)


def pgx_sq(sq_pychess: int, black: bool) -> int:
    rank, file = sq_pychess // 8, sq_pychess % 8
    if black:
        rank = 7 - rank
    return file * 8 + rank


def pychess_sq(sq_pgx: int, black: bool) -> int:
    file, rank = sq_pgx // 8, sq_pgx % 8
    if black:
        rank = 7 - rank
    return rank * 8 + file


def decode(mask, black: bool, board: pychess.Board) -> set:
    moves = set()
    for label in np.nonzero(np.asarray(mask))[0]:
        f, plane = label // 73, label % 73
        t = FROM_PLANE[f, plane]
        fr, to = pychess_sq(int(f), black), pychess_sq(int(t), black)
        promo = None
        if plane < 9:
            promo = UNDER[plane // 3]
        elif board.piece_type_at(fr) == pychess.PAWN and pychess.square_rank(to) in (0, 7):
            promo = pychess.QUEEN
        moves.add(pychess.Move(fr, to, promotion=promo))
    return moves


def encode(mv: pychess.Move, black: bool) -> int:
    f, t = pgx_sq(mv.from_square, black), pgx_sq(mv.to_square, black)
    if mv.promotion in (None, pychess.QUEEN):
        return f * 73 + int(TO_PLANE[f, t])
    direc = {1: 0, 9: 1, -7: 2}[t - f]  # up, right, left
    return f * 73 + {pychess.ROOK: 0, pychess.BISHOP: 1, pychess.KNIGHT: 2}[mv.promotion] * 3 + direc


def state_from_board(board: pychess.Board) -> C.GameState:
    """Build a GameState from a python-chess board (current player always positive/up)."""
    black = board.turn == pychess.BLACK
    arr = np.zeros(64, dtype=np.int32)
    for sq, piece in board.piece_map().items():
        sign = 1 if (piece.color == board.turn) else -1
        arr[pgx_sq(sq, black)] = sign * piece.piece_type
    my, opp = board.turn, not board.turn
    castling = jnp.bool_([
        [board.has_queenside_castling_rights(my), board.has_kingside_castling_rights(my)],
        [board.has_queenside_castling_rights(opp), board.has_kingside_castling_rights(opp)],
    ])
    ep = jnp.int32(-1 if board.ep_square is None else pgx_sq(board.ep_square, black))
    x = C.GameState(
        board=jnp.asarray(arr),
        color=jnp.int32(1 if black else 0),
        castling_rights=castling,
        en_passant=ep,
        hash_history=jnp.zeros_like(C.GameState().hash_history),
        board_history=jnp.zeros_like(C.GameState().board_history),
    )
    return x._replace(legal_action_mask=jit_mask(x))


def compare(board: pychess.Board, x: C.GameState, ctx: str) -> bool:
    black = board.turn == pychess.BLACK
    got = decode(x.legal_action_mask, black, board)
    ref = set(board.legal_moves)
    if got != ref:
        print(f"MISMATCH {ctx}\n  fen={board.fen()}\n  pgx-only={got - ref}\n  ref-only={ref - got}")
        return False
    return True


FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # startpos
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # perft pos3 (ep + pins)
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  # perft pos4
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",  # pos4 mirrored
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  # perft pos5
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",  # perft pos6
    "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",  # 218 legal moves (cap regression)
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",  # ep illegal: horizontal x-ray after both pawns vanish
    "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",  # ep capture of the checking double-pushed pawn
    "8/8/4k3/8/2pP4/8/B7/7K b - d3 0 1",  # ep with diagonal pin geometry
    "3k3r/8/8/8/3n4/8/8/3R2K1 b - - 0 1",  # pinned knight: no moves at all
    "r3k2r/8/5q2/8/8/8/8/R3K2R w KQkq - 0 1",  # castling vs attacked squares
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # symmetric castling, black to move
    "8/P6k/8/8/8/8/7K/8 w - - 0 1",  # promotion (queen + under)
    "8/8/8/8/8/2k5/1q6/K7 w - - 0 1",  # nearly stalemated king
    "4k3/8/8/8/8/8/1q6/R3K2N b - - 0 1",  # contact + discovered check potential
]


def run_fen_suite() -> int:
    bad = 0
    for fen in FENS:
        board = pychess.Board(fen)
        assert board.is_valid(), f"invalid test fen: {fen}"
        x = state_from_board(board)
        if not compare(board, x, "depth1"):
            bad += 1
            continue
        for mv in board.legal_moves:  # depth 2: every child
            child = board.copy()
            child.push(mv)
            x2 = jit_step(x, jnp.int32(encode(mv, board.turn == pychess.BLACK)))
            if not compare(child, x2, f"depth2 after {mv.uci()}"):
                bad += 1
    print(f"FEN suite: {len(FENS)} positions, depth-2 expansion -> {bad} mismatches")
    return bad


def run_random_games(n_games: int = 20, max_plies: int = 160) -> int:
    random.seed(42)
    bad = 0
    for g in range(n_games):
        s = game.init()
        board = pychess.Board()
        for _ in range(max_plies):
            if not compare(board, s, f"random game {g}"):
                bad += 1
                break
            if bool(jit_terminal(s)) or board.is_game_over(claim_draw=True):
                break
            mv = random.choice(sorted(board.legal_moves, key=str))
            s = jit_step(s, jnp.int32(encode(mv, board.turn == pychess.BLACK)))
            board.push(mv)
    print(f"random games: {n_games} -> {bad} mismatches")
    return bad


if __name__ == "__main__":
    failures = run_fen_suite() + run_random_games()
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    raise SystemExit(0 if failures == 0 else 1)
