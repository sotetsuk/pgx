import jax.numpy as jnp

from pgx.shogi import *
from pgx.shogi import _step, _step_move, _step_drop, _flip, _apply_effects, _legal_moves, _legal_drops, _rotate


# check visualization results by image preview plugins
def visualize(state, fname="tests/assets/shogi/xxx.svg"):
    from pgx.visualizer import Visualizer
    v = Visualizer(color_mode="dark")
    v.save_svg(state, fname)


def xy2i(x, y):
    """
    >>> xy2i(2, 6)  # 26歩
    14
    """
    i = (x - 1) * 9 + (y - 1)
    return i


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_step_move():
    s = init()

    # 26歩
    piece, from_, to = PAWN, xy2i(2, 7), xy2i(2, 6)
    assert s.piece_board[from_] == PAWN
    assert s.piece_board[to] == EMPTY
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_001.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 76歩
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)
    assert s.piece_board[from_] == PAWN
    assert s.piece_board[to] == EMPTY
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_002.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 33角成
    piece, from_, to = BISHOP , xy2i(8, 8), xy2i(3, 3)
    assert s.piece_board[from_] == BISHOP
    assert s.piece_board[to] == OPP_PAWN
    assert s.hand[0, PAWN] == 0
    assert (s.hand[0, PAWN:] == 0).all()
    a = Action.make_move(piece=piece, from_=from_, to=to, is_promotion=True)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_003.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == HORSE
    assert s.hand[0, PAWN] == 1
    assert (s.hand[0, :PAWN] == 0).all()


def test_step_drop():
    s = init()
    s = s.replace(hand=s.hand.at[:, :].set(1))  # type: ignore
    # 52飛車打ち
    piece, to = ROOK, xy2i(5, 2)
    a = Action.make_drop(piece, to)
    assert s.piece_board[to] == EMPTY
    assert s.hand[0, ROOK] == 1
    s = _step_drop(s, a)
    visualize(s, "tests/assets/shogi/step_drop_001.svg")
    assert s.piece_board[to] == ROOK
    assert s.hand[0, ROOK] == 0


def test_flip():
    s = init()
     # 26歩
    piece, from_, to = PAWN, xy2i(2, 7), xy2i(2, 6)
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    s = s.replace(hand=s.hand.at[0, :].set(1))  # type: ignore
    visualize(s, "tests/assets/shogi/flip_001.svg")
    assert s.piece_board[xy2i(2, 6)] == PAWN
    assert s.piece_board[xy2i(8, 4)] == EMPTY
    assert (s.hand[0] == 1).all()
    assert (s.hand[1] == 0).all()
    s = _flip(s)
    visualize(s, "tests/assets/shogi/flip_002.svg")
    assert s.piece_board[xy2i(2, 6)] == EMPTY
    assert s.piece_board[xy2i(8, 4)] == OPP_PAWN
    assert (s.hand[0] == 0).all()
    assert (s.hand[1] == 1).all()
    s = _flip(s)
    visualize(s, "tests/assets/shogi/flip_003.svg")
    assert s.piece_board[xy2i(2, 6)] == PAWN
    assert s.piece_board[xy2i(8, 4)] == EMPTY
    assert (s.hand[0] == 1).all()
    assert (s.hand[1] == 0).all()


def test_legal_moves():
    # Promotion
    s = init()
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)  # 77歩
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_001.svg")
    # 33角は成れるが、44では成れない
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert legal_moves[xy2i(8, 8), xy2i(3, 3)]  # 33角
    assert legal_moves[xy2i(8, 8), xy2i(4, 4)]  # 44角
    assert promotion[xy2i(8, 8), xy2i(3, 3)] == 1  # 33角
    assert promotion[xy2i(8, 8), xy2i(4, 4)] == 0  # 44角

    # 11への歩は成らないと行けない
    s = s.replace(hand=s.hand.at[0].set(1))  # type: ignore
    a = Action.make_drop(piece=PAWN, to=xy2i(1, 2))
    s = _step_drop(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_002.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert legal_moves[xy2i(1, 2), xy2i(1, 1)]  # 33角
    assert promotion[xy2i(1, 2), xy2i(1, 1)] == 2  # 33角

    # Suicide action

    # King cannot move into opponent pieces' effect
    s = init()
    s = s.replace(
        piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(6, 8)].set(KING)
        .at[xy2i(5, 9)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_003.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert not legal_moves[xy2i(6, 8), xy2i(5, 8)]

    # Gold is pinned
    s = init()
    s = s.replace(piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE).at[xy2i(5, 7)].set(GOLD))
    visualize(s, "tests/assets/shogi/legal_moves_004.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert not legal_moves[xy2i(5, 7), xy2i(4, 6)]

    # Leave king check

    # King should escape from Lance
    s = init()
    s = s.replace(
        piece_board=s.piece_board
        .at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_005.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert legal_moves[xy2i(5, 9), xy2i(4, 8)]  # 王が逃げるのはOK
    assert legal_moves[xy2i(5, 9), xy2i(6, 8)]  # 王が逃げるのはOK
    assert not legal_moves[xy2i(5, 9), xy2i(5, 8)]  # 自殺手はNG
    assert not legal_moves[xy2i(2, 7), xy2i(2, 6)]  # 王を放置するのはNG

    #
    s = init()
    s = s.replace(
        piece_board=s.piece_board
        .at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(7, 7)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_006.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert not legal_moves[xy2i(8, 8), xy2i(4, 4)]  # 角が香を取る以外の動きは王手放置でNG
    assert legal_moves[xy2i(8, 8), xy2i(5, 5)]      # 角が王手をかけている香を取るのはOK

    # 合駒
    s = init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(2, 9)].set(GOLD)
        .at[xy2i(5, 5)].set(OPP_BISHOP)
    )
    visualize(s, "tests/assets/shogi/legal_moves_006.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert not legal_moves[xy2i(2, 9), xy2i(3, 9)]  # 王手のままなのでNG
    assert legal_moves[xy2i(2, 9), xy2i(2, 8)]  # 角の利きを遮るのでOK


def test_legal_drops():
    # 打ち歩詰
    s = init()
    s = s.replace(hand=s.hand.at[0, PAWN].add(1),
                  piece_board=s.piece_board.at[xy2i(5, 7)].set(EMPTY).at[xy2i(8, 2)].set(EMPTY))
    visualize(s, "tests/assets/shogi/legal_drops_001.svg")

    # 避けられるし金でも取れる
    effects = _apply_effects(s)
    legal_drops = _legal_drops(s, effects)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 片側に避けられるので打ち歩詰でない
    s = s.replace(piece_board=s.piece_board.at[xy2i(4, 1)].set(OPP_PAWN))  # 金を歩に変える
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(EMPTY))  # 金を除く
    s = s.replace(piece_board=s.piece_board.at[xy2i(5, 3)].set(GOLD))  # (5, 3)に金を置く
    visualize(s, "tests/assets/shogi/legal_drops_002.svg")
    effects = _apply_effects(s)
    legal_drops = _legal_drops(s, effects)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 両側に避けられないので打ち歩詰
    s = s.replace(piece_board=s.piece_board.at[xy2i(4, 1)].set(OPP_PAWN))  # 両側に歩を置く
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(OPP_PAWN))
    visualize(s, "tests/assets/shogi/legal_drops_003.svg")
    effects = _apply_effects(s)
    legal_drops = _legal_drops(s, effects)
    assert not legal_drops[PAWN, xy2i(5, 2)]

    # 金で取れるので打ち歩詰でない
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(OPP_GOLD))
    visualize(s, "tests/assets/shogi/legal_drops_004.svg")
    effects = _apply_effects(s)
    legal_drops = _legal_drops(s, effects)
    assert legal_drops[PAWN, xy2i(5, 2)]