import jax.numpy as jnp

from pgx.shogi import *
from pgx.shogi import _init, _step, _step_move, _step_drop, _flip, _apply_effects, _legal_actions, _rotate, _to_direction


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
    s = _init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_step_move():
    s = _init()

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
    s = _init()
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
    s = _init()
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
    s = _init()
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)  # 77歩
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_001.svg")
    # 33角は成れるが、44では成れない
    legal_moves, promotion, _ = _legal_actions(s)
    assert legal_moves[xy2i(8, 8), xy2i(3, 3)]  # 33角
    assert legal_moves[xy2i(8, 8), xy2i(4, 4)]  # 44角
    assert promotion[xy2i(8, 8), xy2i(3, 3)] == 1  # 33角
    assert promotion[xy2i(8, 8), xy2i(4, 4)] == 0  # 44角

    # 11への歩は成らないと行けない
    s = s.replace(hand=s.hand.at[0].set(1))  # type: ignore
    a = Action.make_drop(piece=PAWN, to=xy2i(1, 2))
    s = _step_drop(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_002.svg")
    legal_moves, promotion, _ = _legal_actions(s)
    assert legal_moves[xy2i(1, 2), xy2i(1, 1)]  # 33角
    assert promotion[xy2i(1, 2), xy2i(1, 1)] == 2  # 33角

    # Suicide action

    # King cannot move into opponent pieces' effect
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(6, 8)].set(KING)
        .at[xy2i(5, 9)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_003.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(6, 8), xy2i(5, 8)]

    # Gold is pinned
    s = _init()
    s = s.replace(piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE).at[xy2i(5, 7)].set(GOLD))
    visualize(s, "tests/assets/shogi/legal_moves_004.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(5, 7), xy2i(4, 6)]

    # Gold is not pinned
    s = _init()
    s = s.replace(
        piece_board=s.piece_board
        .at[:].set(EMPTY)
        .at[xy2i(9, 9)].set(KING)
        .at[xy2i(9, 1)].set(OPP_LANCE)
        .at[xy2i(9, 8)].set(GOLD)
    )
    visualize(s, "tests/assets/shogi/legal_moves_006.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(9, 8), xy2i(8, 8)]  # pinned
    s = s.replace(
        piece_board=s.piece_board
        .at[xy2i(9, 5)].set(PAWN)
    )
    visualize(s, "tests/assets/shogi/legal_moves_007.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert legal_moves[xy2i(9, 8), xy2i(8, 8)]  # not pinned

    # Leave king check

    # King should escape from Lance
    s = _init()
    s = s.replace(
        piece_board=s.piece_board
        .at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_008.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert legal_moves[xy2i(5, 9), xy2i(4, 8)]  # 王が逃げるのはOK
    assert legal_moves[xy2i(5, 9), xy2i(6, 8)]  # 王が逃げるのはOK
    assert not legal_moves[xy2i(5, 9), xy2i(5, 8)]  # 自殺手はNG
    assert not legal_moves[xy2i(2, 7), xy2i(2, 6)]  # 王を放置するのはNG

    # Checking piece should be captured
    s = _init()
    s = s.replace(
        piece_board=s.piece_board
        .at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(1, 1)].set(OPP_LANCE)
        .at[xy2i(6, 1)].set(ROOK)
    )
    visualize(s, "tests/assets/shogi/legal_moves_009.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(6, 1), xy2i(2, 1)]  # 飛車が香を取る以外の動きは王手放置でNG
    assert legal_moves[xy2i(6, 1), xy2i(1, 1)]      # 飛車が王手をかけている香を取るのはOK

    # 合駒
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(2, 9)].set(GOLD)
        .at[xy2i(5, 5)].set(OPP_BISHOP)
    )
    visualize(s, "tests/assets/shogi/legal_moves_010.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(2, 9), xy2i(3, 9)]  # 王手のままなのでNG
    assert legal_moves[xy2i(2, 9), xy2i(2, 8)]  # 角の利きを遮るのでOK

    # 両王手
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(5, 9)].set(BISHOP)
        .at[xy2i(9, 1)].set(OPP_BISHOP)
        .at[xy2i(1, 1)].set(OPP_ROOK)
    )
    visualize(s, "tests/assets/shogi/legal_moves_011.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert legal_moves[xy2i(1, 9), xy2i(2, 9)]  # 王が避けるのはOK
    assert not legal_moves[xy2i(1, 9), xy2i(1, 8)]  # 王が避けても相手駒が効いているところはNG
    assert not legal_moves[xy2i(5, 9), xy2i(3, 7)]  # 角の利きを遮るが、両王手なのでNG
    assert not legal_moves[xy2i(5, 9), xy2i(1, 5)]  # 飛の利きを遮るが、両王手なのでNG


def test_legal_drops():
    # 打ち歩詰
    s = _init()
    s = s.replace(hand=s.hand.at[0, PAWN].add(1),
                  piece_board=s.piece_board.at[xy2i(5, 7)].set(EMPTY).at[xy2i(8, 2)].set(EMPTY))
    visualize(s, "tests/assets/shogi/legal_drops_001.svg")

    # 避けられるし金でも取れる
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 片側に避けられるので打ち歩詰でない
    s = s.replace(piece_board=s.piece_board.at[xy2i(4, 1)].set(OPP_PAWN))  # 金を歩に変える
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(EMPTY))  # 金を除く
    s = s.replace(piece_board=s.piece_board.at[xy2i(5, 3)].set(GOLD))  # (5, 3)に金を置く
    visualize(s, "tests/assets/shogi/legal_drops_002.svg")
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 両側に避けられないので打ち歩詰
    s = s.replace(piece_board=s.piece_board.at[xy2i(4, 1)].set(OPP_PAWN))  # 両側に歩を置く
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(OPP_PAWN))
    visualize(s, "tests/assets/shogi/legal_drops_003.svg")
    _, _, legal_drops = _legal_actions(s)
    assert not legal_drops[PAWN, xy2i(5, 2)]

    # 金で取れるので打ち歩詰でない
    s = s.replace(piece_board=s.piece_board.at[xy2i(6, 1)].set(OPP_GOLD))
    visualize(s, "tests/assets/shogi/legal_drops_004.svg")
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 合駒
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(1, 5)].set(OPP_LANCE),
        hand=s.hand.at[0, GOLD].set(1)
    )
    visualize(s, "tests/assets/shogi/legal_drops_005.svg")
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[GOLD, xy2i(1, 6)]  # 合駒はOK
    assert legal_drops[GOLD, xy2i(1, 7)]  # 合駒はOK
    assert legal_drops[GOLD, xy2i(1, 8)]  # 合駒はOK
    assert not legal_drops[GOLD, xy2i(2, 6)]  # 合駒になってないのはNG

    # 両王手
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(9, 1)].set(OPP_BISHOP)
        .at[xy2i(1, 1)].set(OPP_ROOK),
        hand=s.hand.at[0].set(1)
    )
    visualize(s, "tests/assets/shogi/legal_drops_006.svg")
    _, _, legal_drops = _legal_actions(s)
    assert not legal_drops[GOLD, xy2i(1, 5)]  # 角の利きを遮るが、両王手なのでNG
    assert not legal_drops[GOLD, xy2i(5, 5)]  # 飛の利きを遮るが、両王手なのでNG


def test_dlshogi_action():
    # from dlshogi action to Action
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(5, 9)].set(LANCE)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_001.svg")
    dir_ = 0  # UP
    to = xy2i(5, 5)
    dlshogi_action = jnp.int8(dir_ * 81 + to)
    action: Action = Action.from_dlshogi_action(s, dlshogi_action)
    assert not action.is_drop
    assert action.from_ == xy2i(5, 9)
    assert action.to == xy2i(5, 5)
    assert not action.is_promotion

    # check int, int32
    dlshogi_action = jnp.int8(dir_ * 81 + to)  # int
    action: Action = Action.from_dlshogi_action(s, dlshogi_action)
    assert action.from_ == xy2i(5, 9)
    dlshogi_action = dir_ * 81 + to  # int32
    action: Action = Action.from_dlshogi_action(s, dlshogi_action)
    assert action.from_ == xy2i(5, 9)

    # 歩で香車の利きが隠れている場合
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(5, 9)].set(LANCE)
        .at[xy2i(5, 6)].set(PAWN)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_002.svg")
    dir_ = 0  # UP
    to = xy2i(5, 5)
    dlshogi_action = jnp.int32(dir_ * 81 + to)
    action: Action = Action.from_dlshogi_action(s, dlshogi_action)
    assert not action.is_drop
    assert action.from_ != xy2i(5, 9)  # 香ではない
    assert action.from_ == xy2i(5, 6)  # (5, 5)歩から
    assert action.to == xy2i(5, 5)
    assert not action.is_promotion

    # from legal moves to legal action mask
    s = _init()
    s = s.replace(
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(5, 9)].set(LANCE)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_003.svg")
    legal_actions = _legal_actions(s)
    legal_action_mask = _to_direction(legal_actions)
    dir_ = 0  # UP
    assert legal_action_mask.shape == (27 * 81,)
    assert legal_action_mask.sum() != 0
    assert legal_action_mask[dir_ * 81 + xy2i(5, 5)]
    assert legal_action_mask[dir_ * 81 + xy2i(5, 2)]
    assert not legal_action_mask[dir_ * 81 + xy2i(5, 1)]  # have to promote
    assert legal_action_mask[(dir_ + 10) * 81 + xy2i(5, 3)]  # can promote
    assert not legal_action_mask[(dir_ + 10) * 81 + xy2i(5, 4)]  # cannot promote

    # drop
    s = _init()
    s = s.replace(
        piece_board=s.piece_board
        .at[xy2i(1, 7)].set(EMPTY),
        hand=s.hand.at[0, PAWN].set(1)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_004.svg")
    legal_actions = _legal_actions(s)
    legal_action_mask = _to_direction(legal_actions)
    assert legal_action_mask[20 * 81 + xy2i(1, 5)]
    assert not legal_action_mask[20 * 81 + xy2i(2, 5)]


def test_step():
    # init
    s = _init()
    visualize(s, "tests/assets/shogi/step_001.svg")
    s = step(s, 3 * 81 + xy2i(3, 8))
    visualize(s, "tests/assets/shogi/step_002.svg")
    s = step(s, 3 * 81 + xy2i(3, 8))
    visualize(s, "tests/assets/shogi/step_003.svg")
    assert not s.legal_action_mask[3 * 81 + xy2i(3, 8)]
