from functools import partial
import jax.numpy as jnp

from pgx.shogi import *
from pgx.shogi import _step, _step_move, _step_drop, _flip, _effects_all, _legal_actions, _rotate, _to_direction, _from_sfen, _pseudo_legal_drops, _to_sfen


env = Shogi()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def xy2i(x, y):
    """
    >>> xy2i(2, 6)  # 26歩
    14
    """
    i = (x - 1) * 9 + (y - 1)
    return i


# check visualization results by image preview plugins
def visualize(state, fname="tests/assets/shogi/xxx.svg"):
    from pgx._visualizer import Visualizer
    v = Visualizer(color_theme="dark")
    v.save_svg(state, fname)


def update_board(state, piece_board, hand=None):
    state = state.replace(piece_board=piece_board)
    if hand is not None:
        state = state.replace(hand=hand)
    return state


def test_init():
    key = jax.random.PRNGKey(0)
    s = init(key)
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_step_move():
    key = jax.random.PRNGKey(0)
    s = init(key)

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
    key = jax.random.PRNGKey(0)
    s = init(key)
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
    key = jax.random.PRNGKey(0)
    s = init(key)
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
    key = jax.random.PRNGKey(0)
    s = init(key)
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(6, 8)].set(KING)
        .at[xy2i(5, 9)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_003.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(6, 8), xy2i(5, 8)]

    # Gold is pinned
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
                     piece_board=s.piece_board
                     .at[xy2i(5, 5)].set(OPP_LANCE)
                     .at[xy2i(5, 7)].set(GOLD))
    visualize(s, "tests/assets/shogi/legal_moves_004.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(5, 7), xy2i(4, 6)]
    assert legal_moves[xy2i(5, 7), xy2i(5, 6)]

    # Gold is not pinned
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board
        .at[:].set(EMPTY)
        .at[xy2i(9, 9)].set(KING)
        .at[xy2i(9, 1)].set(OPP_LANCE)
        .at[xy2i(9, 8)].set(GOLD)
    )
    visualize(s, "tests/assets/shogi/legal_moves_006.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert not legal_moves[xy2i(9, 8), xy2i(8, 8)]  # pinned
    s = update_board(s,
        piece_board=s.piece_board
        .at[xy2i(9, 5)].set(PAWN)
    )
    visualize(s, "tests/assets/shogi/legal_moves_007.svg")
    legal_moves, _, _ = _legal_actions(s)
    assert legal_moves[xy2i(9, 8), xy2i(8, 8)]  # not pinned

    # Leave king check

    # King should escape from Lance
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
                hand=s.hand.at[0, PAWN].add(1),
                piece_board=s.piece_board.at[xy2i(5, 7)].set(EMPTY).at[xy2i(8, 2)].set(EMPTY))
    visualize(s, "tests/assets/shogi/legal_drops_001.svg")

    # 避けられるし金でも取れる
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 片側に避けられるので打ち歩詰でない
    s = update_board(s,
                 piece_board=s.piece_board
                 .at[xy2i(4, 1)].set(OPP_PAWN)  # 金を歩に変える
                 .at[xy2i(6, 1)].set(EMPTY)  # 金を除く
                 .at[xy2i(5, 3)].set(GOLD)   # (5, 3)に金を置く
                 )
    visualize(s, "tests/assets/shogi/legal_drops_002.svg")
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 両側に避けられないので打ち歩詰
    s = update_board(s,
                     piece_board=s.piece_board
                     .at[xy2i(4, 1)].set(OPP_PAWN)  # 両側に歩を置く
                     .at[xy2i(6, 1)].set(OPP_PAWN)
                     )
    visualize(s, "tests/assets/shogi/legal_drops_003.svg")
    _, _, legal_drops = _legal_actions(s)
    assert not legal_drops[PAWN, xy2i(5, 2)]

    # 金で取れるので打ち歩詰でない
    s = update_board(s,
                     piece_board=s.piece_board
                     .at[xy2i(6, 1)].set(OPP_GOLD))
    visualize(s, "tests/assets/shogi/legal_drops_004.svg")
    _, _, legal_drops = _legal_actions(s)
    assert legal_drops[PAWN, xy2i(5, 2)]

    # 合駒
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[:]
        .set(EMPTY).at[xy2i(5, 9)].set(LANCE)
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
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
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(5, 9)].set(LANCE)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_003.svg")
    legal_actions = _legal_actions(s)
    legal_action_mask = _to_direction(*legal_actions)
    dir_ = 0  # UP
    assert legal_action_mask.shape == (27 * 81,)
    assert legal_action_mask.sum() != 0
    assert legal_action_mask[dir_ * 81 + xy2i(5, 5)]
    assert legal_action_mask[dir_ * 81 + xy2i(5, 2)]
    assert not legal_action_mask[dir_ * 81 + xy2i(5, 1)]  # have to promote
    assert legal_action_mask[(dir_ + 10) * 81 + xy2i(5, 3)]  # can promote
    assert not legal_action_mask[(dir_ + 10) * 81 + xy2i(5, 4)]  # cannot promote

    # drop
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board
        .at[xy2i(1, 7)].set(EMPTY),
        hand=s.hand.at[0, PAWN].set(1)
    )
    visualize(s, "tests/assets/shogi/dlshogi_action_004.svg")
    legal_actions = _legal_actions(s)
    legal_action_mask = _to_direction(*legal_actions)
    assert legal_action_mask[20 * 81 + xy2i(1, 5)]
    assert not legal_action_mask[20 * 81 + xy2i(2, 5)]


def test_step():
    key = jax.random.PRNGKey(0)
    s = init(key)
    visualize(s, "tests/assets/shogi/step_001.svg")
    s = step(s, 3 * 81 + xy2i(3, 8))
    visualize(s, "tests/assets/shogi/step_002.svg")
    s = step(s, 3 * 81 + xy2i(3, 8))
    visualize(s, "tests/assets/shogi/step_003.svg")
    assert not s.legal_action_mask[3 * 81 + xy2i(3, 8)]


def test_legal_action_mask():
    key = jax.random.PRNGKey(0)
    s = init(key)
    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_001.svg")
    assert not s.legal_action_mask[6 * 81 + xy2i(6, 6)]  # 初期盤面では、角の利きは77の歩でとまっている
    assert s.legal_action_mask[0 * 81 + xy2i(7, 6)]  # 76歩の利き
    s = _step(s, Action.make_move(PAWN, xy2i(7, 7), xy2i(7, 6)))  # 76歩

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_002.svg")
    assert s.legal_action_mask[0 * 81 + xy2i(7, 6)]  # 後手34歩の利きがある
    s = _step(s, Action.make_move(PAWN, xy2i(2, 7), xy2i(2, 6)))  # 84歩

    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_003.svg")
    assert not s.legal_action_mask[0 * 81 + xy2i(7, 6)]  # 76歩の利きが消えている
    assert s.legal_action_mask[0 * 81 + xy2i(7, 5)]  # 75歩の利きが増えている
    assert s.legal_action_mask[2 * 81 + xy2i(7, 7)]  # 角の利きが伸びている
    assert s.legal_action_mask[2 * 81 + xy2i(6, 6)]  # 角の利きが伸びている
    assert s.legal_action_mask[2 * 81 + xy2i(3, 3)]  # 角の利きが伸びている
    assert not s.legal_action_mask[2 * 81 + xy2i(2, 2)]  # 角の利きが相手の33歩で止まる
    s = _step(s, Action.make_move(BISHOP, xy2i(8, 8), xy2i(3, 3)))  # 33角

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_004.svg")
    assert not s.legal_action_mask[0 * 81 + xy2i(7, 6)]  # 後手34歩の利きがなくなる
    s = _step(s, Action.make_move(BISHOP, xy2i(8, 8), xy2i(7, 7)))  # 同角


    key = jax.random.PRNGKey(0)
    s = init(key)
    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_005.svg")
    s = _step(s, Action.make_move(PAWN, xy2i(7, 7), xy2i(7, 6)))  # 76歩

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_006.svg")
    s = _step(s, Action.make_move(PAWN, xy2i(7, 7), xy2i(7, 6)))  # 34歩

    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_007.svg")
    assert s.legal_action_mask[2 * 81 + xy2i(6, 6)]  # 後手34歩で角の利きが2までは伸びた
    assert s.legal_action_mask[2 * 81 + xy2i(2, 2)]  # 後手34歩で角の利きが33までは伸びた
    s = _step(s, Action.make_move(PAWN, xy2i(6, 7), xy2i(6, 6)))  # 66歩

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_008.svg")
    assert s.legal_action_mask[2 * 81 + xy2i(4, 4)]  # 後手角の利きが66までは伸びている
    assert not s.legal_action_mask[2 * 81 + xy2i(3, 3)]  # 後手角の利きが77までは届かない
    s = _step(s, Action.make_move(PAWN, xy2i(2, 7), xy2i(2, 6)))  # 84歩

    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_009.svg")
    assert not s.legal_action_mask[2 * 81 + xy2i(6, 6)]  # 角の利きが66歩で止まっている
    assert not s.legal_action_mask[2 * 81 + xy2i(3, 3)]  # 角の利きが33まで止まっている


    key = jax.random.PRNGKey(0)
    s = init(key)
    s = s.replace(hand=s.hand.at[0, GOLD].set(1))
    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_010.svg")
    s = _step(s, Action.make_move(PAWN, xy2i(7, 7), xy2i(7, 6)))  # 76歩

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_011.svg")
    s = _step(s, Action.make_move(PAWN, xy2i(7, 7), xy2i(7, 6)))  # 34歩

    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_012.svg")
    assert s.legal_action_mask[2 * 81 + xy2i(2, 2)]           # 金打の前は角の利きが22までは伸びている
    assert not s.legal_action_mask[0 * 81 + xy2i(4, 3)]       # 金打の前は効きがない
    s = _step(s, Action.make_drop(GOLD, xy2i(4, 4)))  # 44金打

    # 後手
    visualize(s, "tests/assets/shogi/legal_action_mask_013.svg")
    assert s.legal_action_mask[2 * 81 + xy2i(6, 6)]    # 44で金を取るところまでは角が進める
    assert not s.legal_action_mask[2 * 81 + xy2i(5, 5)]    # 真ん中までは角はすすめない
    s = _step(s, Action.make_move(PAWN, xy2i(2, 7), xy2i(2, 6)))  # 84歩

    # 先手
    visualize(s, "tests/assets/shogi/legal_action_mask_014.svg")
    # print(_rotate(s.effects[0, xy2i(8, 8), :]))
    assert s.legal_action_mask[2 * 81 + xy2i(5, 5)]       # 55までは角が進める
    assert not s.legal_action_mask[2 * 81 + xy2i(2, 2)]   # 金打の後は角の利きが止まっている
    assert not s.legal_action_mask[2 * 81 + xy2i(4, 4)]
    assert s.legal_action_mask[0 * 81 + xy2i(4, 3)]       # 金の利きが増える


def test_buggy_samples():
    # 歩以外の持ち駒に対しての二歩判定回避
    sfen = "9/9/9/9/9/9/PPPPPPPPP/9/9 b NLP 1"
    state = _from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_001.svg")

    # 歩は二歩になるので打てない
    assert (~state.legal_action_mask[20 * 81:21 * 81]).all()
    # 香車は2列目には打てるが、1列目と7列目（歩がいる）には打てない
    assert (state.legal_action_mask[21 * 81 + 1:22 * 81:9]).all()
    assert (~state.legal_action_mask[21 * 81:22 * 81:9]).all()
    assert (~state.legal_action_mask[21 * 81 + 6:22 * 81:9]).all()
    # 桂馬は1,2列目に打てないが3列目には打てる
    assert (~state.legal_action_mask[22 * 81:23 * 81:9]).all()
    assert (~state.legal_action_mask[22 * 81 + 1:23 * 81:9]).all()
    assert (state.legal_action_mask[21 * 81 + 2:22 * 81:9]).all()

    # 成駒のpromotion判定
    sfen = "9/2+B1G1+P2/9/9/9/9/9/9/9 b - 1"
    state = _from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_002.svg")
    # promotionは生成されてたらダメ
    assert (state.legal_action_mask[10 * 81:]).sum() == 0

    # 角は成れないはず
    sfen = "l+B6l/6k2/3pg2P1/p6p1/1pP1pB2p/2p3n2/P+r1GP3P/4KS1+s1/LNG5L b RGN2sn6p 1"
    state = _from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_003.svg")
    assert ~state.legal_action_mask[13 * 81 + 72]  # = 1125, promote + left (91角成）

    # #375
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/buggy_samples_004.svg")
    assert (jnp.nonzero(s.legal_action_mask)[0] == jnp.int32([43, 52, 68, 196, 222, 295, 789, 1996, 2004, 2012])).all()

    # #602
    sfen = "9/4R4/9/9/9/9/9/9/9 b 2r2b4g3s4n4l17p 1"
    state = _from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_005.svg")
    dlshogi_action = 846
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_006.svg")
    sfen = "4+R4/9/9/9/9/9/9/9/9 w 2r2b4g3s4n4l7p 1"
    expected_state = _from_sfen(sfen)
    visualize(expected_state, "tests/assets/shogi/buggy_samples_006.svg")
    assert (state.piece_board == expected_state.piece_board).all()

    # #603
    state = _from_sfen("8k/9/9/5b3/9/3B5/9/9/K8 b 2r4g4s4n4l18p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_007.svg")
    dlshogi_action = 202
    a = Action.from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(6, 6)

    # #610
    state = _from_sfen("+PsGg1p2+P/+B1+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 b np 1")
    visualize(state, "tests/assets/shogi/buggy_samples_008.svg")
    dlshogi_action = 225
    a = Action.from_dlshogi_action(state, dlshogi_action)
    effects = _effects_all(state)
    assert effects[xy2i(9, 2), xy2i(8, 1)]
    assert a.from_ == xy2i(9, 2)
    assert a.piece == HORSE
    state = step(state, dlshogi_action)
    expected_state = _from_sfen("+P+BGg1p2+P/2+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 w Snp 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # #613
    state = _from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1g2PL/3KPR2S/1R1G1NG2 b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_009.svg")
    dlshogi_action = 42
    a = Action.from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(5, 8)
    assert a.piece == PAWN
    state = step(state, dlshogi_action)
    expected_state = _from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1P2PL/3K1R2S/1R1G1NG2 w GP4p 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # #618
    state = _from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg+s3L1/p2R2p2/P+n+B+p+ng1+s+p w P 1")
    visualize(state, "tests/assets/shogi/buggy_samples_010.svg")
    dlshogi_action = 28
    legal_moves, *_ = _legal_actions(state)
    assert legal_moves[xy2i(4, 3), xy2i(4, 2)]
    a = Action.from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(4, 3)
    state = step(state, dlshogi_action)
    expected_state = _from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg4L1/p2+s2p2/P+n+B+p+ng1+s+p b Pr 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # 629
    state = _from_sfen("1ns6/+S1p+Ng1p1l/+P2pg1nNS/4k2G1/2L2R2s/p1G2+BPR1/3Pp2+p1/1+p3B1P1/1LPK2+l1+p b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_011.svg")
    dlshogi_action = 1660  # 歩打
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_012.svg")
    assert not state.terminated  # 打ち歩詰でない


def test_observe():
    key = jax.random.PRNGKey(0)
    s: State = init(key)
    obs = observe(s, s.current_player)

    assert obs.shape == (119, 9, 9)

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[PAWN] == expected).all()  # 0

    expected = jnp.bool_([[0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.]])
    assert (obs[DRAGON + 1] == expected).all()  # 14

    expected = jnp.bool_([[0.,0.,0.,0.,0.,1.,1.,1.,0.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.]])
    assert (obs[28] == expected).all()  # 利きの数1以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,1.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[29] == expected).all()  # 利きの数2以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[30] == expected).all()  # 利きの数3以上

    expected = jnp.bool_([[0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.]])
    assert (obs[31] == expected).all()

    expected = jnp.bool_([[0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.]])
    assert (obs[45] == expected).all()

    expected = jnp.bool_([[1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [0.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [0.,1.,1.,1.,0.,0.,0.,0.,0.]])
    assert (obs[59] == expected).all()

    # 駒打ち
    sfen = "1ns4nl/1r4k2/2p1gp3/1p1pp3p/l8/2P2PP2/1PNPP3P/2G2S3/2S1KG2L b BGS3Prbnl2p 1"
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_001.svg")
    obs = observe(s, s.current_player)

    filled = [0, 1, 2, 16, 20, 24, 28, 29, 36, 40, 52, 54]
    for i in range(56):
        if i in filled:
            assert obs[62 + i].all()
        else:
            assert (~obs[62 + i]).all()

    # 王手
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"  # 先手番
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_002.svg")
    obs = observe(s, s.current_player)
    assert (~obs[-1]).all()

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"  # 後手番
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_003.svg")
    obs = observe(s, s.current_player)
    assert obs[-1].all()

    # TODO: player_id != current_player


def test_sfen():
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_001.svg")
    assert _to_sfen(s) == sfen

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = _from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_002.svg")
    assert _to_sfen(s) == sfen


def test_api():
    import pgx
    env = pgx.make("shogi")
    pgx.api_test(env, 10)
