import copy
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


# 指し手のdataclass
@struct.dataclass
class JaxAnimalShogiAction:
    # 上の3つは移動と駒打ちで共用
    # 下の3つは移動でのみ使用
    # 駒打ちかどうか
    is_drop: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # piece: 動かした(打った)駒の種類
    piece: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # final: 移動後の座標
    to: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # 移動前の座標
    from_: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # is_promote: 駒を成るかどうかの判定
    is_promote: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)


# 盤面のdataclass
@struct.dataclass
class JaxAnimalShogiState:
    # turn 先手番なら0 後手番なら1
    turn: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # board 盤面の駒。
    # 空白,先手ヒヨコ,先手キリン,先手ゾウ,先手ライオン,先手ニワトリ,後手ヒヨコ,後手キリン,後手ゾウ,後手ライオン,後手ニワトリ
    # の順で駒がどの位置にあるかをone_hotで記録
    # ヒヨコ: Pawn, キリン: Rook, ゾウ: Bishop, ライオン: King, ニワトリ: Gold　と対応
    board: jnp.ndarray = jnp.zeros((11, 12), dtype=jnp.int32)
    # hand 持ち駒。先手ヒヨコ,先手キリン,先手ゾウ,後手ヒヨコ,後手キリン,後手ゾウの6種の値を増減させる
    hand: jnp.ndarray = jnp.zeros(6, dtype=jnp.int32)
    # legal_actions_black/white: 自殺手や王手放置などの手も含めた合法手の一覧
    # move/dropによって変化させる
    legal_actions_black: jnp.ndarray = jnp.zeros(180, dtype=jnp.int32)
    legal_actions_white: jnp.ndarray = jnp.zeros(180, dtype=jnp.int32)
    # checked: ターンプレイヤーの王に王手がかかっているかどうか
    is_check: jnp.ndarray = jnp.zeros(1, dtype=jnp.int32)
    # checking_piece: ターンプレイヤーに王手をかけている駒の座標
    checking_piece: jnp.ndarray = jnp.zeros(12, dtype=jnp.int32)


# BLACK/WHITE/(NONE)_○○_MOVEは22にいるときの各駒の動き
# 端にいる場合は対応するところに0をかけていけないようにする
BLACK_PAWN_MOVE = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
WHITE_PAWN_MOVE = jnp.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
BLACK_GOLD_MOVE = jnp.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
WHITE_GOLD_MOVE = jnp.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
ROOK_MOVE = jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
BISHOP_MOVE = jnp.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]])
KING_MOVE = jnp.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])


INIT_BOARD = JaxAnimalShogiState(
    turn=jnp.array([0]),
    board=jnp.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    hand=jnp.array([0, 0, 0, 0, 0, 0]),
)


@jax.jit
def jax_init() -> JaxAnimalShogiState:
    return _init_legal_actions(copy.deepcopy(INIT_BOARD))


def jax_step(
    state: JaxAnimalShogiState, action: int
) -> Tuple[JaxAnimalShogiState, int, bool]:
    # state, 勝敗判定,終了判定を返す
    s = copy.deepcopy(state)
    legal_actions = _legal_actions(s)
    # 合法手が存在しない場合、手番側の負けで終了
    if jnp.all(legal_actions == 0):
        print("no legal actions.")
        return s, _turn_to_reward(_another_color(s)), True
    # actionが合法手でない場合、手番側の負けで終了
    _action = _dlaction_to_action(action, s)
    # actionのfromが盤外に存在すると挙動がおかしくなるのでここではじいておく
    if _action.from_ > 11 or _action.from_ < 0:
        print("an illegal action")
        return s, _turn_to_reward(_another_color(s)), True
    if legal_actions[_action_to_dlaction(_action, s.turn)] == 0:
        print("an illegal action")
        return s, _turn_to_reward(_another_color(s)), True
    # actionが合法手の場合
    # 駒打ちの場合の操作
    if _action.is_drop[0] == 1:
        s = _update_legal_drop_actions(s, _action)
        s = _drop(s, _action)
        print("drop: piece =", _action.piece[0], ", to =", _action.to[0])
    # 駒の移動の場合の操作
    else:
        s = _update_legal_move_actions(s, _action)
        s = _move(s, _action)
        print("move: piece =", _action.piece[0], ", to =", _action.to[0])
    s = JaxAnimalShogiState(
        turn=_another_color(s),
        board=s.board,
        hand=s.hand,
        legal_actions_black=s.legal_actions_black,
        legal_actions_white=s.legal_actions_white,
    )
    if _is_check(s):
        # 王手をかけている駒は直前に動かした駒
        # 王手返しの王手の場合があるので一度リセットする
        checking_piece = jnp.zeros(12, dtype=jnp.int32)
        checking_piece = checking_piece.at[_action.to[0]].set(1)
        s = JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=s.legal_actions_black,
            legal_actions_white=s.legal_actions_white,
            is_check=jnp.array([1]),
            checking_piece=checking_piece,
        )
        print("checked!")
    else:
        s = JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=s.legal_actions_black,
            legal_actions_white=s.legal_actions_white,
        )
    return s, 0, False


#@jax.jit
def _turn_to_reward(turn: int) -> int:
    #reward = jax.lax.cond(
     #   turn == 0,  # cond
      #  lambda: 1,  # if true
       # lambda: -1,  # if false
    #)
    #return reward
    if turn == 0:
        return 1
    else:
        return -1


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
@jax.jit
def _dlshogi_action(direction: int, to: int) -> int:
    return direction * 12 + to


# fromの座標とtoの座標からdirを生成
@jax.jit
def _point_to_direction(_from: int, to: int, promote: int, turn: int) -> int:
    direction = -1
    dis = to - _from
    # 後手番の動きは反転させる
    dis = jax.lax.cond(
        turn == 1,
        lambda: -dis,
        lambda: dis
    )
    #if turn == 1:
    #    dis = -dis
    # UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP_PROMOTE... の順でdirを割り振る
    # PROMOTEの場合は+8する処理を入れるが、どうぶつ将棋ではUP_PROMOTEしか存在しない(はず)
    direction = jax.lax.cond(
        dis == -1,
        lambda: 0,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == 3,
        lambda: 1,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == -5,
        lambda: 2,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == 4,
        lambda: 3,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == -4,
        lambda: 4,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == 1,
        lambda: 5,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == 5,
        lambda: 6,
        lambda: direction
    )
    direction = jax.lax.cond(
        dis == -3,
        lambda: 7,
        lambda: direction
    )
    direction = jax.lax.cond(
        promote == 1,
        lambda: direction + 8,
        lambda: direction
    )
    return direction


# 打った駒の種類をdirに変換
@jax.jit
def _hand_to_direction(piece: int) -> int:
    # 移動のdirはPROMOTE_UPの8が最大なので9以降に配置
    # 9: 先手ヒヨコ 10: 先手キリン... 14: 後手ゾウ　に対応させる
    p = jax.lax.cond(
        piece <= 5,
        lambda: 8 + piece,
        lambda: 6 + piece
    )
    return p
    #if piece <= 5:
    #    return 8 + piece
    #else:
    #    return 6 + piece


# AnimalShogiActionをdlshogiのint型actionに変換
def _action_to_dlaction(action: JaxAnimalShogiAction, turn: int) -> int:
    if action.is_drop:
        return _dlshogi_action(
            _hand_to_direction(action.piece[0]), action.to[0]
        )
    else:
        return _dlshogi_action(
            _point_to_direction(
                action.from_[0], action.to[0], action.is_promote[0], turn
            ),
            action.to[0],
        )


# dlshogiのint型actionをdirectionとtoに分解
def _separate_dlaction(action: int) -> Tuple[int, int]:
    # direction, to の順番
    return action // 12, action % 12


# directionからfromがtoからどれだけ離れてるかと成りを含む移動かを得る
# 手番の情報が必要
def _direction_to_from(direction: int, to: int, turn: int) -> Tuple[int, int]:
    dif = 0
    if direction == 0 or direction == 8:
        dif = -1
    if direction == 1:
        dif = 3
    if direction == 2:
        dif = -5
    if direction == 3:
        dif = 4
    if direction == 4:
        dif = -4
    if direction == 5:
        dif = 1
    if direction == 6:
        dif = 5
    if direction == 7:
        dif = -3
    if turn == 0:
        if direction >= 8:
            return to - dif, 1
        else:
            return to - dif, 0
    else:
        if direction >= 8:
            return to + dif, 1
        else:
            return to + dif, 0


def _direction_to_hand(direction: int) -> int:
    if direction <= 11:
        return direction - 8
    else:
        return direction - 6


def _dlaction_to_action(
    action: int, state: JaxAnimalShogiState
) -> JaxAnimalShogiAction:
    direction, to = _separate_dlaction(action)
    if direction <= 8:
        # 駒の移動
        _from, is_promote = _direction_to_from(direction, to, state.turn[0])
        piece = _piece_type(state, _from)
        captured = _piece_type(state, to)
        return JaxAnimalShogiAction(
            jnp.array([0]),
            jnp.array([piece]),
            jnp.array([to]),
            jnp.array([_from]),
            jnp.array([captured]),
            jnp.array([is_promote]),
        )
    else:
        # 駒打ち
        piece = _direction_to_hand(direction)
        return JaxAnimalShogiAction(
            jnp.array([1]), jnp.array([piece]), jnp.array([to])
        )


# 手番側でない色を返す
def _another_color(state: JaxAnimalShogiState) -> int:
    return (state.turn + 1) % 2


# 相手の駒を同じ種類の自分の駒に変換する
def _convert_piece(piece: int) -> int:
    p = (piece + 5) % 10
    if p == 0:
        return 10
    else:
        return p


# 駒から持ち駒への変換
# 先手ひよこが0、後手ぞうが5
def _piece_to_hand(piece: int) -> int:
    if piece % 5 == 0:
        p = piece - 4
    else:
        p = piece
    if p < 6:
        return p - 1
    else:
        return p - 3


#  移動の処理
def _move(
    state: JaxAnimalShogiState,
    action: JaxAnimalShogiAction,
) -> JaxAnimalShogiState:
    board = copy.deepcopy(state.board)
    hand = copy.deepcopy(state.hand)
    # s = copy.deepcopy(state)
    board = board.at[action.piece[0], action.from_[0]].set(0)
    # s.board[action.piece[0]][action.from_[0]] = 0
    board = board.at[0, action.from_[0]].set(1)
    # s.board[0][action.from_[0]] = 1
    board = board.at[action.captured[0], action.to[0]].set(0)
    # s.board[action.captured[0]][action.to[0]] = 0
    if action.is_promote[0] == 1:
        board = board.at[action.piece[0] + 4, action.to[0]].set(1)
        # s.board[action.piece[0] + 4][action.to[0]] = 1
    else:
        board = board.at[action.piece[0], action.to[0]].set(1)
        # s.board[action.piece[0]][action.to[0]] = 1
    if action.captured != 0:
        n = hand[_piece_to_hand(_convert_piece(action.captured[0]))]
        hand = hand.at[_piece_to_hand(_convert_piece(action.captured[0]))].set(
            n + 1
        )
        # s.hand[_piece_to_hand(_convert_piece(action.captured[0]))] += 1
    return JaxAnimalShogiState(
        turn=state.turn,
        board=board,
        hand=hand,
        legal_actions_black=state.legal_actions_black,
        legal_actions_white=state.legal_actions_white,
        is_check=state.is_check,
        checking_piece=state.checking_piece,
    )


#  駒打ちの処理
def _drop(
    state: JaxAnimalShogiState, action: JaxAnimalShogiAction
) -> JaxAnimalShogiState:
    board = copy.deepcopy(state.board)
    hand = copy.deepcopy(state.hand)
    n = hand[_piece_to_hand(action.piece[0])]
    hand = hand.at[_piece_to_hand(action.piece[0])].set(n - 1)
    board = board.at[action.piece[0], action.to[0]].set(1)
    board = board.at[0, action.to[0]].set(0)
    return JaxAnimalShogiState(
        turn=state.turn,
        board=board,
        hand=hand,
        legal_actions_black=state.legal_actions_black,
        legal_actions_white=state.legal_actions_white,
        is_check=state.is_check,
        checking_piece=state.checking_piece,
    )


#  ある座標に存在する駒種を返す
@jax.jit
def _piece_type(state: JaxAnimalShogiState, point: int) -> int:
    return state.board[:, point].argmax()


# ある駒の持ち主を返す
@jax.jit
def _owner(piece: int) -> int:
    p = jax.lax.cond(
        piece == 0,
        lambda: 2,
        lambda: (piece - 1) // 5
    )
    return p
    #if piece == 0:
    #    return 2
    #return (piece - 1) // 5


# 盤面のどこに何の駒があるかをnp.arrayに移したもの
# 同じ座標に複数回piece_typeを使用する場合はこちらを使った方が良い
@jax.jit
def _board_status(state: JaxAnimalShogiState) -> jnp.ndarray:
    board = jnp.zeros(12, dtype=jnp.int32)
    for i in range(12):
        board = board.at[i].set(_piece_type(state, i))
        # board[i] = _piece_type(state, i)
    return board


# 駒の持ち主の判定
def _pieces_owner(state: JaxAnimalShogiState) -> jnp.ndarray:
    board = jnp.zeros(12, dtype=jnp.int32)
    for i in range(12):
        piece = _piece_type(state, i)
        board = board.at[i].set(_owner(piece))
        # board[i] = _owner(piece)
    return board


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
@jax.jit
def _is_side(point: int) -> Tuple[bool, bool, bool, bool]:
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


# point(0~11)を座標((0, 0)~(2, 3))に変換
@jax.jit
def _point_to_location(point: int) -> Tuple[int, int]:
    return point // 4, point % 4


# はみ出す部分をカットする
@jax.jit
def _cut_outside(array: jnp.ndarray, point: int) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    u, d, l, r = _is_side(point)
    for i in range(3):
        new_array = jax.lax.cond(
            u,
            lambda: new_array.at[i, 0].set(0),
            lambda: new_array
        )
        new_array = jax.lax.cond(
            d,
            lambda: new_array.at[i, 2].set(0),
            lambda: new_array
        )
    for i in range(4):
        new_array = jax.lax.cond(
            r,
            lambda: new_array.at[0, i].set(0),
            lambda: new_array
        )
        new_array = jax.lax.cond(
            l,
            lambda: new_array.at[2, i].set(0),
            lambda: new_array
        )
    return new_array


@jax.jit
def _action_board(array: jnp.ndarray, point: int) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    y, t = _point_to_location(point)
    new_array = _cut_outside(new_array, point)
    return jnp.roll(new_array, (y - 1, t - 1), axis=(0, 1))


# 各駒の動き
@jax.jit
def _black_pawn_move(point: int) -> jnp.ndarray:
    return _action_board(BLACK_PAWN_MOVE, point)


@jax.jit
def _white_pawn_move(point: int) -> jnp.ndarray:
    return _action_board(WHITE_PAWN_MOVE, point)


@jax.jit
def _black_gold_move(point: int) -> jnp.ndarray:
    return _action_board(BLACK_GOLD_MOVE, point)


@jax.jit
def _white_gold_move(point: int) -> jnp.ndarray:
    return _action_board(WHITE_GOLD_MOVE, point)


@jax.jit
def _rook_move(point: int) -> jnp.ndarray:
    return _action_board(ROOK_MOVE, point)


@jax.jit
def _bishop_move(point: int) -> jnp.ndarray:
    return _action_board(BISHOP_MOVE, point)


@jax.jit
def _king_move(point: int) -> jnp.ndarray:
    return _action_board(KING_MOVE, point)


#  座標と駒の種類から到達できる座標を列挙する関数
@jax.jit
def _point_moves(_from: int, piece: int) -> jnp.ndarray:
    moves = jnp.zeros((3, 4), dtype=jnp.int32)
    moves = jax.lax.cond(
        piece == 1,
        lambda: _black_pawn_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece == 6,
        lambda: _white_pawn_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece % 5 == 2,
        lambda: _rook_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece % 5 == 3,
        lambda: _bishop_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece % 5 == 4,
        lambda: _king_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece == 5,
        lambda: _black_gold_move(_from),
        lambda: moves,
    )
    moves = jax.lax.cond(
        piece == 10,
        lambda: _white_gold_move(_from),
        lambda: moves,
    )
    return moves
    #if piece == 1:
    #    return _black_pawn_move(point)
    #if piece == 6:
    #    return _white_pawn_move(point)
    #if piece % 5 == 2:
    #    return _rook_move(point)
    #if piece % 5 == 3:
    #    return _bishop_move(point)
    #if piece % 5 == 4:
    #    return _king_move(point)
    #if piece == 5:
    #    return _black_gold_move(point)
    #if piece == 10:
    #    return _white_gold_move(point)
    #return jnp.zeros(12, dtype=jnp.int32)


# 利きの判定
def _effected_positions(state: JaxAnimalShogiState, turn: int) -> jnp.ndarray:
    all_effect = jnp.zeros(12)
    board = _board_status(state)
    piece_owner = _pieces_owner(state)
    for i in range(12):
        own = piece_owner[i]
        if own != turn:
            continue
        piece = board[i]
        effect = _point_moves(piece, i).reshape(12)
        all_effect += effect
    return all_effect


# 王手の判定(turn側の王に王手がかかっているかを判定)
def _is_check(state: JaxAnimalShogiState) -> bool:
    effects = _effected_positions(state, _another_color(state))
    king_location = state.board[4 + 5 * state.turn[0], :].argmax()
    return effects[king_location] != 0


# 成る動きが合法かどうかの判定
@jax.jit
def _can_promote(to: int, piece: int) -> bool:
    can_promote = False
    can_promote = jax.lax.cond(
        piece == 1,
        lambda: jax.lax.cond(
            to % 4 == 0,
            lambda: True,
            lambda: can_promote
        ),
        lambda: can_promote
    )
    can_promote = jax.lax.cond(
        piece == 6,
        lambda: jax.lax.cond(
            to % 4 == 3,
            lambda: True,
            lambda: can_promote
        ),
        lambda: can_promote
    )
    return can_promote


# 駒の種類と位置から生成できるactionのフラグを立てる
@jax.jit
def _create_piece_actions(_from: int, piece: int) -> jnp.ndarray:
    turn = _owner(piece)
    actions = jnp.zeros(180, dtype=jnp.int32)
    motion = _point_moves(_from, piece).reshape(12)
    for i in range(12):
        normal_dir = _point_to_direction(_from, i, False, turn)
        normal_act = _dlshogi_action(normal_dir, i)
        pro_dir = _point_to_direction(_from, i, True, turn)
        pro_act = _dlshogi_action(pro_dir, i)
        actions = jax.lax.cond(
            motion[i] == 0,
            lambda: actions,
            lambda: jax.lax.cond(
                _can_promote(i, piece),
                lambda: actions.at[pro_act].set(1),
                lambda: actions
            ).at[normal_act].set(1)
        )
        #if motion[i] == 0:
        #    continue
        #if _can_promote(i, piece):
        #    actions = actions.at[pro_act].set(1)
            # actions[pro_act] = 1

        #actions = actions.at[normal_act].set(1)
        # actions[normal_act] = 1
    return actions


# 駒の種類と位置から生成できるactionのフラグを立てる
@jax.jit
def _add_move_actions(
    _from: int, piece: int, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(_from, piece)
    for i in range(180):
        new_array = jax.lax.cond(
            actions[i] == 1,
            lambda: new_array.at[i].set(1),
            lambda: new_array
        )
    return new_array


# 駒の種類と位置から生成できるactionのフラグを折る
def _filter_move_actions(
    _from: int, piece: int, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(_from, piece)
    for i in range(180):
        if actions[i] == 1:
            new_array = new_array.at[i].set(0)
    return new_array


# 駒打ちのactionを追加する
@jax.jit
def _add_drop_actions(piece: int, array: jnp.ndarray, **kwargs) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(12):
        action = _dlshogi_action(direction, i)
        new_array = new_array.at[action].set(1)
    return new_array


# 駒打ちのactionを消去する
def _filter_drop_actions(piece: int, array: jnp.ndarray) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(12):
        action = _dlshogi_action(direction, i)
        new_array = new_array.at[action].set(0)
        # new_array[action] = 0
    return new_array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は使わないがlegal_actionsが設定されていない場合に使用
@jax.jit
def _init_legal_actions(state: JaxAnimalShogiState) -> JaxAnimalShogiState:
    s = copy.deepcopy(state)
    bs = _board_status(s)
    legal_black = s.legal_actions_black
    legal_white = s.legal_actions_white
    # 移動の追加
    for i in range(12):
        piece = bs[i]
        legal_black = jax.lax.cond(
            _owner(piece) == 0,
            lambda: _add_move_actions(i, piece, legal_black),
            lambda: legal_black
        )
        legal_white = jax.lax.cond(
            _owner(piece) == 1,
            lambda: _add_move_actions(i, piece, legal_white),
            lambda: legal_white
        )
        #if piece == 0:
        #    continue
        #if piece <= 5:
        #    legal_black = _add_move_actions(i, piece, legal_black)
        #else:
        #    legal_white = _add_move_actions(i, piece, legal_white)
    # 駒打ちの追加
    for i in range(3):
        legal_black = jax.lax.cond(
            s.hand[i] == 0,
            lambda: legal_black,
            lambda: _add_drop_actions(1 + i, legal_black)
        )
        legal_white = jax.lax.cond(
            s.hand[i + 3] == 0,
            lambda: legal_white,
            lambda: _add_drop_actions(1 + i, legal_white)
        )
        #if s.hand[i] != 0:
        #    legal_black = _add_drop_actions(1 + i, legal_black)
        #if s.hand[i + 3] != 0:
        #    legal_white = _add_drop_actions(6 + i, legal_white)
    new_s = JaxAnimalShogiState(
        turn=s.turn,
        board=s.board,
        hand=s.hand,
        legal_actions_black=legal_black,
        legal_actions_white=legal_white,
        is_check=s.is_check,
        checking_piece=s.checking_piece,
    )
    return new_s


# 駒の移動によるlegal_actionsの更新
def _update_legal_move_actions(
    state: JaxAnimalShogiState, action: JaxAnimalShogiAction
) -> JaxAnimalShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
        enemy_actions = s.legal_actions_white
    else:
        player_actions = s.legal_actions_white
        enemy_actions = s.legal_actions_black
    # 元の位置にいたときのフラグを折る
    new_player_actions = _filter_move_actions(
        action.from_[0], action.piece[0], player_actions
    )
    new_enemy_actions = enemy_actions
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = _add_move_actions(
        action.to[0], action.piece[0], new_player_actions
    )
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if action.captured != 0:
        new_enemy_actions = _filter_move_actions(
            action.to[0], action.captured[0], new_enemy_actions
        )
        captured = _convert_piece(action.captured[0])
        # にわとりの場合ひよこに変換
        if captured % 5 == 0:
            captured -= 4
        new_player_actions = _add_drop_actions(captured, new_player_actions)
    if s.turn == 0:
        return JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=new_player_actions,
            legal_actions_white=new_enemy_actions,
            is_check=s.is_check,
            checking_piece=s.checking_piece,
        )
        # s.legal_actions_black = new_player_actions
        # s.legal_actions_white = new_enemy_actions
    else:
        return JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=new_enemy_actions,
            legal_actions_white=new_player_actions,
            is_check=s.is_check,
            checking_piece=s.checking_piece,
        )
        # s.legal_actions_black = new_enemy_actions
        # s.legal_actions_white = new_player_actions


# 駒打ちによるlegal_actionsの更新
def _update_legal_drop_actions(
    state: JaxAnimalShogiState, action: JaxAnimalShogiAction
) -> JaxAnimalShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
    else:
        player_actions = s.legal_actions_white
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = _add_move_actions(
        action.to[0], action.piece[0], player_actions
    )
    # 持ち駒がもうない場合、その駒を打つフラグを折る
    if s.hand[_piece_to_hand(action.piece[0])] == 1:
        new_player_actions = _filter_drop_actions(
            action.piece[0], new_player_actions
        )
    if s.turn == 0:
        return JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=new_player_actions,
            legal_actions_white=s.legal_actions_white,
            is_check=s.is_check,
            checking_piece=s.checking_piece,
        )
        # s.legal_actions_black = new_player_actions
    else:
        return JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=s.legal_actions_black,
            legal_actions_white=new_player_actions,
            is_check=s.is_check,
            checking_piece=s.checking_piece,
        )
        # s.legal_actions_white = new_player_actions


# 自分の駒がある位置への移動を除く
def _filter_my_piece_move_actions(
    turn: int, owner: jnp.ndarray, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    for i in range(12):
        if owner[i] != turn:
            continue
        for j in range(9):
            new_array = new_array.at[12 * j + i].set(0)
            # new_array[12 * j + i] = 0
    return new_array


# 駒がある地点への駒打ちを除く
def _filter_occupied_drop_actions(
    turn: int, owner: jnp.ndarray, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    for i in range(12):
        if owner[i] == 2:
            continue
        for j in range(3):
            new_array = new_array.at[12 * (j + 9 + 3 * turn) + i].set(0)
            # new_array[12 * (j + 9 + 3 * turn) + i] = 0
    return new_array


# 自殺手を除く
def _filter_suicide_actions(
    turn: int, king_sq: int, effects: jnp.ndarray, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    moves = _king_move(king_sq).reshape(12)
    for i in range(12):
        if moves[i] == 0:
            continue
        if effects[i] == 0:
            continue
        direction = _point_to_direction(king_sq, i, False, turn)
        action = _dlshogi_action(direction, i)
        new_array = new_array.at[action].set(0)
        # new_array[action] = 0
    return new_array


# 王手放置を除く
def _filter_leave_check_actions(
    turn: int, king_sq: int, check_piece: jnp.ndarray, array: jnp.ndarray
) -> jnp.ndarray:
    new_array = copy.deepcopy(array)
    moves = _king_move(king_sq).reshape(12)
    for i in range(12):
        # 王手をかけている駒の位置以外への移動は王手放置
        for j in range(15):
            # 駒打ちのフラグは全て折る
            if j > 8:
                new_array = new_array.at[12 * j + i].set(0)
                # new_array[12 * j + i] = 0
            # 王手をかけている駒の場所以外への移動ははじく
            if check_piece[i] == 0:
                new_array = new_array.at[12 * j + i].set(0)
                # new_array[12 * j + i] = 0
        # 玉の移動はそれ以外でも可能だがフラグが折れてしまっているので立て直す
        if moves[i] == 0:
            continue
        direction = _point_to_direction(king_sq, i, False, turn)
        action = _dlshogi_action(direction, i)
        new_array = new_array.at[action].set(1)
        # new_array[action] = 1
    return new_array


# boardのlegal_actionsを利用して合法手を生成する
def _legal_actions(state: JaxAnimalShogiState) -> jnp.ndarray:
    turn = state.turn[0]
    if turn == 0:
        action_array = copy.deepcopy(state.legal_actions_black)
    else:
        action_array = copy.deepcopy(state.legal_actions_white)
    king_sq = state.board[4 + 5 * turn].argmax()
    # 王手放置を除く
    if state.is_check:
        action_array = _filter_leave_check_actions(
            turn, king_sq, state.checking_piece, action_array
        )
    own = _pieces_owner(state)
    # 自分の駒がある位置への移動actionを除く
    action_array = _filter_my_piece_move_actions(turn, own, action_array)
    # 駒がある地点への駒打ちactionを除く
    action_array = _filter_occupied_drop_actions(turn, own, action_array)
    # 自殺手を除く
    effects = _effected_positions(state, _another_color(state))
    action_array = _filter_suicide_actions(
        turn, king_sq, effects, action_array
    )
    # その他の反則手を除く
    # どうぶつ将棋の場合はなし
    return action_array
