from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import jit

seed = 1701
key = jax.random.PRNGKey(seed)


@struct.dataclass
class BackgammonState:

    # 各point(24) bar(2) off(2)にあるcheckerの数 負の値は白, 正の値は黒
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)

    # サイコロの出目 0~5: 1~6
    dice: jnp.ndarray = jnp.zeros(2, dtype=jnp.int8)

    # プレイできるサイコロの目
    playable_dice: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)

    # プレイしたサイコロの目の数
    played_dice_num: jnp.int8 = jnp.int8(0)

    # 白なら-1, 黒なら1
    turn: jnp.int8 = jnp.int8(1)
    """
    合法手
    micro action = 6*src+die
    """
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26 + 6, dtype=jnp.int8)


@jit
def init() -> BackgammonState:
    board: jnp.ndarray = _make_init_board()
    dice: jnp.ndarray = _roll_init_dice()
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.int8 = jnp.int8(0)
    turn: jnp.int8 = _init_turn(dice)
    legal_action_mask: jnp.ndarray = _legal_action_mask(
        board, turn, playable_dice
    )
    state = BackgammonState(  # type: ignore
        board=board,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return state


@jit
def step(
    state: BackgammonState, action: int
) -> Tuple[BackgammonState, int, bool]:
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state.board, state.turn),
        lambda: _winning_step(state),
        lambda: _normal_step(state),
    )


@jit
def _winning_step(state: BackgammonState) -> Tuple[BackgammonState, int, bool]:
    """
    勝利者がいる場合のstep
    """
    reward = _calc_win_score(state.board, state.turn)
    return (state, reward, True)


@jit
def _normal_step(state: BackgammonState) -> Tuple[BackgammonState, int, bool]:
    """
    勝利者がいない場合のstep, ターンが回ってきたが, 動かせる場所がない場合(dance)すぐさまターンを変える必要がある.
    """
    state, has_changed = _change_turn(state)
    return jax.lax.cond(
        has_changed & _is_turn_end(state),
        lambda: (_change_turn(state)[0], 0, False),
        lambda: (state, 0, False),
    )


@jit
def _update_by_action(state: BackgammonState, action: int):
    board: jnp.ndarray = _move(state.board, state.turn, action)
    played_dice_num: jnp.int8 = jnp.int8(state.played_dice_num + 1)
    played_dice: jnp.ndarray = _update_playable_dice(
        state.playable_dice, state.played_dice_num, state.dice, action
    )
    legal_action_mask: jnp.ndarray = _legal_action_mask(
        state.board, state.turn, state.playable_dice
    )
    return BackgammonState(  # type: ignore
        board=board,
        turn=state.turn,
        dice=state.dice,
        playable_dice=played_dice,
        played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


@jit
def _make_init_board() -> jnp.ndarray:
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    board = board.at[0].set(2)
    board = board.at[5].set(-5)
    board = board.at[7].set(-3)
    board = board.at[11].set(5)
    board = board.at[12].set(-5)
    board = board.at[16].set(3)
    board = board.at[18].set(5)
    board = board.at[23].set(-2)
    return board


@jit
def _is_turn_end(state: BackgammonState) -> bool:
    """
    play可能なサイコロ数が0の場合ないしlegal_actionがない場合交代
    """
    return (state.playable_dice.sum() == -4) | (
        state.legal_action_mask.sum() == 0
    )  # type: ignore


@jit
def _change_turn(state: BackgammonState) -> Tuple[BackgammonState, bool]:
    """
    ターンが変わる場合は新しいstateを, そうでない場合は元のstateを返す.
    """
    board: jnp.ndarray = state.board
    turn: jnp.int8 = -state.turn  # turnを変える
    dice: jnp.ndarray = _roll_dice()  # diceを振る
    playable_dice: jnp.ndarray = _set_playable_dice(
        state.dice
    )  # play可能なサイコロを初期化
    played_dice_num: jnp.int8 = jnp.int8(0)
    legal_action_mask: jnp.ndarray = _legal_action_mask(
        state.board, state.turn, state.dice
    )
    return jax.lax.cond(
        _is_turn_end(state),
        lambda: (
            BackgammonState(  # type: ignore
                board=board,
                turn=turn,
                dice=dice,
                playable_dice=playable_dice,
                played_dice_num=played_dice_num,
                legal_action_mask=legal_action_mask,
            ),
            True,
        ),
        lambda: (state, False),
    )


@jax.jit
def _roll_init_dice() -> jnp.ndarray:
    """
    # 違う目が出るまで振り続ける.
    """

    def _cond_fn(roll: jnp.ndarray):
        return roll[0] == roll[1]

    def _body_fn(_roll: jnp.ndarray):
        roll: jnp.ndarray = jax.random.randint(
            key, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int8
        )
        return roll[0]

    return jax.lax.while_loop(
        _cond_fn, _body_fn, jnp.array([0, 0], dtype=jnp.int8)
    )


@jit
def _roll_dice() -> jnp.ndarray:
    roll: jnp.ndarray = jax.random.randint(
        key, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int8
    )
    return roll[0]


@jit
def _init_turn(dice: jnp.ndarray) -> jnp.int8:
    """
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    return jax.lax.cond(diff > 0, lambda: jnp.int8(1), lambda: jnp.int8(-1))


@jit
def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1でemptyを表す.
    """
    return jax.lax.cond(
        dice[0] == dice[1],
        lambda: jnp.array([dice[0]] * 4, dtype=np.int8),
        lambda: jnp.array([dice[0], dice[1], -1, -1], dtype=np.int8),
    )


@jit
def _update_playable_dice(
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.int8,
    dice: jnp.ndarray,
    action: int,
) -> jnp.ndarray:
    _n = played_dice_num
    die = action % 6

    @jit
    def _update_for_diff_dice(die: int, playable_dice: np.ndarray):
        return jax.lax.fori_loop(
            0,
            4,
            lambda i, x: jax.lax.cond(
                die == x[i], lambda: x.at[i].set(-1), lambda: x
            ),
            playable_dice,
        )

    return jax.lax.cond(
        dice[0] == dice[1],
        lambda: playable_dice.at[3 - _n].set(-1),
        lambda: _update_for_diff_dice(die, playable_dice),
    )


@jit
def _home_board(turn: jnp.int8) -> jnp.ndarray:
    """
    白: [18~23], 黒: [0~5]
    """
    return jax.lax.cond(
        turn == -1, lambda: jnp.arange(18, 24), lambda: jnp.arange(0, 6)
    )


@jit
def _off_idx(turn: jnp.int8) -> int:
    """
    白: 26, 黒: 27
    """
    return jax.lax.cond(turn == -1, lambda: 26, lambda: 27)


@jit
def _bar_idx(turn: jnp.int8) -> int:
    """
    白: 24, 黒 25
    """
    return jax.lax.cond(turn == -1, lambda: 24, lambda: 25)


def _rear_distance(board: jnp.ndarray, turn: jnp.int8) -> jnp.int8:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値
    """
    b = board[:24]

    exists: np.ndarray = jnp.where(
        (b * turn > 0), size=24, fill_value=jnp.nan
    )[0]
    return jax.lax.cond(
        turn == 1,
        lambda: jnp.max(jnp.nan_to_num(exists, nan=jnp.int8(-100))) + 1,
        lambda: 24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int8(100))),
    )


@jit
def _is_all_on_homeboad(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    home_board: jnp.ndarray = _home_board(turn)
    on_home_board: int = jnp.clip(
        -1 * board[home_board], a_min=0, a_max=15
    ).sum()
    off: int = board[_off_idx(turn)] * turn
    return (15 - off) == on_home_board


@jit
def _is_open(board: jnp.ndarray, turn: jnp.int8, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    checkers = board[point]
    return turn * checkers >= -1  # 黒と白のcheckerは異符号


@jit
def _exists(board: jnp.ndarray, turn: jnp.int8, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    checkers = board[point]
    return turn * checkers >= 1


@jit
def _calc_src(src: int, turn: jnp.int8) -> jnp.int8:
    """
    boardのindexに合わせる.
    """
    return jax.lax.cond(
        src == 1, lambda: jnp.int8(_bar_idx(turn)), lambda: jnp.int8(src - 2)
    )


@jit
def _calc_tgt(src: int, turn: jnp.int8, die) -> jnp.int8:
    """
    boardのindexに合わせる.
    """
    return jax.lax.cond(
        src >= 24,
        lambda: jnp.int8(
            jnp.clip(24 * turn, a_min=-1, a_max=24) + die * -1 * turn
        ),
        lambda: jnp.int8(_from_other_than_bar(src, turn, die)),
    )


@jit
def _from_other_than_bar(src: int, turn: jnp.int8, die: int) -> int:
    return jax.lax.cond(
        (jnp.abs(src + die * -1 * turn - 25 / 2) < 25 / 2),
        lambda: jnp.int8(src + die * -1 * turn),
        lambda: jnp.int8(_off_idx(turn)),
    )


@jit
def _decompose_action(action: int, turn: jnp.int8) -> Tuple:
    """
    action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(action // 6, turn)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, turn, die)
    return src, die, tgt


@jit
def _is_action_legal(board: jnp.ndarray, turn, action: int) -> bool:
    """
    micro actionの合法判定
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_action(action, turn)
    return jax.lax.cond(
        (0 <= tgt) & (tgt <= 23) & (src >= 0),
        lambda: _is_to_point_legal(board, turn, src, tgt),
        lambda: _is_to_off_legal(board, turn, src, tgt, die),
    )


@jit
def _is_to_off_legal(
    board: jnp.ndarray, turn: np.int8, src: int, tgt: int, die: int
) -> bool:
    """
    boad外への移動についての合法判定
    """
    return jax.lax.cond(
        src < 0,
        lambda: False,
        lambda: _exists(board, turn, src)
        & _is_all_on_homeboad(board, turn)
        & (_rear_distance(board, turn) <= die),
    )


@jit
def _is_to_point_legal(
    board: jnp.ndarray, turn: np.int8, src: int, tgt: int
) -> bool:
    """
    tgtがpointの場合の合法手判定
    """
    return jax.lax.cond(
        src >= 24,
        lambda: (_exists(board, turn, src)) & (_is_open(board, turn, tgt)),
        lambda: (_exists(board, turn, src))
        & (_is_open(board, turn, tgt))
        & (board[_bar_idx(turn)] == 0),
    )


@jit
def _move(board: jnp.ndarray, turn: jnp.int8, action: int) -> jnp.ndarray:
    """
    micro actionに基づく状態更新
    """
    src, _, tgt = _decompose_action(action, turn)
    board = board.at[_bar_idx(-1 * turn)].add(
        -1 * turn
    )  # targetに相手のcheckerが一枚だけある時, それを相手のbarに移動
    board = board.at[src].add(-1 * turn)
    board = board.at[tgt].add(
        turn + (board[tgt] == -1 * turn) * turn
    )  # hitした際は符号が変わるので余分に+1
    return board


@jit
def _is_all_off(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる.
    """
    return board[_off_idx(turn)] * turn == 15


@jit
def _calc_win_score(board: jnp.ndarray, turn: jnp.int8) -> int:
    return jax.lax.cond(
        _is_gammon(board, turn),
        lambda: _score(board, turn),
        lambda: 1,
    )


@jit
def _score(board: jnp.ndarray, turn: jnp.int8) -> int:
    return jax.lax.cond(
        _remains_at_inner(board, turn),
        lambda: 3,
        lambda: 2,
    )


@jit
def _is_gammon(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    相手のoffに一つもcheckerがなければgammon勝ち
    """
    return board[_off_idx(-1 * turn)] == 0  # type: ignore


@jit
def _remains_at_inner(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    相手のoffに一つもcheckerがない && 相手のcheckerが一つでも自分のインナーに残っている
    => backgammon勝ち
    """
    return jnp.take(board, _home_board(-1 * turn)).sum() != 0


@jit
def _legal_action_mask(
    board: jnp.ndarray, turn: jnp.int8, dice: jnp.ndarray
) -> jnp.ndarray:
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int8)

    @jit
    def _update(i: int, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
        return legal_action_mask | _legal_action_mask_for_single_die(
            board, turn, dice[i]
        )

    legal_action_mask = jax.lax.fori_loop(0, 4, _update, legal_action_mask)
    return legal_action_mask


@jit
def _legal_action_mask_for_single_die(
    board: jnp.ndarray, turn: jnp.int8, die: int
) -> jnp.ndarray:
    """
    一つのサイコロの目に対するlegal micro action
    """
    return jax.lax.cond(
        die == -1,
        lambda: jnp.zeros(26 * 6 + 6, dtype=np.int8),
        lambda: _legal_action_mask_for_valid_single_dice(board, turn, die),
    )


@jit
def _legal_action_mask_for_valid_single_dice(
    board: jnp.ndarray, turn: jnp.int8, die: int
) -> jnp.ndarray:
    """
    -1以外のサイコロの目に対して合法判定
    """
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int8)

    @jit
    def _is_legal(i: int, legal_action_mask: jnp.ndarray):
        action = i * 6 + die
        legal_action_mask = legal_action_mask.at[action].set(
            _is_action_legal(board, turn, action)
        )
        return legal_action_mask

    return jax.lax.fori_loop(0, 26, _is_legal, legal_action_mask)
