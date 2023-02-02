from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import jit
from jax.experimental.host_callback import call


@struct.dataclass
class BackgammonState:
    curr_player: jnp.ndarray = jnp.int16(0)
    # 各point(24) bar(2) off(2)にあるcheckerの数 負の値は白, 正の値は黒
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int16)

    # サイコロを振るたびにrngをsplitして更新する.
    rng: jax.random.KeyArray = jnp.zeros(2, dtype=jnp.uint16)

    # 終了しているかどうか.
    terminated: jnp.ndarray = jnp.bool_(False)

    # サイコロの出目 0~5: 1~6
    dice: jnp.ndarray = jnp.zeros(2, dtype=jnp.int16)

    # プレイできるサイコロの目
    playable_dice: jnp.ndarray = jnp.zeros(4, dtype=jnp.int16)

    # プレイしたサイコロの目の数
    played_dice_num: jnp.ndarray = jnp.int16(0)

    # 黒なら-1, 白なら1
    turn: jnp.ndarray = jnp.int16(1)
    """
    合法手
    micro action = 6*src+die
    """
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26 + 6, dtype=jnp.int16)


@jit
def init(rng: jax.random.KeyArray) -> Tuple[jnp.ndarray, BackgammonState]:
    rng1, rng2, rng3 = jax.random.split(rng, num=3)
    curr_player: jnp.ndarray = jnp.int16(jax.random.bernoulli(rng1))
    board: jnp.ndarray = _make_init_board()
    terminated: jnp.ndarray = jnp.bool_(False)
    dice: jnp.ndarray = _roll_init_dice(rng2)
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.ndarray = jnp.int16(0)
    turn: jnp.ndarray = _init_turn(dice)
    legal_action_mask: jnp.ndarray = _legal_action_mask(
        board, turn, playable_dice
    )
    state = BackgammonState(  # type: ignore
        curr_player=curr_player,
        rng=rng3,
        board=board,
        terminated=terminated,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return curr_player, state


@jit
def step(
    state: BackgammonState, action: int
) -> Tuple[BackgammonState, int, bool]:
    return jax.lax.cond(
        _is_turn_end(state),
        lambda: (_change_turn(state).curr_player, _change_turn(state), 0),
        lambda: _normal_step(state, action),
    )


@jit
def observe(state: BackgammonState, curr_player: jnp.ndarray) -> jnp.ndarray:
    """
    手番のplayerに対する観測を返す.
    """
    board: jnp.ndarray = state.board
    turn: jnp.ndarray = state.turn
    _curr_player: jnp.ndarray = state.curr_player
    zero_one_dice_vec: jnp.ndarray = _to_zero_one_dice_vec(state.playable_dice)
    return jax.lax.cond(
        curr_player == _curr_player,
        lambda: jnp.concatenate((turn * board, zero_one_dice_vec), axis=None),  # type: ignore
        lambda: jnp.concatenate(
            (-turn * board, jnp.zeros(6, dtype=jnp.int16)), axis=None  # type: ignore
        ),
    )


@jit
def _to_zero_one_dice_vec(playable_dice: jnp.ndarray) -> jnp.ndarray:
    """
    playできるサイコロを6次元の0-1ベクトルで返す.
    """
    zero_one_dice_vec: jnp.ndarray = jnp.zeros(6, dtype=jnp.int16)
    return jax.lax.fori_loop(
        0,
        4,
        lambda i, x: jax.lax.cond(
            playable_dice[i] != -1,
            lambda: x.at[playable_dice[i]].set(1),
            lambda: x,
        ),
        zero_one_dice_vec,
    )


@jit
def _normal_step(
    state: BackgammonState, action: int
) -> Tuple[BackgammonState, int, bool]:
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state.board, state.turn),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state),
    )


@jit
def _winning_step(
    state: BackgammonState,
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    勝利者がいる場合のstep.
    """
    reward = _calc_win_score(state.board, state.turn)
    state = state.replace(terminated=jnp.bool_(True))  # type: ignore
    return state.curr_player, state, reward


@jit
def _no_winning_step(
    state: BackgammonState,
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    勝利者がいない場合のstep, ターン終了の条件を満たせばターンを変更する.
    """
    return jax.lax.cond(
        _is_turn_end(state),
        lambda: (
            _change_turn(state).curr_player,
            _change_turn(state),
            0,
        ),
        lambda: (state.curr_player, state, 0),
    )


@jit
def _update_by_action(state: BackgammonState, action: int) -> BackgammonState:
    """
    行動を受け取って状態をupdate
    """
    rng = state.rng
    curr_player: jnp.ndarray = state.curr_player
    terminated: jnp.ndarray = state.terminated
    board: jnp.ndarray = _move(state.board, state.turn, action)
    played_dice_num: jnp.ndarray = jnp.int16(state.played_dice_num + 1)
    playable_dice: jnp.ndarray = _update_playable_dice(
        state.playable_dice, state.played_dice_num, state.dice, action
    )
    legal_action_mask: jnp.ndarray = _legal_action_mask(
        board, state.turn, playable_dice
    )
    return BackgammonState(  # type: ignore
        curr_player=curr_player,
        rng=rng,
        terminated=terminated,
        board=board,
        turn=state.turn,
        dice=state.dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


@jit
def _make_init_board() -> jnp.ndarray:
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int16)
    board = board.at[0].set(-2)
    board = board.at[5].set(5)
    board = board.at[7].set(3)
    board = board.at[11].set(-5)
    board = board.at[12].set(5)
    board = board.at[16].set(-3)
    board = board.at[18].set(-5)
    board = board.at[23].set(2)
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
def _change_turn(state: BackgammonState) -> Tuple[BackgammonState]:
    """
    ターンを変更して新しい状態を返す.
    """
    rng1, rng2 = jax.random.split(state.rng)
    board: jnp.ndarray = state.board
    turn: jnp.ndarray = -1 * state.turn  # turnを変える
    curr_player: jnp.ndarray = (state.curr_player + 1) % 2
    terminated: jnp.ndarray = jnp.bool_(False)
    dice: jnp.ndarray = _roll_dice(rng1)  # diceを振る
    playable_dice: jnp.ndarray = _set_playable_dice(dice)  # play可能なサイコロを初期化
    played_dice_num: jnp.ndarray = jnp.int16(0)
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, turn, dice)
    return BackgammonState(  # type: ignore
        curr_player=curr_player,
        rng=rng2,
        board=board,
        terminated=terminated,
        turn=turn,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


@jax.jit
def _roll_init_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    """
    # 違う目が出るまで振り続ける.
    """

    def _cond_fn(roll: jnp.ndarray):
        return roll[0] == roll[1]

    def _body_fn(_roll: jnp.ndarray):
        roll: jnp.ndarray = jax.random.randint(
            rng, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int16
        )
        return roll[0]

    return jax.lax.while_loop(
        _cond_fn, _body_fn, jnp.array([0, 0], dtype=jnp.int16)
    )


@jit
def _roll_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    roll: jnp.ndarray = jax.random.randint(
        rng, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int16
    )
    return roll[0]


@jit
def _init_turn(dice: jnp.ndarray) -> jnp.ndarray:
    """
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    return jax.lax.cond(diff > 0, lambda: jnp.int16(1), lambda: jnp.int16(-1))


@jit
def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1でemptyを表す.
    """
    return jax.lax.cond(
        dice[0] == dice[1],
        lambda: jnp.array([dice[0]] * 4, dtype=np.int16),
        lambda: jnp.array([dice[0], dice[1], -1, -1], dtype=np.int16),
    )


@jit
def _update_playable_dice(
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
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
def _home_board(turn: jnp.ndarray) -> jnp.ndarray:
    """
    黒: [18~23], 白: [0~5]
    """
    return jax.lax.cond(
        turn == -1, lambda: jnp.arange(18, 24), lambda: jnp.arange(0, 6)
    )


@jit
def _off_idx(turn: jnp.ndarray) -> int:
    """
    黒: 26, 白: 27
    """
    return jax.lax.cond(turn == -1, lambda: 26, lambda: 27)


@jit
def _bar_idx(turn: jnp.ndarray) -> int:
    """
    黒: 24, 白 25
    """
    return jax.lax.cond(turn == -1, lambda: 24, lambda: 25)


@jit
def _rear_distance(board: jnp.ndarray, turn: jnp.ndarray) -> jnp.ndarray:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値
    """
    b = board[:24]

    exists = jnp.where((b * turn > 0), size=24, fill_value=jnp.nan)[  # type: ignore
        0
    ]
    return jax.lax.cond(
        turn == 1,
        lambda: jnp.int16(
            jnp.max(jnp.nan_to_num(exists, nan=jnp.int16(-100))) + 1
        ),
        lambda: jnp.int16(
            24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int16(100)))
        ),
    )


@jit
def _is_all_on_homeboad(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    home_board: jnp.ndarray = _home_board(turn)
    on_home_board: int = jnp.clip(
        turn * board[home_board], a_min=0, a_max=15
    ).sum()
    off: int = board[_off_idx(turn)] * turn  # type: ignore
    return (15 - off) == on_home_board


@jit
def _is_open(board: jnp.ndarray, turn: jnp.ndarray, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    checkers = board[point]
    return turn * checkers >= -1  # type: ignore


@jit
def _exists(board: jnp.ndarray, turn: jnp.ndarray, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    checkers = board[point]
    return turn * checkers >= 1  # type: ignore


@jit
def _calc_src(src: int, turn: jnp.ndarray) -> jnp.ndarray:
    """
    boardのindexに合わせる.
    """
    return jax.lax.cond(
        src == 1, lambda: jnp.int16(_bar_idx(turn)), lambda: jnp.int16(src - 2)
    )


@jit
def _calc_tgt(src: int, turn: jnp.ndarray, die) -> jnp.ndarray:
    """
    boardのindexに合わせる.
    """
    return jax.lax.cond(
        src >= 24,
        lambda: jnp.int16(
            jnp.clip(24 * turn, a_min=-1, a_max=24) + die * -1 * turn
        ),
        lambda: jnp.int16(_from_other_than_bar(src, turn, die)),
    )  # type: ignore


@jit
def _from_other_than_bar(src: int, turn: jnp.ndarray, die: int) -> int:
    return jax.lax.cond(
        (src + die * -1 * turn >= 0) & (src + die * -1 * turn <= 23),
        lambda: jnp.int16(src + die * -1 * turn),
        lambda: jnp.int16(_off_idx(turn)),
    )  # type: ignore


@jit
def _decompose_action(action: int, turn: jnp.ndarray) -> Tuple:
    """
    action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(action // 6, turn)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, turn, die)
    return src, die, tgt


@jit
def selu(x, alpha=1.67, lmbda=1.05):
    call(lambda x: print(f"x: {x}"), x)


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
def _distance_to_goal(src: int, turn: jnp.ndarray) -> int:
    return jax.lax.cond(turn == -1, lambda: 24 - src, lambda: src + 1)  # type: ignore


@jit
def _is_to_off_legal(
    board: jnp.ndarray, turn: jnp.ndarray, src: int, tgt: int, die: int
) -> bool:
    """
    board外への移動についての合法判定
    """
    return jax.lax.cond(
        src < 0,
        lambda: False,
        lambda: _exists(board, turn, src)
        & _is_all_on_homeboad(board, turn)
        & (_rear_distance(board, turn) <= die)
        & (_rear_distance(board, turn) == _distance_to_goal(src, turn)),
    )  # type: ignore


@jit
def _is_to_point_legal(
    board: jnp.ndarray, turn: jnp.ndarray, src: int, tgt: int
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
def _move(board: jnp.ndarray, turn: jnp.ndarray, action: int) -> jnp.ndarray:
    """
    micro actionに基づく状態更新
    """
    src, _, tgt = _decompose_action(action, turn)
    board = board.at[_bar_idx(-1 * turn)].add(
        -1 * turn * (board[tgt] == -1 * turn)
    )  # targetに相手のcheckerが一枚だけある時, それを相手のbarに移動
    board = board.at[src].add(-1 * turn)
    board = board.at[tgt].add(
        turn + (board[tgt] == -1 * turn) * turn
    )  # hitした際は符号が変わるので余分に+1
    return board


@jit
def _is_all_off(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる.
    """
    return board[_off_idx(turn)] * turn == 15  # type: ignore


@jit
def _calc_win_score(board: jnp.ndarray, turn: jnp.ndarray) -> int:
    return jax.lax.cond(
        _is_gammon(board, turn),
        lambda: _score(board, turn),
        lambda: 1,
    )


@jit
def _score(board: jnp.ndarray, turn: jnp.ndarray) -> int:
    return jax.lax.cond(
        _remains_at_inner(board, turn),
        lambda: 3,
        lambda: 2,
    )


@jit
def _is_gammon(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがなければgammon勝ち
    """
    return board[_off_idx(-1 * turn)] == 0  # type: ignore


@jit
def _remains_at_inner(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがない && 相手のcheckerが一つでも自分のインナーに残っている
    => backgammon勝ち
    """
    return jnp.take(board, _home_board(-1 * turn)).sum() != 0  # type: ignore


@jit
def _legal_action_mask(
    board: jnp.ndarray, turn: jnp.ndarray, dice: jnp.ndarray
) -> jnp.ndarray:
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int16)

    @jit
    def _update(i: int, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
        return legal_action_mask | _legal_action_mask_for_single_die(
            board, turn, dice[i]
        )

    legal_action_mask = jax.lax.fori_loop(0, 4, _update, legal_action_mask)
    return legal_action_mask


@jit
def _legal_action_mask_for_single_die(
    board: jnp.ndarray, turn: jnp.ndarray, die: int
) -> jnp.ndarray:
    """
    一つのサイコロの目に対するlegal micro action
    """
    return jax.lax.cond(
        die == -1,
        lambda: jnp.zeros(26 * 6 + 6, dtype=np.int16),
        lambda: _legal_action_mask_for_valid_single_dice(board, turn, die),
    )


@jit
def _legal_action_mask_for_valid_single_dice(
    board: jnp.ndarray, turn: jnp.ndarray, die: int
) -> jnp.ndarray:
    """
    -1以外のサイコロの目に対して合法判定
    """
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int16)

    @jit
    def _is_legal(i: int, legal_action_mask: jnp.ndarray):
        action = i * 6 + die
        legal_action_mask = legal_action_mask.at[action].set(
            _is_action_legal(board, turn, action)
        )
        return legal_action_mask

    return jax.lax.fori_loop(0, 26, _is_legal, legal_action_mask)
