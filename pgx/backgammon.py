from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from pgx.flax.struct import dataclass

init_dice_pattern: jnp.ndarray = jnp.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 0],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 0],
        [2, 1],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 4],
        [3, 5],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 5],
        [5, 0],
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4],
    ],
    dtype=jnp.int16,
)


@dataclass
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


def step(
    state: BackgammonState, action: int
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    step 関数.
    terminatedしている場合, 状態をそのまま返す.
    """
    return jax.lax.cond(
        state.terminated,
        lambda: (state.curr_player, state, 0),
        lambda: _normal_step(state, action),
    )


def _normal_step(
    state: BackgammonState, action: int
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    terminated していない場合のstep 関数.
    """
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state.board, state.turn),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state),
    )


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


def _to_zero_one_dice_vec(playable_dice: jnp.ndarray) -> jnp.ndarray:
    """
    playできるサイコロを6次元の0-1ベクトルで返す.
    """
    dice_indices: jnp.ndarray = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int16
    )  # サイコロの数は最大4

    def _insert_dice_num(
        idx: jnp.ndarray, playable_dice: jnp.ndarray
    ) -> jnp.ndarray:
        vec: jnp.ndarray = jnp.zeros(6, dtype=jnp.int16)
        return (playable_dice[idx] != -1) * vec.at[playable_dice[idx]].set(
            1
        ) + (playable_dice[idx] == -1) * vec

    return (
        jax.vmap(_insert_dice_num)(
            dice_indices, jnp.tile(playable_dice, (4, 1))
        )
        .sum(axis=0)
        .astype(jnp.int16)
    )


def _winning_step(
    state: BackgammonState,
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    勝利者がいる場合のstep.
    """
    reward = _calc_win_score(state.board, state.turn)
    state = state.replace(terminated=jnp.bool_(True))  # type: ignore
    return state.curr_player, state, reward


def _no_winning_step(
    state: BackgammonState,
) -> Tuple[jnp.ndarray, BackgammonState, int]:
    """
    勝利者がいない場合のstep, ターン終了の条件を満たせばターンを変更する.
    """
    s = _change_until_legal(state)
    return jax.lax.cond(
        _is_turn_end(state),
        lambda: (
            s.curr_player,
            s,
            0,
        ),
        lambda: (state.curr_player, state, 0),
    )


def _change_until_legal(state: BackgammonState) -> BackgammonState:
    """
    行動可能なplayerが出るまでturnを変え続ける.
    """
    return jax.lax.while_loop(_is_turn_end, _change_turn, state)


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


def _make_init_board() -> jnp.ndarray:
    board: jnp.ndarray = jnp.array(
        [
            -2,
            0,
            0,
            0,
            0,
            5,
            0,
            3,
            0,
            0,
            0,
            -5,
            5,
            0,
            0,
            0,
            -3,
            0,
            -5,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
        ],
        dtype=jnp.int16,
    )
    return board


def _is_turn_end(state: BackgammonState) -> bool:
    """
    play可能なサイコロ数が0の場合ないしlegal_actionがない場合交代
    """
    return (state.playable_dice.sum() == -4) | (
        state.legal_action_mask.sum() == 0
    )  # type: ignore


def _change_turn(state: BackgammonState) -> BackgammonState:
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


def _roll_init_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    """
    # 違う目が出るまで振り続ける.
    """

    return jax.random.choice(rng, init_dice_pattern)


def _roll_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    roll: jnp.ndarray = jax.random.randint(
        rng, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int16
    )
    return roll[0]


def _init_turn(dice: jnp.ndarray) -> jnp.ndarray:
    """
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    return (diff > 0) * jnp.int16(1) + (diff <= 0) * jnp.int16(-1)


def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1でemptyを表す.
    """
    return (dice[0] == dice[1]) * jnp.array([dice[0]] * 4, dtype=jnp.int16) + (
        dice[0] != dice[1]
    ) * jnp.array([dice[0], dice[1], -1, -1], dtype=jnp.int16)


def _update_playable_dice(
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    dice: jnp.ndarray,
    action: int,
) -> jnp.ndarray:
    _n = played_dice_num
    die_array = jnp.array([action % 6] * 4, dtype=jnp.int16)
    dice_indices: jnp.ndarray = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int16
    )  # サイコロの数は最大4

    def _update_for_diff_dice(
        die: jnp.ndarray, idx: jnp.ndarray, playable_dice: jnp.ndarray
    ):
        return (die == playable_dice[idx]) * -1 + (
            die != playable_dice[idx]
        ) * playable_dice[idx]

    return (dice[0] == dice[1]) * playable_dice.at[3 - _n].set(-1) + (
        dice[0] != dice[1]
    ) * jax.vmap(_update_for_diff_dice)(
        die_array, dice_indices, jnp.tile(playable_dice, (4, 1))
    ).astype(
        jnp.int16
    )


def _home_board(turn: jnp.ndarray) -> jnp.ndarray:
    """
    黒: [18~23], 白: [0~5]
    """
    return (turn == -1) * jnp.arange(18, 24) + (turn == 1) * jnp.arange(0, 6)  # type: ignore


def _off_idx(turn: jnp.ndarray) -> int:
    """
    黒: 26, 白: 27
    """
    return (turn == -1) * 26 + (turn == 1) * 27  # type: ignore


def _bar_idx(turn: jnp.ndarray) -> int:
    """
    黒: 24, 白 25
    """
    return (turn == -1) * 24 + (turn == 1) * 25  # type: ignore


def _rear_distance(board: jnp.ndarray, turn: jnp.ndarray) -> jnp.ndarray:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値
    """
    b = board[:24]
    exists = jnp.where((b * turn > 0), size=24, fill_value=jnp.nan)[  # type: ignore
        0
    ]
    return (turn == 1) * jnp.int16(
        jnp.max(jnp.nan_to_num(exists, nan=jnp.int16(-100))) + 1
    ) + (turn == -1) * jnp.int16(
        24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int16(100)))
    )


def _is_all_on_home_board(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    home_board: jnp.ndarray = _home_board(turn)
    on_home_board: int = jnp.clip(
        turn * board[home_board], a_min=0, a_max=15
    ).sum()
    off: int = board[_off_idx(turn)] * turn  # type: ignore
    return (15 - off) == on_home_board


def _is_open(board: jnp.ndarray, turn: jnp.ndarray, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    checkers = board[point]
    return turn * checkers >= -1  # type: ignore


def _exists(board: jnp.ndarray, turn: jnp.ndarray, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    checkers = board[point]
    return turn * checkers >= 1  # type: ignore


def _calc_src(src: int, turn: jnp.ndarray) -> int:
    """
    boardのindexに合わせる.
    """
    return (src == 1) * jnp.int16(_bar_idx(turn)) + (src != 1) * jnp.int16(
        src - 2
    )  # type: ignore


def _calc_tgt(src: int, turn: jnp.ndarray, die) -> int:
    """
    boardのindexに合わせる. actionは src*6 + dieの形になっている. targetは黒ならsrcからdie分+白ならdie分-(目的地が逆だから.)
    """
    return (src >= 24) * jnp.int16(
        jnp.clip(24 * turn, a_min=-1, a_max=24) + die * -1 * turn
    ) + (src < 24) * jnp.int16(
        _from_other_than_bar(src, turn, die)
    )  # type: ignore


def _from_other_than_bar(src: int, turn: jnp.ndarray, die: int) -> int:
    _is_from_board = (src + die * -1 * turn >= 0) & (
        src + die * -1 * turn <= 23
    )
    return _is_from_board * jnp.int16(src + die * -1 * turn) + (
        (~_is_from_board)
    ) * jnp.int16(
        _off_idx(turn)
    )  # type: ignore


def _decompose_action(action: int, turn: jnp.ndarray) -> Tuple:
    """
    action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(action // 6, turn)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, turn, die)
    return src, die, tgt


def _is_action_legal(board: jnp.ndarray, turn, action: int) -> bool:
    """
    micro actionの合法判定
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_action(action, turn)
    _is_to_point = (0 <= tgt) & (tgt <= 23) & (src >= 0)
    return _is_to_point & _is_to_point_legal(board, turn, src, tgt) | (
        ~_is_to_point
    ) & _is_to_off_legal(
        board, turn, src, tgt, die
    )  # type: ignore


def _distance_to_goal(src: int, turn: jnp.ndarray) -> int:
    return (turn == -1) * (24 - src) + (turn == 1) * (src + 1)  # type: ignore


def _is_to_off_legal(
    board: jnp.ndarray, turn: jnp.ndarray, src: int, tgt: int, die: int
):
    """
    board外への移動についての合法判定
    """
    r = _rear_distance(board, turn)
    d = _distance_to_goal(src, turn)
    return (
        (src >= 0)
        & _exists(board, turn, src)
        & _is_all_on_home_board(board, turn)
        & (r <= die)
        & (r == d)
    )  # type: ignore


def _is_to_point_legal(
    board: jnp.ndarray, turn: jnp.ndarray, src: int, tgt: int
) -> bool:
    """
    tgtがpointの場合の合法手判定
    """
    e = _exists(board, turn, src)
    o = _is_open(board, turn, tgt)
    return ((src >= 24) & e & o) | (
        (src < 24) & e & o & (board[_bar_idx(turn)] == 0)
    )  # type: ignore


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


def _is_all_off(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる.
    """
    return board[_off_idx(turn)] * turn == 15  # type: ignore


def _calc_win_score(board: jnp.ndarray, turn: jnp.ndarray) -> int:
    """
    通常勝ち: 1点
    gammon勝ち: 2点
    backgammon勝ち: 3点
    """
    g = _is_gammon(board, turn)
    return 1 + g + (g & _remains_at_inner(board, turn))


def _is_gammon(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがなければgammon勝ち
    """
    return board[_off_idx(-1 * turn)] == 0  # type: ignore


def _remains_at_inner(board: jnp.ndarray, turn: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがない && 相手のcheckerが一つでも自分のインナーに残っている
    => backgammon勝ち
    """
    return jnp.take(board, _home_board(-1 * turn)).sum() != 0  # type: ignore


def _legal_action_mask(
    board: jnp.ndarray, turn: jnp.ndarray, dice: jnp.ndarray
) -> jnp.ndarray:

    legal_action_mask = jax.vmap(
        partial(_legal_action_mask_for_single_die, board=board, turn=turn)
    )(die=dice).any(
        axis=0
    )  # (26*6 + 6)
    return legal_action_mask


def _legal_action_mask_for_single_die(
    board: jnp.ndarray, turn: jnp.ndarray, die
) -> jnp.ndarray:
    """
    一つのサイコロの目に対するlegal micro action
    """
    return (die == -1) * jnp.zeros(26 * 6 + 6, dtype=jnp.int16) + (
        die != -1
    ) * _legal_action_mask_for_valid_single_dice(board, turn, die)


def _legal_action_mask_for_valid_single_dice(
    board: jnp.ndarray, turn: jnp.ndarray, die
) -> jnp.ndarray:
    """
    -1以外のサイコロの目に対して合法判定
    """
    src_indices = jnp.arange(
        26, dtype=jnp.int16
    )  # 26パターンのsrcに対してlegal_actionを求める.

    def _is_legal(idx: jnp.ndarray):
        action: int = idx * 6 + die
        legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=jnp.int16)
        legal_action_mask = legal_action_mask.at[action].set(
            _is_action_legal(board, turn, action)
        )
        return legal_action_mask

    legal_action_mask = jax.vmap(_is_legal)(src_indices).any(
        axis=0
    )  # (26*6 + 6)
    return legal_action_mask
