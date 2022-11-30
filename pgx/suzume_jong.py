"""Suzume-Jong

すずめ雀のルール
  * 2-6人
  * 牌はソウズと發中のみの11種44枚
  * 手牌は基本5枚、ツモ直後6枚
  * 順子か刻子を2つ完成で手牌完成
  * チーポンカンはなし
  * ドラは表示牌がそのままドラ
  * 中はすべてドラ、各牌ひとつはドラがある
  * フリテンは自分が捨てた牌はあがれないが、他の牌ではあがれる

Pgx実装での違い
  * 3人のみ
  * 一局のみ
  * 行動選択は打牌選択のみ
    * 打牌選択時、赤牌とそうでない牌がある場合には、常に赤牌でないほうを打牌する
    * ロン・ツモは自動判定

実装TODO
  * [x] フリテン
  * [x] ドラ
  * [x] 赤牌
  * [ ] 手牌点数計算
    * [x] 順子
    * [x] 刻子
    * [x] タンヤオ
    * [x] チャンタ
    * [x] チンヤオ
    * [x] all green
    * [ ] super red
    * [x] ドラ
    * [x] 赤ドラ
    * [x] 親
  * [ ] 点数移動計算（ロン・ツモ）
  * [ ] 5ポイント縛り
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import struct

NUM_TILES = 44
NUM_TILE_TYPES = 11
N_PLAYER = 3
MAX_RIVER_LENGTH = 10
NUM_CACHE = 160
WIN_HANDS = jnp.int32([18, 78, 90, 378, 390, 450, 778, 790, 850, 1150, 1550, 1878, 1890, 1950, 2250, 2650, 3878, 3890, 3950, 4250, 4650, 5750, 7750, 9378, 9390, 9450, 9750, 10150, 11250, 13250, 19378, 19390, 19450, 19750, 20150, 21250, 23250, 28750, 38750, 46878, 46890, 46950, 47250, 47650, 48750, 50750, 56250, 66250, 96878, 96890, 96950, 97250, 97650, 98750, 100750, 106250, 116250, 143750, 193750, 234378, 234390, 234450, 234750, 235150, 236250, 238250, 243750, 253750, 281250, 331250, 484378, 484390, 484450, 484750, 485150, 486250, 488250, 493750, 503750, 531250, 581250, 718750, 968750, 1171878, 1171890, 1171950, 1172250, 1172650, 1173750, 1175750, 1181250, 1191250, 1218750, 1268750, 1406250, 1656250, 2421878, 2421890, 2421950, 2422250, 2422650, 2423750, 2425750, 2431250, 2441250, 2468750, 2518750, 2656250, 2906250, 3593750, 4843750, 5859378, 5859390, 5859450, 5859750, 5860150, 5861250, 5863250, 5868750, 5878750, 5906250, 5956250, 6093750, 6343750, 7031250, 8281250, 12109378, 12109390, 12109450, 12109750, 12110150, 12111250, 12113250, 12118750, 12128750, 12156250, 12206250, 12343750, 12593750, 13281250, 14531250, 17968750, 24218750, 29296878, 29296890, 29296950, 29297250, 29297650, 29298750, 29300750, 29306250, 29316250, 29343750, 29393750, 29531250, 29781250, 30468750, 31718750, 35156250, 41406250])  # type: ignore
BASE_SCORES = jnp.int8([4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3])  # type: ignore
YAKU_SCORES = jnp.int8([15, 15, 15, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 10, 0, 10, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 10, 10, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # type: ignore


@struct.dataclass
class State:
    curr_player: jnp.ndarray = jnp.int8(0)
    legal_action_mask: jnp.ndarray = jnp.zeros(9, jnp.bool_)
    terminated: jnp.ndarray = jnp.bool_(False)
    turn: jnp.ndarray = jnp.int8(0)  # 0 = dealer
    rivers: jnp.ndarray = -jnp.ones(
        (N_PLAYER, MAX_RIVER_LENGTH), dtype=jnp.int8
    )  # tile type (0~10) is set
    last_discard: jnp.ndarray = jnp.int8(-1)  # tile type (0~10) is set
    hands: jnp.ndarray = jnp.zeros(
        (N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8
    )  # tile type (0~10) is set
    n_red_in_hands: jnp.ndarray = jnp.zeros(
        (N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8
    )
    is_red_in_river: jnp.ndarray = jnp.zeros(
        (N_PLAYER, MAX_RIVER_LENGTH), dtype=jnp.bool_
    )
    wall: jnp.ndarray = jnp.zeros(
        NUM_TILES, dtype=jnp.int8
    )  # tile id (0~43) is set
    draw_ix: jnp.ndarray = jnp.int8(N_PLAYER * 5)
    shuffled_players: jnp.ndarray = jnp.zeros(N_PLAYER)  # 0: dealer, ...
    dora: jnp.ndarray = jnp.int8(0)  # tile type (0~10) is set
    scores: jnp.ndarray = jnp.zeros(3, dtype=jnp.int8)  # 0 = dealer


# TODO: avoid Tenhou
@jax.jit
def init(rng: jax.random.KeyArray):
    # shuffle players and wall
    key1, key2 = jax.random.split(rng)
    shuffled_players = jnp.arange(N_PLAYER)
    shuffled_players = jax.random.shuffle(key1, shuffled_players)
    wall = jnp.arange(NUM_TILES, dtype=jnp.int8)
    wall = jax.random.shuffle(key2, wall)
    curr_player = shuffled_players[0]  # dealer
    dora = wall[-1] // 4
    # set hands (hands[0] = dealer's hand)
    hands = jnp.zeros((N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8)
    hands = lax.fori_loop(
        0, N_PLAYER * 5, lambda i, x: x.at[i // 5, wall[i] // 4].add(1), hands
    )
    n_red_in_hands = jnp.zeros((N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8)
    n_red_in_hands = lax.fori_loop(
        0,
        N_PLAYER * 5,
        lambda i, x: x.at[i // 5, wall[i] // 4].add(
            (wall[i] % 4 == 0) | (wall[i] >= 40)
        ),
        n_red_in_hands,
    )
    # first draw
    draw_ix = jnp.int8(N_PLAYER * 5)
    hands = hands.at[0, wall[draw_ix] // 4].add(1)
    draw_ix += 1
    legal_action_mask = hands[0] > 0
    state = State(
        curr_player=curr_player,
        legal_action_mask=legal_action_mask,
        hands=hands,
        n_red_in_hands=n_red_in_hands,
        wall=wall,
        draw_ix=draw_ix,
        shuffled_players=shuffled_players,
        dora=dora,
    )  # type: ignore
    return curr_player, state


@jax.jit
def _to_base5(hand: jnp.ndarray):
    b = jnp.int32(
        [9765625, 1953125, 390625, 78125, 15625, 3125, 625, 125, 25, 5, 1]
    )
    return (hand * b).sum()


@jax.jit
def _is_completed(hand: jnp.ndarray):
    return jnp.any(_to_base5(hand) == WIN_HANDS)


@jax.jit
def _hand_to_score(hand: jnp.ndarray):
    # behavior for incomplete hand is undefined
    ix = jnp.argmin(jnp.abs(WIN_HANDS - _to_base5(hand)))
    return BASE_SCORES[ix], YAKU_SCORES[ix]


@jax.jit
def _hands_to_score(state: State) -> jnp.ndarray:
    scores = jnp.zeros(3, dtype=jnp.int8)
    for i in range(N_PLAYER):
        hand = state.hands[i]
        hand = jax.lax.cond(
            hand.sum() == 5,
            lambda: hand.at[state.last_discard].add(1),
            lambda: hand,
        )
        bs, ys = _hand_to_score(hand)
        n_doras = hand[state.dora]
        n_red_doras = state.n_red_in_hands[i].sum().astype(jnp.int8)
        s = lax.cond(
            ys >= 10,  # yakuman
            lambda: bs + ys,
            lambda: bs + ys + n_doras + n_red_doras
        )
        scores = scores.at[i].set(s)
    scores = scores.at[0].add(2)
    return scores


@jax.jit
def _check_ron(state: State) -> jnp.ndarray:
    # TODO: furiten
    # TODO: 5-fan limit
    winning_players = jax.lax.fori_loop(
        0,
        N_PLAYER,
        lambda i, x: x.at[i].set(
            _is_completed(state.hands.at[i, state.last_discard].add(1)[i])
        ),
        jnp.zeros(N_PLAYER, dtype=jnp.bool_),
    )
    winning_players = winning_players.at[state.turn].set(False)
    is_furiten = (state.rivers == state.last_discard).sum(axis=1) > 0
    winning_players = winning_players & jnp.logical_not(is_furiten)
    return winning_players


@jax.jit
def _check_tsumo(state: State) -> jnp.ndarray:
    return _is_completed(state.hands[state.turn])


@jax.jit
def _order_by_player_idx(x, shuffled_players):
    return lax.fori_loop(
        0,
        N_PLAYER,
        lambda i, e: e.at[shuffled_players[i]].set(x[i]),
        jnp.zeros_like(x),
    )


@jax.jit
def _step_by_ron(state: State):
    winning_players = _check_ron(state)
    scores = _hands_to_score(state)
    scores = scores * winning_players
    scores = scores.at[state.turn % N_PLAYER].set(-scores.sum())
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
        scores=scores,
    )
    r = _order_by_player_idx(scores, state.shuffled_players)
    return curr_player, state, r


@jax.jit
def _step_by_tsumo(state: State):
    winner_score = _hands_to_score(state)[state.turn]
    loser_score = jnp.ceil(winner_score / (N_PLAYER - 1)).astype(jnp.int8)
    scores = -jnp.ones(N_PLAYER, dtype=jnp.int8) * loser_score
    scores = scores.at[state.turn % N_PLAYER].set(winner_score)
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
        scores=scores,
    )
    r = _order_by_player_idx(scores, state.shuffled_players)
    return curr_player, state, r


def _step_by_tie(state):
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
    )
    r = jnp.zeros(3, dtype=jnp.float16)
    return curr_player, state, r


def _draw_tile(state: State) -> State:
    turn = state.turn + 1
    curr_player = state.shuffled_players[turn % N_PLAYER]
    tile_id = state.wall[state.draw_ix]
    tile_type = tile_id // 4
    is_red = (tile_id % 4 == 0) | (tile_id >= 40)
    hands = state.hands.at[turn % N_PLAYER, tile_type].add(1)
    n_red_in_hands = state.n_red_in_hands.at[turn % N_PLAYER, tile_type].add(
        is_red
    )
    draw_ix = state.draw_ix + 1
    legal_action_mask = hands[turn % N_PLAYER] > 0
    state = state.replace(  # type: ignore
        turn=turn,
        curr_player=curr_player,
        hands=hands,
        n_red_in_hands=n_red_in_hands,
        draw_ix=draw_ix,
        legal_action_mask=legal_action_mask,
        terminated=jnp.bool_(False),
    )
    return state


def _step_non_terminal(state: State):
    r = jnp.zeros(3, dtype=jnp.float16)
    return state.curr_player, state, r


def _step_non_tied(state: State):
    state = _draw_tile(state)
    is_tsumo = _check_tsumo(state)
    if is_tsumo:
        return _step_by_tsumo(state)
    else:
        return _step_non_terminal(state)


def step(state: State, action: jnp.ndarray):
    # discard tile
    hands = state.hands.at[state.turn % N_PLAYER, action].add(-1)
    is_red_discarded = (
        hands[state.turn % N_PLAYER, action]
        < state.n_red_in_hands[state.turn % N_PLAYER, action]
    )
    n_red_in_hands = state.n_red_in_hands.at[
        state.turn % N_PLAYER, action
    ].add(-is_red_discarded.astype(jnp.int8))
    rivers = state.rivers.at[
        state.turn % N_PLAYER, state.turn // N_PLAYER
    ].set(action)
    is_red_in_river = state.is_red_in_river.at[
        state.turn % N_PLAYER, state.turn // N_PLAYER
    ].set(is_red_discarded)
    last_discard = action
    state = state.replace(  # type: ignore
        hands=hands,
        n_red_in_hands=n_red_in_hands,
        rivers=rivers,
        is_red_in_river=is_red_in_river,
        last_discard=last_discard,
    )

    win_players = _check_ron(state)
    if jnp.any(win_players):
        return _step_by_ron(state)
    else:
        if jnp.bool_(NUM_TILES - 1 <= state.draw_ix):
            return _step_by_tie(state)
        else:
            return _step_non_tied(state)


def _tile_type_to_str(tile_type) -> str:
    if tile_type < 9:
        s = str(tile_type + 1)
    elif tile_type == 9:
        s = "g"
    elif tile_type == 10:
        s = "r"
    return s


def _hand_to_str(hand: jnp.ndarray, n_red_in_hands: jnp.ndarray) -> str:
    s = ""
    for i in range(NUM_TILE_TYPES):
        for j in range(hand[i]):
            s += _tile_type_to_str(i)
            if j < n_red_in_hands[i]:
                s += "*"
            else:
                s += " "
    return s.ljust(12)


def _river_to_str(river: jnp.ndarray, is_red_in_river: jnp.ndarray) -> str:
    s = ""
    for i in range(MAX_RIVER_LENGTH):
        tile_type = river[i]
        if tile_type >= 0:
            s += _tile_type_to_str(tile_type)
            s += "*" if is_red_in_river[i] else " "
        else:
            s += "_ "
    return s.ljust(20)


def _to_str(state: State):
    s = f"{'[terminated]' if state. terminated else ''} dora: {_tile_type_to_str(state.dora)}\n"
    for i in range(N_PLAYER):
        s += f"{'*' if not state.terminated and state.turn % N_PLAYER == i else ' '}[{state.shuffled_players[i]}] "
        s += f"{_hand_to_str(state.hands[i], state.n_red_in_hands[i])}: "
        s += f"{_river_to_str(state.rivers[i], state.is_red_in_river[i])} "
        s += "\n"
    return s


def _validate(state: State) -> bool:
    if state.dora < 0 or 10 < state.dora:
        assert False
    if 10 < state.last_discard:
        assert False
    if state.last_discard < 0 and state.rivers[0, 0] >= 0:
        assert False
    if jnp.any(state.hands < 0):
        assert False
    counts = state.hands.sum(axis=0)
    for i in range(N_PLAYER):
        for j in range(MAX_RIVER_LENGTH):
            if state.rivers[i, j] >= 0:
                counts = counts.at[state.rivers[i, j]].add(1)
    if jnp.any(counts > 4):
        assert False

    # check legal_action_mask
    if not state.terminated:
        for i in range(NUM_TILE_TYPES):
            if (
                state.legal_action_mask[i]
                and state.hands[state.turn % N_PLAYER, i] <= 0
            ):
                assert (
                    False
                ), f"\n{state.legal_action_mask[i]}\n{state.hands[state.turn % N_PLAYER, i]}\n{_to_str(state)}"
            if (
                not state.legal_action_mask[i]
                and state.hands[state.turn % N_PLAYER, i] > 0
            ):
                assert (
                    False
                ), f"\n{state.legal_action_mask}\n{state.hands[state.turn % N_PLAYER]}\n{_to_str(state)}"

    if not jnp.all(state.n_red_in_hands[:, :-2] <= 1):
        assert False
    if (state.n_red_in_hands.sum() + state.is_red_in_river.sum()) > 14:
        assert (
            False
        ), f"\n{state.n_red_in_hands}\n{state.is_red_in_river}\n{_to_str(state)}"
    return True
