from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, tree_util


class Tile:
    @staticmethod
    def to_str(tile: int) -> str:
        suit, num = tile // 9, tile % 9 + 1
        return str(num) + ["m", "p", "s", "z"][suit]


class Action:
    # 手出し: 0~33
    RON = 34
    PON = 35
    CHI_R = 36  # 45[6]
    CHI_M = 37  # 4[5]6
    CHI_L = 38  # [4]56
    PASS = 39
    TSUMO = 40
    TSUMOGIRI = 41
    RIICHI = 42
    NONE = 43


@dataclass
class Deck:
    idx: int
    arr: jnp.ndarray

    @jit
    def is_empty(self) -> bool:
        return self.size() == 0

    @jit
    def size(self) -> int:
        return 122 - self.idx

    def _tree_flatten(self):
        children = (self.idx, self.arr)
        aux_data = {}
        return (children, aux_data)

    @staticmethod
    @jit
    def init(key: jnp.ndarray) -> Deck:
        arr = jax.random.permutation(key, jnp.arange(136) // 4)
        return Deck(0, arr)

    @staticmethod
    @jit
    def draw(deck: Deck) -> tuple[Deck, int]:
        # assert not deck.is_empty()
        tile = deck.arr[deck.idx]
        deck.idx += 1
        return deck, tile

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


class Hand:
    # fmt: off
    CACHE_SUITED = jnp.array(
        [1385955477, 2569143448, 0, 0, 3686684160, 2569143708, 5413232, 1385976476, 2569142272, 2569143448, 9961984, 0, 301312, 39200, 2569143440, 2569143448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14326640, 1385976476, 3686719232, 2569182652, 2574425584, 3686684316, 2579103744, 2569143512, 1385976464, 0, 301312, 39200, 2569143440, 421659800, 0, 1386011580, 2569143696, 2569143448, 5413120, 1385976476, 0, 0, 3686662144, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 0, 0, 301312, 39200, 2569143440, 2569143448, 2569142272, 2569143448, 9961984, 0, 301312, 39200, 2569143440, 421659800, 0, 4, 39200, 0, 2569143296, 2569143448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2569143440, 2569143448, 9961984, 0, 0, 0, 262144, 39200, 2569143440, 421659800, 0, 0, 0, 0, 0, 2569143448, 9961984, 0, 301312, 39200, 421659792, 287442072, 2569142272, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1386011580, 2569182640, 2569143448, 2574425344, 3686684316, 0, 0, 3686662144, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 2583338992, 3686684316, 3686719232, 2569182652, 2574425584, 1539200668, 2579365888, 2569182712, 3686684304, 421659800, 301312, 39200, 421659792, 287442072, 0, 3686719420, 2579105680, 421659800, 5675264, 1386011580, 421659792, 287442072, 3686662144, 421660060, 14326640, 303846044, 1539235584, 287448508, 292691440, 18224668, 0, 0, 0, 0, 2569444608, 2569182648, 2569143440, 421659800, 2574385152, 3686684316, 9961984, 0, 301312, 39200, 421659792, 287442072, 0, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2569143440, 421659800, 9961984, 0, 0, 0, 262144, 39200, 421659792, 287442072, 0, 0, 0, 0, 0, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 421658624, 287442072, 1573376, 0, 268544, 4384, 16909328, 16, 0, 0, 0, 0, 0, 0, 0, 0, 3686662144, 2569182652, 2574425584, 1539200668, 3686684160, 421660060, 5413232, 303846044, 0, 2569143512, 1385976464, 0, 301312, 39200, 421659792, 287442072, 2569142272, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 0, 0, 2569143696, 421659800, 5413120, 1385976476, 0, 0, 3686662144, 421660060, 5413232, 303846044, 1539200512, 287442332, 5380464, 1315356, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3686719232, 421699004, 426941936, 322852508, 2579103744, 421659864, 1385976464, 0, 301312, 6432, 287442064, 16909336, 0, 1386011580, 421660048, 287442072, 5413120, 303846044, 0, 0, 1539178496, 287442332, 5380464, 1315356, 322852352, 16909596, 1184112, 20, 0, 0, 14326640, 303846044, 1539235584, 287448508, 292691440, 18224668, 431620096, 287442136, 303846032, 0, 268544, 4384, 16909328, 16, 0, 303848380, 287442320, 16909336, 5380352, 1315356, 0, 0, 322830336, 16909596, 1184112, 20, 18224640, 276, 272]
        , dtype=jnp.uint32,
    )
    # fmt: on
    CACHE_HONOR = jnp.array([1, 0, 1, 1, 0])

    @staticmethod
    @jit
    def cache_suited(code: int) -> int:
        return (Hand.CACHE_SUITED[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    @jit
    def can_ron(hand: jnp.ndarray, tile: int) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        # assert hand[tile] < 4
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    @jit
    def can_riichi(hand: jnp.ndarray) -> bool:
        # assert: hand is menzen
        return jax.lax.fori_loop(
            0,
            34,
            lambda i, sum: jax.lax.cond(
                hand[i] == 0,
                lambda: sum,
                lambda: sum | Hand.is_tenpai(Hand.sub(hand, i)),
            ),
            False,
        )

    @staticmethod
    @jit
    def is_tenpai(hand: jnp.ndarray) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        return jax.lax.fori_loop(
            0,
            34,
            lambda tile, sum: jax.lax.cond(
                hand[tile] == 4,
                lambda: False,
                lambda: sum | Hand.can_ron(hand, tile),
            ),
            False,
        )

    @staticmethod
    @jit
    def can_tsumo(hand: jnp.ndarray) -> bool:
        # assert jnp.sum(hand) % 3 == 2
        heads = 0
        valid = True

        for i in range(3):
            heads, valid, code, size = jax.lax.fori_loop(
                0,
                9,
                lambda j, tpl: jax.lax.cond(
                    hand[9 * i + j] == 0,
                    lambda: (
                        tpl[0] + (tpl[3] % 3 == 2),
                        tpl[1] & (Hand.cache_suited(tpl[2]) != 0),
                        0,
                        0,
                    ),
                    lambda: (
                        tpl[0],
                        tpl[1],
                        ((tpl[2] << 1) + 1)
                        << (hand[9 * i + j].astype(int) - 1),
                        tpl[3] + hand[9 * i + j].astype(int),
                    ),
                ),
                (heads, valid, 0, 0),
            )

            heads += size % 3 == 2
            valid &= Hand.cache_suited(code) != 0

        heads, valid = jax.lax.fori_loop(
            27,
            34,
            lambda i, tpl: (
                tpl[0] + (hand[i] % 3 == 2),
                tpl[1] & (Hand.CACHE_HONOR[hand[i]] != 0),
            ),
            (heads, valid),
        )

        return valid & (heads == 1)

    @staticmethod
    @jit
    def can_pon(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] >= 2

    @staticmethod
    @jit
    def can_chi(hand: jnp.ndarray, tile: int, action: int) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        # assert action is Action.CHI_R, CHI_M or CHI_L
        return jax.lax.switch(
            action - Action.CHI_R,
            [
                lambda: (
                    (tile < 27)
                    & (tile % 9 > 1)
                    & (hand[tile - 2] > 0)
                    & (hand[tile - 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 > 0)
                    & (tile % 9 < 8)
                    & (hand[tile - 1] > 0)
                    & (hand[tile + 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 < 7)
                    & (hand[tile + 1] > 0)
                    & (hand[tile + 2] > 0)
                ),
            ],
        )

    @staticmethod
    @jit
    def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        # assert 0 <= hand[tile] + x <= 4
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    @jit
    def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        # assert 0 <= hand[tile] - x <= 4
        return Hand.add(hand, tile, -x)

    @staticmethod
    @jit
    def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        # assert Hand.can_pon(hand, tile)
        return Hand.sub(hand, tile, 2)

    @staticmethod
    @jit
    def chi(hand: jnp.ndarray, tile: int, action: int) -> jnp.ndarray:
        # assert Hand.can_chi(hand, tile, action)
        # assert action is Action.CHI_R, CHI_M or CHI_L
        return jax.lax.switch(
            action - Action.CHI_R,
            [
                lambda: Hand.sub(Hand.sub(hand, tile - 2), tile - 1),
                lambda: Hand.sub(Hand.sub(hand, tile - 1), tile + 1),
                lambda: Hand.sub(Hand.sub(hand, tile + 1), tile + 2),
            ],
        )

    @staticmethod
    def to_str(hand: jnp.ndarray) -> str:
        s = ""
        for i in range(4):
            t = ""
            for j in range(9 if i < 3 else 7):
                t += str(j + 1) * hand[9 * i + j]
            if t:
                t += ["m", "p", "s", "t"][i]
            s += t
        return s


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int
    last_draw: int


class Meld:
    @staticmethod
    @jit
    def init(action: int, target: int, src: int) -> int:
        return (src << 12) | (target << 6) | action

    @staticmethod
    def to_str(meld: int) -> str:
        action = Meld.action(meld)
        target = Meld.target(meld)
        suit, num = target // 9, target % 9 + 1
        if action == Action.PON:
            return "{}{}{}{}".format(num, num, num, ["m", "p", "s", "z"][suit])
        if action == Action.CHI_R:
            return "{}{}{}{}".format(
                num - 2, num - 1, num, ["m", "p", "s", "z"][suit]
            )
        if action == Action.CHI_M:
            return "{}{}{}{}".format(
                num - 1, num, num + 1, ["m", "p", "s", "z"][suit]
            )
        if action == Action.CHI_L:
            return "{}{}{}{}".format(
                num, num + 1, num + 2, ["m", "p", "s", "z"][suit]
            )
        return ""

    @staticmethod
    @jit
    def target(meld: int) -> int:
        return (meld >> 6) & 0b111111

    @staticmethod
    @jit
    def action(meld: int) -> int:
        return meld & 0b111111


@dataclass
class State:
    deck: Deck
    hand: jnp.ndarray
    turn: int  # 手牌が3n+2枚, もしくは直前に牌を捨てたplayer
    target: int  # 直前に捨てられてron,pon,chi の対象になっている牌. 存在しなければ-1
    last_draw: int  # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    riichi_declared: bool  # state.turn がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi: jnp.ndarray  # 各playerのリーチが成立しているかどうか
    meld_num: jnp.ndarray  # 各playerの副露回数
    melds: jnp.ndarray
    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0

    # reward:
    # - player0 がplayer1 からロン => [ 2,-2, 0, 0]
    # - player0 がツモ             => [ 2,-2,-2,-2]
    # - 流局時 全員聴牌            => [ 1, 1, 1, 1]
    # - 流局時 全員ノー聴          => [-1,-1,-1,-1]
    # - 流局時 player0 だけ聴牌    => [ 1,-1,-1,-1]

    @jit
    def legal_actions(self) -> jnp.ndarray:
        legal_actions = jnp.full((4, 43), False)

        # リーチ
        legal_actions = jax.lax.cond(
            (self.last_draw == -1)
            | self.riichi_declared
            | self.riichi[self.turn]
            | (self.deck.size() < 4)
            # | (jnp.sum(self.hand[self.turn]) < 14),
            | (self.meld_num[self.turn]),
            lambda: legal_actions,
            lambda: legal_actions.at[(self.turn, Action.RIICHI)].set(
                Hand.can_riichi(self.hand[self.turn])
            ),
        )

        # リーチ宣言直後の打牌
        legal_actions = jax.lax.cond(
            (self.last_draw != -1) & self.riichi_declared,
            lambda arr: jax.lax.fori_loop(
                0,
                34,
                lambda i, arr: arr.at[(self.turn, i)].set(
                    jax.lax.cond(
                        self.hand[self.turn][i] > (i == self.last_draw),
                        lambda: Hand.is_tenpai(
                            Hand.sub(self.hand[self.turn], i)
                        ),
                        lambda: False,
                    )
                ),
                arr,
            )
            .at[(self.turn, Action.TSUMOGIRI)]
            .set(
                Hand.is_tenpai(Hand.sub(self.hand[self.turn], self.last_draw))
            ),
            lambda arr: arr,
            legal_actions,
        )

        # ツモ切り, ツモ
        legal_actions = jax.lax.cond(
            (self.last_draw == -1) | self.riichi_declared,
            lambda: legal_actions,
            lambda: legal_actions.at[(self.turn, Action.TSUMOGIRI)]
            .set(True)
            .at[(self.turn, Action.TSUMO)]
            .set(Hand.can_tsumo(self.hand[self.turn])),
        )

        # 手出し, 鳴いた後の手出し
        legal_actions = jax.lax.cond(
            (self.target != -1)
            | self.riichi_declared
            | self.riichi[self.turn],
            lambda arr: arr,
            lambda arr: jax.lax.fori_loop(
                0,
                34,
                lambda i, arr: arr.at[(self.turn, i)].set(
                    self.hand[self.turn][i] > (i == self.last_draw),
                ),
                arr,
            ),
            legal_actions,
        )

        for player in range(4):
            # ロン
            legal_actions = jax.lax.cond(
                (self.target == -1) | (player == self.turn),
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.RON)].set(
                    Hand.can_ron(self.hand[player], self.target)
                ),
            )
            # ポン
            legal_actions = jax.lax.cond(
                (self.target == -1)
                | (player == self.turn)
                | self.deck.is_empty()
                | self.riichi[player],
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.PON)].set(
                    Hand.can_pon(self.hand[player], self.target)
                ),
            )
            # チー
            legal_actions = jax.lax.cond(
                (self.target == -1)
                | (player != (self.turn + 1) % 4)
                | self.deck.is_empty()
                | self.riichi[player],
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.CHI_R)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_R)
                )
                .at[(player, Action.CHI_M)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_M)
                )
                .at[(player, Action.CHI_L)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_L)
                ),
            )
            legal_actions = legal_actions.at[(player, Action.PASS)].set(
                (player != self.turn) & jnp.any(legal_actions[player])
            )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand[player], self.target, self.last_draw)

    @staticmethod
    @jit
    def init(key) -> State:
        deck = Deck.init(key)
        hand = jnp.zeros((4, 34), dtype=jnp.uint8)

        for i in range(4):
            for _ in range(13):
                deck, tile = Deck.draw(deck)
                hand = hand.at[i].set(Hand.add(hand[i], tile))

        deck, tile = Deck.draw(deck)
        hand = hand.at[0].set(Hand.add(hand[0], tile))

        turn = 0
        target = -1
        last_draw = tile
        riichi_declared = False
        riichi = jnp.full(4, False)
        meld_num = jnp.zeros(4, dtype=jnp.uint8)
        melds = jnp.zeros((4, 4), dtype=jnp.uint32)
        return State(
            deck,
            hand,
            turn,
            target,
            last_draw,
            riichi_declared,
            riichi,
            meld_num,
            melds,
        )

    @staticmethod
    @jit
    def step(
        state: State, actions: jnp.ndarray
    ) -> tuple[State, jnp.ndarray, bool]:
        player = jnp.argmin(actions)
        return State._step(state, player, actions[player])

    @staticmethod
    @jit
    def _step(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        return jax.lax.cond(
            action < 34,
            lambda: State._discard(state, action),
            lambda: jax.lax.switch(
                action - 34,
                [
                    lambda: State._ron(state, player),
                    lambda: State._pon(state, player),
                    lambda: State._chi(state, player, Action.CHI_R),
                    lambda: State._chi(state, player, Action.CHI_M),
                    lambda: State._chi(state, player, Action.CHI_L),
                    lambda: State._try_draw(state),
                    lambda: State._tsumo(state),
                    lambda: State._tsumogiri(state),
                    lambda: State._riichi(state),
                ],
            ),
        )

    @staticmethod
    @jit
    def _tsumogiri(state: State) -> tuple[State, jnp.ndarray, bool]:
        return State._discard(state, state.last_draw)

    @staticmethod
    @jit
    def _discard(state: State, tile: int) -> tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[state.turn].set(
            Hand.sub(state.hand[state.turn], tile)
        )
        state.target = tile
        state.last_draw = -1
        return jax.lax.cond(
            jnp.any(state.legal_actions()),
            lambda: (state, jnp.full(4, 0), False),
            lambda: State._try_draw(state),
        )

    @staticmethod
    @jit
    def _try_draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.target = -1
        return jax.lax.cond(
            state.deck.is_empty(),
            lambda: State._ryukyoku(state),
            lambda: State._draw(state),
        )

    @staticmethod
    @jit
    def _accept_riichi(state: State) -> State:
        state.riichi = state.riichi.at[state.turn].set(
            state.riichi[state.turn] | state.riichi_declared
        )
        state.riichi_declared = False
        return state

    @staticmethod
    @jit
    def _draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        state.turn += 1
        state.turn %= 4
        state.deck, tile = Deck.draw(state.deck)
        state.last_draw = tile
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _ryukyoku(state: State) -> tuple[State, jnp.ndarray, bool]:
        reward = jnp.array(
            [2 * Hand.is_tenpai(state.hand[i]) - 1 for i in range(4)]
        )
        return state, reward, True

    @staticmethod
    @jit
    def _ron(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        return (
            state,
            jnp.full(4, 0).at[state.turn].set(-2).at[player].set(2),
            True,
        )

    @staticmethod
    @jit
    def _append_meld(state: State, meld: int, player: int) -> State:
        state.melds = state.melds.at[(player, state.meld_num[player])].set(
            meld
        )
        state.meld_num = state.meld_num.at[player].set(
            state.meld_num[player] + 1
        )
        return state

    @staticmethod
    @jit
    def _pon(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(Action.PON, state.target, state.turn)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.pon(state.hand[player], state.target)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _chi(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(action, state.target, state.turn)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.chi(state.hand[player], state.target, action)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _tsumo(state: State) -> tuple[State, jnp.ndarray, bool]:
        return state, jnp.full(4, -2).at[state.turn].set(2), True

    @staticmethod
    @jit
    def _riichi(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.riichi_declared = True
        return state, jnp.full(4, 0), False

    def _tree_flatten(self):
        children = (
            self.deck,
            self.hand,
            self.turn,
            self.target,
            self.last_draw,
            self.riichi_declared,
            self.riichi,
            self.meld_num,
            self.melds,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    State, State._tree_flatten, State._tree_unflatten
)


@jit
def init(key) -> State:
    return State.init(key)


@jit
def step(
    state: State, actions: jnp.ndarray
) -> tuple[State, jnp.ndarray, bool]:
    return State.step(state, actions)
