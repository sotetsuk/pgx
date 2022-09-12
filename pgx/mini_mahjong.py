import numpy as np
from typing import Dict, List

from shanten_tools import shanten

np.random.seed(0)

# TODO: 赤牌
# TODO: リーチ
# TODO: フリテン
# TODO: 槓
# TODO: 役
# TODO: 西入


class Deck:
    def __init__(self):
        # TODO: 赤牌
        self.deck = np.array([i//4 for i in range(136)])
        np.random.shuffle(self.deck)
        self.idx = 0
        self.end = 136 - 14

    def draw(self):
        assert self.idx < self.end
        tile = self.deck[self.idx]
        self.idx += 1
        return tile

    def is_empty(self):
        return self.idx == self.end


class Hand:
    def __init__(self):
        self.hand = np.zeros(34, dtype=np.uint8)

    def size(self):
        return np.sum(self.hand)

    def add(self, tile: int):
        assert self.hand[tile] < 4
        self.hand[tile] += 1

    def sub(self, tile: int):
        assert self.hand[tile] > 0
        self.hand[tile] -= 1

    def to_indices(self):
        return np.where(self.hand > 0)[0].tolist()

    def can_ron(self, tile: int):
        assert self.hand[tile] < 4
        self.hand[tile] += 1
        can_ron = shanten(self.hand) == -1
        self.hand[tile] -= 1
        return can_ron

    def can_tsumo(self):
        can_tsumo = shanten(self.hand) == -1
        return can_tsumo

    def can_pon(self, tile: int):
        return self.hand[tile] >= 2

    def legal_chis(self, tile: int):
        if tile >= 27:
            return []

        chis = []
        if tile % 9 > 1 and self.hand[tile-2] > 0 and self.hand[tile-1] > 0:
            chis.append(CHI_R)
        if tile % 9 > 0 and tile % 9 < 8 and self.hand[tile-1] > 0 and self.hand[tile+1] > 0:
            chis.append(CHI_M)
        if tile % 9 < 7 and self.hand[tile+1] > 0 and self.hand[tile+2] > 0:
            chis.append(CHI_L)

        return chis

    def pon(self, tile: int):
        assert self.hand[tile] >= 2
        self.hand[tile] -= 2

    def chi(self, tile: int, chi: int):
        assert tile < 27
        if chi == CHI_R:
            assert tile % 9 > 1
            assert self.hand[tile-2] > 0
            assert self.hand[tile-1] > 0
            self.hand[tile-2] -= 1
            self.hand[tile-1] -= 1
        if chi == CHI_M:
            assert tile % 9 > 0 and tile % 9 < 8
            assert self.hand[tile-1] > 0
            assert self.hand[tile+1] > 0
            self.hand[tile-1] -= 1
            self.hand[tile+1] -= 1
        if chi == CHI_L:
            assert tile % 9 < 7
            assert self.hand[tile+1] > 0
            assert self.hand[tile+2] > 0
            self.hand[tile+1] -= 1
            self.hand[tile+2] -= 1



# Event/Action
# 0~33: discard
# 34: ron
# 35: pon
# 36,37,38: chi
# 39: pass(action only)
# 40: tsumo agari

RON = 34
PON = 35
CHI_R = 36  # 45[6]
CHI_M = 37  # 4[5]6
CHI_L = 38  # [4]56
PASS = 39
TSUMO = 40


class MiniMahjong:
    def __init__(self):
        self.ba = 0
        self.kyoku = 0
        self.honba = 0
        self.tens = np.full(4, 25000, dtype=np.int32)
        self._reset_round()

    def _reset_round(self):
        if self.is_terminal():
            return

        print('{}{}局{}本場'.format(
            '東' if self.ba == 0 else '南',
            self.kyoku + 1,
            self.honba
            ))

        self.deck = Deck()
        self.hand = [Hand() for _ in range(4)]
        for i in range(4):
            for j in range(13):
                self.hand[i].add(self.deck.draw())

        self.turn = self.kyoku
        # 手牌が3n+2枚のplayer.
        # 存在しなければ直前にdiscardしたplayer.

        self.target = None
        # 直前に捨てられた牌.
        # TODO: 加槓も対象にする.

        self._draw()

    def is_terminal(self):
        return self.ba == 2


    def legal_actions(self) -> Dict[int,int]:
        if self.hand[self.turn].size() % 3 == 2:
            # 手牌が3n+2
            return {
                    self.turn: self.hand[self.turn].to_indices() + \
                    self._legal_tsumo()
                    }
            # TODO: 暗/加槓の追加
            # TODO: 喰い変えNGによる捨て牌制限

        # self.turn がdiscard した直後
        # TODO: 暗/加槓の直後の場合の処理

        ret = {}
        for player in range(4):
            if player == self.turn:
                continue
            actions = self._legal_ron(player) + \
                      self._legal_pon(player) + \
                      self._legal_chis(player)
            # TODO: 明槓の追加

            if len(actions) > 0:
                actions.append(PASS)
                ret[player] = actions

        return ret


    def step(self, actions: Dict[int,int]):
        assert len(actions) > 0

        legal_actions = self.legal_actions()
        assert len(legal_actions) > 0

        for player in actions:
            assert actions[player] in legal_actions[player]

        # action(int) の小さいものが優先される.
        player, action = min(
                filter(lambda x:x[1] is not None, actions.items()),
                key=lambda x:x[1])
        self._step(player, action)

    def _step(self, player: int, action: int):
        if action < 34:
            # discard
            assert player == self.turn
            self._discard(action)
            # 割り込みがなければ次の人のdrawまで進む
            if len(self.legal_actions()) == 0:
                self.turn += 1
                self.turn %= 4
                self.target = None
                if self.deck.is_empty():
                    self._ryukyoku()
                else:
                    self._draw()

        elif action == 34: 
            # ron
            self._ron(player)

        elif action == 35:
            # pon
            self._pon(player)

        elif action < 39:
            # chi
            self._chi(player, action)

        elif action == 39:
            # pass
            # 次の人のdrawまで進む
            self.turn += 1
            self.turn %= 4
            self.target = None
            if self.deck.is_empty():
                self._ryukyoku()
            else:
                self._draw()

        elif action == 40:
            # tsumo
            assert player == self.turn
            self._tsumo()

    def _ryukyoku(self):
        print('流局')
        print('-' * 20)
        is_tenpai = np.array([shanten(self.hand[i].hand) == 0 for i in range(4)])
        is_not_tenpai = np.array([shanten(self.hand[i].hand) > 0 for i in range(4)])

        # 点棒移動
        if sum(is_tenpai) == 1:
            self.tens[is_tenpai] += 3000
            self.tens[is_not_tenpai] -= 1000

        if sum(is_tenpai) == 2:
            self.tens[is_tenpai] += 1500
            self.tens[is_not_tenpai] -= 1500

        if sum(is_tenpai) == 3:
            self.tens[is_tenpai] += 1000
            self.tens[is_not_tenpai] -= 3000

        if is_tenpai[self.kyoku]:
            # 連チャン
            self.honba += 1
        else:
            self.kyoku += 1
            self.honba += 1

        if self.kyoku == 4:
            self.kyoku = 0
            self.ba += 1

        if self.ba == 2:
            print('終局')
            return

        self._reset_round()

    def _draw(self):
        self.hand[self.turn].add(self.deck.draw())

    def _discard(self, tile: int):
        self.hand[self.turn].sub(tile)
        self.target = tile

    def _tsumo(self):
        print('自摸')
        print('self.hand[self.turn].hand:', self.hand[self.turn].hand)
        print('-' * 20)

        # 一律30符1翻
        if player == self.kyoku:
            self.tens[np.arange(4) != self.turn] -= 500 + self.honba * 100
            self.tens[self.turn] += 1500 + self.honba * 300
        else:
            self.tens[
                    (np.arange(4) != self.turn) * (np.arange(4) != self.kyoku)
                    ] -= 300 + self.honba * 100
            self.tens[self.kyoku] -= 500 + self.honba * 100
            self.tens[self.turn] += 1100 + self.honba * 300

        if player == self.kyoku:
            # 連チャン
            self.honba += 1
        else:
            self.kyoku += 1
            self.honba = 0

        if self.kyoku == 4:
            self.kyoku = 0
            self.ba += 1

        if self.ba == 2:
            print('終局')
            return

        self._reset_round()

    def _ron(self, player: int):
        print('ロン')
        print('self.hand[player].hand:', self.hand[player].hand)
        print('self.target:', self.target)
        print('-' * 20)

        # 一律30符1翻
        ten = 1500 if player == self.kyoku else 1000
        ten += self.honba * 300

        self.tens[self.turn] -= ten
        self.tens[player] += ten

        if player == self.kyoku:
            # 連チャン
            self.honba += 1
        else:
            self.kyoku += 1
            self.honba = 0

        if self.kyoku == 4:
            self.kyoku = 0
            self.ba += 1

        if self.ba == 2:
            print('終局')
            return

        self._reset_round()

    def _pon(self, player: int):
        self.hand[player].pon(self.target)
        self.target = None
        self.turn = player

    def _chi(self, player: int, chi: int):
        self.hand[player].chi(self.target, chi)
        self.target = None
        self.turn = player

    def _legal_tsumo(self):
        if self.hand[self.turn].can_tsumo():
            return [TSUMO]
        return []

    def _legal_ron(self, player: int):
        if self.target is None:
            return []
        if self.hand[player].can_ron(self.target):
            return [RON]
        return []

    def _legal_pon(self, player: int):
        if self.target is None:
            return []
        if self.hand[player].can_pon(self.target):
            return [PON]
        return []

    def _legal_chis(self, player: int):
        if self.target is None:
            return []
        return self.hand[player].legal_chis(self.target)

    def observation(self, player: int):
        # TODO: 自分のhand以外の情報
        return {
                'hand': self.hand[player].hand,
                'target': self.target
                }


class BasicAgent:
    def act(self, actions: List[int], obs: Dict) -> int:
        if TSUMO in actions:
            return TSUMO
        if RON in actions:
            return RON

        if np.sum(obs['hand']) % 3 == 2:
            # discard
            # shanten数が小さくなるように選択
            min_shanten = 9999
            discard = None
            for tile in actions:
                obs['hand'][tile] -= 1
                s = shanten(obs['hand'])
                if s < min_shanten:
                    s = min_shanten
                    discard = tile
                obs['hand'][tile] += 1
            return discard

        target = obs['target']
        
        if PON in actions:
            obs['hand'][target] -= 2
            s = shanten(obs['hand'])
            obs['hand'][target] += 2
            if s < shanten(obs['hand']):
                return PON

        if CHI_R in actions:
            obs['hand'][target-2] -= 1
            obs['hand'][target-1] -= 1
            s = shanten(obs['hand'])
            obs['hand'][target-2] += 1
            obs['hand'][target-1] += 1
            if s < shanten(obs['hand']):
                return CHI_R

        if CHI_M in actions:
            obs['hand'][target-1] -= 1
            obs['hand'][target+1] -= 1
            s = shanten(obs['hand'])
            obs['hand'][target-1] += 1
            obs['hand'][target+1] += 1
            if s < shanten(obs['hand']):
                return CHI_M

        if CHI_L in actions:
            obs['hand'][target+1] -= 1
            obs['hand'][target+2] -= 1
            s = shanten(obs['hand'])
            obs['hand'][target+1] += 1
            obs['hand'][target+2] += 1
            if s < shanten(obs['hand']):
                return CHI_L

        return PASS
        


if __name__ == '__main__':
    env = MiniMahjong()
    agent = BasicAgent()

    while not env.is_terminal():
        selected = {}
        for player, actions in env.legal_actions().items():
            selected[player] = agent.act(actions, env.observation(player))

        env.step(selected)

    print(env.tens)

    # print(env.legal_actions())
    # print(env.observation(0))
    # print(agent.act(
    #     env.legal_actions()[0],
    #     env.observation(0)
    #     ))

    # env.step({0:28})

    # print(env.legal_actions())
    # print(env.observation(2))
    # print(agent.act(
    #     env.legal_actions()[2],
    #     env.observation(2)
    #     ))
    # env.step({2:35})

    # print(env.legal_actions())
    # print(env.observation(2))
    # print(agent.act(
    #     env.legal_actions()[2],
    #     env.observation(2)
    #     ))
    # env.step({})

    #env.step({0:2})
    #print(env.legal_actions())
    #print(env.hand[2].hand)
    #env.step({2:36})
    #print(env.hand[2].hand)
    #print(env.legal_actions())
    #env.step({2:28})
    #print(env.legal_actions())
    #print(env.hand[3].hand)
    #env.step({3:0})
    #print(env.hand[3].hand)
    #print(env.legal_actions())
    #print(env.observation(0))
