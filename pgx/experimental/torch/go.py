from functools import partial
from typing import NamedTuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from lax import select, cond, scan, switch, fori_loop, while_loop
from utils import At


class GameState(NamedTuple):
    step_count: Tensor = torch.tensor(0, dtype=torch.int32)
    chain_id_board: Tensor = torch.zeros(19 * 19, dtype=torch.int32)
    board_history: Tensor = torch.full((8, 19 * 19), 2, dtype=torch.int32)
    num_captured_stones: Tensor = torch.zeros(2, dtype=torch.int32)
    consecutive_pass_count: Tensor = torch.tensor(0, dtype=torch.int32)
    ko: Tensor = torch.tensor(-1, dtype=torch.int32)
    is_psk: Tensor = torch.tensor(False, dtype=torch.bool)

    @property
    def color(self) -> Tensor:
        return self.step_count % 2

    @property
    def size(self) -> Tensor:
        return torch.sqrt(torch.tensor(self.chain_id_board.shape[-1], dtype=torch.float32)).to(torch.int32)


class Game:
    def __init__(self, size: int = 19, komi: float = 7.5, history_length: int = 8, max_termination_steps: Optional[int] = None):
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = size * size * 2 if max_termination_steps is None else max_termination_steps

    def init(self) -> GameState:
        return GameState(
            chain_id_board=torch.zeros(self.size**2, dtype=torch.int32),
            board_history=torch.full((self.history_length, self.size**2), 2, dtype=torch.int32),
        )

    def step(self, state: GameState, action: Tensor) -> GameState:
        state = state._replace(ko=torch.tensor(-1, dtype=torch.int32))
        is_pass = torch.tensor(action == self.size * self.size)
        pass_count = select(is_pass, state.consecutive_pass_count + 1, torch.tensor(0, dtype=torch.int32))
        state = state._replace(consecutive_pass_count=pass_count)
        if not is_pass:
            state = _apply_action(state, action, self.size)
        state = state._replace(step_count=state.step_count + 1)
        board_history = torch.roll(state.board_history, self.size**2)
        board_history = At(board_history)[0].set(torch.clamp(state.chain_id_board, -1, 1).to(torch.int32))
        state = state._replace(board_history=board_history)
        state = state._replace(is_psk=_check_PSK(state))
        return state

    def observe(self, state: GameState, color: Optional[Tensor] = None) -> Tensor:
        if color is None:
            color = state.color

        my_color_sign = torch.tensor([1, -1], dtype=torch.int32)[color]

        def _make(i):
            c = torch.tensor([1, -1], dtype=torch.int32)[i % 2] * my_color_sign
            return state.board_history[i // 2] == c

        log = torch.vmap(_make)(torch.arange(self.history_length * 2))
        color = torch.full_like(log[0], color)

        return torch.stack([log, color]).transpose(0, 1).reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Tensor:
        is_empty = state.chain_id_board == 0

        my_color = _my_color(state)
        opp_color = _opponent_color(state)
        num_pseudo, idx_sum, idx_squared_sum = _count(state, self.size)

        chain_ix = torch.abs(state.chain_id_board) - 1
        in_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
        has_liberty = (state.chain_id_board * my_color > 0) & ~in_atari
        kills_opp = (state.chain_id_board * opp_color > 0) & in_atari

        def is_neighbor_ok(xy):
            neighbors = _neighbour(xy, self.size)
            on_board = neighbors != -1
            _has_empty = is_empty[neighbors]
            _has_liberty = has_liberty[neighbors]
            _kills_opp = kills_opp[neighbors]
            return (on_board & _has_empty).any() | (on_board & _kills_opp).any() | (on_board & _has_liberty).any()

        neighbor_ok = torch.vmap(is_neighbor_ok)(torch.arange(self.size**2))
        legal_action_mask = is_empty & neighbor_ok

        legal_action_mask = cond(
            (state.ko == -1),
            lambda: legal_action_mask,
            lambda: At(legal_action_mask)[state.ko].set(False),
        )
        return torch.cat([legal_action_mask, torch.tensor([True])])

    def is_terminal(self, state: GameState) -> Tensor:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Tensor:
        score = _count_point(state, self.size)
        rewards = select(
            score[0] - self.komi > score[1],
            torch.tensor([1, -1], dtype=torch.float32),
            torch.tensor([-1, 1], dtype=torch.float32),
        )
        to_play = state.color
        rewards = select(state.is_psk, At(torch.tensor([-1, -1], dtype=torch.float32))[to_play].set(1.0), rewards)
        rewards = select(self.is_terminal(state), rewards, torch.zeros(2, dtype=torch.float32))
        return rewards


def _apply_action(state: GameState, action, size) -> GameState:
    xy = action
    num_captured_stones_before = state.num_captured_stones[state.color]

    ko_may_occur = _ko_may_occur(state, xy, size)

    adj_xy = _neighbour(xy, size)
    oppo_color = _opponent_color(state)
    chain_id = state.chain_id_board[adj_xy]
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    chain_ix = torch.abs(chain_id) - 1
    is_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
    z = idx_sum[chain_ix]
    z = torch.where(z == 0, z + 1, z)
    single_liberty = (idx_squared_sum[chain_ix] // z) - 1
    is_killed = (adj_xy != -1) & (chain_id * oppo_color > 0) & is_atari & (single_liberty == xy)
    state = fori_loop(
        0,
        4,
        lambda i, s: _remove_stones(is_killed[i], s, chain_id[i], adj_xy[i], ko_may_occur),
        state,
    )
    state = _set_stone(state, xy)

    state = fori_loop(0, 4, lambda i, s: _merge_around_xy(i, s, xy, size), state)

    ko = select(
        state.num_captured_stones[state.color] - num_captured_stones_before == 1,
        state.ko,
        torch.tensor(-1, dtype=torch.int32),
    )

    return state._replace(ko=ko)


def _merge_around_xy(i, state: GameState, xy, size):
    my_color = _my_color(state)
    adj_xy = _neighbour(xy, size)[i]
    is_off = adj_xy == -1
    is_my_chain = state.chain_id_board[adj_xy] * my_color > 0
    chain_id_board = cond(
        ((~is_off) & is_my_chain),
        lambda: _merge_chain(state, xy, adj_xy),
        lambda: state.chain_id_board,
    )
    return state._replace(chain_id_board=chain_id_board)


def _set_stone(state: GameState, xy) -> GameState:
    my_color = _my_color(state)
    return state._replace(
        chain_id_board=At(state.chain_id_board)[xy].set((xy + 1) * my_color),
    )


def _merge_chain(state: GameState, xy, adj_xy):
    my_color = _my_color(state)
    new_id = torch.abs(state.chain_id_board[xy])
    adj_chain_id = torch.abs(state.chain_id_board[adj_xy])
    small_id = torch.minimum(new_id, adj_chain_id) * my_color
    large_id = torch.maximum(new_id, adj_chain_id) * my_color

    chain_id_board = torch.where(
        state.chain_id_board == large_id,
        small_id,
        state.chain_id_board,
    )

    return chain_id_board


def _remove_stones(killed, state: GameState, rm_chain_id, rm_stone_xy, ko_may_occur) -> GameState:
    surrounded_stones = state.chain_id_board == rm_chain_id
    num_captured_stones = torch.count_nonzero(surrounded_stones)
    chain_id_board = torch.where(surrounded_stones, torch.tensor(0, dtype=torch.int32), state.chain_id_board)
    ko = cond(
        ko_may_occur & (num_captured_stones == 1),
        lambda: torch.tensor(rm_stone_xy, dtype=torch.int32),
        lambda: state.ko,
    )
    return state._replace(
        chain_id_board=select(killed, chain_id_board, state.chain_id_board),
        num_captured_stones=select(killed, At(state.num_captured_stones)[state.color].add(num_captured_stones), state.num_captured_stones),
        ko=select(killed, ko, state.ko),
    )


def _count(state: GameState, size):
    ZERO = torch.tensor(0, dtype=torch.int32)
    chain_id_board = torch.abs(state.chain_id_board)
    is_empty = chain_id_board == 0
    idx_sum = torch.where(is_empty, torch.arange(1, size**2 + 1, dtype=torch.int32), ZERO)
    idx_squared_sum = torch.where(is_empty, torch.arange(1, size**2 + 1, dtype=torch.int32) ** 2, ZERO)

    def _count_neighbor(xy):
        neighbors = _neighbour(xy, size)
        on_board = neighbors != -1
        return (
            torch.where(on_board, is_empty[neighbors], ZERO).sum(),
            torch.where(on_board, idx_sum[neighbors], ZERO).sum(),
            torch.where(on_board, idx_squared_sum[neighbors], ZERO).sum(),
        )

    idx = torch.arange(size**2, dtype=torch.int32)
    num_pseudo, idx_sum, idx_squared_sum = torch.vmap(_count_neighbor)(idx)

    def _num_pseudo(x):
        return torch.where(chain_id_board == (x + 1), num_pseudo, ZERO).sum()

    def _idx_sum(x):
        return torch.where(chain_id_board == (x + 1), idx_sum, ZERO).sum()

    def _idx_squared_sum(x):
        return torch.where(chain_id_board == (x + 1), idx_squared_sum, ZERO).sum()

    return torch.vmap(_num_pseudo)(idx), torch.vmap(_idx_sum)(idx), torch.vmap(_idx_squared_sum)(idx)


def _my_color(state: GameState):
    return torch.tensor([1, -1], dtype=torch.int32)[state.color]


def _opponent_color(state: GameState):
    return torch.tensor([-1, 1], dtype=torch.int32)[state.color]


def _ko_may_occur(state: GameState, xy: int, size: int) -> Tensor:
    x = xy // size
    y = xy % size
    oob = torch.tensor([x - 1 < 0, x + 1 >= size, y - 1 < 0, y + 1 >= size], dtype=torch.bool)
    oppo_color = _opponent_color(state)
    is_occupied_by_opp = state.chain_id_board[_neighbour(xy, size)] * oppo_color > 0
    return (oob | is_occupied_by_opp).all()


def _neighbour(xy, size):
    dx = torch.tensor([-1, +1, 0, 0], dtype=torch.int32)
    dy = torch.tensor([0, 0, -1, +1], dtype=torch.int32)
    xs = xy // size + dx
    ys = xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return torch.where(on_board, xs * size + ys, -1)


def _neighbours(size):
    return torch.vmap(partial(_neighbour, size=size))(torch.arange(size**2, dtype=torch.int32))


def _check_PSK(state: GameState):
    not_passed = state.consecutive_pass_count == 0
    is_psk = not_passed & (torch.abs(state.board_history[0] - state.board_history[1:]).sum(axis=1) == 0).any()
    return is_psk


def _count_point(state: GameState, size):
    return torch.tensor(
        [
            _count_ji(state, 1, size) + torch.count_nonzero(state.chain_id_board > 0),
            _count_ji(state, -1, size) + torch.count_nonzero(state.chain_id_board < 0),
        ],
        dtype=torch.float32,
    )


def _count_ji(state: GameState, color: int, size: int):
    board = torch.zeros_like(state.chain_id_board)
    board = torch.where(state.chain_id_board * color > 0, 1, board)
    board = torch.where(state.chain_id_board * color < 0, -1, board)

    neighbours = _neighbours(size)

    def is_opp_neighbours(b):
        return (b == 0) & ((b[neighbours.flatten()] == -1).reshape(size**2, 4) & (neighbours != -1)).any(axis=1)

    def fill_opp(x):
        b, _ = x
        mask = is_opp_neighbours(b)
        return torch.where(mask, -1, b), mask.any()

    b, _ = while_loop(lambda x: x[1], fill_opp, (board, True))

    return (b == 0).sum()


if __name__ == '__main__':
    actions = [45, 49, 17, 12, 22, 29, 31, 28, 46, 41, 33, 3, 8, 80, 13, 57, 48, 56, 7, 9, 75, 71, 58, 78, 23, 66, 77, 81, 79, 27, 47, 16, 72, 36, 21, 62, 14, 68, 54, 2, 5, 53, 76, 6, 64, 37, 15, 43, 70, 52, 40, 0, 74, 44, 10, 59, 60, 67, 63, 51, 25, 50, 18, 34, 11, 55, 6, 26, 39, 58, 32, 35, 16, 61, 73, 19, 30, 1, 81, 38, 42, 18, 81, 81]
    expected = [-1, -1, -1, -1, 0, 6, 6, 6, 6, -1, 11, 11, -1, 6, 6, 6, 6, 6, -1, -1, 0, 6, 6, 6, 0, 6, -27, -1, -1, -1, 6, 6, 6, 6, -27, -27, -1, -1, -1, 6, 6, -27, 6, -27, -27, 6, 6, 6, 6, -27, -27, -27, -27, -27, 6, -27, -27, -27, -27, -27, 61, -27, -27, 6, 6, 0, -27, -27, -27, 0, 71, -27, 6, 6, 6, 6, 6, 6, -79, 71, -27]

    game = Game(size=9)
    init_fn = game.init
    step_fn = game.step
    legal_action_fn = game.legal_action_mask
    terminal_fn = game.is_terminal

    state = init_fn()
    for i, action in enumerate(actions):
        legal_action_mask = legal_action_fn(state)
        assert legal_action_mask[action], f"action {action} is not legal"
        assert not terminal_fn(state)
        state = step_fn(state, action)

    print("expected = ")
    print(torch.tensor(expected).reshape(9, 9))
    print("actual = ")
    print(state.chain_id_board.reshape(9, 9))

    assert torch.all(state.chain_id_board == torch.tensor(expected))
    assert game.is_terminal(state)

