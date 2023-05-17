import jax
import jax.numpy as jnp

import pgx.v1 as v1
from pgx._src.gardner_chess_utils import (
    BETWEEN,
    CAN_MOVE,
    CAN_MOVE_ANY,
    INIT_LEGAL_ACTION_MASK,
    PLANE_MAP,
    TO_MAP,
)
from pgx._src.struct import dataclass

MAX_TERMINATION_STEPS = 250


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int8(0)
PAWN = jnp.int8(1)
KNIGHT = jnp.int8(2)
BISHOP = jnp.int8(3)
ROOK = jnp.int8(4)
QUEEN = jnp.int8(5)
KING = jnp.int8(6)
# OPP_PAWN = -1
# OPP_KNIGHT = -2
# OPP_BISHOP = -3
# OPP_ROOK = -4
# OPP_QUEEN = -5
# OPP_KING = -6


# board index (white view)
# 5  4  9 14 19 24
# 4  3  8 13 18 23
# 3  2  7 12 17 22
# 2  1  6 11 16 21
# 1  0  5 10 15 20
#    a  b  c  d  e
# board index (flipped black view)
# 5  0  5 10 15 20
# 4  1  6 11 16 21
# 3  2  7 12 17 22
# 2  3  8 13 18 23
# 1  4  9 14 19 24
#    a  b  c  d  e
# fmt: off
INIT_BOARD = jnp.int8([
    4, 1, 0, -1, -4,
    2, 1, 0, -1, -2,
    3, 1, 0, -1, -3,
    5, 1, 0, -1, -5,
    6, 1, 0, -1, -6,
])
# fmt: on


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = INIT_LEGAL_ACTION_MASK
    observation: jnp.ndarray = jnp.zeros((5, 5, 19), dtype=jnp.float32)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Chess specific ---
    _turn: jnp.ndarray = jnp.int8(0)
    _board: jnp.ndarray = INIT_BOARD  # 左上からFENと同じ形式で埋めていく
    # # of moves since the last piece capture or pawn move
    _halfmove_count: jnp.ndarray = jnp.int32(0)
    _fullmove_count: jnp.ndarray = jnp.int32(1)  # increase every black move
    _zobrist_hash: jnp.ndarray = jnp.uint32([1429435994, 901419182])
    _hash_history: jnp.ndarray = (
        jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32)
        .at[0]
        .set(jnp.uint32([1429435994, 901419182]))
    )

    @staticmethod
    def _from_fen(fen: str):
        return _from_fen(fen)

    def _to_fen(self) -> str:
        return _to_fen(self)

    @property
    def env_id(self) -> v1.EnvId:
        return "gardner_chess"


# Action
# 0 ... 9 = underpromotions
# plane // 3 == 0: rook
# plane // 3 == 1: bishop
# plane // 3 == 2: knight
# plane % 3 == 0: forward
# plane % 3 == 1: right
# plane % 3 == 2: left
# 33          16          32
#    34       15       31
#       35 44 14 48 30
#       42 36 13 29 46
# 17 18 19 20  X 21 22 23 24
#       41 28 12 37 45
#       27 43 11 47 38
#    26       10       39
# 25           9          40
@dataclass
class Action:
    from_: jnp.ndarray = jnp.int8(-1)
    to: jnp.ndarray = jnp.int8(-1)
    underpromotion: jnp.ndarray = jnp.int8(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: jnp.ndarray):
        """We use AlphaZero style label with channel-last representation: (5, 5, 49)

        49 = queen moves (32) + knight moves (8) + underpromotions (3 * 3)
        """
        from_, plane = label // 49, label % 49
        return Action(  # type: ignore
            from_=from_,
            to=TO_MAP[from_, plane],  # -1 if impossible move
            underpromotion=jax.lax.select(
                plane >= 9, jnp.int8(-1), jnp.int8(plane // 3)
            ),
        )

    def _to_label(self):
        plane = PLANE_MAP[self.from_, self.to]
        # plane = jax.lax.select(self.underpromotion >= 0, ..., plane)
        return jnp.int32(self.from_) * 49 + jnp.int32(plane)


class GardnerChess(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        state = State(current_player=current_player)  # type: ignore
        return state

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        state = _step(state, action)
        state = jax.lax.cond(
            MAX_TERMINATION_STEPS <= state._step_count,
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state)

    @property
    def id(self) -> v1.EnvId:
        return "gardner_chess"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: jnp.ndarray):
    a = Action._from_label(action)
    # state = _update_zobrist_hash(state, a)
    state = _apply_move(state, a)
    state = _flip(state)
    state = state.replace(  # type: ignore
        legal_action_mask=_legal_action_mask(state)
    )
    state = _check_termination(state)
    return state


def _apply_move(state: State, a: Action):
    # apply move action
    piece = state._board[a.from_]

    # update counters
    captured = state._board[a.to] < 0
    state = state.replace(  # type: ignore
        _halfmove_count=jax.lax.select(
            captured | (piece == PAWN), 0, state._halfmove_count + 1
        ),
        _fullmove_count=state._fullmove_count + jnp.int32(state._turn == 1),
    )

    # promotion to queen
    piece = jax.lax.select(
        piece == PAWN & (a.from_ % 5 == 3) & (a.underpromotion < 0),
        QUEEN,
        piece,
    )
    # underpromotion
    piece = jax.lax.select(
        a.underpromotion < 0,
        piece,
        jnp.int8([ROOK, BISHOP, KNIGHT])[a.underpromotion],
    )

    # actually move
    state = state.replace(  # type: ignore
        _board=state._board.at[a.from_].set(EMPTY).at[a.to].set(piece)
    )

    return state


def _check_termination(state: State):
    has_legal_action = state.legal_action_mask.any()
    # rep = (state._hash_history == state._zobrist_hash).any(axis=1).sum() - 1
    terminated = ~has_legal_action
    terminated |= state._halfmove_count >= 100
    terminated |= has_insufficient_pieces(state)
    # terminated |= rep >= 2

    is_checkmate = (~has_legal_action) & _is_checking(_flip(state))
    # fmt: off
    reward = jax.lax.select(
        is_checkmate,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # fmt: on
    return state.replace(  # type: ignore
        terminated=terminated,
        rewards=reward,
    )


def has_insufficient_pieces(state: State):
    # Uses the same condition as OpenSpiel.
    # See https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/chess/chess_board.cc#L724
    num_pieces = (state._board != EMPTY).sum()
    num_pawn_rook_queen = (
        (jnp.abs(state._board) >= ROOK) | (jnp.abs(state._board) == PAWN)
    ).sum() - 2  # two kings
    num_bishop = (jnp.abs(state._board) == 3).sum()
    coords = jnp.arange(25).reshape((5, 5))
    # [ 0  2  4  6 16 18 20 22 32 34 36 38 48 50 52 54 9 11 13 15 25 27 29 31 41 43 45 47 57 59 61 63]
    black_coords = jnp.arange(0, 25, 2)
    num_bishop_on_black = (jnp.abs(state._board[black_coords]) == BISHOP).sum()
    is_insufficient = FALSE
    # King vs King
    is_insufficient |= num_pieces <= 2
    # King + X vs King. X == KNIGHT or BISHOP
    is_insufficient |= (num_pieces == 3) & (num_pawn_rook_queen == 0)
    # King + Bishop* vs King + Bishop* (Bishops are on same color tile)
    is_bishop_all_on_black = num_bishop_on_black == num_bishop
    is_bishop_all_on_white = num_bishop_on_black == 0
    is_insufficient |= (num_pieces == num_bishop + 2) & (
        is_bishop_all_on_black | is_bishop_all_on_white
    )

    return is_insufficient


def _legal_action_mask(state):
    def is_legal(a: Action):
        ok = _is_pseudo_legal(state, a)
        next_s = _flip(_apply_move(state, a))
        ok &= ~_is_checking(next_s)

        return ok

    @jax.vmap
    def legal_normal_moves(from_):
        piece = state._board[from_]

        @jax.vmap
        def legal_label(to):
            a = Action(from_=from_, to=to)
            return jax.lax.select(
                (from_ >= 0) & (piece > 0) & (to >= 0) & is_legal(a),
                a._to_label(),
                jnp.int32(-1),
            )

        return legal_label(CAN_MOVE[piece, from_])

    def legal_underpromotions(mask):
        # from_ = 3, 8, 13, 18, 23
        # plane = 0 ... 8
        @jax.vmap
        def make_labels(from_):
            return from_ * 49 + jnp.arange(9)

        labels = make_labels(jnp.int32([3, 8, 13, 18, 23])).flatten()

        @jax.vmap
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state._board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return jax.lax.select(ok, label, -1)

        ok_labels = legal_labels(labels)
        return ok_labels.flatten()

    actions = legal_normal_moves(jnp.arange(25)).flatten()
    # +1 is to avoid setting True to the last element
    mask = jnp.zeros(25 * 49 + 1, dtype=jnp.bool_)
    mask = mask.at[actions].set(TRUE)

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(TRUE)

    return mask[:-1]


def _is_attacking(state: State, pos):
    @jax.vmap
    def can_move(from_):
        a = Action(from_=from_, to=pos)
        return (from_ != -1) & _is_pseudo_legal(state, a)

    return can_move(CAN_MOVE_ANY[pos, :]).any()


def _is_checking(state: State):
    """True if possible to capture the opponent king"""
    opp_king_pos = jnp.argmin(jnp.abs(state._board - -KING))
    return _is_attacking(state, opp_king_pos)


def _is_pseudo_legal(state: State, a: Action):
    piece = state._board[a.from_]
    ok = (piece >= 0) & (state._board[a.to] <= 0)
    ok &= (CAN_MOVE[piece, a.from_] == a.to).any()
    between_ixs = BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state._board[between_ixs] == EMPTY)).all()
    # filter pawn move
    ok &= ~(
        (piece == PAWN)
        & (a.to // 5 == a.from_ // 5)
        & (state._board[a.to] < 0)
    )
    ok &= ~(
        (piece == PAWN)
        & (a.to // 5 != a.from_ // 5)
        & (state._board[a.to] >= 0)
    )
    return (a.to >= 0) & ok


def _observe(state):
    return jnp.zeros(1)


def _flip_pos(x):
    """
    >>> _flip_pos(jnp.int8(0))
    Array(4, dtype=int8)
    >>> _flip_pos(jnp.int8(4))
    Array(0, dtype=int8)
    >>> _flip_pos(jnp.int8(-1))
    Array(-1, dtype=int8)
    """
    return jax.lax.select(x == -1, x, (x // 5) * 5 + (4 - (x % 5)))


def _flip(state: State) -> State:
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        _board=-jnp.flip(state._board.reshape(5, 5), axis=1).flatten(),
        _turn=(state._turn + 1) % 2,
    )


def _rotate(board):
    return jnp.rot90(board, k=1)


def _from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen("rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1")
    >>> _rotate(state._board.reshape(5, 5))
    Array([[-4, -2, -3, -5, -6],
           [-1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6]], dtype=int8)
    >>> state = _from_fen("bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1")
    >>> _rotate(state._board.reshape(5, 5))
    Array([[-4, -2, -3, -5, -6],
           [ 0, -1, -1, -1,  0],
           [ 0,  0,  0,  0,  0],
           [-1,  1,  1,  1,  1],
           [ 3,  3,  6,  5,  4]], dtype=int8)
    """
    board, turn, castling, en_passant, halfmove_cnt, fullmove_cnt = fen.split()
    arr = []
    for line in board.split("/"):
        for c in line:
            if str.isnumeric(c):
                for _ in range(int(c)):
                    arr.append(0)
            else:
                ix = "pnbrqk".index(str.lower(c)) + 1
                if str.islower(c):
                    ix *= -1
                arr.append(ix)
    mat = jnp.int8(arr).reshape(5, 5)
    if turn == "b":
        mat = -jnp.flip(mat, axis=0)
    state = State(  # type: ignore
        _board=jnp.rot90(mat, k=3).flatten(),
        _turn=jnp.int8(0) if turn == "w" else jnp.int8(1),
        _halfmove_count=jnp.int32(halfmove_cnt),
        _fullmove_count=jnp.int32(fullmove_cnt),
    )
    return state


def _to_fen(state: State):
    """Convert state into FEN expression.

    - ポーン:P ナイト:N ビショップ:B ルーク:R クイーン:Q キング:K
    - 先手の駒は大文字、後手の駒は小文字で表現
    - 空白の場合、連続する空白の数を入れて次の駒にシフトする。P空空空RならP3R
    - 左上から開始して右に見ていく
    - 段が変わるときは/を挿入
    - 盤面の記入が終わったら手番（w/b）
    - キャスリングの可否。できる場合はQまたはqを先後それぞれ書く(キングサイドのキャスリングは不可)。全部不可なら-
    - アンパッサン可能な位置。ポーンが2マス動いた場合はそのポーンが通過した位置を記録
    - 最後にポーンの移動および駒取りが発生してからの手数と通常の手数

    >>> s = State()
    >>> _to_fen(s)
    'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1'
    >>> _to_fen(_from_fen("bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1"))
    'bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1'
    """
    pb = jnp.rot90(state._board.reshape(5, 5), k=1)
    if state._turn == 1:
        pb = -jnp.flip(pb, axis=0)
    fen = ""
    # 盤面
    for i in range(5):
        space_length = 0
        for j in range(5):
            piece = pb[i, j]
            if piece == 0:
                space_length += 1
            elif space_length != 0:
                fen += str(space_length)
                space_length = 0
            if piece != 0:
                if piece > 0:
                    fen += "PNBRQK"[piece - 1]
                else:
                    fen += "pnbrqk"[-piece - 1]
        if space_length != 0:
            fen += str(space_length)
        if i != 4:
            fen += "/"
        else:
            fen += " "
    # 手番
    fen += "w " if state._turn == 0 else "b "
    # キャスリング、アンパッサン(なし)
    fen += "- - "
    fen += str(state._halfmove_count.item())
    fen += " "
    fen += str(state._fullmove_count.item())
    return fen
