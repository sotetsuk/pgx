import jax
import jax.numpy as jnp

import pgx.v1 as v1
from pgx._src.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

# fmt: off
INIT_BOARD = jnp.int8([
    4, 1, 0, -1, -4,
    2, 1, 0, -1, -2,
    3, 1, 0, -1, -3,
    5, 1, 0, -1, -5,
    6, 1, 0, -1, -6,
])
# fmt: o


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(1)  # TODO: fix me
    observation: jnp.ndarray = jnp.zeros((8, 8, 19), dtype=jnp.float32)
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
        jnp.zeros((1001, 2), dtype=jnp.uint32)
            .at[0]
            .set(jnp.uint32([1429435994, 901419182]))
    )

    @property
    def env_id(self) -> v1.EnvId:
        return "gardner_chess"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def _flip_pos(x):
    """
    >>> _flip_pos(jnp.int8(11))
    Array(13, dtype=int8)
    >>> _flip_pos(jnp.int8(13))
    Array(11, dtype=int8)
    >>> _flip_pos(jnp.int8(-1))
    Array(-1, dtype=int8)
    """
    return jax.lax.select(x == -1, x, (x // 5) * 5 + (4 - (x % 5)))


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
    Array([[-3, -3, -6, -5, -4],
           [ 1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0],
           [ 0,  1,  1,  1,  0],
           [ 4,  2,  3,  5,  6]], dtype=int8)
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
    # state = state.replace(  # type: ignore
    #     _possible_piece_positions=jax.jit(_possible_piece_positions)(state)
    # )
    # state = state.replace(  # type: ignore
    #     legal_action_mask=jax.jit(_legal_action_mask)(state),
    # )
    # state = jax.jit(_check_termination)(state)
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
