import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.chess import Game, GameState, _flip_pos, _legal_action_mask, _update_history
from pgx.chess import State

TRUE = jnp.bool_(True)
game = Game()


def from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR w KQkq e3 0 1")
    >>> jnp.rot90(state._x.board.reshape(8, 8), k=1)
    Array([[-4, -2, -3, -5, -6, -3, -2, -4],
           [-1, -1, -1, -1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  1,  1,  1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6,  3,  2,  4]], dtype=int32)
    >>> state._en_passant
    Array(34, dtype=int32)
    >>> state = _from_fen("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1")
    >>> jnp.rot90(state._x.board.reshape(8, 8), k=1)
    Array([[-4, -2, -3, -5, -6, -3, -2, -4],
           [ 0, -1, -1, -1, -1, -1, -1, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6,  3,  2,  4]], dtype=int32)
    >>> state._en_passant
    Array(37, dtype=int32)
    """
    board, color, castling, en_passant, halfmove_cnt, fullmove_cnt = fen.split()
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
    castling_rights = jnp.bool_([["Q" in castling, "K" in castling], ["q" in castling, "k" in castling]])
    if color == "b":
        castling_rights = castling_rights[::-1]
    mat = jnp.int32(arr).reshape(8, 8)
    if color == "b":
        mat = -jnp.flip(mat, axis=0)
    ep = jnp.int32(-1) if en_passant == "-" else jnp.int32("abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1)
    if color == "b" and ep >= 0:
        ep = _flip_pos(ep)
    x = GameState(
        board=jnp.rot90(mat, k=3).flatten(),
        color=jnp.int32(0) if color == "w" else jnp.int32(1),
        castling_rights=castling_rights,
        en_passant=ep,
        halfmove_count=jnp.int32(halfmove_cnt),
        fullmove_count=jnp.int32(fullmove_cnt),
    )
    legal_action_mask = jax.jit(_legal_action_mask)(x)
    x = x._replace(legal_action_mask=legal_action_mask)
    x = _update_history(x)

    player_order = jnp.int32([0, 1])
    state = State(
        _player_order=player_order,
        _x=x,
        current_player=player_order[x.color],
        legal_action_mask=legal_action_mask,
        terminated=jax.jit(game.is_terminal)(x),
        rewards=jax.jit(game.rewards)(x)[player_order],
        observation=jax.jit(game.observe)(x, x.color),
    )
    return state


def to_fen(state: State):
    """Convert state into FEN expression.

    - Board
        - Pawn:P Knight:N Bishop:B ROok:R Queen:Q King:K
        - The pice of th first player is capitalized
        - If empty, the number of consecutive spaces is inserted and shifted to the next piece. (e.g., P Empty Empty Empty R is P3R)
        - Starts from the upper left and looks to the right
        - When the row changes, insert /
    - Turn (w/b) comes after the board
    - Castling availability. K for King side, Q for Queen side. If both are not available, -
    - The place where en passant is possible. If the pawn moves 2 squares, record the position where the pawn passed
    - At last, the number of moves since the last pawn move or capture and the normal number of moves (fixed at 0 and 1 here)

    >>> s = State(_en_passant=jnp.int32(34))
    >>> _to_fen(s)
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1'
    >>> _to_fen(_from_fen("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1"))
    'rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1'
    """
    pb = np.rot90(state._x.board.reshape(8, 8), k=1)
    if state._x.color == 1:
        pb = -np.flip(pb, axis=0)
    fen = ""
    # board
    for i in range(8):
        space_length = 0
        for j in range(8):
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
        if i != 7:
            fen += "/"
        else:
            fen += " "
    # turn
    fen += "w " if state._x.color == 0 else "b "
    # castling
    castling_rights = state._x.castling_rights
    if state._x.color == 1:
        castling_rights = castling_rights[::-1]
    if not (np.any(castling_rights)):
        fen += "-"
    else:
        if castling_rights[0, 1]:
            fen += "K"
        if castling_rights[0, 0]:
            fen += "Q"
        if castling_rights[1, 1]:
            fen += "k"
        if castling_rights[1, 0]:
            fen += "q"
    fen += " "
    # em passant
    en_passant = state._x.en_passant
    if state._x.color == 1:
        en_passant = -1 if en_passant == -1 else (en_passant // 8) * 8 + (7 - (en_passant % 8))
    ep = int(en_passant)
    if ep == -1:
        fen += "-"
    else:
        fen += "abcdefgh"[ep // 8]
        fen += str(ep % 8 + 1)
    fen += " "
    fen += str(state._x.halfmove_count)
    fen += " "
    fen += str(state._x.fullmove_count)
    return fen
