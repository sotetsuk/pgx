import jax
import jax.numpy as jnp

from pgx.chess import (
    State,
    _check_termination,
    _flip_pos,
    _legal_action_mask,
    _observe,
    _possible_piece_positions,
    _update_history,
    _zobrist_hash,
)

TRUE = jnp.bool_(True)


def from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR w KQkq e3 0 1")
    >>> _rotate(state._board.reshape(8, 8))
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
    >>> _rotate(state._board.reshape(8, 8))
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
    can_castle_queen_side = jnp.zeros(2, dtype=jnp.bool_)
    can_castle_king_side = jnp.zeros(2, dtype=jnp.bool_)
    if "Q" in castling:
        can_castle_queen_side = can_castle_queen_side.at[0].set(TRUE)
    if "q" in castling:
        can_castle_queen_side = can_castle_queen_side.at[1].set(TRUE)
    if "K" in castling:
        can_castle_king_side = can_castle_king_side.at[0].set(TRUE)
    if "k" in castling:
        can_castle_king_side = can_castle_king_side.at[1].set(TRUE)
    if turn == "b":
        can_castle_queen_side = can_castle_queen_side[::-1]
        can_castle_king_side = can_castle_king_side[::-1]
    mat = jnp.int32(arr).reshape(8, 8)
    if turn == "b":
        mat = -jnp.flip(mat, axis=0)
    ep = jnp.int32(-1) if en_passant == "-" else jnp.int32("abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1)
    if turn == "b" and ep >= 0:
        ep = _flip_pos(ep)
    state = State(  # type: ignore
        _board=jnp.rot90(mat, k=3).flatten(),
        _turn=jnp.int32(0) if turn == "w" else jnp.int32(1),
        _can_castle_queen_side=can_castle_queen_side,
        _can_castle_king_side=can_castle_king_side,
        _en_passant=ep,
        _halfmove_count=jnp.int32(halfmove_cnt),
        _fullmove_count=jnp.int32(fullmove_cnt),
    )
    state = state.replace(_possible_piece_positions=jax.jit(_possible_piece_positions)(state))  # type: ignore
    state = state.replace(  # type: ignore
        legal_action_mask=jax.jit(_legal_action_mask)(state),
    )
    state = state.replace(_zobrist_hash=_zobrist_hash(state))  # type: ignore
    state = _update_history(state)
    state = jax.jit(_check_termination)(state)
    state = state.replace(observation=jax.jit(_observe)(state, state.current_player))  # type: ignore
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
    pb = jnp.rot90(state._board.reshape(8, 8), k=1)
    if state._turn == 1:
        pb = -jnp.flip(pb, axis=0)
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
    fen += "w " if state._turn == 0 else "b "
    # castling
    can_castle_queen_side = state._can_castle_queen_side
    can_castle_king_side = state._can_castle_king_side
    if state._turn == 1:
        can_castle_queen_side = can_castle_queen_side[::-1]
        can_castle_king_side = can_castle_king_side[::-1]
    if not (can_castle_queen_side.any() | can_castle_king_side.any()):
        fen += "-"
    else:
        if can_castle_king_side[0]:
            fen += "K"
        if can_castle_queen_side[0]:
            fen += "Q"
        if can_castle_king_side[1]:
            fen += "k"
        if can_castle_queen_side[1]:
            fen += "q"
    fen += " "
    # em passant
    en_passant = state._en_passant
    if state._turn == 1:
        en_passant = _flip_pos(en_passant)
    ep = int(en_passant.item())
    if ep == -1:
        fen += "-"
    else:
        fen += "abcdefgh"[ep // 8]
        fen += str(ep % 8 + 1)
    fen += " "
    fen += str(state._halfmove_count.item())
    fen += " "
    fen += str(state._fullmove_count.item())
    return fen
