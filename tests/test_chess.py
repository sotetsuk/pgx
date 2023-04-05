import pgx
from pgx._chess import State, Action, KING, _rotate

pgx.set_visualization_config(color_theme="dark")

def test_action():
    state = State._from_fen("k7/8/8/8/8/8/1Q6/7K w - - 0 1")
    print(_rotate(state.board.reshape(8, 8)))
    assert state.board[56] == KING
    assert state.board[7] == -KING
    state.save_svg("tests/assets/chess/action_001.svg")