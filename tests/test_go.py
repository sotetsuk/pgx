from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from pgx.go import _count_ji, _count_point, Go, State, _show

BOARD_SIZE = 5
env = Go(size=BOARD_SIZE)
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 1


def test_end_by_pass():
    key = jax.random.PRNGKey(0)

    state = init(key=key)
    state = step(state=state, action=25)
    assert state.passed
    assert not state.terminated
    state = step(state=state, action=0)
    assert not state.passed
    assert not state.terminated
    state = step(state=state, action=25)
    assert state.passed
    assert not state.terminated
    state = step(state=state, action=25)
    assert state.passed
    assert state.terminated


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 1

    state = step(state=state, action=12)  # BLACK
    state = step(state=state, action=11)  # WHITE
    state = step(state=state, action=17)
    state = step(state=state, action=7)
    state = step(state=state, action=8)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=16)
    state = step(state=state, action=21)
    state = step(state=state, action=2)
    state = step(state=state, action=10)
    state = step(state=state, action=5)
    state = step(state=state, action=14)
    state = step(state=state, action=15)
    state = step(state=state, action=23)
    state = step(state=state, action=20)
    state = step(state=state, action=25)  # pass
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=21)
    state = step(state=state, action=25)  # pass
    state = step(state=state, action=25)  # pass

    expected_board: jnp.ndarray = jnp.array(
        [
            [ 0, -1, -1, 1, 0],
            [-1,  0, -1, 1, 0],
            [ 0, -1,  1, 0, 1],
            [-1, -1,  1, 0, 1],
            [-1, -1, -1, 1, 0],
        ]
    )  # type:ignore
    """
      [ 0 1 2 3 4 ]
    [0] + O O @ +
    [1] O + O @ +
    [2] + O @ + @
    [3] O O @ + @
    [4] O O O @ +
    """
    assert (jnp.clip(state.chain_id_board, -1, 1) == expected_board.ravel()).all()
    assert state.terminated

    # 同点なのでコミの分 黒 == player_1 の負け
    assert (state.reward == jnp.array([1, -1])).all()


def test_ko():
    key = jax.random.PRNGKey(0)

    state: State = init(key=key)
    assert state.current_player == 1
    state = step(state=state, action=2)  # BLACK
    state = step(state=state, action=17)  # WHITE
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=13)  # WHITE
    state = step(state=state, action=8)  # BLACK
    state = step(state=state, action=11)  # WHITE
    state = step(state=state, action=12)  # BLACK
    state = step(state=state, action=7)  # WHITE

    """
    ===========
    + + @ + +
    + @ + @ +
    + O @ O +
    + + O + +
    + + + + +
    ===========
    + + @ + +
    + @ O @ +
    + O + O +
    + + O + +
    + + + + +
    """
    assert state.ko == 12

    loser = state.current_player
    state1: State = step(
        state=state, action=12
    )  # BLACK
    # ルール違反により黒 = player_id=1 の負け
    assert state1.terminated
    assert state1.reward[loser] == -1.
    assert state1.reward.sum() == 0.

    state2: State = step(state=state, action=0)  # BLACK
    # 回避した場合
    assert not state2.terminated
    assert state2.ko == -1

    # see #468
    state: State = init(key=key)
    state = step(state, action=2)
    state = step(state, action=9)
    state = step(state, action=18)
    state = step(state, action=5)
    state = step(state, action=11)
    state = step(state, action=22)
    state = step(state, action=8)
    state = step(state, action=14)
    state = step(state, action=25)
    state = step(state, action=1)
    state = step(state, action=24)
    state = step(state, action=23)
    state = step(state, action=7)
    state = step(state, action=4)
    state = step(state, action=16)
    state = step(state, action=15)
    state = step(state, action=19)
    state = step(state, action=6)
    state = step(state, action=20)
    state = step(state, action=25)
    state = step(state, action=12)
    state = step(state, action=3)
    state = step(state, action=21)
    state = step(state, action=10)
    state = step(state, action=17)
    state = step(state, action=25)
    state = step(state, action=13)
    state = step(state, action=4)
    state = step(state, action=14)
    state = step(state, action=23)
    state = step(state, action=0)
    assert state.ko == -1

    # see #468
    state: State = init(key=key)
    state = step(state, action=1)
    state = step(state, action=16)
    state = step(state, action=9)
    state = step(state, action=11)
    state = step(state, action=14)
    state = step(state, action=6)
    state = step(state, action=24)
    state = step(state, action=4)
    state = step(state, action=25)
    state = step(state, action=17)
    state = step(state, action=21)
    state = step(state, action=18)
    state = step(state, action=13)
    state = step(state, action=23)
    state = step(state, action=8)
    state = step(state, action=0)
    state = step(state, action=5)
    state = step(state, action=25)
    state = step(state, action=15)
    state = step(state, action=19)
    state = step(state, action=22)
    state = step(state, action=25)
    state = step(state, action=12)
    state = step(state, action=10)
    state = step(state, action=2)
    state = step(state, action=25)
    state = step(state, action=3)
    state = step(state, action=20)
    assert state.ko == -1

    # Ko after pass
    state: State = init(key=key)
    state = step(state, action=17)
    state = step(state, action=25)
    state = step(state, action=20)
    state = step(state, action=24)
    state = step(state, action=6)
    state = step(state, action=13)
    state = step(state, action=12)
    state = step(state, action=18)
    state = step(state, action=22)
    state = step(state, action=5)
    state = step(state, action=8)
    state = step(state, action=10)
    state = step(state, action=11)
    state = step(state, action=14)
    state = step(state, action=2)
    state = step(state, action=9)
    state = step(state, action=1)
    state = step(state, action=23)
    state = step(state, action=16)
    state = step(state, action=4)
    state = step(state, action=0)
    state = step(state, action=19)
    state = step(state, action=15)
    state = step(state, action=25)
    state = step(state, action=21)
    state = step(state, action=25)
    state = step(state, action=5)
    state = step(state, action=25)
    state = step(state, action=3)
    state = step(state, action=14)
    state = step(state, action=4)
    state = step(state, action=19)
    state = step(state, action=7)
    state = step(state, action=23)
    state = step(state, action=18)
    state = step(state, action=13)
    state = step(state, action=24)
    state = step(state, action=25)  # pass
    assert state.ko == -1

    # see #479
    actions = [107, 11, 56, 41, 300, 19, 228, 231, 344, 257, 35, 32, 57, 276, 0, 277, 164, 15, 187, 179, 357, 255, 150, 211, 256,
     190, 297, 303, 358, 189, 322, 3, 129, 64, 13, 336, 22, 286, 264, 192, 55, 360, 23, 31, 113, 119, 195, 98, 208, 294,
     240, 241, 149, 280, 118, 296, 245, 99, 335, 226, 29, 287, 84, 248, 225, 351, 202, 20, 137, 274, 232, 85, 36, 141,
     108, 95, 282, 93, 337, 216, 58, 131, 283, 10, 106, 243, 318, 220, 136, 34, 127, 293, 80, 165, 125, 83, 114, 105,
     30, 61, 147, 71, 109, 173, 87, 233, 76, 361, 66, 115, 212, 200, 346, 197, 54, 326, 298, 167, 347, 4, 354, 16, 140,
     144, 68, 178, 24, 204, 285, 203, 316, 307, 146, 37, 201, 268, 176, 133, 25, 227, 310, 291, 132, 352, 123, 184, 343,
     299, 90, 267, 334, 134, 7, 110, 321, 182, 281, 92, 222, 96, 329, 70, 340, 207, 323, 138, 308, 100, 49, 78, 5, 126,
     317, 17, 349, 160, 261, 266, 306, 221, 355, 327, 324, 284, 236, 60, 359, 174, 252, 46, 260, 114, 163, 235, 250,
     206, 239, 2, 166, 328, 128, 104, 341, 224, 74, 198, 304, 295, 101, 88, 360, 325, 199, 38, 263, 270, 151, 331, 230,
     33, 152, 48, 47, 28, 122, 161, 273, 103, 143, 238, 121, 52, 333, 244, 218, 265, 361, 77, 275, 185, 172, 350, 194,
     59, 53, 21, 272, 319, 320, 158, 251, 253, 135, 27, 196, 180, 188, 345, 254, 130, 42, 156, 259, 332, 361, 18, 82,
     86, 191, 249, 51, 45, 348, 217, 63, 302, 292, 155, 313, 205, 6, 237, 279, 229, 258, 234, 262, 40, 73, 142, 219,
     330, 111, 186, 153, 311, 336, 44, 12, 62, 215, 39, 299, 9, 269, 275, 157, 225, 361, 177, 361, 162, 81, 76, 183,
     168, 247, 309, 145, 210, 221, 65, 301, 1, 289, 120, 315, 353, 305, 67, 214, 79, 314, 290, 47, 181, 346, 175, 89,
     312, 43, 231, 329, 102, 91, 208, 139, 236, 348, 66, 26, 8, 94, 169, 271, 339, 58, 69, 80, 349, 170, 23, 159, 347,
     288, 154, 270, 6, 187, 22, 42, 148, 193, 346, 126, 116, 242, 124, 159, 14, 12, 144, 26, 24, 361, 223, 7, 361, 63,
     117, 112, 5, 81, 118, 135, 82, 92, 140, 123, 97, 278, 47, 361, 137, 230, 220]
    env = Go(size=19)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(0))
    for a in actions:
        state = env.step(state, a)
    assert state.ko == -1
    assert state.legal_action_mask[231]

def test_observe():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 1
    # player 0 is white, player 1 is black

    state = step(state=state, action=0)
    state = step(state=state, action=1)
    state = step(state=state, action=2)
    state = step(state=state, action=3)
    state = step(state=state, action=4)
    state = step(state=state, action=5)
    state = step(state=state, action=6)
    state = step(state=state, action=7)
    # ===========
    # + O + O @
    # O @ O + +
    # + + + + +
    # + + + + +
    # + + + + +
    # fmt: off
    curr_board = jnp.int8(
        [[ 0, -1,  0, -1, 1],
         [-1,  1, -1,  0, 0],
         [ 0,  0,  0,  0, 0],
         [ 0,  0,  0,  0, 0],
         [ 0,  0,  0,  0, 0]]
    )
    # fmt: on
    assert state.current_player == 1
    assert state.turn % 2 == 0  # black turn
    obs = observe(state, 0)   # white
    assert obs.shape == (5, 5, 17)
    assert (obs[:, :, 0] == (curr_board == -1)).all()
    assert (obs[:, :, 1] == (curr_board == 1)).all()
    assert (obs[:, :, -1] == 0).all()

    obs = observe(state, 1)  # black
    assert obs.shape == (5, 5, 17)
    assert (obs[:, :, 0] == (curr_board == 1)).all()
    assert (obs[:, :, 1] == (curr_board == -1)).all()
    assert (obs[:, :, -1] == 1).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)

    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        False, True, False, True, False,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    # white 8
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 13
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=13)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 9
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=13)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    # fmt:off
    expected_b = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    expected_w = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, True, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert jnp.all(state.legal_action_mask == expected_b)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(state.legal_action_mask == expected_w)

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    # fmt:off
    # black 24
    expected_w1 = jnp.array([
        True, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # white pass
    expected_b = jnp.array([
        True, False, False, False, True,
        False, False, False, True, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # black 8
    expected_w2 = jnp.array([
        False, False, False, False, False,
        False, True, False, False, False,
        False, True, False, True, False,
        False, True, False, True, False,
        False, True, True, True, False, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    assert jnp.all(state.legal_action_mask == expected_w1)
    state = step(state=state, action=25)
    assert jnp.all(state.legal_action_mask == expected_b)
    state = step(state=state, action=8)
    assert jnp.all(state.legal_action_mask == expected_w2)

    # =====
    # random
    env = Go(size=BOARD_SIZE)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = env.init(key=key)
    for _ in range(50):  # 5 * 5 * 2 = 50
        assert np.where(state.legal_action_mask)[0][-1]

        legal_actions = np.where(state.legal_action_mask)[0][:-1]
        illegal_actions = np.where(~state.legal_action_mask)[0][:-1]
        for action in legal_actions:
            _state = env.step(state=state, action=action)
            if _state._step_count < 50:
                assert not _state.terminated
            else:
                assert _state.terminated
        for action in illegal_actions:
            _state = env.step(state=state, action=action)
            assert _state.terminated
        if len(legal_actions) == 0:
            a = BOARD_SIZE * BOARD_SIZE
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, legal_actions)
        state = env.step(state=state, action=a)


def test_counting_ji():
    key = jax.random.PRNGKey(0)
    count_ji = jax.jit(_count_ji, static_argnums=(2,))
    BLACK, WHITE = 1, -1

    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert count_ji(state, BLACK, BOARD_SIZE) == 17
    assert count_ji(state, WHITE, BOARD_SIZE) == 0
    state = step(state=state, action=24)  # WHITE
    assert count_ji(state, BLACK, BOARD_SIZE) == 5
    assert count_ji(state, WHITE, BOARD_SIZE) == 0

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert count_ji(state, BLACK, BOARD_SIZE) == 14
    assert count_ji(state, WHITE, BOARD_SIZE) == 0

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert count_ji(state, BLACK, BOARD_SIZE) == 1
    assert count_ji(state, WHITE, BOARD_SIZE) == 12

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    state = step(state=state, action=25)
    assert count_ji(state, BLACK, BOARD_SIZE) == 2
    assert count_ji(state, WHITE, BOARD_SIZE) == 0
    state = step(state=state, action=8)
    assert count_ji(state, BLACK, BOARD_SIZE) == 10
    assert count_ji(state, WHITE, BOARD_SIZE) == 0

    # セキ判定
    # =====
    # + @ O + +
    # O @ O + +
    # O @ O + +
    # O @ O + +
    # + @ O + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=6)
    state = step(state=state, action=5)
    state = step(state=state, action=11)
    state = step(state=state, action=7)
    state = step(state=state, action=16)
    state = step(state=state, action=10)
    state = step(state=state, action=21)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    assert count_ji(state, BLACK, BOARD_SIZE) == 0
    assert count_ji(state, WHITE, BOARD_SIZE) == 10

    # =====
    # O O O O +
    # O @ @ @ O
    # O @ + @ O
    # O @ @ @ O
    # + O O O +
    state = init(key=key)
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=0)  # WHITE
    state = step(state=state, action=7)
    state = step(state=state, action=1)
    state = step(state=state, action=8)
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=9)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=18)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=21)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    assert count_ji(state, BLACK, BOARD_SIZE) == 1
    assert count_ji(state, WHITE, BOARD_SIZE) == 3

    # =====
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    state = init(key=key)
    #assert count_ji(state, 0, BOARD_SIZE) == 0
    #assert count_ji(state, 1, BOARD_SIZE) == 0

    # =====
    # + + + + +
    # + @ @ @ +
    # + @ + @ +
    # + @ @ @ +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=7)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=13)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    assert count_ji(state, BLACK, BOARD_SIZE) == 17
    assert count_ji(state, WHITE, BOARD_SIZE) == 0


def test_counting_point():
    key = jax.random.PRNGKey(0)
    count_point = jax.jit(_count_point, static_argnums=(1,))
    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))
    state = step(state=state, action=24)  # WHITE
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([13, 1], dtype=jnp.float32))

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([22, 2], dtype=jnp.float32))

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([5, 20], dtype=jnp.float32))

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    state = step(state=state, action=25)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([16, 8], dtype=jnp.float32))
    state = step(state=state, action=8)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))

    # セキ判定
    # =====
    # + @ O + +
    # O @ O + +
    # O @ O + +
    # O @ O + +
    # + @ O + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=6)
    state = step(state=state, action=5)
    state = step(state=state, action=11)
    state = step(state=state, action=7)
    state = step(state=state, action=16)
    state = step(state=state, action=10)
    state = step(state=state, action=21)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([5, 18], dtype=jnp.float32))

    # =====
    # O O O O +
    # O @ @ @ O
    # O @ + @ O
    # O @ @ @ O
    # + O O O +
    state = init(key=key)
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=0)  # WHITE
    state = step(state=state, action=7)
    state = step(state=state, action=1)
    state = step(state=state, action=8)
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=9)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=18)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=21)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([9, 16], dtype=jnp.float32))

    # =====
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    state = init(key=key)
    # 本当は[0, 0]
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([25, 25], dtype=jnp.float32))

    # =====
    # + + + + +
    # + @ @ @ +
    # + @ + @ +
    # + @ @ @ +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=7)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=13)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))
    # =====
    # + @ @ O +
    # + + @ O +
    # + + @ O O
    # + + @ O O
    # + + @ O O
    # Tromp-Taylor rule: Black 15, White 10 → White Win
    # Japanese rule: Black 9, White 2 → Black Win
    state = init(key=key)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=7)
    state = step(state=state, action=13)
    state = step(state=state, action=12)
    state = step(state=state, action=14)
    state = step(state=state, action=17)
    state = step(state=state, action=18)
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([15, 10], dtype=jnp.float32))

    # =====
    # + @ @ O +
    # @ + @ O +
    # + + @ O O
    # + + @ O O
    # + + @ O O
    # Agehama: Black 1, White 0
    # Tromp-Taylor rule: Black 15, White 10 → White Win
    # Japanese rule: Black 9, White 2 → Black Win
    state = init(key=key)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=7)
    state = step(state=state, action=13)
    state = step(state=state, action=12)
    state = step(state=state, action=14)
    state = step(state=state, action=17)
    state = step(state=state, action=18)
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=0)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    assert jnp.all(count_point(state, BOARD_SIZE) == jnp.array([15, 10], dtype=jnp.float32))


def test_PSK():
    env = Go(size=5)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, 20)  # BLACK
    state = env.step(state, 17)  # WHITE
    state = env.step(state, 6)  # BLACK
    state = env.step(state, 9)  # WHITE
    state = env.step(state, 11)  # BLACK
    state = env.step(state, 1)  # WHITE
    state = env.step(state, 25)  # BLACK
    state = env.step(state, 4)  # WHITE
    state = env.step(state, 24)  # BLACK
    state = env.step(state, 19)  # WHITE
    state = env.step(state, 16)  # BLACK
    state = env.step(state, 18)  # WHITE
    state = env.step(state, 5)  # BLACK
    state = env.step(state, 0)  # WHITE
    state = env.step(state, 3)  # BLACK
    state = env.step(state, 21)  # WHITE
    state = env.step(state, 12)  # BLACK
    state = env.step(state, 13)  # WHITE
    state = env.step(state, 7)  # BLACK
    state = env.step(state, 8)  # WHITE
    state = env.step(state, 22)  # BLACK
    state = env.step(state, 25)  # WHITE
    state = env.step(state, 21)  # BLACK
    state = env.step(state, 23)  # WHITE
    state = env.step(state, 10)  # BLACK
    #  O O + @ O
    #  @ @ @ O O
    #  @ @ @ O +
    #  + @ O O O
    #  @ @ @ O +
    state = env.step(state, 2)  # WHITE
    state = env.step(state, 3)  # BLACK
    state = env.step(state, 0)  # WHITE
    assert not state.terminated
    state = env.step(state, 25)  # BLACK
    assert not state.terminated
    state = env.step(state, 1)  # WHITE
    #  O O + @ O
    #  @ @ @ O O
    #  @ @ @ O +
    #  + @ O O O
    #  @ @ @ O +
    assert state.terminated
    assert state.black_player == 1
    assert (state.reward == jnp.float32([-1, 1])).all()  # black wins


def test_random_play_5():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    while not state.terminated:
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = 25
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])

        state = step(state=state, action=a)

        if state.turn > 100:
            break
    assert state.passed or state.turn > 100


def test_api():
    import pgx
    env = pgx.make("go-9x9")
    pgx.api_test(env, 10)
    env = pgx.make("go-19x19")
    pgx.api_test(env, 10)
