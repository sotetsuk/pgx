import jax

from pgx.go import Go


def from_sgf(sgf: str):
    indexes = "abcdefghijklmnopqrs"
    infos = sgf.split(";")
    game_info = infos[1]
    game_record = infos[2:]
    # assert game_info[game_info.find('GM') + 3] == "1"
    # set default to 19
    size = 19
    if game_info.find("SZ") != -1:
        sz = game_info[game_info.find("SZ") + 3 : game_info.find("SZ") + 5]
        if sz[1] == "]":
            sz = sz[0]
        size = int(sz)
    env = Go(size=size)
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = init(key)
    has_branch = False
    for reco in game_record:
        if reco[-2] == ")":
            # The end of main branch
            print("this sgf has some branches")
            print("loaded main branch")
            has_branch = True
        if reco[2] == "]":
            # pass
            state = step(state, size * size)
            # check branches
            if has_branch:
                return state
            continue
        pos = reco[2:4]
        yoko = indexes.index(pos[0])
        tate = indexes.index(pos[1])
        action = yoko + size * tate
        state = step(state, action)
        # We only follow the main branch
        # Return when the main branch ends
        if has_branch:
            return state
    return state
