from minatar import Environment
import numpy as np
import random


def test_step():
    env = Environment("breakout")
    num_actions = env.num_actions()
    
    env.reset()
    s = env.state()
    a = random.randrange(num_actions)
    env.act(a)
    s_next = env.state()
    assert np.allclose(s, s_next)
