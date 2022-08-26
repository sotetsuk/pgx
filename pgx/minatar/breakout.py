"""A fork of MinAtar environment distributed at https://github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""

import numpy as np


#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=None, random_state=None):
        self.channels = {
            "paddle": 0,
            "ball": 1,
            "trail": 2,
            "brick": 3,
        }
        self.action_map = ["n", "l", "u", "r", "d", "f"]
        if random_state is None:
            self.random = np.random.RandomState()
        else:
            self.random = random_state
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal

        a = self.action_map[a]

        # Resolve player action
        if a == "l":
            self.pos = max(0, self.pos - 1)
        elif a == "r":
            self.pos = min(9, self.pos + 1)

        # Update ball position
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        if self.ball_dir == 0:
            new_x = self.ball_x - 1
            new_y = self.ball_y - 1
        elif self.ball_dir == 1:
            new_x = self.ball_x + 1
            new_y = self.ball_y - 1
        elif self.ball_dir == 2:
            new_x = self.ball_x + 1
            new_y = self.ball_y + 1
        elif self.ball_dir == 3:
            new_x = self.ball_x - 1
            new_y = self.ball_y + 1

        strike_toggle = False
        if new_x < 0 or new_x > 9:
            if new_x < 0:
                new_x = 0
            if new_x > 9:
                new_x = 9
            self.ball_dir = [1, 0, 3, 2][self.ball_dir]
        if new_y < 0:
            new_y = 0
            self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif self.brick_map[new_y, new_x] == 1:
            strike_toggle = True
            if not self.strike:
                r += 1
                self.strike = True
                self.brick_map[new_y, new_x] = 0
                new_y = self.last_y
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif new_y == 9:
            if np.count_nonzero(self.brick_map) == 0:
                self.brick_map[1:4, :] = 1
            if self.ball_x == self.pos:
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
                new_y = self.last_y
            elif new_x == self.pos:
                self.ball_dir = [2, 3, 0, 1][self.ball_dir]
                new_y = self.last_y
            else:
                self.terminal = True

        if not strike_toggle:
            self.strike = False

        self.ball_x = new_x
        self.ball_y = new_y
        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

        # Process the game-state into the 10x10xn state provided to the agent and return

    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.ball_y, self.ball_x, self.channels["ball"]] = 1
        state[9, self.pos, self.channels["paddle"]] = 1
        state[self.last_y, self.last_x, self.channels["trail"]] = 1
        state[:, :, self.channels["brick"]] = self.brick_map
        return state

    # Reset to start state for new episode
    def reset(self):
        self.ball_y = 3
        ball_start = self.random.choice(2)
        self.ball_x, self.ball_dir = [(0, 2), (9, 3)][ball_start]
        self.pos = 4
        self.brick_map = np.zeros((10, 10))
        self.brick_map[1:4, :] = 1
        self.strike = False
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ["n", "l", "r"]
        return [self.action_map.index(x) for x in minimal_actions]
