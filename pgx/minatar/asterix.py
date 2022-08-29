################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np

#####################################################################################################################
# Constants
#
#####################################################################################################################
ramp_interval = 100
init_spawn_speed = 10
init_move_interval = 5
shot_cool_down = 5


#####################################################################################################################
# Env
#
# The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of
# +1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and
# treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed
# and spawn rate of enemies and treasure.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=True, random_state=None):
        self.channels = {"player": 0, "enemy": 1, "trail": 2, "gold": 3}
        self.action_map = ["n", "l", "u", "r", "d", "f"]
        self.ramping = ramping
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

        # Spawn enemy if timer is up
        if self.spawn_timer == 0:
            self._spawn_entity()
            self.spawn_timer = self.spawn_speed

        # Resolve player action
        if a == "l":
            self.player_x = max(0, self.player_x - 1)
        elif a == "r":
            self.player_x = min(9, self.player_x + 1)
        elif a == "u":
            self.player_y = max(1, self.player_y - 1)
        elif a == "d":
            self.player_y = min(8, self.player_y + 1)

        # Update entities
        for i in range(len(self.entities)):
            x = self.entities[i]
            if x is not None:
                if x[0:2] == [self.player_x, self.player_y]:
                    if self.entities[i][3]:
                        self.entities[i] = None
                        r += 1
                    else:
                        self.terminal = True
        if self.move_timer == 0:
            self.move_timer = self.move_speed
            for i in range(len(self.entities)):
                x = self.entities[i]
                if x is not None:
                    x[0] += 1 if x[2] else -1
                    if x[0] < 0 or x[0] > 9:
                        self.entities[i] = None
                    if x[0:2] == [self.player_x, self.player_y]:
                        if self.entities[i][3]:
                            self.entities[i] = None
                            r += 1
                        else:
                            self.terminal = True

        # Update various timers
        self.spawn_timer -= 1
        self.move_timer -= 1

        # Ramp difficulty if interval has elapsed
        if self.ramping and (self.spawn_speed > 1 or self.move_speed > 1):
            if self.ramp_timer >= 0:
                self.ramp_timer -= 1
            else:
                if self.move_speed > 1 and self.ramp_index % 2:
                    self.move_speed -= 1
                if self.spawn_speed > 1:
                    self.spawn_speed -= 1
                self.ramp_index += 1
                self.ramp_timer = ramp_interval
        return r, self.terminal

    # Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
    def _spawn_entity(self):
        lr = self.random.choice([True, False])
        is_gold = self.random.choice([True, False], p=[1 / 3, 2 / 3])
        x = 0 if lr else 9
        slot_options = [
            i for i in range(len(self.entities)) if self.entities[i] == None
        ]
        if not slot_options:
            return
        slot = self.random.choice(slot_options)
        self.entities[slot] = [x, slot + 1, lr, is_gold]

    # Query the current level of the difficulty ramp, could be used as additional input to agent for example
    def difficulty_ramp(self):
        return self.ramp_index

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.player_y, self.player_x, self.channels["player"]] = 1
        for x in self.entities:
            if x is not None:
                c = self.channels["gold"] if x[3] else self.channels["enemy"]
                state[x[1], x[0], c] = 1
                back_x = x[0] - 1 if x[2] else x[0] + 1
                if back_x >= 0 and back_x <= 9:
                    state[x[1], back_x, self.channels["trail"]] = 1
        return state

    # Reset to start state for new episode
    def reset(self):
        self.player_x = 5
        self.player_y = 5
        self.entities = [None] * 8
        self.shot_timer = 0
        self.spawn_speed = init_spawn_speed
        self.spawn_timer = self.spawn_speed
        self.move_speed = init_move_interval
        self.move_timer = self.move_speed
        self.ramp_timer = ramp_interval
        self.ramp_index = 0
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ["n", "l", "u", "r", "d"]
        return [self.action_map.index(x) for x in minimal_actions]
