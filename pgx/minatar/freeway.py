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
player_speed = 3
time_limit = 2500


#####################################################################################################################
# Env
#
# The player begins at the bottom of the screen and motion is restricted to traveling up and down. Player speed is
# also restricted such that the player can only move every 3 frames. A reward of +1 is given when the player reaches
# the top of the screen, at which point the player is returned to the bottom. Cars travel horizontally on the screen
# and teleport to the other side when the edge is reached. When hit by a car, the player is returned to the bottom of
# the screen. Car direction and speed is indicated by 5 trail channels, the location of the trail gives direction
# while the specific channel indicates how frequently the car moves (from once every frame to once every 5 frames).
# Each time the player successfully reaches the top of the screen, the car speeds are randomized. Termination occurs
# after 2500 frames have elapsed.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=None, random_state=None):
        self.channels = {
            "chicken": 0,
            "car": 1,
            "speed1": 2,
            "speed2": 3,
            "speed3": 4,
            "speed4": 5,
            "speed5": 6,
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

        if a == "u" and self.move_timer == 0:
            self.move_timer = player_speed
            self.pos = max(0, self.pos - 1)
        elif a == "d" and self.move_timer == 0:
            self.move_timer = player_speed
            self.pos = min(9, self.pos + 1)

        # Win condition
        if self.pos == 0:
            r += 1
            self._randomize_cars(initialize=False)
            self.pos = 9

        # Update cars
        for car in self.cars:
            if car[0:2] == [4, self.pos]:
                self.pos = 9
            if car[2] == 0:
                car[2] = abs(car[3])
                car[0] += 1 if car[3] > 0 else -1
                if car[0] < 0:
                    car[0] = 9
                elif car[0] > 9:
                    car[0] = 0
                if car[0:2] == [4, self.pos]:
                    self.pos = 9
            else:
                car[2] -= 1

        # Update various timers
        self.move_timer -= self.move_timer > 0
        self.terminate_timer -= 1
        if self.terminate_timer < 0:
            self.terminal = True
        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

        # Process the game-state into the 10x10xn state provided to the agent and return

    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.pos, 4, self.channels["chicken"]] = 1
        for car in self.cars:
            state[car[1], car[0], self.channels["car"]] = 1
            back_x = car[0] - 1 if car[3] > 0 else car[0] + 1
            if back_x < 0:
                back_x = 9
            elif back_x > 9:
                back_x = 0
            if abs(car[3]) == 1:
                trail = self.channels["speed1"]
            elif abs(car[3]) == 2:
                trail = self.channels["speed2"]
            elif abs(car[3]) == 3:
                trail = self.channels["speed3"]
            elif abs(car[3]) == 4:
                trail = self.channels["speed4"]
            elif abs(car[3]) == 5:
                trail = self.channels["speed5"]
            state[car[1], back_x, trail] = 1
        return state

    # Randomize car speeds and directions, also reset their position if initialize=True
    def _randomize_cars(self, initialize=False):
        speeds = self.random.randint(1, 6, 8)
        directions = self.random.choice([-1, 1], 8)
        speeds *= directions
        if initialize:
            self.cars = []
            for i in range(8):
                self.cars += [[0, i + 1, abs(speeds[i]), speeds[i]]]
        else:
            for i in range(8):
                self.cars[i][2:4] = [abs(speeds[i]), speeds[i]]

    # Reset to start state for new episode
    def reset(self):
        self._randomize_cars(initialize=True)
        self.pos = 9
        self.move_timer = player_speed
        self.terminate_timer = time_limit
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ["n", "u", "d"]
        return [self.action_map.index(x) for x in minimal_actions]
