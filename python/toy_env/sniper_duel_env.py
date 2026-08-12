import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SniperDuelEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # move_x, move_y, turn_delta, scope_signal, fire_signal — all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # observation_space goes here next
