import numpy as np
import gymnasium as gym
from gymnasium import spaces

# custom 1v1map.vmf (width=1020, length=1345 Hammer units).
ARENA_HALF_WIDTH = 1020.0 / 2.0   # x-axis
ARENA_HALF_LENGTH = 1345.0 / 2.0  # y-axis

POSITION_LOW = np.array([-ARENA_HALF_WIDTH, -ARENA_HALF_LENGTH], dtype=np.float32)
POSITION_HIGH = np.array([ARENA_HALF_WIDTH, ARENA_HALF_LENGTH], dtype=np.float32)


class SniperDuelEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, # for movement. like an analog stick
            high=1.0,
            shape=(5,), # 5 elements for the bot. x, y, turn, scope, fire
            dtype=np.float32,
        )
        #basically a set of rules for the ai to abide
        self.observation_space = spaces.Dict({
            "self_pos": spaces.Box(
                low=POSITION_LOW, high=POSITION_HIGH, shape=(2,), dtype=np.float32
            ),
            "self_angle": spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            "scope_active": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "scope_charge": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "opponent_pos": spaces.Box(
                low=POSITION_LOW, high=POSITION_HIGH, shape=(2,), dtype=np.float32
            ),
            "opponent_visible": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "time_left": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })