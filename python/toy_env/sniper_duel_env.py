import numpy as np
import gymnasium as gym
from gymnasium import spaces

# custom 1v1map.vmf (width=1020, length=1345 Hammer units).
ARENA_HALF_WIDTH = 1020.0 / 2.0   # x-axis
ARENA_HALF_LENGTH = 1345.0 / 2.0  # y-axis

POSITION_LOW = np.array([-ARENA_HALF_WIDTH, -ARENA_HALF_LENGTH], dtype=np.float32)
POSITION_HIGH = np.array([ARENA_HALF_WIDTH, ARENA_HALF_LENGTH], dtype=np.float32)

# opposite ends of the arena along the long (length) axis, facing each other.
SELF_SPAWN = np.array([0.0, -ARENA_HALF_LENGTH * 0.8], dtype=np.float32)
OPPONENT_SPAWN = np.array([0.0, ARENA_HALF_LENGTH * 0.8], dtype=np.float32)
SPAWN_JITTER = 40.0  # random +/- offset added to spawn position each reset

SELF_SPAWN_ANGLE = 90.0    # facing +y, toward opponent
OPPONENT_SPAWN_ANGLE = -90.0  # facing -y, toward self

MAX_EPISODE_STEPS = 300


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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._self_pos = SELF_SPAWN + self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER, size=2)
        self._opponent_pos = OPPONENT_SPAWN + self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER, size=2)

        self._self_angle = SELF_SPAWN_ANGLE
        self._opponent_angle = OPPONENT_SPAWN_ANGLE

        self._self_scope_active = False
        self._self_scope_charge = 0.0
        self._opponent_scope_active = False
        self._opponent_scope_charge = 0.0

        self._step_count = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self): #get observation. basically packaging all the data so it fits the observation rulespace
        time_left = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
        return {
            "self_pos": self._self_pos.astype(np.float32),
            "self_angle": np.array([self._self_angle], dtype=np.float32),
            "scope_active": np.array([1.0 if self._self_scope_active else 0.0], dtype=np.float32),
            "scope_charge": np.array([self._self_scope_charge], dtype=np.float32),
            # TODO: mask position and force opponent_visible=0.0 once line-of-sight/cover exists.
            "opponent_pos": self._opponent_pos.astype(np.float32),
            "opponent_visible": np.array([1.0], dtype=np.float32),
            "time_left": np.array([time_left], dtype=np.float32),
        }


if __name__ == "__main__":
    env = SniperDuelEnv()
    obs, info = env.reset(seed=42)
    print("observation:", obs)
    print("info:", info)
    print("valid according to observation_space?", env.observation_space.contains(obs))