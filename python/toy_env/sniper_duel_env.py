import numpy as np
import gymnasium as gym
from gymnasium import spaces

# custom 1v1map.vmf 
ARENA_HALF_WIDTH = 1280.0 / 2.0   # x-axis
ARENA_HALF_LENGTH = 920.0 / 2.0  # y-axis

POSITION_LOW = np.array([-ARENA_HALF_WIDTH, -ARENA_HALF_LENGTH], dtype=np.float32)
POSITION_HIGH = np.array([ARENA_HALF_WIDTH, ARENA_HALF_LENGTH], dtype=np.float32)

# opposite ends of the arena along the x axis (RED/BLU spawns in 1v1map.vmf),
# facing each other.
SELF_SPAWN = np.array([-500.0, 0.0], dtype=np.float32)
OPPONENT_SPAWN = np.array([500.0, 0.0], dtype=np.float32)
SPAWN_JITTER = 40.0  # random +/- offset added to spawn position each reset

SELF_SPAWN_ANGLE = 0.0       # facing +x (RED), toward opponent
OPPONENT_SPAWN_ANGLE = 180.0  # facing -x (BLU), toward self

# top-down (x,y) footprints of the cover brushes in 1v1map.vmf, as
# (x_min, x_max, y_min, y_max). taken directly from the .vmf solids.
BARRIERS = np.array([
    [-476.0, -412.0, 144.667, 374.0],   # upper-left pillar
    [-476.0, -412.0, -330.0, -100.667], # lower-left pillar
    [-28.0, 36.0, -74.0, 118.0],        # middle crate
    [400.0, 464.0, 141.667, 371.0],     # upper-right pillar
    [400.0, 464.0, -333.0, -103.667],   # lower-right pillar
], dtype=np.float32)

LOS_SAMPLE_COUNT = 40  # points checked along the shooter->target line for line-of-sight

MAX_EPISODE_STEPS = 300

# how far a full-strength (1.0) action value moves/turns an agent in one step
MAX_MOVE_PER_STEP = 20.0   # hammer units
MAX_TURN_PER_STEP_DEG = 15.0

NOOP_ACTION = np.zeros(5, dtype=np.float32)  # placeholder opponent action

FULL_CHARGE_STEPS = 30  # steps of holding scope to reach full (1.0) charge

MAX_HEALTH = 125.0  # TF2 Sniper base health
AIM_TOLERANCE_DEG = 5.0  # target must be within this many degrees of facing to be hit

MIN_CHARGE_FOR_HEADSHOT = 0.1  # ~3 steps of scoping before a headshot can register
UNSCOPED_HIT_DAMAGE = 50.0     # flat body-shot damage: unscoped, or scoped but under-charged
MIN_HEADSHOT_DAMAGE = 150.0    # headshot damage at MIN_CHARGE_FOR_HEADSHOT
MAX_HEADSHOT_DAMAGE = 450.0    # headshot damage at full (1.0) charge

TERMINAL_REWARD = 100.0  # magnitude of the win/loss reward, must dominate shaping
SHAPING_SCALE = 0.01     # small per-step reward for being aimed at the opponent


class SniperDuelEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, # for movement. like an analog stick
            high=1.0,
            shape=(5,), # 5 elements: strafe, forward/back, turn, scope, fire (strafe/forward are relative to facing)
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
            "self_health": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
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

        self._self_health = MAX_HEALTH
        self._opponent_health = MAX_HEALTH

        self._step_count = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def _move_agent(self, pos, angle, action):

        yaw_rad = np.radians(angle)
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float32)
        right = np.array([np.sin(yaw_rad), -np.cos(yaw_rad)], dtype=np.float32)

        strafe, fwd_back = action[0], action[1]
        move = (strafe * right + fwd_back * forward) * MAX_MOVE_PER_STEP
        new_pos = np.clip(pos + move, POSITION_LOW, POSITION_HIGH)

        if self._point_in_any_barrier(new_pos):
            new_pos = pos  # movement blocked by cover, stay put

        angle = angle + action[2] * MAX_TURN_PER_STEP_DEG
        angle = ((angle + 180.0) % 360.0) - 180.0  # wrap to [-180, 180]

        return new_pos, angle

    def _update_scope(self, scope_active, scope_charge, action):
        scoping_now = action[3] > 0.0

        if scoping_now:
            scope_charge = min(1.0, scope_charge + 1.0 / FULL_CHARGE_STEPS)
        else:
            scope_charge = 0.0

        return scoping_now, scope_charge

    def _point_in_barrier(self, point, barrier):
        x, y = point
        x_min, x_max, y_min, y_max = barrier

        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    def _lerp_point(self, a, b, fraction):
        return a + fraction * (b - a)

    def _point_in_any_barrier(self, point):
        for barrier in BARRIERS:
            if self._point_in_barrier(point, barrier):
                return True

        return False

    def _line_of_sight_clear(self, shooter_pos, target_pos):
        for i in range(LOS_SAMPLE_COUNT + 1):
            fraction = i / LOS_SAMPLE_COUNT
            point = self._lerp_point(shooter_pos, target_pos, fraction)

            if self._point_in_any_barrier(point):
                return False

        return True

    def _is_on_target(self, shooter_pos, shooter_angle, target_pos):
        to_target = target_pos - shooter_pos
        angle_to_target = np.degrees(np.arctan2(to_target[1], to_target[0]))
        angle_diff = ((angle_to_target - shooter_angle + 180.0) % 360.0) - 180.0
        return abs(angle_diff) <= AIM_TOLERANCE_DEG

    def _resolve_fire(self, shooter_pos, shooter_angle, shooter_scope_active, shooter_scope_charge, fire_signal, target_pos):
        if fire_signal <= 0.0:
            return 0.0
        if not self._is_on_target(shooter_pos, shooter_angle, target_pos):
            return 0.0
        if not self._line_of_sight_clear(shooter_pos, target_pos):
            return 0.0

        if shooter_scope_active and shooter_scope_charge >= MIN_CHARGE_FOR_HEADSHOT:
            charge_t = (shooter_scope_charge - MIN_CHARGE_FOR_HEADSHOT) / (1.0 - MIN_CHARGE_FOR_HEADSHOT)
            return MIN_HEADSHOT_DAMAGE + charge_t * (MAX_HEADSHOT_DAMAGE - MIN_HEADSHOT_DAMAGE)

        return UNSCOPED_HIT_DAMAGE

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        self._self_pos, self._self_angle = self._move_agent(
            self._self_pos, self._self_angle, action
        )
        self._opponent_pos, self._opponent_angle = self._move_agent(
            self._opponent_pos, self._opponent_angle, NOOP_ACTION
        )

        self._self_scope_active, self._self_scope_charge = self._update_scope(
            self._self_scope_active, self._self_scope_charge, action
        )
        self._opponent_scope_active, self._opponent_scope_charge = self._update_scope(
            self._opponent_scope_active, self._opponent_scope_charge, NOOP_ACTION
        )

        damage_to_opponent = self._resolve_fire(
            self._self_pos, self._self_angle,
            self._self_scope_active, self._self_scope_charge,
            action[4], self._opponent_pos,
        )
        damage_to_self = self._resolve_fire(
            self._opponent_pos, self._opponent_angle,
            self._opponent_scope_active, self._opponent_scope_charge,
            NOOP_ACTION[4], self._self_pos,
        )
        self._opponent_health = max(0.0, self._opponent_health - damage_to_opponent)
        self._self_health = max(0.0, self._self_health - damage_to_self)

        self._step_count += 1

        self_dead = self._self_health <= 0.0
        opponent_dead = self._opponent_health <= 0.0

        if self_dead and opponent_dead:
            reward = 0.0  # simultaneous kill, draw
        elif opponent_dead:
            reward = TERMINAL_REWARD
        elif self_dead:
            reward = -TERMINAL_REWARD
        else:
            aimed_at_opponent = self._is_on_target(self._self_pos, self._self_angle, self._opponent_pos)
            reward = SHAPING_SCALE if aimed_at_opponent else 0.0

        terminated = self_dead or opponent_dead
        truncated = self._step_count >= MAX_EPISODE_STEPS
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self): #get observation. basically packaging all the data so it fits the observation rulespace
        time_left = 1.0 - (self._step_count / MAX_EPISODE_STEPS)

        opponent_visible = self._line_of_sight_clear(self._self_pos, self._opponent_pos)
        if opponent_visible:
            opponent_pos_obs = self._opponent_pos.astype(np.float32)
        else:
            opponent_pos_obs = np.zeros(2, dtype=np.float32)

        return {
            "self_pos": self._self_pos.astype(np.float32),
            "self_angle": np.array([self._self_angle], dtype=np.float32),
            "scope_active": np.array([1.0 if self._self_scope_active else 0.0], dtype=np.float32),
            "scope_charge": np.array([self._self_scope_charge], dtype=np.float32),
            "opponent_pos": opponent_pos_obs,
            "opponent_visible": np.array([1.0 if opponent_visible else 0.0], dtype=np.float32),
            "time_left": np.array([time_left], dtype=np.float32),
            "self_health": np.array([self._self_health / MAX_HEALTH], dtype=np.float32),
        }


if __name__ == "__main__":
    env = SniperDuelEnv()
    obs, info = env.reset(seed=42)
    print("observation:", obs)
    print("info:", info)
    print("valid according to observation_space?", env.observation_space.contains(obs))

    print("\nself_pos before step:", env._self_pos)
    move_right_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(move_right_action)
    print("self_pos after step:", obs["self_pos"])
    print("valid according to observation_space?", env.observation_space.contains(obs))