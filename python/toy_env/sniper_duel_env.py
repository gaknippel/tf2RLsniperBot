import numpy as np
import gymnasium as gym
from gymnasium import spaces

# custom 1v1map.vmf -- surveyed directly from the compiled map's actual wall
# brushes (not guessed): inner wall faces are at x=-639/647, y=-479/459. A
# real TF2 player's collision hull stops ~24 units short of a wall (confirmed
# empirically via sniperbot_debug: both bots got physically stuck sliding
# along a wall at x=-614.7/622.8, exactly wall_face -+ ~24.3). The previous
# symmetric +-640/+-460 box was close to the wall faces themselves but didn't
# account for that hull radius, and -- more importantly -- assumed ~140 units
# of retreat room behind each spawn (500 vs 640) when the real map only has
# ~55-80 (spawn_red is at x=-559.61, only ~55 units from where a bot actually
# gets stuck). The policy's learned "back away while shooting" style had
# never experienced running out of room, so it got stuck the first time it
# tried it for real. These bounds are the real stopping points minus a small
# safety margin, asymmetric to match the real (not perfectly centered) map.
POSITION_LOW = np.array([-610.0, -450.0], dtype=np.float32)
POSITION_HIGH = np.array([615.0, 430.0], dtype=np.float32)

# opposite ends of the arena along the x axis (RED/BLU spawns in 1v1map.vmf),
# facing each other. The trained agent always plays RED/SELF_SPAWN now -- see
# the 2026-08-25 rewrite note below -- so there is exactly one canonical role
# and no more spawn-side randomization or observation mirroring to keep in
# sync with the C++ bridge.
SELF_SPAWN = np.array([-500.0, 0.0], dtype=np.float32)
OPPONENT_SPAWN = np.array([500.0, 0.0], dtype=np.float32)
SPAWN_JITTER = 40.0  # random +/- offset added to spawn position each reset

# curriculum-only spawn y (see reset()) -- confirmed via direct LOS scan to
# sit in the one horizontal gap with clear sightline the full width of the
# arena, clear of both the crate (top at y=109) and the lower pillars
# (bottom at y=144.667).
EASY_SPAWN_Y = 125.0

SELF_SPAWN_ANGLE = 0.0       # facing +x (RED), toward opponent
OPPONENT_SPAWN_ANGLE = 180.0  # facing -x (BLU), toward self

# top-down (x,y) footprints of the cover brushes in 1v1map.vmf, as
# (x_min, x_max, y_min, y_max). taken directly from the .vmf solids.
BARRIERS = np.array([
    [-476.0, -412.0, 144.667, 374.0],   # upper-left pillar
    [-476.0, -412.0, -330.0, -100.667], # lower-left pillar
    [-65.0, 64.0, -67.0, 109.0],        # middle crate
    [400.0, 464.0, 141.667, 371.0],     # upper-right pillar
    [400.0, 464.0, -333.0, -103.667],   # lower-right pillar
], dtype=np.float32)

LOS_SAMPLE_COUNT = 40  # points checked along the shooter->target line for line-of-sight

# a real TF2 player has a collision hull, so their origin stops this far short
# of any solid -- confirmed empirically via sniperbot_debug: both bots got
# stuck jammed against the middle crate at exactly crate_face -+ 24 units
# (same radius independently confirmed at the outer walls too, see
# POSITION_LOW/POSITION_HIGH above). The toy env otherwise treats agents as
# zero-radius points, which let training walk right up to a barrier's exact
# mathematical edge -- a standoff distance real physics never allows. Applies
# only to movement blocking, never to line-of-sight/aim/fire checks -- a
# bullet or sightline isn't blocked by the shooter's own hull.
PLAYER_COLLISION_RADIUS = 24.0

MAX_EPISODE_STEPS = 300

# how far a full-strength (1.0) action value moves/turns an agent in one step
MAX_MOVE_PER_STEP = 20.0   # hammer units
MAX_TURN_PER_STEP_DEG = 15.0

FULL_CHARGE_STEPS = 30  # steps of holding scope to reach full (1.0) charge

MAX_HEALTH = 125.0  # TF2 Sniper base health
AIM_TOLERANCE_DEG = 5.0  # target must be within this many degrees of facing to be hit

MIN_CHARGE_FOR_HEADSHOT = 0.1  # ~3 steps of scoping before a headshot can register
UNSCOPED_HIT_DAMAGE = 50.0     # flat body-shot damage: unscoped, or scoped but under-charged
MIN_HEADSHOT_DAMAGE = 150.0    # headshot damage at MIN_CHARGE_FOR_HEADSHOT
MAX_HEADSHOT_DAMAGE = 450.0    # headshot damage at full (1.0) charge

# 2026-09-03: a shot used to connect off a single-frame flicker of on-target,
# which meant a policy could win purely by spinning at max turn rate and
# holding fire down -- a fast spin sweeps back across the opponent's bearing
# often enough to intermittently satisfy _is_on_target AND farm SHAPING_SCALE,
# without ever needing to actually settle onto target. With the scripted
# opponent dying in ~3 unscoped body shots inside a ~16-step average
# engagement, that exploit was cheaper for PPO (ent_coef=0.0, 50M steps, no
# exploration pressure fighting it) to fully collapse onto than learning real
# tracking -- confirmed live via sniperbot_debug: the deployed bot span
# continuously at ~TURN_RATE_DEG_PER_SEC while firing almost every tick, and
# the toy sim's own 600/600 eval was the exact same behavior, just never
# rendered/inspected, only win/loss-counted. Requiring a held aim closes the
# exploit at the mechanic, not just the reward -- a fast spin resets the
# streak every frame, so it can never accumulate enough to land a hit. 3
# matches the "~3 steps of scoping" feel MIN_CHARGE_FOR_HEADSHOT already
# established elsewhere in this file.
#
# 2026-09-03: gating BOTH sides' fire on this (N=3, then N=2 -- two 2M-step
# trials each) reverted to the exact "avoid engaging entirely" passivity floor
# this project already fought once (ep_rew_mean flat negative, std creeping UP
# instead of shrinking). Root cause was the *symmetry*, not the threshold
# value -- see the step()-site comment where this is applied. Fixed by only
# gating the agent's own shots (required_streak=REQUIRE_SUSTAINED_AIM_STEPS)
# and leaving the scripted opponent's shots ungated (required_streak=1, i.e.
# its original, already-stable single-on-target-frame behavior).
#
# 2026-09-04: even asymmetric, N=3 still failed at the difficulty curriculum's
# high end -- confirmed via a diagnostic trial that pinned difficulty at 1.0
# for a full 1M steps (ruling out "curriculum ramps too fast for a short
# trial" as the cause): std never recovered, climbing 0.94->1.13 and
# plateauing there instead of shrinking, reward stuck negative throughout.
# Also tried adding STREAK_SHAPING_SCALE (reward that ramps with streak
# progress, see below) to rule out "no gradient toward holding aim" -- same
# failure curve almost exactly, so that wasn't it either. Conclusion: holding
# 3 consecutive on-target+LOS frames while a difficulty=1.0 scripted opponent
# (near-zero aim noise, ~70%/frame fire chance once lined up, no sustained-aim
# gate on ITS shots) shoots back is genuinely too hard to learn, independent
# of training-time/pacing/shaping-gradient fixes. Dropped to 2 -- still blocks
# the original single-frame spin exploit (a spin only sweeps across the
# target for one frame per rotation, so back-to-back frames still forces an
# actual stop) while roughly halving exposure time against the hardest
# opponent. Re-validate before trusting this.
REQUIRE_SUSTAINED_AIM_STEPS = 2

TERMINAL_REWARD = 100.0  # magnitude of the win/loss reward, must dominate shaping
# 2026-08-25: raised SHAPING_SCALE/LOS_SHAPING_SCALE ~5x (were 0.01/0.003) after two
# curriculum runs (ent_coef=0.0 and 0.003) both converged to the exact "never get LOS,
# just eat STEP_PENALTY" floor (-0.9 == -STEP_PENALTY*300) within ~700k steps and never
# escaped it for the rest of a 1M-step run. Root cause: getting LOS only beat passivity by
# LOS_SHAPING_SCALE - STEP_PENALTY = 0.000/step (they were equal), and it came with real
# risk (opponent can shoot back once LOS is mutual) for that zero net gain -- nothing
# pulled the policy out of the safe corner once it found it, regardless of entropy. Now
# LOS-only clearly beats passivity (+0.017 vs -0.003/step) and being fully aimed is a
# clear win (+0.047/step), while staying tiny next to TERMINAL_REWARD=100 for the
# actual win/loss decision (0.05 * 300 = 15, well under 100).
SHAPING_SCALE = 0.05     # per-step reward for being aimed at the opponent (with LOS)
MISS_PENALTY = 0.02      # small per-shot cost when firing lands no damage -- discourages
                          # constant spam-fire (the toy env has no ammo limit, so without
                          # this a policy has zero incentive to hold fire until actually aimed)
# 2026-08-25: tried splitting this into a heavier BLIND_FIRE_PENALTY for firing with no LOS
# at all, after live in-game testing showed the bot holding the fire button constantly
# including through cover. Reverted -- even a mild 1.5x delta (0.03 vs 0.02) reproduced the
# exact "never get LOS at all" passivity floor this project fought earlier (see reset()'s
# spawn-curriculum comment), isolated via a 1M-step trial with SCOPE_CHARGE_SHAPING_SCALE
# disabled to rule that out first. Root cause: blind fire is the *majority* case during
# exploration in the harder curriculum phase (before positioning is learned), so even a
# small per-shot increase raises the *average* cost of ever attempting to engage enough to
# teach full retreat again -- this axis is too fragile to tune via reward shaping without
# more risk than it's worth. Fixed at the deployment layer instead: tf_sniper_bot.cpp's
# ApplyAction now hard-gates the fire button on the engine's real FVisible() check,
# regardless of what the policy outputs, so this doesn't need to be solved by training at
# all.
STEP_PENALTY = 0.003     # tiny constant per-step cost, regardless of action -- passivity
                          # (never engaging, riding out the timeout) nets exactly 0 reward
                          # otherwise, an easy local optimum to slide into once engaging gets
                          # even slightly harder. Makes standing around strictly worse than
                          # trying, without being large enough to distort the terminal
                          # win/loss incentive (0.003 * 300 max steps = 0.9, tiny next to
                          # TERMINAL_REWARD=100).
LOS_SHAPING_SCALE = 0.02  # reward for LOS alone, independent of being on-target -- without
                            # this, "blocked by cover" and "actively maneuvering toward an
                            # open sightline" score identically (0 reward) right up until the
                            # instant everything lines up. SHAPING_SCALE (on-target AND LOS
                            # together) is worth more, so this only adds a gradient toward
                            # "getting closer", it doesn't replace the incentive to actually
                            # finish aiming.
# 2026-09-04: added after the asymmetric REQUIRE_SUSTAINED_AIM_STEPS fix (see above) still
# failed a full 2M-step trial -- reward/std looked healthy while the curriculum kept the
# opponent weak (difficulty <0.5), then reward went back to flat-negative and std climbed
# right back up (0.85->1.16) once difficulty ramped past 0.5. Hypothesis: SHAPING_SCALE pays
# the exact same flat reward for a single on-target frame as for a held one, so there was no
# gradient actually teaching the policy to hold aim through REQUIRE_SUSTAINED_AIM_STEPS --
# only the terminal +-100 ever distinguished them, and that signal gets sparser as a sharper
# opponent makes full engagements harder to complete. This adds reward that scales with how
# far into the required streak the agent already is, on top of (not instead of) the existing
# flat SHAPING_SCALE, so committing to a hold is visibly better every single step, not just
# on the eventual payoff.
STREAK_SHAPING_SCALE = 0.03
# 2026-08-25: live testing also showed the bot never scoping in, even though a scoped
# headshot deals 3-9x an unscoped body shot (MIN/MAX_HEADSHOT_DAMAGE vs
# UNSCOPED_HIT_DAMAGE) -- the scripted opponent dies in ~3 unscoped body shots well within
# an average ~16-step engagement, so the terminal +100 never needed the scope to arrive.
# Tried adding a direct per-step bonus for holding scope charge while aimed+LOS
# (SCOPE_CHARGE_SHAPING_SCALE=0.03) to make charging worth more than firing immediately
# unscoped -- reverted after a 1M-step validation reproduced the exact same "never get LOS
# at all" passivity collapse as the abandoned BLIND_FIRE_PENALTY attempt above, isolated
# with that change alone (no blind-fire penalty active). Likely mechanism: rewarding
# lingering in the aimed+LOS state to build charge also means lingering exposed to the
# opponent's return fire for longer, and during the still-learning mid-curriculum phase
# that extra death risk apparently taught avoidance again rather than patience. Both of
# this project's two attempts at rewarding "better" combat habits beyond the proven
# SHAPING_SCALE/LOS_SHAPING_SCALE/MISS_PENALTY/STEP_PENALTY set destabilized training the
# same way -- that reward function is more fragile to additions than it looks, so treat
# further changes here as high-risk and validate hard before trusting them.

# --- scripted opponent (2026-08-25 rewrite) ---------------------------------
# Self-play (mirrored dual-role training against a frozen copy of itself) was
# scrapped: it produced a whole class of bugs (spawn-side asymmetry, then the
# observation/action mirroring needed to fix that, then both sides
# co-evolving into the same degenerate wall-hugging equilibrium since they're
# literally the same weights) and, even when "working", only ever optimized
# the policy to beat *its own reflection* -- not something that generalizes
# to a human opponent. Training now always puts the learning agent at
# SELF_SPAWN in the canonical frame (matching bot_rl_solo's real deployment:
# one RL bot on RED, a human on BLU) against a hand-scripted BLU opponent
# whose skill ramps up over training (see set_difficulty / train.py's
# curriculum callback) -- so the final policy has to get good at beating
# increasingly sharp, human-like play, not just itself.
OPPONENT_ENGAGE_RANGE = 250.0  # scripted opponent closes distance until within this range

# 2026-08-25: live testing surfaced a real generalization gap, not a training-stability
# bug like the ones above. The scripted opponent always closed distance when far away --
# in a live console session where the human player just stood still at spawn, the bot
# wandered aimlessly and never got LOS for 10+ ticks (opp_real_pos frozen the whole time),
# because "opponent eventually approaches" was baked into every single training episode.
# Same root cause explains it retreating to the exact map corner at low HP and camping
# there until it died to a real player who just walked up -- retreat-and-snipe only ever
# had to survive an opponent that kept obligingly closing the distance itself. Randomizing
# the scripted opponent's whole approach style per episode forces the policy to learn to
# hunt/reposition on its own instead of just reacting to a predictable approach pattern.
OPPONENT_STYLES = ("aggressive", "camper", "retreater")


class SniperDuelEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 0.0 = slow, noisy, hesitant scripted opponent (easy on-ramp early
        # in training); 1.0 = sharp aim, fast turns, fires the instant it's
        # lined up -- the "OP" endpoint the curriculum ramps toward. See
        # set_difficulty and _scripted_opponent_action.
        self.difficulty = 0.0

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

    def set_difficulty(self, difficulty):
        # called from train.py's curriculum callback via env_method, so each
        # SubprocVecEnv worker ramps its own scripted opponent in step.
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # environment-side curriculum, not just opponent-side: three straight
        # 1M-step trials (ent_coef 0.0, 0.003, and boosted SHAPING_SCALE on
        # top of that) all converged to the exact "never once get LOS"
        # floor and stayed there. Root cause turned out to be the map, not
        # the reward -- a direct scan showed the real spawn line (y=0) has
        # almost no reachable position with clear LOS to the opponent (the
        # crate+pillars block nearly the whole corridor except thin slivers
        # near the arena edges), so undirected Gaussian exploration was
        # essentially never going to stumble into the flanking position
        # needed to earn *any* shaping reward, regardless of its magnitude.
        # EASY_SPAWN_Y sits in the one confirmed-clear horizontal gap
        # between the crate's top (y=109) and the lower pillars' bottom
        # (y=144.667) -- at difficulty=0 both spawns sit on it, so the agent
        # can learn "see it, aim, shoot" while stationary, before difficulty
        # blends the spawn back down to the real (cover-blocked) line, by
        # which point engaging is already a learned habit worth navigating
        # cover to keep doing. Y jitter scales in alongside the blend so an
        # easy-mode reset doesn't occasionally jitter back into the crate's
        # shadow.
        spawn_y = EASY_SPAWN_Y * (1.0 - self.difficulty)
        y_jitter = SPAWN_JITTER * self.difficulty

        self._self_pos = np.array([SELF_SPAWN[0], spawn_y], dtype=np.float32)
        self._self_pos[0] += self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER)
        self._self_pos[1] += self.np_random.uniform(-y_jitter, y_jitter)

        self._opponent_pos = np.array([OPPONENT_SPAWN[0], spawn_y], dtype=np.float32)
        self._opponent_pos[0] += self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER)
        self._opponent_pos[1] += self.np_random.uniform(-y_jitter, y_jitter)

        self._self_angle = SELF_SPAWN_ANGLE
        self._opponent_angle = OPPONENT_SPAWN_ANGLE

        self._self_scope_active = False
        self._self_scope_charge = 0.0
        self._opponent_scope_active = False
        self._opponent_scope_charge = 0.0

        self._self_health = MAX_HEALTH
        self._opponent_health = MAX_HEALTH

        # see REQUIRE_SUSTAINED_AIM_STEPS above
        self._self_aim_streak = 0
        self._opponent_aim_streak = 0

        self._step_count = 0

        # see OPPONENT_STYLES comment above -- picked fresh each episode so
        # training sees a genuine mix of approach patterns, not one habit.
        self._opponent_style = self.np_random.choice(OPPONENT_STYLES)

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

        if self._point_in_any_barrier(new_pos, padding=PLAYER_COLLISION_RADIUS):
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

    def _point_in_barrier(self, point, barrier, padding=0.0):
        x, y = point
        x_min, x_max, y_min, y_max = barrier

        if (x_min - padding) <= x <= (x_max + padding) and (y_min - padding) <= y <= (y_max + padding):
            return True
        else:
            return False

    def _lerp_point(self, a, b, fraction):
        return a + fraction * (b - a)

    def _point_in_any_barrier(self, point, padding=0.0):
        for barrier in BARRIERS:
            if self._point_in_barrier(point, barrier, padding):
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

    def _resolve_fire(self, aimed, aim_streak, shooter_scope_active, shooter_scope_charge, fire_signal, required_streak):
        if fire_signal <= 0.0:
            return 0.0
        if not aimed:
            return 0.0

        if shooter_scope_active and shooter_scope_charge >= MIN_CHARGE_FOR_HEADSHOT:
            # 2026-09-04: NOT gated on aim_streak -- see REQUIRE_SUSTAINED_AIM_STEPS
            # above. A headshot already needs several steps of holding scope charge
            # (MIN_CHARGE_FOR_HEADSHOT) before it can land at all, which is its own
            # windup a fast spin can't fake; the original spin-and-spray exploit was
            # entirely on UNSCOPED fire (zero windup of any kind). Two rounds of
            # trials (N=3, then N=2, both asymmetric) gating headshots too still
            # failed once the opponent curriculum reached full difficulty -- std
            # never stopped climbing even given a full 1M held-difficulty steps to
            # recover. Exempting headshots removes a redundant tax on the one shot
            # type the agent most needs to stay competitive against a razor-sharp
            # opponent, without reopening the exploit this was meant to close.
            charge_t = (shooter_scope_charge - MIN_CHARGE_FOR_HEADSHOT) / (1.0 - MIN_CHARGE_FOR_HEADSHOT)
            return MIN_HEADSHOT_DAMAGE + charge_t * (MAX_HEADSHOT_DAMAGE - MIN_HEADSHOT_DAMAGE)

        # unscoped (or scoped-but-undercharged) fire has zero windup otherwise --
        # see REQUIRE_SUSTAINED_AIM_STEPS above -- a single on-target frame is not
        # enough to land this shot, closing the spin-and-spray exploit at its
        # actual source. required_streak is 1 (i.e. no extra gate) for the
        # scripted opponent's own shots -- see the step()-site comment for why.
        if aim_streak < required_streak:
            return 0.0

        return UNSCOPED_HIT_DAMAGE

    def _scripted_opponent_action(self):
        # hand-authored BLU opponent, not a learned policy -- see the
        # "scripted opponent" block comment above for why. Turns toward self
        # and fires when aimed with LOS regardless of style; `self.difficulty`
        # (0..1, ramped by train.py's curriculum callback) scales aim noise/
        # turn speed/shot commitment so early training faces an easy, sloppy
        # opponent and late training faces a sharp one. `self._opponent_style`
        # (picked fresh each episode, see OPPONENT_STYLES) controls whether it
        # closes distance, holds ground, or keeps its distance -- see the
        # forward-movement branch below.
        to_self = self._self_pos - self._opponent_pos
        distance = np.linalg.norm(to_self)
        angle_to_self = np.degrees(np.arctan2(to_self[1], to_self[0]))
        angle_diff = ((angle_to_self - self._opponent_angle + 180.0) % 360.0) - 180.0

        aim_noise_deg = (1.0 - self.difficulty) * 30.0
        noisy_diff = angle_diff + self.np_random.uniform(-aim_noise_deg, aim_noise_deg)
        turn = np.clip(noisy_diff / MAX_TURN_PER_STEP_DEG, -1.0, 1.0)
        turn *= 0.4 + 0.6 * self.difficulty  # sluggish turn-in at low difficulty

        has_los = self._line_of_sight_clear(self._opponent_pos, self._self_pos)
        on_target = abs(angle_diff) <= AIM_TOLERANCE_DEG

        if has_los and on_target:
            # lined up -- hold ground and peek/strafe a little instead of
            # standing bolt still.
            strafe = self.np_random.uniform(-1.0, 1.0) * 0.3
            fwd = 0.0
        else:
            strafe = self.np_random.uniform(-1.0, 1.0) * 0.5
            if self._opponent_style == "aggressive":
                # closes distance until within range -- the only style this
                # project originally trained against exclusively, which is
                # exactly what taught the policy to just wait for the
                # opponent to come to it.
                fwd = 1.0 if distance > OPPONENT_ENGAGE_RANGE else 0.0
            elif self._opponent_style == "camper":
                # holds ground regardless of distance -- forces the RL agent
                # to learn to close distance and hunt for LOS itself instead
                # of relying on the opponent to walk into view.
                fwd = 0.0
            else:  # "retreater"
                # keeps its distance, backing off if the agent gets close --
                # a kiting opponent the agent has to run down, not just wait
                # out.
                fwd = -1.0 if distance < OPPONENT_ENGAGE_RANGE else 0.0

        want_scope = 1.0 if (has_los and abs(angle_diff) <= AIM_TOLERANCE_DEG * 2.0) else -1.0

        # quadratic, not linear, in difficulty -- at difficulty=0 this was
        # 0.3 (still real lethality on day one of training), which combined
        # with LOS being risky punished the agent for ever exploring into a
        # sightline before it had any chance to discover engaging pays off.
        # Squaring keeps early-curriculum near-harmless (0.1 floor) and only
        # ramps real threat in during the back half of training, once
        # engaging is already a learned habit.
        fire_chance = 0.1 + 0.6 * (self.difficulty ** 2)
        fire = 1.0 if (has_los and on_target and self.np_random.uniform() < fire_chance) else -1.0

        return np.array([strafe, fwd, turn, want_scope, fire], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        opponent_action = self._scripted_opponent_action()

        self._self_pos, self._self_angle = self._move_agent(
            self._self_pos, self._self_angle, action
        )
        self._opponent_pos, self._opponent_angle = self._move_agent(
            self._opponent_pos, self._opponent_angle, opponent_action
        )

        self._self_scope_active, self._self_scope_charge = self._update_scope(
            self._self_scope_active, self._self_scope_charge, action
        )
        self._opponent_scope_active, self._opponent_scope_charge = self._update_scope(
            self._opponent_scope_active, self._opponent_scope_charge, opponent_action
        )

        # sustained-aim tracking (see REQUIRE_SUSTAINED_AIM_STEPS) -- computed
        # once here so both fire resolution and the reward block below (which
        # used to recompute the same has_los/on_target checks independently)
        # share a single, consistent source of truth per step.
        #
        # 2026-09-04: decays by 1 on a miss instead of hard-resetting to 0 --
        # a control trial (gate fully removed) proved the gate itself, not
        # difficulty or training pacing, was what destabilized training at
        # high opponent difficulty (std climbed 0.9->1.3 with the hard-reset
        # gate in three different variants, but stayed flat ~0.85-0.87 for
        # the whole held-at-max-difficulty window with no gate at all). A
        # hard reset punishes one bobbled frame while genuinely tracking a
        # moving, shooting-back target exactly as hard as never having aimed
        # at all, forcing a full rebuild from scratch -- against a
        # difficulty=1.0 opponent (near-zero aim noise, fast turn-in) that's
        # an unrealistic bar. Decay still can't be satisfied by a spin or a
        # single flicker (off-frames dominate on-frames in both, so the
        # streak nets toward 0 over time), but tolerates a single miss
        # without erasing an otherwise-real hold. `aimed` (this frame) is
        # still a separate hard requirement in _resolve_fire regardless of
        # streak value, so this can't let a shot land while off-target.
        self_has_los = self._line_of_sight_clear(self._self_pos, self._opponent_pos)
        self_aimed = self_has_los and self._is_on_target(self._self_pos, self._self_angle, self._opponent_pos)
        self._self_aim_streak = self._self_aim_streak + 1 if self_aimed else max(0, self._self_aim_streak - 1)

        opponent_has_los = self._line_of_sight_clear(self._opponent_pos, self._self_pos)
        opponent_aimed = opponent_has_los and self._is_on_target(self._opponent_pos, self._opponent_angle, self._self_pos)
        self._opponent_aim_streak = self._opponent_aim_streak + 1 if opponent_aimed else max(0, self._opponent_aim_streak - 1)

        # 2026-09-03: gating BOTH sides on REQUIRE_SUSTAINED_AIM_STEPS
        # (validated via two 2M-step trials, N=3 and N=2) reintroduced the
        # exact "avoid LOS entirely" passivity collapse this project already
        # fought once -- ep_rew_mean stayed negative and std crept UP
        # instead of shrinking in both trials. Root cause: the scripted
        # opponent already tracks near-perfectly and holds position once
        # locked on (see _scripted_opponent_action's on-target branch), so it
        # builds its own streak easily even against an early, undertrained
        # agent -- while the *agent* has to learn to hold still to ever
        # build its own streak. That's an asymmetric nerf: engaging got
        # riskier for the agent without getting any less dangerous from the
        # opponent, undoing the "engaging clearly beats passivity" balance
        # SHAPING_SCALE/STEP_PENALTY were tuned around. Only gating the
        # agent's own shots leaves the opponent's threat model exactly as it
        # was in the last known-stable config, closing the actual complained-
        # about behavior (the agent's own spin-and-spray) without touching
        # what wasn't broken.
        damage_to_opponent = self._resolve_fire(
            self_aimed, self._self_aim_streak,
            self._self_scope_active, self._self_scope_charge,
            action[4], REQUIRE_SUSTAINED_AIM_STEPS,
        )
        damage_to_self = self._resolve_fire(
            opponent_aimed, self._opponent_aim_streak,
            self._opponent_scope_active, self._opponent_scope_charge,
            opponent_action[4], 1,
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
            # reuse the has_los/aimed values computed above, alongside the
            # aim-streak tracking -- require actual LOS, not just angle,
            # otherwise "facing the opponent's raw bearing through a wall"
            # farms the same reward as genuinely having them in your sights,
            # with none of the risk.
            has_los = self_has_los
            aimed_at_opponent = self_aimed
            if aimed_at_opponent:
                # see STREAK_SHAPING_SCALE above -- ramps 0 -> full across the
                # first REQUIRE_SUSTAINED_AIM_STEPS frames of a held aim, on
                # top of the flat SHAPING_SCALE every aimed frame already got.
                streak_progress = min(self._self_aim_streak, REQUIRE_SUSTAINED_AIM_STEPS) / REQUIRE_SUSTAINED_AIM_STEPS
                reward = SHAPING_SCALE + STREAK_SHAPING_SCALE * streak_progress
            elif has_los:
                # partial credit for having a real sightline even before
                # being aimed -- otherwise "blocked by cover" and "actively
                # maneuvering toward an open angle" both score 0, with no
                # gradient telling the policy it's making progress.
                reward = LOS_SHAPING_SCALE
            else:
                reward = 0.0

            # discourage constant spam-fire -- the env has no ammo limit, so
            # without a cost the policy has no reason to ever hold fire.
            if action[4] > 0.0 and damage_to_opponent <= 0.0:
                reward -= MISS_PENALTY

            # discourage riding out the clock -- passivity would otherwise
            # net exactly 0 reward forever, an easy local optimum once
            # engaging gets even slightly harder.
            reward -= STEP_PENALTY

        terminated = self_dead or opponent_dead
        truncated = self._step_count >= MAX_EPISODE_STEPS
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self): #get observation. basically packaging all the data so it fits the observation rulespace
        time_left = 1.0 - (self._step_count / MAX_EPISODE_STEPS)

        other_visible = self._line_of_sight_clear(self._self_pos, self._opponent_pos)
        if other_visible:
            other_pos_obs = self._opponent_pos.astype(np.float32).copy()
        else:
            other_pos_obs = np.zeros(2, dtype=np.float32)

        return {
            "self_pos": self._self_pos.astype(np.float32).copy(),
            "self_angle": np.array([self._self_angle], dtype=np.float32),
            "scope_active": np.array([1.0 if self._self_scope_active else 0.0], dtype=np.float32),
            "scope_charge": np.array([self._self_scope_charge], dtype=np.float32),
            "opponent_pos": other_pos_obs,
            "opponent_visible": np.array([1.0 if other_visible else 0.0], dtype=np.float32),
            "time_left": np.array([time_left], dtype=np.float32),
            "self_health": np.array([self._self_health / MAX_HEALTH], dtype=np.float32),
        }


if __name__ == "__main__":
    env = SniperDuelEnv()
    env.set_difficulty(1.0)
    obs, info = env.reset(seed=42)
    print("observation:", obs)
    print("info:", info)
    print("valid according to observation_space?", env.observation_space.contains(obs))

    print("\nself_pos before step:", env._self_pos)
    move_right_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(move_right_action)
    print("self_pos after step:", obs["self_pos"])
    print("valid according to observation_space?", env.observation_space.contains(obs))
