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

# 2026-09-09: THE core mechanic that was missing, and the reason spin-and-spray
# was viable at all. Until now either side could fire on EVERY tick -- there is
# no ammo limit here, so "hold the trigger down forever" was legal. TF2's real
# Sniper Rifle has ~1.5s of refire time, which at this env's step rate
# (EPISODE_DURATION 20s / MAX_EPISODE_STEPS 300 = 0.067 s/step) is ~22 steps.
# So the sim was permitting something physically impossible in the deployment
# target -- a sim-to-real mismatch, not just a balance quirk.
#
# That mismatch is what made the exploit pay: sweeping a crosshair across the
# target while firing every tick lands roughly 8 hits per episode (on-target is
# ~10 of 360 degrees, ~2.8% of ticks, over 300 ticks) when only 3 body shots
# are needed to kill. With a real cooldown the same spin gets ~13 shots at
# ~2.8% blind hit chance -- about 0.4 expected hits, i.e. useless -- while
# actually aiming gets 3 aimed shots and a kill. The exploit dies on economics,
# with no need to forbid the agent from turning.
#
# Crucially this is applied SYMMETRICALLY. Every previous attempt to kill the
# exploit constrained only the agent's fire, which made engaging strictly
# riskier for it while the opponent stayed unconstrained -- and at high
# difficulty that flipped engagement to negative expected value and collapsed
# the policy into never taking line of sight at all (confirmed behaviorally:
# 0.0% LOS, 0% fire rate, 60/60 timeouts). A symmetric cooldown instead cuts
# the scripted opponent's damage output hard (from up to ~0.7 shots/tick once
# lined up down to one shot per cooldown), which relieves that pressure rather
# than adding to it.
FIRE_COOLDOWN_STEPS = 22

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
# rendered/inspected, only win/loss-counted.
#
# 2026-09-03 through 2026-09-08: spent roughly a dozen 2M-6M-step trials
# trying to close this by gating the agent's own fire on SOME condition tied
# to how it was aiming -- N consecutive on-target+LOS frames (hard-reset at
# N=3 and N=2, symmetric and asymmetric), the same streak but decaying by 1
# on a miss instead of hard-resetting, an added reward gradient toward
# holding aim, headshots exempted from the gate, and finally an instantaneous
# (no history at all) check on the shooter's current turn-rate at the moment
# of firing. One combination (decay-on-miss + N=2 + headshot-exempt) even
# validated clean on a full 2M-step held-at-max-difficulty run. Every single
# one of these -- all the way down to the memory-free turn-rate version --
# failed the same way once tested against the REAL curriculum's actual
# shape: a continuous 0->1 ramp with no hold. std climbed steadily and never
# recovered as difficulty rose past ~0.1-0.3, reproduced in the real 50M-step
# run (0.24->1.51 by difficulty 0.53), a 6M-step trial of the decay/N=2/
# headshot-exempt fix (0.77->1.01+ by difficulty ~0.5), and again in a 6M-step
# trial of the turn-rate version (0.79->1.3+ by difficulty ~0.68) -- three
# structurally different mechanics, identical failure curve. A held-difficulty
# harness made some of these look fixed only because its ramp physically
# stopped moving partway through and held there; that's an artifact of the
# harness, not evidence any of the fixes were sound, and it never applies to
# the real, continuously-ramping curriculum.
#
# The decisive test: run the ORIGINAL, fully ungated mechanic (below) through
# that same continuous-ramp harness. std shrank smoothly and monotonically
# the entire way (0.92 -> 0.47 by difficulty 0.68, still falling) -- clean,
# with zero instability, at exactly the difficulty range where every gated
# variant blew up. That rules out the ramp itself, or ent_coef=0.0, or the
# reward economy as the cause: this exact env, unmodified, trains fine.
# It's specifically ANY constraint placed on the agent's own fire -- no
# matter how it's implemented -- that breaks PPO's training dynamics here.
# Best guess why: it's an asymmetric nerf that compounds. The opponent's
# shots stay ungated the whole time (by design, to protect its threat
# model), so as its own aim/fire-chance sharpen with difficulty, the agent
# is simultaneously being asked to clear an extra bar just to return fire at
# all -- and unlike a fixed-difficulty opponent, there's no difficulty level
# where that tradeoff ever settles into an equilibrium the policy can lock
# onto; the target keeps moving. A single-step, history-free check should in
# principle have been the easiest version of this to learn, and it failed
# just as hard as the others -- strong evidence the complexity of the gate
# was never the actual problem, just any bar placed on it at all.
#
# 2026-09-09: the above conclusion was WRONG, and the way it was reached is
# worth recording. "Revert the gate, fix it at the deployment layer" was
# validated on std/ep_rew_mean/ep_len_mean only -- never on what the policy
# actually DID. A behavioral probe of the resulting 50M-step policy (see
# scratchpad probe_behavior.py) found textbook spin-and-spray: yaw sweeping
# monotonically without ever settling, fire held on 100% of ticks, mean aim
# error 94 deg, on-target just 2.4% of ticks. The ungated run's smooth,
# fast-shrinking std was never "healthy training" -- it was rapid, clean
# convergence onto the exploit, which genuinely IS near-optimal when a single
# on-target frame is enough to land a shot. That also flips the read on the
# gated runs: their climbing std was more likely a policy still searching
# under a hard, sparse reward, not broken training dynamics. This is the same
# mistake as the original 600/600 eval that hid this behavior in the first
# place -- judging by aggregate metrics that a degenerate policy scores well
# on. Any verdict here needs the behavioral probe, not just the curves.
#
# It also sinks the deployment-layer plan on its own terms: with the policy
# spinning on essentially every tick (|turn| < 0.35 on 0.0% of them), a
# turn-rate gate in tf_sniper_bot.cpp suppresses 100% of its shots -- the bot
# would never fire at all. A deployment gate can only clean up behavior the
# policy already exhibits sometimes; it cannot create it.
#
# 2026-09-09, final: tried one more gate (instantaneous turn-rate) together
# with the dense ALIGN_SHAPING_SCALE slope described below, on the theory that
# the gates had failed only for lack of a learnable gradient. It failed too,
# and the behavioral probe showed the OTHER failure mode this project knows
# well: the agent stopped spinning (|turn| median 0.267, under the gate) but
# answered "you must hold still to shoot" with "then I never engage" -- 0.0%
# of ticks with line of sight, 0% fire rate, 60/60 timeouts, zero kills either
# way. That is the documented passivity floor, and it makes the mechanism
# behind all five gate attempts concrete: gating only the AGENT's fire, while
# the scripted opponent shoots freely, means engaging costs the agent more
# exactly as the opponent grows more lethal, until hiding is simply the better
# play. The gate was never the wrong idea in isolation -- the ASYMMETRY was.
#
# So: no gate on the agent at all. FIRE_COOLDOWN_STEPS (see above) removes the
# exploit's payoff symmetrically instead, by fixing the underlying sim-to-real
# bug that made spamming fire legal in the first place. The dense alignment
# slope below stays, since aiming now genuinely pays and the policy needs a
# gradient to learn it.

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
# 2026-09-09: was 0.02, a per-shot cost for firing without landing damage. Its
# stated purpose was to discourage constant spam-fire, "the toy env has no ammo
# limit, so without this a policy has zero incentive to hold fire until actually
# aimed" -- which FIRE_COOLDOWN_STEPS now enforces directly and much better. Left
# at 0.02 alongside the cooldown it became actively harmful: the cooldown cuts
# shots per episode from 300 to ~13, so positive reinforcement for "firing
# sometimes connects" got ~23x sparser, while this penalty stayed immediate and
# certain. A 6M-step trial with both active learned the obvious response and
# stopped firing altogether -- trigger pulled on 0.0% of ticks at every
# difficulty, zero shots, zero wins. Zeroed out rather than deleted so the
# reasoning stays attached to the knob if anyone reaches for it again.
MISS_PENALTY = 0.0
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

# 2026-09-09: the missing piece behind every failed attempt at gating fire.
# Until now the aim reward was BINARY: SHAPING_SCALE the instant |aim error|
# fell under AIM_TOLERANCE_DEG (5 deg), LOS_SHAPING_SCALE otherwise. Going
# from 94 deg of error to 10 deg earned exactly zero extra reward, so with a
# fire gate active the policy had no gradient telling it it was getting
# warmer -- it had to randomly stumble into a 10-deg-wide window out of a
# full 360 to get any signal at all. That needle-in-a-haystack search, not
# "gates break PPO", is the far better explanation for why four different
# fire gates all failed to converge: they made the cheap exploit unavailable
# without ever making real aiming *learnable*.
#
# This adds a dense term that grows smoothly as aim error shrinks, scaled
# from 0 at ALIGN_SHAPING_MAX_ERROR_DEG of error up to its full value at
# perfectly on-target, and only paid while LOS is clear (same reasoning as
# LOS_SHAPING_SCALE -- don't pay for lining up on a wall). Deliberately kept
# smaller than SHAPING_SCALE so actually reaching on-target still dominates;
# this only supplies the slope leading there.
#
# Worth distinguishing from the two reward additions this file already
# records as failures (BLIND_FIRE_PENALTY, SCOPE_CHARGE_SHAPING_SCALE): both
# of those added *cost or risk* to engaging, which is what tipped training
# back into the passivity floor. This adds gradient toward the objective and
# no new downside, which is the standard fix for exactly this sparse-binary
# -reward problem.
ALIGN_SHAPING_SCALE = 0.03
ALIGN_SHAPING_MAX_ERROR_DEG = 90.0  # aim error at/above which the dense term pays 0

# 2026-09-09: paid per step for holding scope while genuinely lined up, ramping
# with charge up to MIN_CHARGE_FOR_HEADSHOT.
#
# With FIRE_COOLDOWN_STEPS in place, scoping stopped being optional and became
# the only strategy that actually wins. Measured on this map, hand-written
# strategies over 60 episodes on identical seeds:
#
#   stand + unscoped body shots : 27W/19L (diff 0), 9W/12L (0.5), 0W/5L (1.0)
#   scope + charged headshot    : 59W/0L  (diff 0), 26W/0L  (0.5), 7W/0L  (1.0)
#
# The body-shot line can't win at high difficulty at all: line of sight is only
# available ~6% of ticks there, and three cooldown-spaced body shots need 45+
# steps of it. A charged headshot deals 150-450 against 125 max health, so it
# one-shots from a single LOS window -- which is what a cover-heavy map like
# this actually offers. Note the scoped line takes ZERO losses at every
# difficulty; it kills before the opponent can stack up three body shots.
#
# This file already records an abandoned SCOPE_CHARGE_SHAPING_SCALE=0.03 from
# 2026-08-25 that caused a passivity collapse, reasoned as "lingering in the
# aimed+LOS state to build charge also means lingering exposed to the
# opponent's return fire". That objection was correct then and is obsolete
# now: back then fire had no cooldown, so lingering meant absorbing up to one
# opponent shot per tick, and body-shot spam was good enough that scoping was
# a pure cost. With the cooldown the opponent fires at most once per
# FIRE_COOLDOWN_STEPS, and scoping is the winning line rather than a detour --
# so this reward now points at the true optimum instead of away from it.
SCOPE_SHAPING_SCALE = 0.04
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

# 2026-09-09: floor on how often an episode still starts on the clear-sightline
# spawn (EASY_SPAWN_Y), no matter how high difficulty has climbed. See reset().
#
# The old scheme interpolated the spawn y coordinate itself --
# spawn_y = EASY_SPAWN_Y * (1 - difficulty) -- intending a smooth blend from
# "guaranteed line of sight" to "the real, cover-blocked spawn line". In
# practice the map geometry turns that into a cliff, not a blend: the middle
# crate's top edge is at y=109, so the sightline is clear only while
# spawn_y > 109, i.e. difficulty < ~0.13. Measured directly, LOS at spawn goes
# 100% at difficulty 0.00 -> 0% by difficulty 0.25 and stays at 0% forever
# after. So the agent gets a reliable engagement signal for the first ~13% of
# training and then never sees one again, which extinguishes whatever it had
# learned and leaves nothing to relearn from -- a very good candidate for the
# "never get LOS" passivity floor this file keeps rediscovering.
#
# Mixing whole episodes instead of interpolating a coordinate gives a curriculum
# that is actually gradual in the thing that matters (how often engaging is
# even possible), and the floor guarantees the signal never disappears
# entirely: even at difficulty 1.0 this fraction of episodes still start with a
# clear shot, so "engage" stays reinforced while the rest of the episodes teach
# fighting from the real cover-blocked spawns.
MIN_EASY_SPAWN_FRACTION = 0.15


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
        # (y=144.667), so an episode starting there has a clear shot the full
        # width of the arena and the agent can learn "see it, aim, shoot"
        # while stationary.
        #
        # 2026-09-09: this used to interpolate the coordinate itself
        # (spawn_y = EASY_SPAWN_Y * (1 - difficulty)), which reads like a
        # smooth blend but isn't one -- the crate's top edge at y=109 means
        # the sightline is clear only while spawn_y > 109, i.e. difficulty
        # below ~0.13. Measured, LOS at spawn was 100% at difficulty 0.00 and
        # already 0% by 0.25, staying there for the rest of the run. Now the
        # curriculum mixes whole episodes instead: each reset independently
        # picks the easy line or the real one, with the easy share falling
        # from 1.0 to MIN_EASY_SPAWN_FRACTION as difficulty climbs. That makes
        # the ramp gradual in the quantity that actually matters -- how often
        # engaging is possible at all -- and the floor keeps the signal alive
        # instead of switching it off entirely partway through training.
        easy_fraction = max(MIN_EASY_SPAWN_FRACTION, 1.0 - self.difficulty)
        easy_spawn = self.np_random.uniform() < easy_fraction

        if easy_spawn:
            # jitter stays small here so an easy reset can't land in the
            # crate's shadow and quietly stop being an easy reset.
            spawn_y = EASY_SPAWN_Y
            y_jitter = SPAWN_JITTER * 0.25
        else:
            spawn_y = 0.0  # the real RED/BLU spawn line
            y_jitter = SPAWN_JITTER

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

        # see FIRE_COOLDOWN_STEPS -- both start ready to fire
        self._self_fire_cooldown = 0
        self._opponent_fire_cooldown = 0

        self._step_count = 0

        # see OPPONENT_STYLES comment above -- picked fresh each episode so
        # training sees a genuine mix of approach patterns, not one habit.
        self._opponent_style = self.np_random.choice(OPPONENT_STYLES)

        # 2026-09-09: whether this opponent scopes at all, decided per episode
        # so scope charge can build consistently within one.
        #
        # The difficulty curriculum scaled the opponent's aim noise, turn rate
        # and fire chance, but never its scoping -- so even at difficulty 0.0
        # it was a headshotting sniper, and a charged headshot (150-450) one-
        # shots a 125 HP target. Measured against an agent standing still in
        # the open, the difficulty-0.0 "easy on-ramp" opponent killed it in 31
        # of 60 episodes in ~70 steps, landing more headshots (30) than body
        # shots (18). That leaves no phase of training where engaging is
        # survivable while the agent is still learning to aim, which is the
        # bootstrap trap behind the passivity collapses: die instantly on
        # contact, learn to avoid contact, never learn to fight. Ramping scope
        # use with difficulty restores what the curriculum was supposed to
        # provide -- an early opponent that can only body-shot, needing three
        # hits and giving the agent room to make mistakes and still learn.
        self._opponent_uses_scope = self.np_random.uniform() < self.difficulty

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

    def _aim_error_deg(self, shooter_pos, shooter_angle, target_pos):
        to_target = target_pos - shooter_pos
        angle_to_target = np.degrees(np.arctan2(to_target[1], to_target[0]))
        angle_diff = ((angle_to_target - shooter_angle + 180.0) % 360.0) - 180.0
        return abs(angle_diff)

    def _is_on_target(self, shooter_pos, shooter_angle, target_pos):
        return self._aim_error_deg(shooter_pos, shooter_angle, target_pos) <= AIM_TOLERANCE_DEG

    def _resolve_fire(self, aimed, shooter_scope_active, shooter_scope_charge, fire_signal):
        """Damage dealt this step. Callers must check/clear the shooter's own
        cooldown -- see _try_fire, which owns that bookkeeping for both sides."""
        if fire_signal <= 0.0:
            return 0.0
        if not aimed:
            return 0.0

        if shooter_scope_active and shooter_scope_charge >= MIN_CHARGE_FOR_HEADSHOT:
            charge_t = (shooter_scope_charge - MIN_CHARGE_FOR_HEADSHOT) / (1.0 - MIN_CHARGE_FOR_HEADSHOT)
            return MIN_HEADSHOT_DAMAGE + charge_t * (MAX_HEADSHOT_DAMAGE - MIN_HEADSHOT_DAMAGE)

        return UNSCOPED_HIT_DAMAGE

    def _try_fire(self, cooldown, aimed, scope_active, scope_charge, fire_signal):
        """Resolve one side's shot subject to FIRE_COOLDOWN_STEPS.

        Returns (damage, new_cooldown, shot_taken). The cooldown only resets
        when a shot is actually taken (trigger pulled while off cooldown) --
        holding the trigger down through the cooldown doesn't stack up extra
        shots, and a shot that's taken but misses still spends the cooldown,
        so blind firing costs real opportunity instead of being free.

        shot_taken is what MISS_PENALTY keys off, so that holding the trigger
        during cooldown -- which the game simply ignores -- isn't punished. The
        agent has no cooldown field in its observation (deliberately: the real
        weapon in tf_sniper_bot.cpp handles refire itself while the bot just
        holds +attack, so this matches deployment), and penalizing every
        ignored trigger tick would have forced it to time shots blind.
        """
        if cooldown > 0:
            return 0.0, cooldown - 1, False
        if fire_signal <= 0.0:
            return 0.0, 0, False

        damage = self._resolve_fire(aimed, scope_active, scope_charge, fire_signal)
        # -1 so the shot-to-shot cadence is exactly FIRE_COOLDOWN_STEPS: this
        # tick fires, the next FIRE_COOLDOWN_STEPS-1 ticks tick the counter
        # down, and the one after that is free to fire again.
        return damage, FIRE_COOLDOWN_STEPS - 1, True

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

        # see _opponent_uses_scope in reset() -- scoping ramps in with
        # difficulty rather than being on from the very first training step.
        want_scope = 1.0 if (self._opponent_uses_scope
                             and has_los
                             and abs(angle_diff) <= AIM_TOLERANCE_DEG * 2.0) else -1.0

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

        # computed once here so both fire resolution and the reward block
        # below (which used to recompute the same has_los/on_target checks
        # independently) share a single, consistent source of truth per step.
        self_has_los = self._line_of_sight_clear(self._self_pos, self._opponent_pos)
        self_aim_error = self._aim_error_deg(self._self_pos, self._self_angle, self._opponent_pos)
        self_aimed = self_has_los and self_aim_error <= AIM_TOLERANCE_DEG

        opponent_has_los = self._line_of_sight_clear(self._opponent_pos, self._self_pos)
        opponent_aimed = opponent_has_los and self._is_on_target(self._opponent_pos, self._opponent_angle, self._self_pos)

        # symmetric -- both sides pay the same FIRE_COOLDOWN_STEPS. See its
        # comment above for why symmetry is the whole point here.
        damage_to_opponent, self._self_fire_cooldown, self_shot_taken = self._try_fire(
            self._self_fire_cooldown, self_aimed,
            self._self_scope_active, self._self_scope_charge, action[4],
        )
        damage_to_self, self._opponent_fire_cooldown, _ = self._try_fire(
            self._opponent_fire_cooldown, opponent_aimed,
            self._opponent_scope_active, self._opponent_scope_charge, opponent_action[4],
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
            # reuse the has_los/aimed values computed above -- require actual
            # LOS, not just angle, otherwise "facing the opponent's raw
            # bearing through a wall" farms the same reward as genuinely
            # having them in your sights, with none of the risk.
            has_los = self_has_los
            aimed_at_opponent = self_aimed
            if aimed_at_opponent:
                reward = SHAPING_SCALE
            elif has_los:
                # partial credit for having a real sightline even before
                # being aimed -- otherwise "blocked by cover" and "actively
                # maneuvering toward an open angle" both score 0, with no
                # gradient telling the policy it's making progress.
                reward = LOS_SHAPING_SCALE
            else:
                reward = 0.0

            # see ALIGN_SHAPING_SCALE -- dense slope toward being on-target,
            # so closing from 94 deg of aim error to 10 deg is visibly better
            # than not, instead of paying nothing until the 5-deg window is
            # hit exactly. LOS-gated for the same reason as LOS_SHAPING_SCALE:
            # lining up on a wall shouldn't pay.
            if has_los:
                align_t = 1.0 - min(self_aim_error, ALIGN_SHAPING_MAX_ERROR_DEG) / ALIGN_SHAPING_MAX_ERROR_DEG
                reward += ALIGN_SHAPING_SCALE * align_t

            # see SCOPE_SHAPING_SCALE -- pays for charging a scope while a
            # sightline is open, which is the one line that reliably wins on
            # this map.
            #
            # 2026-09-09: gated on has_los rather than on being aimed. Gating
            # it on aim created a chicken-and-egg the policy could not cross:
            # the bonus only paid once already lined up, so a policy that had
            # not yet learned to engage never experienced it and never learned
            # that scoping is what makes engaging pay. Measured across two
            # otherwise-identical runs, one found scoping (99.6% of ticks) and
            # one never did (13.8% at 1M steps, decaying to 0) -- pure
            # exploration luck on a skill that should not need luck. LOS is
            # still required so this cannot be farmed while hiding behind
            # cover, which is what keeps the anti-passivity pressure intact.
            if has_los and self._self_scope_active:
                charge_t = min(self._self_scope_charge, MIN_CHARGE_FOR_HEADSHOT) / MIN_CHARGE_FOR_HEADSHOT
                reward += SCOPE_SHAPING_SCALE * charge_t

            # cost a shot that was actually taken and missed. Keyed on
            # self_shot_taken, not the raw trigger, so trigger ticks the
            # cooldown swallowed aren't punished -- see _try_fire.
            if self_shot_taken and damage_to_opponent <= 0.0:
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
