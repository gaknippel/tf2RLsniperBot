"""Warm-start the PPO policy by imitating a hand-written expert.

    python pretrain_policy.py [out_path.zip]

Why this exists
---------------
Winning on this map requires a four-part conjunction: reach a sightline, aim
onto the target, hold scope until charged, and fire on the right tick. Each
piece pays off only when the others are already in place, so undirected
exploration has to find all four together before any of it looks good.

Blind PPO runs demonstrably do not do this reliably. Across three otherwise
identical 6M-step runs the policy converged on a different partial solution
each time -- one scoped on 99.7% of ticks but never took a sightline, another
took sightlines but stopped scoping, a third stopped firing altogether. That
is exploration luck, not a reward-shaping problem, and re-rolling it at 35
minutes a run is not a strategy.

The expert below is the same scripted policy measured in the strategy
comparison: standing still, tracking the target, scoping, and firing only once
charged enough for a headshot. It wins 59/60 at difficulty 0.0, 26/60 at 0.5
and 7/60 at 1.0 -- and loses ZERO games at every difficulty, because a charged
headshot (150-450) one-shots a 125 HP target before the opponent can stack up
three body shots.

Cloning it gives PPO a competent starting point instead of a random one. PPO
then does what it is actually good at: refining a working strategy (notably
the movement/positioning the expert does not even attempt, since it never
moves) rather than discovering one from scratch.
"""
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sniper_duel_env import (
    SniperDuelEnv, AIM_TOLERANCE_DEG, MAX_TURN_PER_STEP_DEG, MIN_CHARGE_FOR_HEADSHOT,
    MAX_HEALTH, MAX_EPISODE_STEPS, POSITION_LOW, POSITION_HIGH,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "models", "sniper_duel_pretrained")

EPISODES = 1200          # expert episodes to collect
EPOCHS = 12
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
# collect across the whole curriculum so the clone isn't specialised to the
# easy end, and so it sees plenty of no-line-of-sight states too
DIFFICULTIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def expert_action(env):
    """Track the target, scope, and fire only when charged for a headshot.

    Also nudges forward when there is no sightline -- the pure turret version
    never moves at all, which would teach the clone to stand still even when
    blocked by cover. This keeps the demonstrated behavior 'seek a sightline,
    then hold it'.
    """
    to_opp = env._opponent_pos - env._self_pos
    bearing = np.degrees(np.arctan2(to_opp[1], to_opp[0]))
    diff = ((bearing - env._self_angle + 180.0) % 360.0) - 180.0
    turn = float(np.clip(diff / MAX_TURN_PER_STEP_DEG, -1.0, 1.0))

    has_los = env._line_of_sight_clear(env._self_pos, env._opponent_pos)
    aimed = abs(diff) <= AIM_TOLERANCE_DEG
    charged = env._self_scope_charge >= MIN_CHARGE_FOR_HEADSHOT

    if has_los:
        strafe, fwd = 0.0, 0.0           # hold the sightline
    else:
        strafe, fwd = 0.0, 0.7           # go looking for one

    # 2026-09-10: fire only on a charged headshot, and only while on target.
    # Firing now zeroes scope charge (matching the real rifle), so holding the
    # trigger would reset the charge every cooldown and make a headshot
    # impossible -- the expert has to deliberately wait for the charge instead.
    # This is also what teaches trigger discipline rather than "hold it down",
    # which is what the previous clone learned when charge was free.
    fire = 1.0 if (has_los and aimed and charged) else -1.0
    return np.array([strafe, fwd, turn, 1.0, fire], dtype=np.float32)


def collect(episodes):
    obs_batches, act_batches = [], []
    env = SniperDuelEnv()
    per_diff = max(1, episodes // len(DIFFICULTIES))
    for d in DIFFICULTIES:
        env.set_difficulty(d)
        for ep in range(per_diff):
            obs, _ = env.reset(seed=hash((d, ep)) % (2**31))
            while True:
                a = expert_action(env)
                obs_batches.append({k: v.copy() for k, v in obs.items()})
                act_batches.append(a)
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
    return obs_batches, np.array(act_batches, dtype=np.float32)


def collect_aim_coverage(n_states=40000):
    """Synthetic state coverage for target acquisition, off-trajectory.

    2026-09-10: trajectory data alone does not teach aiming. The expert corrects
    its aim within a few ticks, so rollouts are overwhelmingly small-error
    states and a clone fit to them turns correctly in only ~40% of swept aim
    errors -- the same blind spot that let the previous 50M policy ship unable
    to aim at all.

    The expert is a known function of state rather than something that has to
    be rolled out, so the aim mapping can be taught directly: sample positions
    and yaws across the whole arena, keep the ones with a real sightline, and
    label each with what the expert would do there. This covers the large-error
    states a trajectory almost never visits, which is exactly what aiming at a
    human who can be at any bearing requires.
    """
    env = SniperDuelEnv()
    env.set_difficulty(0.5)
    rng = np.random.default_rng(12345)
    obs_batches, act_batches = [], []
    attempts = 0
    while len(act_batches) < n_states and attempts < n_states * 60:
        attempts += 1
        slf = rng.uniform(POSITION_LOW, POSITION_HIGH).astype(np.float32)
        opp = rng.uniform(POSITION_LOW, POSITION_HIGH).astype(np.float32)
        if not env._line_of_sight_clear(slf, opp):
            continue
        env._self_pos, env._opponent_pos = slf, opp
        env._self_angle = float(rng.uniform(-180.0, 180.0))
        env._self_scope_active = True
        env._self_scope_charge = float(rng.uniform(0.0, 1.0))
        env._self_health = float(rng.uniform(0.3, 1.0)) * MAX_HEALTH
        env._self_fire_cooldown = int(rng.integers(0, 2))
        env._step_count = int(rng.integers(0, MAX_EPISODE_STEPS))
        obs_batches.append({k: v.copy() for k, v in env._get_obs().items()})
        act_batches.append(expert_action(env))
    return obs_batches, np.array(act_batches, dtype=np.float32)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    print(f"collecting {EPISODES} expert episodes across difficulties {DIFFICULTIES} ...")
    obs_list, actions = collect(EPISODES)
    print(f"  {len(actions)} trajectory pairs")

    # see collect_aim_coverage -- trajectories barely contain large aim errors,
    # so the aim mapping is taught from direct state-space coverage as well.
    cov_obs, cov_actions = collect_aim_coverage()
    print(f"  {len(cov_actions)} synthetic aim-coverage pairs")
    obs_list = obs_list + cov_obs
    actions = np.concatenate([actions, cov_actions], axis=0)
    print(f"  {len(actions)} total\n")

    vec_env = DummyVecEnv([lambda: Monitor(SniperDuelEnv())])
    model = PPO("MultiInputPolicy", vec_env, verbose=0, ent_coef=0.0)
    policy = model.policy
    device = policy.device

    keys = obs_list[0].keys()
    obs_tensor = {
        k: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device)
        for k in keys
    }
    act_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)

    # The expert only pulls the trigger when line of sight, aim and charge all
    # line up -- a small minority of ticks. Plain MSE over that column collapses
    # to the majority value ("never fire"), which is exactly what a first pass
    # produced: a clone that scoped 99.6% of the time and fired on 0.0% of ticks.
    # Weighting the rare firing samples back up to parity fixes the imbalance.
    fire_positive = actions[:, 4] > 0.0
    pos_frac = float(fire_positive.mean())
    fire_weight = (1.0 - pos_frac) / max(pos_frac, 1e-6)
    print(f"expert fires on {pos_frac * 100:.1f}% of ticks -> weighting those "
          f"samples x{fire_weight:.1f}\n")
    sample_w = torch.as_tensor(
        np.where(fire_positive, fire_weight, 1.0), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    n = len(actions)
    print(f"cloning for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch_obs = {k: v[idx] for k, v in obs_tensor.items()}
            batch_act = act_tensor[idx]

            # match the expert's action directly. Using the distribution mean
            # (rather than sampled log-prob) keeps this a plain regression and
            # avoids the entropy term fighting the fit.
            latent_pi, _, _ = policy._get_latent(batch_obs) if hasattr(policy, "_get_latent") else (None, None, None)
            if latent_pi is None:
                features = policy.extract_features(batch_obs)
                if isinstance(features, tuple):
                    features = features[0]
                latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)

            per_sample = torch.nn.functional.mse_loss(
                mean_actions, batch_act, reduction="none").mean(dim=1)
            w = sample_w[idx]
            loss = (per_sample * w).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1:>2}/{EPOCHS}  mse={total / n:.5f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"\nsaved warm-started policy to {out_path}.zip")
    print("run eval_policy.py on it to confirm the clone actually plays like the expert")


if __name__ == "__main__":
    main()
