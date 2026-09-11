"""Behavioral evaluation for a trained policy. Judges what it DOES, not just what it scores.

    python eval_policy.py [path/to/model.zip]

This project twice shipped/nearly-shipped a policy that scored beautifully and
behaved terribly:

  * a "600/600 win rate" eval that was actually a bot spinning at max turn rate
    with the trigger held, winning by sweeping its crosshair across the target;
  * a 50M-step run whose reward/std curves looked textbook-healthy and which
    turned out to be doing exactly the same thing (mean aim error 94 deg,
    on-target 2.4% of ticks, fire held on 100% of ticks).

Both were missed because every verdict came from ep_rew_mean / ep_len_mean /
std, which a degenerate policy scores well on. Aggregate metrics cannot tell
"wins by aiming" apart from "wins by exploiting". This script exists so that
question gets answered directly, every time, before anything is deployed.

The two known failure modes have opposite fingerprints, so both are checked:

  SPIN-AND-SPRAY : |turn| pinned near max, fire rate near 100%, aim error huge,
                   on-target only a few percent of ticks.
  PASSIVITY      : never takes line of sight, never fires, every episode times
                   out -- the policy hiding rather than engaging.
"""
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from sniper_duel_env import (
    SniperDuelEnv, AIM_TOLERANCE_DEG, FIRE_COOLDOWN_STEPS, MAX_EPISODE_STEPS,
    MIN_CHARGE_FOR_HEADSHOT, UNSCOPED_HIT_DAMAGE, MAX_HEALTH,
    POSITION_LOW, POSITION_HIGH,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo.zip")

DIFFICULTIES = (0.0, 0.25, 0.5, 0.75, 1.0)
EPISODES_PER_DIFFICULTY = 60


def evaluate(model, difficulty, n_episodes, seed_base=0):
    env = SniperDuelEnv()
    env.set_difficulty(difficulty)

    turns, aim_errs, ep_lengths = [], [], []
    los_ticks = on_target_ticks = trigger_ticks = total_ticks = 0
    shots_taken = shots_hit = 0
    scope_ticks = charged_shots = headshot_kills = 0
    wins = losses = draws = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            # sampled before step() so they describe the state the policy
            # actually chose this action in
            has_los = env._line_of_sight_clear(env._self_pos, env._opponent_pos)
            err = env._aim_error_deg(env._self_pos, env._self_angle, env._opponent_pos)
            ready = env._self_fire_cooldown == 0
            hp_before = env._opponent_health

            turns.append(abs(float(action[2])))
            aim_errs.append(err)
            total_ticks += 1
            if has_los:
                los_ticks += 1
                if err <= AIM_TOLERANCE_DEG:
                    on_target_ticks += 1
            if env._self_scope_active:
                scope_ticks += 1
            charged = (env._self_scope_active
                       and env._self_scope_charge >= MIN_CHARGE_FOR_HEADSHOT)
            if action[4] > 0.0:
                trigger_ticks += 1
                if ready:
                    shots_taken += 1
                    if charged:
                        charged_shots += 1

            obs, _, term, trunc, _ = env.step(action)
            dealt = hp_before - env._opponent_health
            if dealt > 0:
                shots_hit += 1
                if dealt > UNSCOPED_HIT_DAMAGE:
                    headshot_kills += 1

            steps += 1
            if term or trunc:
                if term and env._opponent_health <= 0.0 and env._self_health > 0.0:
                    wins += 1
                elif term and env._self_health <= 0.0 and env._opponent_health > 0.0:
                    losses += 1
                else:
                    draws += 1
                ep_lengths.append(steps)
                break

    turns = np.array(turns)
    aim_errs = np.array(aim_errs)
    return {
        "difficulty": difficulty,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / n_episodes,
        "ep_len": float(np.mean(ep_lengths)),
        "los_pct": los_ticks / total_ticks * 100,
        "on_target_pct": on_target_ticks / total_ticks * 100,
        "aim_err": float(aim_errs.mean()),
        "turn_mean": float(turns.mean()),
        "turn_median": float(np.median(turns)),
        "trigger_pct": trigger_ticks / total_ticks * 100,
        "shots": shots_taken,
        "hit_rate": (shots_hit / shots_taken * 100) if shots_taken else 0.0,
        "shots_per_ep": shots_taken / n_episodes,
        "scope_pct": scope_ticks / total_ticks * 100,
        "charged_shot_pct": (charged_shots / shots_taken * 100) if shots_taken else 0.0,
        "headshots": headshot_kills,
    }


def aim_generalization(model, n_positions=6, errors=(-120, -90, -45, -20, -5, 5, 20, 45, 90, 120)):
    """Does the policy actually turn toward the target, anywhere on the map?

    2026-09-10: added after a policy with a 98% in-sim hit rate shipped and sat
    73 degrees off target in game, outputting turn=0.00. In-sim metrics could
    not see it: the agent always spawned facing the opponent with the bearing
    permanently near 0, so "drift slowly one way and fire when the target
    crosses the crosshair" scored just as well as aiming. This probes the skill
    directly and independently of the episode dynamics that hid its absence --
    place the pair at many positions, sweep the aim error, and check the turn
    output actually points the right way.
    """
    env = SniperDuelEnv()
    env.set_difficulty(0.5)
    rng = np.random.default_rng(0)
    good = total = 0
    for _ in range(n_positions):
        # sample a pair with a clear sightline so the aim cue is live
        for _attempt in range(200):
            slf = rng.uniform(POSITION_LOW, POSITION_HIGH).astype(np.float32)
            opp = rng.uniform(POSITION_LOW, POSITION_HIGH).astype(np.float32)
            if env._line_of_sight_clear(slf, opp):
                break
        else:
            continue
        bearing = np.degrees(np.arctan2(opp[1] - slf[1], opp[0] - slf[0]))
        for err in errors:
            env._self_pos, env._opponent_pos = slf.copy(), opp.copy()
            env._self_angle = float(((bearing - err + 180.0) % 360.0) - 180.0)
            env._self_scope_active, env._self_scope_charge = True, 0.5
            env._self_health = MAX_HEALTH
            env._self_fire_cooldown = 0
            env._step_count = 60
            action, _ = model.predict(env._get_obs(), deterministic=True)
            turn = float(np.asarray(action)[2])
            # correct means: turns the right way, with some actual magnitude
            if np.sign(turn) == np.sign(err) and abs(turn) > 0.05:
                good += 1
            total += 1
    return good / total if total else 0.0


def verdict(rows, aim_score=None):
    """Explicit pass/fail on the two known degenerate behaviors.

    Thresholds are deliberately loose -- they are meant to catch a policy that
    has collapsed into an exploit or into hiding, not to grade fine play.
    """
    problems = []
    hard = [r for r in rows if r["difficulty"] >= 0.5]

    if all(r["turn_median"] > 0.8 for r in hard):
        problems.append(
            "SPIN: |turn| median > 0.8 at every hard difficulty -- policy is "
            "holding max turn rate, i.e. spinning rather than aiming")
    if all(r["aim_err"] > 45.0 for r in hard):
        errs = ", ".join(f"{r['aim_err']:.0f}" for r in hard)
        problems.append(
            f"SPIN: mean aim error > 45 deg at every hard difficulty ({errs}) "
            "-- policy is rarely pointed anywhere near the target")
    if all(r["trigger_pct"] > 95.0 for r in hard) and all(r["hit_rate"] < 25.0 for r in hard):
        problems.append(
            "SPRAY: trigger held > 95% of ticks with hit rate < 25% -- "
            "policy is spamming fire rather than picking shots")

    if all(r["los_pct"] < 2.0 for r in hard):
        problems.append(
            "PASSIVITY: line of sight on < 2% of ticks at every hard difficulty "
            "-- policy is avoiding engagement entirely")
    if all(r["shots"] == 0 for r in hard):
        problems.append("PASSIVITY: never fires a single shot at hard difficulties")
    if all(r["ep_len"] > MAX_EPISODE_STEPS * 0.97 for r in hard) and all(r["win_rate"] < 0.05 for r in hard):
        problems.append(
            "PASSIVITY: episodes run to the timeout with almost no wins -- "
            "policy is riding out the clock")

    # 2026-09-10: added after this returned PASS on a policy that won 0% at
    # every difficulty with a 0% hit rate. The two signature checks above look
    # for specific collapse shapes and a thoroughly broken policy can slip
    # between them -- it fired occasionally (so not "never fires"), held some
    # line of sight (so not strictly passive) and didn't spin, while being
    # completely ineffective. Effectiveness is worth asserting directly rather
    # than inferring from behavior fingerprints.
    if all(r["win_rate"] < 0.05 for r in rows):
        problems.append(
            "INEFFECTIVE: wins < 5% at EVERY difficulty, including the easiest "
            "-- whatever it is doing, it does not work")
    if aim_score is not None and aim_score < 0.6:
        problems.append(
            f"CANNOT AIM: turns the correct way in only {aim_score*100:.0f}% of "
            "swept aim errors across random map positions -- it has not learned "
            "general target acquisition, it has fit the spawn geometry")
    if all(r["hit_rate"] < 10.0 for r in rows):
        problems.append(
            "INEFFECTIVE: hit rate < 10% at every difficulty -- shots are not "
            "connecting, so aim is broken regardless of the other stats")

    return problems


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else EPISODES_PER_DIFFICULTY

    print(f"model    : {model_path}")
    print(f"episodes : {n_eps} per difficulty")
    print(f"(fire cooldown {FIRE_COOLDOWN_STEPS} steps -> at most "
          f"{MAX_EPISODE_STEPS // FIRE_COOLDOWN_STEPS} shots per episode)\n")

    model = PPO.load(model_path)
    rows = [evaluate(model, d, n_eps, seed_base=1000 + int(d * 1000)) for d in DIFFICULTIES]

    hdr = (f"{'diff':>5} {'win%':>6} {'eplen':>6} {'LOS%':>6} {'onTgt%':>7} "
           f"{'aimErr':>7} {'turn~':>6} {'trig%':>6} {'shots/ep':>9} {'hit%':>6} "
           f"{'scope%':>7} {'chgShot%':>9} {'HS':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['difficulty']:>5.2f} {r['win_rate']*100:>6.1f} {r['ep_len']:>6.0f} "
              f"{r['los_pct']:>6.1f} {r['on_target_pct']:>7.1f} {r['aim_err']:>7.1f} "
              f"{r['turn_median']:>6.2f} {r['trigger_pct']:>6.1f} "
              f"{r['shots_per_ep']:>9.1f} {r['hit_rate']:>6.1f} "
              f"{r['scope_pct']:>7.1f} {r['charged_shot_pct']:>9.1f} {r['headshots']:>4}")

    aim_score = aim_generalization(model)
    print()
    print(f"aim generalization: turns the correct way in {aim_score*100:.0f}% of "
          f"swept aim errors at random map positions (want >=60%)")

    problems = verdict(rows, aim_score)
    print()
    if problems:
        print("VERDICT: FAIL -- degenerate behavior detected")
        for p in problems:
            print(f"  - {p}")
    else:
        print("VERDICT: PASS -- no degenerate behavior signature detected")
        print("  (this checks for collapse into spin-and-spray or passivity;")
        print("   it does not by itself certify the policy plays well)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
