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


def evaluate(model, difficulty, n_episodes, seed_base=0, deterministic=True):
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
            action, _ = model.predict(obs, deterministic=deterministic)
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

            obs, _, term, trunc, _ = env.step(action)

            # 2026-09-12: a "shot" is now detected from the cooldown actually
            # being spent, not from "trigger pulled while off cooldown".
            #
            # Those used to be the same thing. They stopped being the same when
            # _try_fire gained bridge parity: a pull with no line of sight is a
            # free no-op now, so it leaves the cooldown at 0 and the old test
            # counted it as a shot. That inflated shots/ep past the cooldown's
            # own ceiling (15.2 per episode against a hard cap near 13, which is
            # how the bug announced itself) and correspondingly deflated hit%,
            # since every no-op counted as a miss. Both columns were describing
            # trigger pulls while claiming to describe shots.
            if env._self_fire_cooldown == FIRE_COOLDOWN_STEPS - 1 and ready:
                shots_taken += 1
                if charged:
                    charged_shots += 1
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
        # 2026-09-12: hit_rate is ~100% by construction now -- _try_fire only
        # spends a shot when the bot is on target and charged, so a shot that
        # exists is a shot that connected. It stays as a column for sanity, but
        # it can no longer distinguish a good policy from a bad one.
        #
        # conversion replaces it as the discipline signal: of all the ticks the
        # policy held the trigger, how many became actual shots. Wasted pulls are
        # free in game (the bridge drops them), so a low number is not a bug --
        # but it does say the policy is squeezing the trigger blind rather than
        # choosing moments, which is what the spray checks below look for.
        "conversion": (shots_taken / trigger_ticks * 100) if trigger_ticks else 0.0,
        "shots_per_ep": shots_taken / n_episodes,
        "scope_pct": scope_ticks / total_ticks * 100,
        "charged_shot_pct": (charged_shots / shots_taken * 100) if shots_taken else 0.0,
        "headshots": headshot_kills,
    }


def verdict(rows):
    """Explicit pass/fail on the two known degenerate behaviors.

    Thresholds are deliberately loose -- they are meant to catch a policy that
    has collapsed into an exploit or into hiding, not to grade fine play.
    """
    problems = []
    hard = [r for r in rows if r["difficulty"] >= 0.5]

    # 2026-09-10: the "|turn| median > 0.8 means it's spinning" check is gone.
    # It was the original spin-and-spray detector and it did its job, but the
    # env and the bridge now both ignore action[2] -- a policy could emit 1.0
    # there forever and the view would not move, so the check can only produce
    # false failures. The turn_* columns are still reported for information.
    #
    # Aim error is still worth failing on, but only together with a poor hit
    # rate: it averages over no-line-of-sight ticks too, where the bot has
    # nothing to aim at and simply holds its last yaw, so a large mean is normal
    # for a policy that spends time repositioning behind cover.
    # 2026-09-12: these two used to be gated on hit_rate, which is now ~100% by
    # construction (see "conversion" above) and so could never fail again.
    # Rewritten against shots actually taken, which is the quantity that still
    # separates a working policy from a broken one.
    if all(r["aim_err"] > 60.0 for r in hard) and all(r["shots_per_ep"] < 0.2 for r in hard):
        errs = ", ".join(f"{r['aim_err']:.0f}" for r in hard)
        problems.append(
            f"NOT AIMING: mean aim error > 60 deg ({errs}) and under 0.2 shots "
            "per episode at every hard difficulty -- the bot is never settling "
            "on target long enough to take a shot, which points at the aim path "
            "rather than at the policy")
    if all(r["trigger_pct"] > 95.0 for r in hard) and all(r["conversion"] < 5.0 for r in hard):
        problems.append(
            "SPRAY: trigger held > 95% of ticks while under 5% of pulls become "
            "shots -- policy is squeezing blind rather than picking moments")

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
    # 2026-09-10: the aim-generalization probe that used to live here is gone.
    # It swept aim error and checked the sign of action[2], which was the right
    # check while the policy steered its own view -- it caught a policy that
    # turned correctly in only 40% of cases. Both the env and the bridge now aim
    # analytically and ignore action[2] entirely, so that probe could only ever
    # report ~0% and fail every healthy policy. Aim quality is no longer the
    # policy's responsibility, and the aim_err column below reflects the
    # analytic aim rather than anything learned.
    if all(r["shots_per_ep"] < 0.1 for r in rows):
        problems.append(
            "INEFFECTIVE: under 0.1 shots per episode at every difficulty -- the "
            "bot almost never gets a real shot away, whatever the win rate says")

    return problems


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else EPISODES_PER_DIFFICULTY

    print(f"model    : {model_path}")
    print(f"episodes : {n_eps} per difficulty")
    print(f"(fire cooldown {FIRE_COOLDOWN_STEPS} steps -> at most "
          f"{MAX_EPISODE_STEPS // FIRE_COOLDOWN_STEPS} shots per episode)\n")

    model = PPO.load(model_path)

    # 2026-09-11: both action-selection modes are reported, because the gap
    # between them WAS the bug that kept shipping broken bots.
    #
    # PPO learns a Gaussian over actions; model.predict(deterministic=True)
    # returns its mean, deterministic=False samples it. The C++ bridge ran the
    # mean for this project's entire history, while every training metric
    # described the samples -- so training could honestly report a 90%+ win rate
    # while the deployed bot stood still and never fired. The fire dimension is
    # the reason: its mean sits near -0.8 against a 0.0 threshold and never
    # crosses, so the learned ~12% trigger rate exists only in the variance.
    #
    # The bridge now samples too (SniperPolicy::Forward, bStochastic), so
    # STOCHASTIC is the row that predicts in-game behaviour. The deterministic
    # row is kept as the diagnostic: a large gap means the policy is leaning on
    # sampling noise, which is worth knowing even though it is now faithfully
    # reproduced.
    std = np.exp(model.policy.log_std.detach().cpu().numpy())
    print("action std (the sampling the bridge reproduces): " + ", ".join(
        f"{n}={v:.3f}" for n, v in zip(("strafe", "fwd", "turn", "scope", "fire"), std)))
    print()

    rows = [evaluate(model, d, n_eps, seed_base=1000 + int(d * 1000),
                     deterministic=False) for d in DIFFICULTIES]
    det_rows = [evaluate(model, d, n_eps, seed_base=1000 + int(d * 1000),
                         deterministic=True) for d in DIFFICULTIES]

    hdr = (f"{'diff':>5} {'win%':>6} {'eplen':>6} {'LOS%':>6} {'onTgt%':>7} "
           f"{'aimErr':>7} {'turn~':>6} {'trig%':>6} {'shots/ep':>9} {'conv%':>6} "
           f"{'scope%':>7} {'chgShot%':>9} {'HS':>4}")
    print("STOCHASTIC -- matches how the DLL runs the policy in-game:")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['difficulty']:>5.2f} {r['win_rate']*100:>6.1f} {r['ep_len']:>6.0f} "
              f"{r['los_pct']:>6.1f} {r['on_target_pct']:>7.1f} {r['aim_err']:>7.1f} "
              f"{r['turn_median']:>6.2f} {r['trigger_pct']:>6.1f} "
              f"{r['shots_per_ep']:>9.1f} {r['conversion']:>6.1f} "
              f"{r['scope_pct']:>7.1f} {r['charged_shot_pct']:>9.1f} {r['headshots']:>4}")

    print()
    print("DETERMINISTIC (network mean) -- diagnostic only, not what ships:")
    print(f"{'diff':>5} {'win%':>6} {'trig%':>6} {'shots/ep':>9} {'hit%':>6}   delta win%")
    for r, d in zip(rows, det_rows):
        print(f"{d['difficulty']:>5.2f} {d['win_rate']*100:>6.1f} "
              f"{d['trigger_pct']:>6.1f} {d['shots_per_ep']:>9.1f} {d['hit_rate']:>6.1f}   "
              f"{(r['win_rate'] - d['win_rate'])*100:>+8.1f}")

    problems = verdict(rows)

    # A policy whose competence lives entirely in its sampling noise is fragile:
    # it is correct in-game only as long as the bridge keeps sampling, and it
    # says the mean never learned the behaviour. Reported, not failed -- the
    # bridge does sample, so this is a note about robustness rather than a bug.
    mean_det_trig = float(np.mean([d["trigger_pct"] for d in det_rows]))
    mean_sto_trig = float(np.mean([r["trigger_pct"] for r in rows]))
    if mean_sto_trig > 2.0 and mean_det_trig < mean_sto_trig * 0.25:
        print()
        print(f"NOTE: trigger rate is {mean_sto_trig:.1f}% sampled vs "
              f"{mean_det_trig:.1f}% at the mean -- firing is driven by sampling",
              "noise rather than by the mean crossing its threshold. Fine as long",
              "as the bridge samples (it does), but the mean has not learned it.")
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
