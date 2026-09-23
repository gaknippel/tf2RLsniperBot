"""Short continuation of a finished run, spent almost entirely on hard opponents.

    python finetune_hard.py [path/to/model.zip]

Why this is not just "run train.py for longer"
----------------------------------------------
train.py sets opponent difficulty from num_timesteps / TOTAL_TIMESTEPS, so a
50M run only crosses difficulty 0.8 around 40M steps and only reaches 1.0 in the
final stretch. Roughly a tenth of the policy's life was spent against the
hardest opponent, which is exactly the regime the deployed bot has to hold up
in.

Extending the curriculum run to 100M does the opposite of what you would want:
raising TOTAL_TIMESTEPS to 100M recomputes the current position as 0.5, so the
opponent gets EASIER for the next 25M steps and the run re-covers ground it has
already learned. Measured returns across the 50M run were ~+20 points of win
rate at difficulty 1.00 for 49.5M steps, and roughly logarithmic, so another
full run buys very little even if the schedule were fixed.

This instead pins difficulty near the top and runs a short pass. It is the
targeted version of "more steps": same compute buys training where the policy is
actually weak.

What this cannot fix
--------------------
The bot's main remaining flaw is positional -- it sprints into open ground and
trades first shots rather than holding an angle it has not been seen on. Nothing
in the reward pays for an unseen sightline over any sightline, so no amount of
training at any difficulty will produce that. It needs a reward change and a
fresh run. Do not expect this pass to change how the bot fights, only how well
it executes the fight it already knows.
"""
import glob
import os
import re
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sniper_duel_env import SniperDuelEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_MODEL = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
# deliberately NOT sniper_duel_ppo.zip: that model is verified and already baked
# into tf_sniper_policy_weights.h, and overwriting it would silently replace a
# known-good policy with an unevaluated one.
OUTPUT_MODEL = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo_hard")
# separate from snapshots/ so these don't collide with the curriculum run's
# checkpoints -- resume_train.py picks its checkpoint out of that directory by
# step number, and dropping differently-trained files in there would corrupt it.
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "snapshots_hard")

TOTAL_TIMESTEPS = 10_000_000
CHUNK_TIMESTEPS = 100_000
N_ENVS = 8

# Per-env difficulty. Most of the compute goes to the hardest opponent, but not
# all of it: training exclusively at 1.0 invites the policy to forget how to
# handle the sloppier, less predictable opponents, and a human is closer to
# those than to the max-difficulty scripted aimbot. Six of eight at the top,
# two held back, keeps the pressure where it belongs without narrowing the
# policy onto one opponent style.
ENV_DIFFICULTIES = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5)

# This is fine-tuning a converged policy, which is the exact regime where an
# earlier resumed run destroyed itself: near-deterministic actions make PPO's
# importance ratios explode on small mean shifts, clipping stops protecting the
# update, and one bad batch flattens the policy (approx_kl was seen at 1.2-2.0
# against a normal 0.01-0.03). target_kl aborts an epoch once measured KL passes
# it, and the reduced rate keeps updates small -- see resume_train.py.
TARGET_KL = 0.03
LEARNING_RATE = 1e-4

TENSORBOARD_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")
TENSORBOARD_RUN_NAME = f"ppo_sniper_hard_{time.strftime('%Y%m%d_%H%M%S')}"


def make_env():
    # Monitor is what surfaces per-episode return/length as rollout/ep_rew_mean;
    # without it the logs show loss stats and nothing about whether the policy
    # is actually winning.
    return Monitor(SniperDuelEnv())


# Where the source model's own step count is recorded on the first attempt.
#
# Checkpoints here are named with ABSOLUTE step counts (reset_num_timesteps is
# False, so the counter continues from the 50M the source model already has).
# That is what makes crash-resume work: names keep increasing, so "newest" is
# unambiguous. The tradeoff is that this pass's own progress is
# (absolute - baseline), and the baseline has to survive a restart -- hence the
# file. Without it a resumed attempt could not tell 10M of fine-tuning from the
# 50M that came before it.
BASELINE_FILE = os.path.join(CHECKPOINT_DIR, "baseline_steps.txt")


def find_latest_checkpoint():
    """Newest checkpoint from a previous attempt at THIS pass, if any.

    The run wrapper restarts this script if it dies; without this it would
    reload the source model every time and throw away the fine-tuning done so
    far.
    """
    best_steps, best_path = 0, None
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, "sniper_duel_ppo_hard_*_steps.zip")):
        match = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        if match and int(match.group(1)) > best_steps:
            best_steps, best_path = int(match.group(1)), path
    return best_steps, best_path


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

    _, resume_path = find_latest_checkpoint()
    if len(sys.argv) > 1:
        source, resume_path = sys.argv[1], None
        print(f"[hard] starting from {source} (explicit)")
    elif resume_path:
        source = resume_path
        print(f"[hard] resuming from {os.path.basename(resume_path)}")
    else:
        source = SOURCE_MODEL
        print(f"[hard] continuing from {source}")

    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    model = PPO.load(source, env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR)

    # see BASELINE_FILE -- written once, on the attempt that starts the pass
    if resume_path and os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = int(f.read().strip())
    else:
        baseline = int(model.num_timesteps)
        with open(BASELINE_FILE, "w") as f:
            f.write(str(baseline))

    done_steps = int(model.num_timesteps) - baseline
    remaining = TOTAL_TIMESTEPS - done_steps
    print(f"[hard] {done_steps}/{TOTAL_TIMESTEPS} of this pass done "
          f"(absolute step count {model.num_timesteps}, baseline {baseline})")
    if remaining <= 0:
        print("[hard] already at the step budget, nothing to do")
        model.save(OUTPUT_MODEL)
        print(f"[hard] done, saved to {OUTPUT_MODEL}")
        return

    model.target_kl = TARGET_KL
    model.learning_rate = LEARNING_RATE
    model.lr_schedule = lambda _progress_remaining: LEARNING_RATE

    # PPO.load restores verbose from the checkpoint, and these carry verbose=0
    # from the behavior-cloning warm start. A silent run is how an earlier
    # collapse went unnoticed for 10M steps.
    model.verbose = 1
    model.set_logger(configure(None, ["stdout"]))

    # set each worker independently -- env_method with indices addresses one env
    for i, d in enumerate(ENV_DIFFICULTIES[:N_ENVS]):
        vec_env.env_method("set_difficulty", d, indices=[i])
    print(f"[hard] per-env difficulty: {ENV_DIFFICULTIES[:N_ENVS]}")
    print(f"[hard] target_kl={TARGET_KL}, lr={LEARNING_RATE}, "
          f"{remaining} steps remaining of {TOTAL_TIMESTEPS}")

    callback = CheckpointCallback(
        # save_freq counts per-env steps under a VecEnv
        save_freq=max(CHUNK_TIMESTEPS // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="sniper_duel_ppo_hard",
    )

    model.learn(
        total_timesteps=remaining,
        callback=callback,
        tb_log_name=TENSORBOARD_RUN_NAME,
        # keep the absolute step counter running -- see BASELINE_FILE. Resetting
        # it would restart checkpoint numbering at 0 on every crash-resume, so
        # "newest checkpoint" would stop being well-defined and the wrapper would
        # reload the same stale file forever.
        reset_num_timesteps=False,
    )

    model.save(OUTPUT_MODEL)
    print(f"[hard] done, saved to {OUTPUT_MODEL}")
    print("[hard] evaluate it against the 50M model before deploying:")
    print(f"[hard]   python eval_policy.py {OUTPUT_MODEL}.zip 50")


if __name__ == "__main__":
    main()
