import glob
import os
import re
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sniper_duel_env import SniperDuelEnv

# 2026-09-08: resumes the real 50M-step run (train.py) from its latest
# CheckpointCallback snapshot -- lets training be paused (e.g. to free up
# the machine) and picked back up later with at most CHUNK_TIMESTEPS (100k)
# of lost progress, the interval between checkpoints.
TOTAL_TIMESTEPS = 50_000_000  # must match train.py -- the curriculum callback's
                                # progress ratio (num_timesteps/TOTAL_TIMESTEPS)
                                # has to line up with the original run's scale,
                                # not just the remaining step count.
CHUNK_TIMESTEPS = 100_000
N_ENVS = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "snapshots")
FINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
TENSORBOARD_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")
TENSORBOARD_RUN_NAME = f"ppo_sniper_duel_resume_{time.strftime('%Y%m%d_%H%M%S')}"

CHECKPOINT_PATTERN = re.compile(r"sniper_duel_ppo_(\d+)_steps\.zip$")

# 2026-09-10: a resumed run silently destroyed a policy that had been winning
# 90% at max difficulty. Action noise spiked between 30M and 34.4M steps --
# turn std tripled, 0.076 -> 0.238 -- and since precise aim is the whole game
# here, that wrecked aim, then hit rate, then everything downstream. By 43M it
# had re-converged to a degenerate policy winning 0% at every difficulty.
#
# Cause: by 30M the policy was nearly deterministic (turn std 0.076). At that
# scale PPO's importance ratios explode on tiny mean shifts, clipping stops
# protecting the update, and a single bad batch can blow the policy apart --
# approx_kl was seen hitting 1.2-2.0 against a normal 0.01-0.03.
#
# target_kl makes SB3 abandon the rest of an epoch's updates once measured KL
# exceeds it, which is the direct guard against exactly that. The reduced
# learning rate is the matching fix for fine-tuning an already-good policy:
# late training should be refining, not taking full-size steps.
TARGET_KL = 0.03
FINE_TUNE_LEARNING_RATE = 1e-4


def find_latest_checkpoint():
    candidates = []
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, "sniper_duel_ppo_*_steps.zip")):
        match = CHECKPOINT_PATTERN.search(os.path.basename(path))
        if match:
            # mtime, not just the step count in the filename -- this repo has
            # leftover checkpoints from older, unrelated training runs sitting
            # in the same snapshots/ dir with overlapping/higher step numbers
            # in their names, so the filename's step count alone isn't a
            # reliable "most recent" signal.
            candidates.append((os.path.getmtime(path), int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found in {CHECKPOINT_DIR}")
    candidates.sort()
    return candidates[-1][1], candidates[-1][2]  # (step_count, path)


def make_env():
    return Monitor(SniperDuelEnv())


class DifficultyCurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, update_freq):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self._last_update_step = 0

    def _on_step(self):
        if self.num_timesteps - self._last_update_step >= self.update_freq:
            self._last_update_step = self.num_timesteps
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            self.training_env.env_method("set_difficulty", progress)
            print(f"[resume] {self.num_timesteps}/{self.total_timesteps} timesteps, "
                  f"opponent difficulty -> {progress:.2f}")
        return True


def main():
    # allow resuming from an explicit checkpoint rather than the newest one --
    # needed when the newest is a collapsed policy and the good one is older.
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        match = CHECKPOINT_PATTERN.search(os.path.basename(checkpoint_path))
        if not match:
            raise SystemExit(f"can't read a step count out of {checkpoint_path}")
        step_count = int(match.group(1))
    else:
        step_count, checkpoint_path = find_latest_checkpoint()
    remaining = TOTAL_TIMESTEPS - step_count
    print(f"[resume] loading {checkpoint_path} ({step_count}/{TOTAL_TIMESTEPS} steps already done, "
          f"{remaining} remaining)")

    if remaining <= 0:
        print("[resume] checkpoint already at or past TOTAL_TIMESTEPS, nothing to do")
        return

    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    model = PPO.load(checkpoint_path, env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR)

    # see TARGET_KL above -- guards against the destructive-update collapse.
    model.target_kl = TARGET_KL
    model.learning_rate = FINE_TUNE_LEARNING_RATE
    model.lr_schedule = lambda _progress_remaining: FINE_TUNE_LEARNING_RATE

    # PPO.load restores verbose from the checkpoint, and these checkpoints carry
    # verbose=0 from the behavior-cloning warm start -- which is why the run
    # that destroyed the policy printed no std/approx_kl at all and the damage
    # wasn't noticed until a checkpoint was evaluated by hand.
    model.verbose = 1
    model.set_logger(configure(None, ["stdout"]))
    print(f"[resume] target_kl={TARGET_KL}, lr={FINE_TUNE_LEARNING_RATE}")

    # seed the opponent curriculum back to where it actually was -- workers
    # are fresh processes with difficulty=0.0 by default otherwise, which
    # would suddenly hand a partially-trained agent an easy-mode opponent
    # for the first update_freq stretch instead of resuming where it left off.
    progress_now = min(1.0, step_count / TOTAL_TIMESTEPS)
    vec_env.env_method("set_difficulty", progress_now)
    print(f"[resume] restored opponent difficulty -> {progress_now:.2f}")

    callback = CallbackList([
        DifficultyCurriculumCallback(TOTAL_TIMESTEPS, update_freq=CHUNK_TIMESTEPS),
        CheckpointCallback(
            save_freq=max(CHUNK_TIMESTEPS // N_ENVS, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="sniper_duel_ppo",
        ),
    ])

    model.learn(
        total_timesteps=remaining,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=TENSORBOARD_RUN_NAME,
    )

    model.save(FINAL_MODEL_PATH)
    print(f"[resume] done, final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
