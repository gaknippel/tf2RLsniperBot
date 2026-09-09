import glob
import os
import re
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
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
    step_count, checkpoint_path = find_latest_checkpoint()
    remaining = TOTAL_TIMESTEPS - step_count
    print(f"[resume] loading {checkpoint_path} ({step_count}/{TOTAL_TIMESTEPS} steps already done, "
          f"{remaining} remaining)")

    if remaining <= 0:
        print("[resume] checkpoint already at or past TOTAL_TIMESTEPS, nothing to do")
        return

    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    model = PPO.load(checkpoint_path, env=vec_env)

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
