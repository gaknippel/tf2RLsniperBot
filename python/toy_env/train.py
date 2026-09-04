import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sniper_duel_env import SniperDuelEnv

# 2026-08-25: switched from self-play (mirrored dual-role training against a
# frozen copy of itself) to training against a scripted opponent whose
# difficulty ramps over the run. Self-play was the root cause of most of
# this project's bugs (spawn-side asymmetry -> mirroring -> both sides
# co-evolving into the same degenerate wall-hugging equilibrium) and, even
# working as intended, only ever optimized the policy to beat its own
# reflection -- not something that transfers to a human opponent. Now the
# agent always plays the one canonical role (matches bot_rl_solo: one RL bot
# on RED, a human on BLU) against sniper_duel_env.py's hand-scripted BLU
# opponent, which gets sharper as training progresses (DifficultyCurriculumCallback
# below), so the final policy has to be genuinely good, not just
# self-consistent.
TOTAL_TIMESTEPS = 50_000_000  # 2026-08-25: sized for an overnight run (~8h at this
                               # machine's observed 2500-5000 fps) once opponent-style
                               # diversity (see sniper_duel_env.py's OPPONENT_STYLES) made
                               # the task genuinely harder than a 5M run could fully solve
CHUNK_TIMESTEPS = 100_000  # checkpoint + curriculum-update interval
N_ENVS = 8  # matches this machine's logical core count

# anchor output paths to this script's own folder, not the caller's cwd, so
# `python train.py` and `python toy_env/train.py` (run from elsewhere) both
# save to the same place.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "snapshots")
FINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
TENSORBOARD_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")
# timestamped so each script execution gets its own run folder -- SB3 reuses
# the latest existing run folder for a given tb_log_name otherwise, which
# silently mixed a brand new run's data in with a stale previous run's event
# files the first time this happened.
TENSORBOARD_RUN_NAME = f"ppo_sniper_duel_{time.strftime('%Y%m%d_%H%M%S')}"


def make_env():
    # Monitor tracks per-episode return/length and reports it back through
    # info["episode"] -- without it SB3 has nothing to compute
    # rollout/ep_rew_mean or ep_len_mean from, so TensorBoard silently shows
    # only generic PPO loss stats with zero visibility into whether the
    # policy is actually winning duels (this bit us on the first real run).
    return Monitor(SniperDuelEnv())


class DifficultyCurriculumCallback(BaseCallback):
    """Linearly ramps the scripted opponent's difficulty 0 -> 1 over the
    course of training, dispatched to every SubprocVecEnv worker's own
    SniperDuelEnv via env_method (each worker process has its own env
    instance, same reason self-play's load_opponent used env_method).
    """

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
            print(f"[train] {self.num_timesteps}/{self.total_timesteps} timesteps, "
                  f"opponent difficulty -> {progress:.2f}")
        return True


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    # ent_coef=0.0 (SB3's default, no entropy bonus). This project spent a
    # lot of cycles on this knob before landing back on the default -- worth
    # recording why. SB3's continuous-action Gaussian has *unbounded*
    # entropy (bigger std always scores higher), so any nonzero ent_coef is
    # a knife-edge: 0.01 ran the full 5M curriculum here and, once reward
    # signal got sparse, had nothing counteracting it -- std grew
    # monotonically 1.0 -> 79.7, i.e. the policy degenerated into clipped
    # uniform noise. 0.003 looked safer in a couple of 1M-step trials but
    # turned out to still be seed-sensitive over the full 5M run: one run
    # converged beautifully (std settling to ~0.6, 299/300 eval win rate),
    # a second nominally-identical run drifted (std creeping to 2.79) and
    # collapsed back into the passivity floor. The actual fix for *that*
    # local optimum was never the entropy bonus anyway -- it was the
    # SHAPING_SCALE/LOS_SHAPING_SCALE increase and the spawn-position
    # curriculum in sniper_duel_env.py's reset() (env-side curriculum, not
    # just opponent-side), which made LOS genuinely reachable through
    # exploration in the first place. With that fixed, a 2M-step run at
    # ent_coef=0.0 converged cleanly and *monotonically* shrinking std
    # (0.96 -> 0.53) with reward climbing to 80-98/100 by max difficulty --
    # no entropy bonus needed, and no runaway risk since there's nothing
    # pushing std up at all.
    model = PPO("MultiInputPolicy", vec_env, verbose=1, ent_coef=0.0, tensorboard_log=TENSORBOARD_LOG_DIR)

    callback = CallbackList([
        DifficultyCurriculumCallback(TOTAL_TIMESTEPS, update_freq=CHUNK_TIMESTEPS),
        CheckpointCallback(
            save_freq=max(CHUNK_TIMESTEPS // N_ENVS, 1),  # save_freq counts per-env steps under VecEnv
            save_path=CHECKPOINT_DIR,
            name_prefix="sniper_duel_ppo",
        ),
    ])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name=TENSORBOARD_RUN_NAME,
    )

    model.save(FINAL_MODEL_PATH)
    print(f"[train] done, final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
