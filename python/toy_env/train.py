import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sniper_duel_env import SniperDuelEnv

# real training run -- the earlier 200k smoke test confirmed the
# train/snapshot/self-play loop works end to end. this scales up timesteps
# and runs N_ENVS environments in parallel (separate processes) so it
# actually finishes in a reasonable amount of wall-clock time.
TOTAL_TIMESTEPS = 5_000_000
CHUNK_TIMESTEPS = 100_000  # train this many steps, then refresh the opponent snapshot
N_ENVS = 8  # matches this machine's logical core count

# anchor output paths to this script's own folder, not the caller's cwd, so
# `python train.py` and `python toy_env/train.py` (run from elsewhere) both
# save to the same place.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "snapshots")
FINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
TENSORBOARD_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")
# timestamped so each script execution gets its own run folder -- the loop
# below always passes reset_num_timesteps=False (needed so all chunks within
# *this* run share one continuous logger), but that same flag also makes SB3
# reuse the latest existing run folder instead of creating a new one, which
# silently mixed a brand new run's data in with a stale previous run's event
# files the first time this happened. A unique name per execution sidesteps
# that regardless of the reset_num_timesteps value.
TENSORBOARD_RUN_NAME = f"ppo_sniper_duel_{time.strftime('%Y%m%d_%H%M%S')}"


def make_env():
    # Monitor tracks per-episode return/length and reports it back through
    # info["episode"] -- without it SB3 has nothing to compute
    # rollout/ep_rew_mean or ep_len_mean from, so TensorBoard silently shows
    # only generic PPO loss stats with zero visibility into whether the
    # policy is actually winning duels (this bit us on the first real run).
    return Monitor(SniperDuelEnv())


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)

    timesteps_done = 0
    chunk_index = 0

    while timesteps_done < TOTAL_TIMESTEPS:
        # reset_num_timesteps=False keeps SB3's internal step counter (and
        # logging) continuous across chunks instead of restarting at 0 each time.
        model.learn(
            total_timesteps=CHUNK_TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=TENSORBOARD_RUN_NAME,
        )
        timesteps_done += CHUNK_TIMESTEPS
        chunk_index += 1

        snapshot_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{chunk_index}")
        model.save(snapshot_path)

        # opponent becomes a frozen copy of the model as of *this* snapshot --
        # each worker process loads it independently (via env.load_opponent,
        # dispatched through env_method) so the opponent stays fixed for the
        # rest of the chunk instead of drifting alongside `model`.
        vec_env.env_method("load_opponent", snapshot_path)

        print(f"[train] {timesteps_done}/{TOTAL_TIMESTEPS} timesteps done, "
              f"opponent updated to snapshot_{chunk_index}")

    model.save(FINAL_MODEL_PATH)
    print(f"[train] done, final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
