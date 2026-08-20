import os

from stable_baselines3 import PPO

from sniper_duel_env import SniperDuelEnv

# small numbers for now -- this is a first smoke-test-scale run to confirm the
# train/snapshot/self-play loop works end to end. bump these up once we've
# confirmed it runs and the reward trend looks sane.
TOTAL_TIMESTEPS = 200_000
CHUNK_TIMESTEPS = 20_000  # train this many steps, then refresh the opponent snapshot

# anchor output paths to this script's own folder, not the caller's cwd, so
# `python train.py` and `python toy_env/train.py` (run from elsewhere) both
# save to the same place.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "snapshots")
FINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
TENSORBOARD_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")
TENSORBOARD_RUN_NAME = "ppo_sniper_duel"


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

    env = SniperDuelEnv()
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)

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
        # loading it separately (rather than reusing `model`) keeps it from
        # continuing to change underneath the opponent as training proceeds.
        env.opponent_policy = PPO.load(snapshot_path)

        print(f"[train] {timesteps_done}/{TOTAL_TIMESTEPS} timesteps done, "
              f"opponent updated to snapshot_{chunk_index}")

    model.save(FINAL_MODEL_PATH)
    print(f"[train] done, final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
