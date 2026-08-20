import json
import os

#this is a verifier to see that our brains math that it does (los, angles, moving, etc.)
# is actually correct before doing big tests and stuff :D

import numpy as np
from stable_baselines3 import PPO

from sniper_duel_env import SniperDuelEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_policy.json")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")


def load_exported_policy():
    with open(EXPORT_PATH) as f:
        return json.load(f)


def flatten_obs(obs, obs_key_order):
    return np.concatenate([np.asarray(obs[key], dtype=np.float64).flatten() for key in obs_key_order])


def forward(exported, obs):
    x = flatten_obs(obs, exported["obs_key_order"])
    for layer in exported["layers"]:
        w = np.array(layer["weight"], dtype=np.float64)
        b = np.array(layer["bias"], dtype=np.float64)
        x = w @ x + b
        if layer["activation"] == "tanh":
            x = np.tanh(x)
    action = np.clip(x, exported["action_low"], exported["action_high"])
    return action


def main():
    exported = load_exported_policy()
    model = PPO.load(MODEL_PATH)
    env = SniperDuelEnv()

    obs, _ = env.reset(seed=123)
    max_abs_diff = 0.0

    for step in range(20):
        sb3_action, _ = model.predict(obs, deterministic=True)
        our_action = forward(exported, obs)

        diff = np.max(np.abs(sb3_action - our_action))
        max_abs_diff = max(max_abs_diff, diff)
        print(f"step {step:>2}  sb3={np.round(sb3_action, 4)}  ours={np.round(our_action, 4)}  max_diff={diff:.2e}")

        obs, reward, terminated, truncated, info = env.step(sb3_action)
        if terminated or truncated:
            obs, _ = env.reset(seed=123 + step)

    print(f"\nmax abs diff across all steps: {max_abs_diff:.2e}")
    if max_abs_diff < 1e-5:
        print("PASS: pure-numpy forward pass matches SB3 exactly")
    else:
        print("FAIL: mismatch -- do not trust the export yet")


if __name__ == "__main__":
    main()
