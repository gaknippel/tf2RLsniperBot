import json
import os
import sys

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
    # Must be the SAME checkpoint export_policy.py was run on, or this compares
    # two different policies and fails for the wrong reason. Same argv override
    # as export_policy.py so a snapshot can be verified directly.
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    print(f"verifying export against {model_path}")
    exported = load_exported_policy()
    model = PPO.load(model_path)
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
    ok = max_abs_diff < 1e-5
    if ok:
        print("PASS: pure-numpy forward pass matches SB3's MEAN exactly")
    else:
        print("FAIL: mismatch -- do not trust the export yet")

    # 2026-09-11: matching the mean is necessary but not sufficient. The mean is
    # only half the policy -- PPO acts by sampling N(mean, exp(log_std)), and an
    # export carrying perfect weights but no std shipped a bot that never fired
    # (its fire mean sits near -0.8 against a 0.0 threshold). So the std has to
    # round-trip too, and its absence is a hard failure, not a warning.
    expected_std = np.exp(model.policy.log_std.detach().numpy().astype(np.float64))
    if "action_std" not in exported:
        print("FAIL: export has no 'action_std' -- the DLL would run the "
              "deterministic policy, which does not fire. Re-run export_policy.py.")
        ok = False
    else:
        std_diff = float(np.max(np.abs(
            np.array(exported["action_std"], dtype=np.float64) - expected_std)))
        names = ("strafe", "fwd", "turn", "scope", "fire")
        print("\naction std: " + ", ".join(
            f"{n}={v:.4f}" for n, v in zip(names, expected_std)))
        if std_diff < 1e-6:
            print(f"PASS: exported action_std matches the model (max diff {std_diff:.2e})")
        else:
            print(f"FAIL: action_std mismatch (max diff {std_diff:.2e})")
            ok = False

    print("\nOVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
