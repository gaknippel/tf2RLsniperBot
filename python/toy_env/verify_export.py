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


def forward(exported, obs, layers=None):
    x = flatten_obs(obs, exported["obs_key_order"])
    for layer in (layers if layers is not None else exported["layers"]):
        w = np.array(layer["weight"], dtype=np.float64)
        b = np.array(layer["bias"], dtype=np.float64)
        x = w @ x + b
        if layer["activation"] == "tanh":
            x = np.tanh(x)
    action = np.clip(x, exported["action_low"], exported["action_high"])
    return action


def main():
    """Verify the exported JSON against the model(s) it came from.

        python verify_export.py                              # default model
        python verify_export.py path/to/model.zip            # one model
        python verify_export.py early=a.zip late=b.zip       # one per variant

    The named form is the one to use once export_policy.py has been given
    several checkpoints: each variant is checked against the model it was
    actually exported from. Checking a multi-variant export against a single
    model would compare "late" weights to the "early" block and fail for a
    reason that has nothing to do with the export being wrong.
    """
    exported = load_exported_policy()
    variants = exported.get("variants") or [dict(exported, name="default")]

    args = sys.argv[1:]
    if args and all("=" in a for a in args):
        specs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args]
    elif len(args) == 1:
        if len(variants) > 1:
            raise SystemExit(
                f"this export has {len(variants)} variants "
                f"({', '.join(v['name'] for v in variants)}) -- pass one model per "
                "variant as name=path pairs, or the check is meaningless")
        specs = [(variants[0]["name"], args[0])]
    elif not args:
        if len(variants) > 1:
            raise SystemExit(
                f"this export has {len(variants)} variants "
                f"({', '.join(v['name'] for v in variants)}) -- pass one model per "
                "variant as name=path pairs")
        specs = [(variants[0]["name"], MODEL_PATH)]
    else:
        raise SystemExit("pass a single model path, or name=path pairs")

    by_name = {v["name"]: v for v in variants}
    ok = True
    for name, model_path in specs:
        if name not in by_name:
            raise SystemExit(f"no variant named {name!r} in the export "
                             f"(have: {', '.join(by_name)})")
        variant = by_name[name]
        print(f"\n=== variant {name!r} vs {model_path} ===")
        model = PPO.load(model_path)
        env = SniperDuelEnv()
        obs, _ = env.reset(seed=123)
        max_abs_diff = 0.0
        for step in range(20):
            sb3_action, _ = model.predict(obs, deterministic=True)
            our_action = forward(exported, obs, variant["layers"])
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(sb3_action - our_action))))
            obs, _, terminated, truncated, _ = env.step(sb3_action)
            if terminated or truncated:
                obs, _ = env.reset(seed=123 + step)

        if max_abs_diff < 1e-5:
            print(f"  PASS: forward pass matches SB3's MEAN (max diff {max_abs_diff:.2e})")
        else:
            print(f"  FAIL: weight mismatch (max diff {max_abs_diff:.2e})")
            ok = False

        # 2026-09-11: matching the mean is necessary but not sufficient. PPO acts
        # by sampling N(mean, exp(log_std)); an export carrying perfect weights
        # but no std shipped a bot that never fired (fire mean sits near -0.8
        # against a 0.0 threshold). The std has to round-trip too, and its
        # absence is a hard failure rather than a warning.
        expected_std = np.exp(model.policy.log_std.detach().numpy().astype(np.float64))
        if "action_std" not in variant:
            print("  FAIL: no 'action_std' -- the DLL would run the deterministic "
                  "policy, which does not fire. Re-run export_policy.py.")
            ok = False
        else:
            std_diff = float(np.max(np.abs(
                np.array(variant["action_std"], dtype=np.float64) - expected_std)))
            names = ("strafe", "fwd", "turn", "scope", "fire")
            print("  std: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, expected_std)))
            if std_diff < 1e-6:
                print(f"  PASS: action_std matches (max diff {std_diff:.2e})")
            else:
                print(f"  FAIL: action_std mismatch (max diff {std_diff:.2e})")
                ok = False

    print("\nOVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
