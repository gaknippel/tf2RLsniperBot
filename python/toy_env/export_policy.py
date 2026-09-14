import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

#this is basically a translator so TF2 can understand what the brain thinks by reading 
# the JSON file exported from this file.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_ppo")
EXPORT_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_policy.json")

# gymnasium.spaces.Dict sorts keys alphabetically internally -- this is NOT
# the order we declared them in SniperDuelEnv.__init__. Confirmed empirically
# via model.policy.features_extractor.extractors.keys(); do not assume
# insertion order here or in the C++ port.
OBS_KEY_ORDER = [
    "aim_error_cos",      # 1
    "aim_error_sin",      # 1
    "opponent_pos",       # 2
    "opponent_visible",   # 1
    "scope_active",       # 1
    "scope_charge",       # 1
    "self_angle",         # 1
    "self_health",        # 1
    "self_pos",           # 2
    "time_left",          # 1
]


def main():
    # optional argv override so a specific snapshot can be exported without
    # first having to shuffle files into models/sniper_duel_ppo.zip -- handy for
    # deploying a mid-run checkpoint while training continues.
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    print(f"exporting from {model_path}")
    model = PPO.load(model_path)
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    # policy_net = [Linear(10,64), Tanh, Linear(64,64), Tanh]
    linear1, _, linear2, _ = policy_net
    layers = [
        {"weight": linear1.weight.detach().numpy().tolist(), "bias": linear1.bias.detach().numpy().tolist(), "activation": "tanh"},
        {"weight": linear2.weight.detach().numpy().tolist(), "bias": linear2.bias.detach().numpy().tolist(), "activation": "tanh"},
        {"weight": action_net.weight.detach().numpy().tolist(), "bias": action_net.bias.detach().numpy().tolist(), "activation": "none"},
    ]

    # 2026-09-11: log_std is as much a part of the policy as the weights are,
    # and leaving it out was this project's single most expensive bug.
    #
    # PPO's actor outputs the MEAN of a Gaussian; the action actually taken is
    # sampled from N(mean, exp(log_std)) and clipped to the action bounds. The
    # bridge only ever ran the mean network, i.e. it deployed the deterministic
    # policy while training had optimised the stochastic one. Measured on the
    # 19.7M checkpoint, the two are completely different agents:
    #
    #        difficulty   deterministic   stochastic
    #             0.00         26.7%         96.7%
    #             0.50         60.0%         93.3%
    #             1.00         53.3%         66.7%
    #
    # The trigger is where this bites hardest. Its mean sits at -0.71..-0.86
    # with a 95th percentile of -0.12 -- it NEVER crosses the fire threshold of
    # 0.0, so every shot the policy has ever taken came from sampling noise.
    # "Fire on ~12% of ticks" is encoded purely in the variance, and taking the
    # mean throws it away: in-game that is a bot that holds a scoped, charged,
    # on-target shot and never pulls the trigger. Which is exactly what shipped.
    #
    # So the std ships with the weights, and the C++ side samples (see
    # SniperPolicy::Forward's bStochastic argument).
    log_std = model.policy.log_std.detach().numpy()
    export = {
        "obs_key_order": OBS_KEY_ORDER,
        "action_low": model.action_space.low.tolist(),
        "action_high": model.action_space.high.tolist(),
        "log_std": log_std.tolist(),
        "action_std": np.exp(log_std).tolist(),
        "layers": layers,
    }

    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f)

    print(f"exported policy to {EXPORT_PATH}")
    print("  action std: " + ", ".join(
        f"{n}={v:.3f}" for n, v in zip(
            ("strafe", "fwd", "turn", "scope", "fire"), np.exp(log_std))))


if __name__ == "__main__":
    main()
