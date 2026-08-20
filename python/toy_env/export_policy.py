import json
import os

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
    model = PPO.load(MODEL_PATH)
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    # policy_net = [Linear(10,64), Tanh, Linear(64,64), Tanh]
    linear1, _, linear2, _ = policy_net
    layers = [
        {"weight": linear1.weight.detach().numpy().tolist(), "bias": linear1.bias.detach().numpy().tolist(), "activation": "tanh"},
        {"weight": linear2.weight.detach().numpy().tolist(), "bias": linear2.bias.detach().numpy().tolist(), "activation": "tanh"},
        {"weight": action_net.weight.detach().numpy().tolist(), "bias": action_net.bias.detach().numpy().tolist(), "activation": "none"},
    ]

    export = {
        "obs_key_order": OBS_KEY_ORDER,
        "action_low": model.action_space.low.tolist(),
        "action_high": model.action_space.high.tolist(),
        "layers": layers,
    }

    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f)

    print(f"exported policy to {EXPORT_PATH}")


if __name__ == "__main__":
    main()
