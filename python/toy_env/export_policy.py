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


# 2026-09-13: the DLL can carry several policies at once, selected in game by
# the sniperbot_policy convar. This exists for showing the bot's progression --
# an early checkpoint and a late one behave visibly differently (the 500k-step
# policy holds its scope on ~90% of ticks and crawls; the 50M one scopes on
# ~55% and sprints between engagements).
#
# It is NOT a difficulty setting. Measured win rates at 500k vs 50M steps are
# 92/68/48 vs 98/78/68 across difficulties 0.00/0.50/1.00 -- the warm start from
# behavior cloning means even a barely-trained policy duels competently. What
# actually makes the bot easy or hard to fight is the analytic aim, which was
# never learned; see sniperbot_aim_skill in tf_sniper_bot.cpp.
DEFAULT_VARIANTS = ("default",)


def extract_policy(model_path):
    """Pull the weights and action std out of one saved model."""
    print(f"  reading {model_path}")
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
    print("    action std: " + ", ".join(
        f"{n}={v:.3f}" for n, v in zip(
            ("strafe", "fwd", "turn", "scope", "fire"), np.exp(log_std))))
    return {
        "action_low": model.action_space.low.tolist(),
        "action_high": model.action_space.high.tolist(),
        "log_std": log_std.tolist(),
        "action_std": np.exp(log_std).tolist(),
        "layers": layers,
    }


def main():
    """Export one or more policies into a single JSON.

        python export_policy.py                          # models/sniper_duel_ppo
        python export_policy.py path/to/model.zip        # one specific model
        python export_policy.py early=a.zip late=b.zip   # several, named

    The named form is what feeds the in-game sniperbot_policy convar; names
    become the values that convar accepts. Order is preserved, and index 0 is
    what the bot uses by default.
    """
    args = sys.argv[1:]
    if not args:
        specs = [("default", MODEL_PATH)]
    elif all("=" in a for a in args):
        specs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args]
    elif len(args) == 1:
        specs = [("default", args[0])]
    else:
        raise SystemExit(
            "pass either a single model path, or several as name=path pairs "
            "(e.g. early=snapshots/a.zip late=models/sniper_duel_ppo.zip)")

    seen = set()
    for name, _ in specs:
        if not name.replace("_", "").isalnum():
            raise SystemExit(f"variant name {name!r} must be alphanumeric/underscore "
                             "-- it becomes a C++ identifier and a convar value")
        if name in seen:
            raise SystemExit(f"duplicate variant name {name!r}")
        seen.add(name)

    print(f"exporting {len(specs)} policy variant(s)")
    variants = []
    for name, path in specs:
        print(f"  [{name}]")
        entry = extract_policy(path)
        entry["name"] = name
        entry["source"] = os.path.basename(str(path))
        variants.append(entry)

    # every variant shares one architecture and one observation layout -- the
    # C++ side has a single Forward() and a single BuildObservation, so a
    # mismatch here would be silently wrong rather than a build error.
    shapes = {tuple(len(l["weight"]) for l in v["layers"]) for v in variants}
    if len(shapes) != 1:
        raise SystemExit(f"variants have different layer shapes {shapes} -- "
                         "they must all come from the same architecture")

    export = {
        "obs_key_order": OBS_KEY_ORDER,
        # top-level copies of the first variant keep the single-policy consumers
        # (verify_export.py, anything reading this file directly) working
        # unchanged rather than forcing every reader to learn about variants.
        "action_low": variants[0]["action_low"],
        "action_high": variants[0]["action_high"],
        "log_std": variants[0]["log_std"],
        "action_std": variants[0]["action_std"],
        "layers": variants[0]["layers"],
        "variants": variants,
    }

    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f)

    print()
    print(f"exported to {EXPORT_PATH}")
    for i, v in enumerate(variants):
        print(f"  [{i}] {v['name']:<10} <- {v['source']}")


if __name__ == "__main__":
    main()
