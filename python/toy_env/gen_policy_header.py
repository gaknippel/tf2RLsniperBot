import json
import os

# Turns the exported policy JSON (see export_policy.py) into a C++ header of
# static const float arrays, so the mod_tf server DLL can run the trained
# policy's forward pass with zero runtime file I/O or JSON parsing. Re-run
# this and rebuild the DLL any time the model is retrained -- the header is
# generated, not hand-edited.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_JSON_PATH = os.path.join(SCRIPT_DIR, "models", "sniper_duel_policy.json")

REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_HEADER_PATH = os.path.join(
    REPO_ROOT, "src", "game", "server", "tf", "tf_sniper_policy_weights.h"
)


def format_float(v):
    # %.9g drops the decimal point for whole numbers (e.g. -1.0 -> "-1"),
    # which C++ then parses as an int literal with an invalid "f" suffix --
    # force a decimal point so it's always a valid float literal.
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def format_float_array(values, indent="\t"):
    return indent + ", ".join(format_float(v) for v in values)


def format_matrix(rows, indent="\t"):
    lines = []
    for row in rows:
        lines.append(indent + "{ " + ", ".join(format_float(v) for v in row) + " },")
    return "\n".join(lines)


def main():
    with open(POLICY_JSON_PATH) as f:
        policy = json.load(f)

    layers = policy["layers"]
    obs_size = len(layers[0]["weight"][0])
    action_size = len(layers[-1]["weight"])

    lines = []
    lines.append("// GENERATED FILE -- do not hand-edit.")
    lines.append(f"// Produced by python/toy_env/gen_policy_header.py from {os.path.basename(POLICY_JSON_PATH)}.")
    lines.append("// Re-run that script and rebuild the DLL after retraining the policy.")
    lines.append("#ifndef TF_SNIPER_POLICY_WEIGHTS_H")
    lines.append("#define TF_SNIPER_POLICY_WEIGHTS_H")
    lines.append("#ifdef _WIN32")
    lines.append("#pragma once")
    lines.append("#endif")
    lines.append("")
    lines.append("namespace SniperPolicy")
    lines.append("{")
    lines.append(f"\tconst int kObsSize = {obs_size};")
    lines.append(f"\tconst int kActionSize = {action_size};")
    lines.append(f"\tconst int kLayerCount = {len(layers)};")
    lines.append("")

    for i, layer in enumerate(layers):
        w = layer["weight"]  # [out][in]
        b = layer["bias"]
        out_dim = len(w)
        in_dim = len(w[0])
        activation = layer["activation"]  # "tanh" or "none"

        lines.append(f"\t// layer {i}: Linear({in_dim} -> {out_dim}), activation = {activation}")
        lines.append(f"\tconst int kLayer{i}InputSize = {in_dim};")
        lines.append(f"\tconst int kLayer{i}OutputSize = {out_dim};")
        lines.append(f"\tconst bool kLayer{i}Tanh = {'true' if activation == 'tanh' else 'false'};")
        lines.append(f"\tconst float kLayer{i}Weight[{out_dim}][{in_dim}] =")
        lines.append("\t{")
        lines.append(format_matrix(w, indent="\t\t"))
        lines.append("\t};")
        lines.append(f"\tconst float kLayer{i}Bias[{out_dim}] =")
        lines.append("\t{")
        lines.append(format_float_array(b, indent="\t\t"))
        lines.append("\t};")
        lines.append("")

    lines.append(f"\tconst float kActionLow[{action_size}] =")
    lines.append("\t{")
    lines.append(format_float_array(policy["action_low"], indent="\t\t"))
    lines.append("\t};")
    lines.append(f"\tconst float kActionHigh[{action_size}] =")
    lines.append("\t{")
    lines.append(format_float_array(policy["action_high"], indent="\t\t"))
    lines.append("\t};")
    lines.append("")
    lines.append("\t// obs_key_order from export_policy.py, for reference when building the")
    lines.append("\t// C++ observation vector -- MUST match this order exactly:")
    for key in policy["obs_key_order"]:
        lines.append(f"\t// - {key}")
    lines.append("}")
    lines.append("")
    lines.append("#endif // TF_SNIPER_POLICY_WEIGHTS_H")

    os.makedirs(os.path.dirname(OUTPUT_HEADER_PATH), exist_ok=True)
    with open(OUTPUT_HEADER_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {OUTPUT_HEADER_PATH}")


if __name__ == "__main__":
    main()
