"""Build a zip of the mod for a friend to install and connect to your server.

    python package_mod.py

Output: dist/mod_tf_friend.zip, containing a mod_tf/ folder that goes in
    <Steam>/steamapps/sourcemods/mod_tf

Re-run this after every rebuild. The friend's client.dll has to match the one
you're hosting with (it includes shared code such as the sniper rifle changes),
so an old zip plus a new build is a mismatch.

What differs from the folder you play from:

* gameinfo.txt uses |appid_440| / |appid_243750| instead of the absolute
  E:\\SteamLibrary\\... paths. The engine resolves those through Steam to
  wherever TF2 and the SDK are installed on THAT machine. Your own gameinfo is
  left alone on purpose: filesystem_init.cpp only supports appid mounting inside
  the engine -- vbsp/vvis/vrad/Hammer hit "Appid based mounting is not supported
  on non-engine DLL projects", so switching yours would break map compiling.

* Only what a connecting player needs. No .pdb debug symbols (~180 MB), demos,
  map sources, or your personal cfgs -- autoexec.cfg's HLAE commands would just
  spew "Unknown command" on a machine without HLAE, and config.cfg is your binds.
  The server side (Jerry's brain, the round manager) runs on YOUR machine; the
  friend only needs enough to render the mod and join.

* The folder is still named mod_tf. A client can only join a server running the
  same game directory, so renaming it would stop the friend connecting.
"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MOD = os.path.join(REPO, "game", "mod_tf")
DIST = os.path.join(REPO, "dist")
STAGE = os.path.join(DIST, "mod_tf")
ZIP_PATH = os.path.join(DIST, "mod_tf_friend.zip")

# folders copied whole
DIRS = ["maps", "materials", "resource", "scripts", "media", "sound"]
# individual files
FILES = [
    "steam.inf",
    os.path.join("bin", "x64", "client.dll"),
    os.path.join("bin", "x64", "server.dll"),
    os.path.join("bin", "x64", "game_shader_generic_example.dll"),
]
# never ship these even if they turn up inside a copied folder
SKIP_EXT = {".pdb", ".log", ".prt", ".vmx", ".dem"}

TF2_ABS = r"E:\SteamLibrary\steamapps\common\Team Fortress 2" + "\\"
SDK_ABS = r"E:\SteamLibrary\steamapps\common\Source SDK Base 2013 Multiplayer" + "\\"


def portable_gameinfo(text):
    """Swap this machine's absolute install paths for Steam appid lookups."""
    out = text.replace(TF2_ABS, "|appid_440|").replace(SDK_ABS, "|appid_243750|")
    # match the SDK's own template (mod_hl2mp) and keep separators uniform
    lines = []
    for line in out.splitlines():
        if "|appid_" in line:
            line = line.replace("\\", "/")
        lines.append(line)
    out = "\n".join(lines) + "\n"
    if "SteamLibrary" in out:
        raise SystemExit("gameinfo still contains a local SteamLibrary path -- "
                         "the substitution missed one, check gameinfo.txt")
    return out


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    for d in DIRS:
        src = os.path.join(MOD, d)
        if not os.path.isdir(src):
            continue
        shutil.copytree(src, os.path.join(STAGE, d),
                        ignore=lambda _dir, names: [n for n in names
                                                    if os.path.splitext(n)[1].lower() in SKIP_EXT])

    for f in FILES:
        src = os.path.join(MOD, f)
        if not os.path.exists(src):
            raise SystemExit(f"missing {src} -- build the solution first")
        dst = os.path.join(STAGE, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    with open(os.path.join(MOD, "gameinfo.txt")) as fh:
        gameinfo = portable_gameinfo(fh.read())
    with open(os.path.join(STAGE, "gameinfo.txt"), "w", newline="\n") as fh:
        fh.write(gameinfo)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    total = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(STAGE):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, DIST))
                total += os.path.getsize(full)

    print(f"staged  {STAGE}  ({total / 1e6:.1f} MB)")
    print(f"zipped  {ZIP_PATH}  ({os.path.getsize(ZIP_PATH) / 1e6:.1f} MB)")
    print("friend extracts it so the folder lands at <Steam>/steamapps/sourcemods/mod_tf")


if __name__ == "__main__":
    main()
