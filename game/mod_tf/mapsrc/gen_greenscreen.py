"""Generate greenscreen.vmf -- a sealed green box for keyable footage.

    python gen_greenscreen.py

Generated rather than hand-written because brush planes are easy to get wrong:
each face is three points whose winding decides which way it faces, and one
inverted face leaks the map. Every brush here is an axis-aligned box, written
with the same point order Hammer itself uses for a block.

Layout mirrors 1v1map where it matters for the RL bot:
  * spawns named spawn_red / spawn_blu at 1v1map's x/y, since round_manager.nut
    looks them up by name and tf_sniper_bot.cpp pins Jerry to spawn_red;
  * the same logic_script running round_manager, so Jerry spawns and the duel
    runs exactly as it does on 1v1map;
  * an interior larger than 1v1map's playable bounds (x -610..615,
    y -450..430), because the bridge clamps Jerry to those bounds -- the walls
    must sit outside them or he'd be pinned against green.
"""
import os

GREEN = "GREENSCREEN/GREEN"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "greenscreen.vmf")

# interior extents
X0, X1 = -1024, 1024
Y0, Y1 = -768, 768
Z0, Z1 = 0, 640
T = 32  # wall thickness

_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def side(p1, p2, p3, uaxis, vaxis):
    pts = " ".join("(%g %g %g)" % p for p in (p1, p2, p3))
    return (
        '\t\tside\n\t\t{\n'
        f'\t\t\t"id" "{nid()}"\n'
        f'\t\t\t"plane" "{pts}"\n'
        f'\t\t\t"material" "{GREEN}"\n'
        f'\t\t\t"uaxis" "{uaxis} 0.25"\n'
        f'\t\t\t"vaxis" "{vaxis} 0.25"\n'
        '\t\t\t"rotation" "0"\n'
        '\t\t\t"lightmapscale" "16"\n'
        '\t\t\t"smoothing_groups" "0"\n'
        '\t\t}\n'
    )


def block(x1, y1, z1, x2, y2, z2):
    """Axis-aligned box brush, faces in Hammer's own block order."""
    XY_U, XY_V = "[1 0 0 0]", "[0 -1 0 0]"
    X_U, X_V = "[0 1 0 0]", "[0 0 -1 0]"
    Y_U, Y_V = "[1 0 0 0]", "[0 0 -1 0]"
    s = '\tsolid\n\t{\n' + f'\t\t"id" "{nid()}"\n'
    s += side((x1, y2, z2), (x2, y2, z2), (x2, y1, z2), XY_U, XY_V)  # top
    s += side((x1, y1, z1), (x2, y1, z1), (x2, y2, z1), XY_U, XY_V)  # bottom
    s += side((x1, y2, z2), (x1, y1, z2), (x1, y1, z1), X_U, X_V)    # -x
    s += side((x2, y2, z1), (x2, y1, z1), (x2, y1, z2), X_U, X_V)    # +x
    s += side((x2, y2, z2), (x1, y2, z2), (x1, y2, z1), Y_U, Y_V)    # +y
    s += side((x2, y1, z1), (x1, y1, z1), (x1, y1, z2), Y_U, Y_V)    # -y
    return s + '\t}\n'


def entity(kv):
    body = "".join(f'\t"{k}" "{v}"\n' for k, v in kv.items())
    return 'entity\n{\n' + f'\t"id" "{nid()}"\n' + body + '}\n'


def main():
    # six slabs that overlap at the corners, so the box is sealed
    brushes = [
        block(X0 - T, Y0 - T, Z0 - T, X1 + T, Y1 + T, Z0),       # floor
        block(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T),       # ceiling
        block(X0 - T, Y0 - T, Z0, X0, Y1 + T, Z1),               # -x wall
        block(X1, Y0 - T, Z0, X1 + T, Y1 + T, Z1),               # +x wall
        block(X0, Y0 - T, Z0, X1, Y0, Z1),                       # -y wall
        block(X0, Y1, Z0, X1, Y1 + T, Z1),                       # +y wall
    ]

    world = (
        'world\n{\n'
        f'\t"id" "{nid()}"\n'
        '\t"mapversion" "1"\n'
        '\t"classname" "worldspawn"\n'
        '\t"skyname" "sky_day01_01"\n'
        '\t"maxpropscreenwidth" "-1"\n'
        + "".join(brushes) + '}\n'
    )

    ents = [
        # same positions as 1v1map; z on the floor
        entity({"classname": "info_player_teamspawn", "targetname": "spawn_red",
                "TeamNum": "2", "origin": "-559.61 33 8", "angles": "0 0 0"}),
        entity({"classname": "info_player_teamspawn", "targetname": "spawn_blu",
                "TeamNum": "3", "origin": "532.387 -9.667 8", "angles": "0 180 0"}),
        entity({"classname": "logic_script", "vscripts": "round_manager",
                "origin": "0 0 64"}),
    ]
    # The green surfaces are unlit, so these only light the PLAYERS -- without
    # them models render near-black, since a model's lighting comes from the
    # world lights vrad bakes. Spread out so nobody walks into a dark patch.
    for x, y in ((0, 0), (-640, -448), (640, -448), (-640, 448), (640, 448)):
        ents.append(entity({"classname": "light", "origin": f"{x} {y} 560",
                            "_light": "255 255 255 900",
                            "_lightHDR": "-1 -1 -1 1", "_lightscaleHDR": "1"}))

    header = (
        'versioninfo\n{\n\t"editorversion" "400"\n\t"editorbuild" "8000"\n'
        '\t"mapversion" "1"\n\t"formatversion" "100"\n\t"prefab" "0"\n}\n'
        'visgroups\n{\n}\n'
        'viewsettings\n{\n\t"bSnapToGrid" "1"\n\t"bShowGrid" "1"\n'
        '\t"bShowLogicalGrid" "0"\n\t"nGridSpacing" "64"\n\t"bShow3DGrid" "0"\n}\n'
    )
    footer = 'cameras\n{\n\t"activecamera" "-1"\n}\ncordon\n{\n\t"mins" "(-1024 -1024 -1024)"\n\t"maxs" "(1024 1024 1024)"\n\t"active" "0"\n}\n'

    with open(OUT, "w", newline="\n") as f:
        f.write(header + world + "".join(ents) + footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
