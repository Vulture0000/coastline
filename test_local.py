"""Local half of coastline.py, exercised with a synthetic coast.

The fake sea polygon has a straight waterline at x = 1000 pre-event and
x = 975 post-event, i.e. 25 m of landward retreat. The test asserts that
section generation and the crossing logic recover exactly 25 m, and that
sections lying on the artificial AOI clip edges are discarded.
"""

from shapely.geometry import Point, box

import coastline as C

WX_PRE, WX_POST = 1000.0, 975.0
AOI_X0, AOI_X1, AOI_Y0, AOI_Y1 = 200.0, 4000.0, 400.0, 5600.0


def fake_water(wx):
    return Polygon_water(wx)


def Polygon_water(wx):
    from shapely.geometry import Polygon

    return Polygon([(wx, AOI_Y0), (AOI_X1, AOI_Y0), (AOI_X1, AOI_Y1), (wx, AOI_Y1)])


def crossings(stations, wx):
    out = []
    for p, n in stations:
        found = None
        for f in C.transect_points(p, n):
            x, _ = f["geometry"]["coordinates"]
            if x <= wx:
                found = f["properties"]["d"]
        out.append(found)
    return out


def principal_of(poly):
    return C.principal_line(poly)


def assert_water_poly_at_call_sites():
    """Every build_sections call must pass a water polygon, by name.

    build_sections reads its second argument as "which side is water". Handing
    it land is silent and inverts every normal, so the failure is a sign flip
    in the results rather than an exception. Checking the call sites is the
    only guard that catches it, since the synthetic suite is happy either way.
    """
    import ast
    import os

    offenders = []
    for fname in ("coastline.py", "prestorm.py", "chennai.py"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        with open(path) as fh:
            tree = ast.parse(fh.read(), fname)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", getattr(fn, "id", ""))
            if name != "build_sections" or len(node.args) < 2:
                continue
            arg = ast.dump(node.args[1])
            if "water_poly" not in arg and "sea" not in arg:
                offenders.append("{}:{} {}".format(fname, node.lineno, name))
    print("build_sections call sites passing a water polygon: {}".format(
        "all" if not offenders else offenders))
    assert not offenders, "build_sections called with a non-water polygon at " + ", ".join(offenders)


def main():
    pre, post = fake_water(WX_PRE), fake_water(WX_POST)
    keepaway = box(AOI_X0, AOI_Y0, AOI_X1, AOI_Y1)

    line = C.principal_line(pre)
    print("boundary type:", line.geom_type, "| perimeter:", round(line.length, 1), "m")

    no_margin = C.build_sections(line, pre, keepaway=None)
    stations = C.build_sections(line, pre, keepaway=keepaway)
    print("sections without keep-away:", len(no_margin))
    print("sections with keep-away:   ", len(stations))

    assert len(no_margin) > len(stations), "keep-away should drop AOI-edge sections"
    for p, _ in stations:
        assert abs(p[0] - WX_PRE) < 1.0, "a section landed off the true waterline"
        assert keepaway.exterior.distance(Point(p)) >= C.AOI_MARGIN, "margin not respected"
    print("all sections on the true waterline, AOI clip edges discarded")

    for p, n in stations:
        assert pre.contains(Point(p[0] + n[0] * 60, p[1] + n[1] * 60)), "normal not seaward"
        assert not pre.contains(Point(p[0] - n[0] * 60, p[1] - n[1] * 60)), "normal not landward"
    print("all normals point seaward")

    pre_d = crossings(stations, WX_PRE)
    post_d = crossings(stations, WX_POST)
    assert all(d is not None for d in pre_d + post_d), "every transect must cross"
    assert abs(pre_d[0]) < 1e-9, "baseline itself must be sampled at d=0, got {}".format(pre_d[0])
    print("transect offsets along baseline:", pre_d[0], "->", pre_d[-1], "m")

    # Regression guard. build_sections takes a WATER polygon; every real-data
    # caller used to hand it the land polygon instead, which is not a crash --
    # it silently returns normals pointing inland and reverses the sign of
    # every shoreline change. demo_run.py builds its polygons as water, so it
    # never exercised the land path and the suite stayed green.
    land_poly = box(AOI_X0, AOI_Y0, AOI_X1, AOI_Y1).difference(pre)
    flipped = C.build_sections(principal_of(pre), land_poly, keepaway=None)
    bad = [n for p, n in flipped if pre.contains(Point(p[0] - n[0] * 60, p[1] - n[1] * 60))]
    print("land polygon as water_poly -> {} of {} normals point inland".format(
        len(bad), len(flipped)))
    for p, n in stations:
        assert pre.contains(Point(p[0] + n[0] * 60, p[1] + n[1] * 60)), "sea_poly must agree with water_poly"

    assert_water_poly_at_call_sites()

    retreats = sorted({round(b - a, 2) for a, b in zip(pre_d, post_d)})
    print("retreat values:", retreats)
    assert len(retreats) == 1, "retreat should be uniform on a straight coast"
    err = abs(retreats[0] + 25.0)
    print("retreat error vs true -25.0 m: {:.2f} m (STEP/2 = {:.2f})".format(err, C.STEP / 2))
    assert err <= C.STEP / 2 + 1e-9, "retreat error exceeds half a sample step"

    print("\nALL LOCAL CHECKS PASSED")


if __name__ == "__main__":
    main()