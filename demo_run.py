"""End-to-end dry run of the coastline pipeline on a synthetic shore.

This exercises the same section/transect/crossing/CSV code as coastline.py
but fabricates the Sentinel-2 water mask locally, so it runs with no Earth
Engine account. Output goes to out_demo/ and is clearly not real data.

It answers one question: does the geometry and CSV stage work, or is it still
broken? Once `earthengine authenticate` is done, coastline.py runs the same
code against real imagery.
"""

import json
import math
import os

from shapely.geometry import Polygon, box

import coastline as C

OUTD = os.path.join(C.OUT, "..", "out_demo")
UTM_X0, UTM_X1 = 400000.0, 440000.0
OFFSET = 320000.0


def shore(x, retreat=0.0):
    t = (x - UTM_X0) / 1000.0
    base = 600.0 * math.sin(t / 6.0) + 120.0 * math.sin(t / 2.3)
    return base - retreat


def retreat_at(x):
    t = (x - UTM_X0) / 1000.0
    return 18.0 + 14.0 * math.sin(t / 4.0)


def water_poly(retreat_fn):
    pts_top = []
    pts_bot = []
    x = UTM_X0
    while x <= UTM_X1:
        pts_top.append((x, OFFSET + shore(x, retreat_fn(x))))
        pts_bot.append((x, OFFSET - 900.0))
        x += 20.0
    return Polygon(pts_top + pts_bot[::-1])


def mask_at(x, y, retreat_fn):
    return 1 if y < OFFSET + shore(x, retreat_fn(x)) else 0


def main():
    os.makedirs(OUTD, exist_ok=True)

    f_pre = lambda x: 0.0
    f_post = retreat_at

    pre_poly = water_poly(f_pre)
    post_poly = water_poly(f_post)

    pre_line = C.principal_line(pre_poly)
    post_line = C.principal_line(post_poly)

    all_stations = C.build_sections(pre_line, pre_poly, keepaway=None)
    xs = [p[0] for p, _ in all_stations]
    ys = [c[1] for c in pre_poly.exterior.coords]
    ka = box(min(xs) - 2000, min(ys) - 2000, max(xs) + 2000, max(ys) + 2000)
    stations = C.build_sections(pre_line, pre_poly, keepaway=ka)
    print("stations generated:", len(stations))

    rows, feats = [], []
    errors = []
    for idx, (p, n) in enumerate(stations, start=1):
        entry = C.transect_points(p, n)
        hit = {}
        for ft in entry:
            x, y = ft["geometry"]["coordinates"]
            d = ft["properties"]["d"]
            for epoch, fn in (("pre", f_pre), ("post", f_post)):
                if mask_at(x, y, fn) == 0:
                    hit[epoch] = max(hit.get(epoch, -1e9), d)
            ft["properties"]["pre"] = mask_at(x, y, f_pre)
            ft["properties"]["post"] = mask_at(x, y, f_post)
            ft["properties"]["section"] = idx
        pre_d, post_d = hit.get("pre"), hit.get("post")
        if pre_d is None or post_d is None:
            continue
        got = round(post_d - pre_d, 2)
        want = round(retreat_at(p[0]), 2)
        errors.append(abs(got - want))
        wgs = C.to_wgs(C.Point(p))
        rows.append(
            {
                "Section_ID": "S{:03d}".format(idx),
                "Latitude": round(wgs.y, 6),
                "Longitude": round(wgs.x, 6),
                "Storm": "Gaja",
                "Storm_Date": C.STORM_DATE,
                "Retreat_base_post_m": got,
            }
        )
        vals=[got]
        rows[-1].update({
            "MaxRetreat_m": round(max(vals),2),
            "MeanRetreat_m": round(sum(vals)/len(vals),2),
            "NRate_m_per_yr": round(sum(vals)/len(vals)/0.2028,2),
        })
        feats.extend(entry)

    print("rows written:", len(rows))
    print("abs error vs truth: mean {:.2f} m  max {:.2f} m  (STEP={:.0f})".format(
        sum(errors) / len(errors), max(errors), C.STEP))

    from shapely.geometry import mapping

    header = ["Section_ID", "Latitude", "Longitude", "Storm", "Storm_Date",
              "Retreat_base_post_m", "MaxRetreat_m", "MeanRetreat_m", "NRate_m_per_yr"]
    for name, geom in (("coast_base", pre_line), ("coast_post", post_line)):
        with open(os.path.join(OUTD, name + ".geojson"), "w") as fh:
            json.dump(mapping(C.to_wgs(geom)), fh)
    with open(os.path.join(OUTD, "transects.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    csv = os.path.join(OUTD, "shoreline_retreat.csv")
    with open(csv, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r[k]) for k in header) + "\n")

    print("\n--- shoreline_retreat.csv (first 6) ---")
    for r in rows[:6]:
        print("   ", ", ".join("" if r.get(k) is None else str(r[k]) for k in header))
    print("\nwrote", os.path.abspath(OUTD))
    print("NOTE: synthetic. Not Sentinel-2. Run coastline.py after authenticating.")


if __name__ == "__main__":
    main()