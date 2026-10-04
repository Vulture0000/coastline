"""Sentinel-2 coastline extraction, pre/post Cyclone Gaja (Tamil Nadu coast).

Usage:
    python -m pip install earthengine-api shapely pyproj numpy
    earthengine authenticate
    python coastline.py

Outputs (in ./out):
    coast_pre.geojson          pre-event coastline LineString (WGS84)
    coast_post.geojson         post-event coastline LineString (WGS84)
    transects.geojson         transect sample points, both epochs labelled
    shoreline_retreat.csv     per-section dataset

Uncertainty:
    Each epoch's waterline is found by sampling the water mask along the
    transect, so it is quantised to +/- STEP/2. The two epochs quantise
    independently, so the RETREAT DIFFERENCE carries up to +/- STEP of error.
    With STEP=10 that is +/- 10 m, which is the method's real floor and is
    already worse than the 10 m Sentinel-2 pixel. Do not set STEP below 10.

    INLAND must exceed the largest retreat you expect to measure, otherwise
    the landward search never terminates and the section is dropped.
    OFFSHORE only needs to be far enough to be safely seaward.

Epochs, and why:
    Google Earth Engine has NO T44PLS (the main study tile) imagery between
    2018-09-13 and Cyclone Gaja's 2018-11-16 landfall, so no tight pre-storm
    baseline exists there. The 2018-11-12 date returns zero scenes.
    base -> pre is therefore only 5 days with no storm in between, so
    Retreat_base_pre_m is a REPEATABILITY measurement: whatever it shows is
    measurement noise plus tidal stage, and it is the noise floor that the
    storm signal has to beat. Retreat_pre_post_m spans the cyclone plus about
    2.5 months of post-monsoon drift, so it is an upper bound on storm damage,
    not a clean storm-only figure.

Tide:
    TIDE_F is absent from every 2018 L2A product, so tide stage is unknown.
    Retreat_base_pre_m doubles as a tide diagnostic: a systematic landward bias
    there indicates the two scenes were taken at different tidal stages.
"""

import json
import math
import os
from functools import lru_cache

import ee
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

AOI_LL = [79.4, 10.05, 80.10, 10.75]
CRS = "EPSG:32644"
SCALE = 10

VECTOR_SCALE = 30
CLOUD_MAX = 100.0
NDWI_THRESH = 0.0
SPACING = 500.0
OFFSHORE = 150.0
INLAND = 100.0
STEP = 10.0
AOI_MARGIN = 300.0
SIMPLIFY = 30.0
MIN_AREA = 3.6e5  # 36 ha at 30 m; drops cloud-gap blobs, keeps real coast
MAX_SECTIONS = 200
API_TIMEOUT = 1800
BATCH = 1000
SKIP_REPORT = True

EPOCHS = [
    ("base", "2018-09-08"),
    ("post", "2018-11-27"),
    ("recover", "2019-01-01"),
    ("q1", "2019-01-06"),
]

PAIRS = [
    ("base", "post", "Storm_base_post_m"),
    ("post", "recover", "Recovery_post_recover_m"),
    ("recover", "q1", "Noise_recover_q1_m"),
]

STORM_DATE = "2018-11-16"

_to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
_to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform


def to_wgs(geom):
    return shp_transform(_to_wgs, geom)


def to_utm(geom):
    return shp_transform(_to_utm, geom)


@lru_cache(maxsize=1)
def aoi():
    return ee.Geometry.Rectangle(AOI_LL)


def scl_mask(image):
    scl = image.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    bad = bad.focalMax(3).focalMax(3).focalMax(3)
    return image.updateMask(good.And(bad.Not()))


def prep(image):
    image = scl_mask(image)
    ten = image.select(["B3", "B4", "B8"]).multiply(0.0001).reproject(CRS, scale=SCALE)
    return ten.copyProperties(image, ["system:time_start", "SPACECRAFT_NAME"])


def ndwi(image):
    # NDWI (McFeeters): green vs NIR. Both 10 m bands, so no mixed-resolution
    # band math. Chosen over MNDWI because 2018 post-Gaja turbid water has
    # SWIR darker than NIR, which breaks MNDWI but not NDWI.
    return image.normalizedDifference(["B3", "B8"]).rename("WATER")


def collection(date):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi())
        .filterDate(date, ee.Date(date).advance(1, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX))
        .map(prep)
    )


def report(date, col):
    n = col.size().getInfo()
    print("\n{} -> {} scene(s), cloud filter < {}%".format(date, n, CLOUD_MAX))
    if not n:
        return

    def row(im):
        clear = (
            im.select("B8")
            .mask()
            .unmask(0)
            .reduceRegion(ee.Reducer.mean(), aoi(), 120)
            .get("B8")
        )
        tide = ee.Algorithms.If(
            im.bandNames().contains(ee.String("TIDE_F")),
            im.select("TIDE_F").reduceRegion(ee.Reducer.mean(), aoi(), 120).get("TIDE_F"),
            ee.Number(-9999),
        )
        return ee.Feature(None, {
            "id": im.get("system:index"),
            "system": im.get("SPACECRAFT_NAME"),
            "granule_cloud": im.get("CLOUDY_PIXEL_PERCENTAGE"),
            "aoi_clear_pct": ee.Number(clear).multiply(100),
            "tide_cm": ee.Number(tide),
        })

    def fmt(v, spec="{:.1f}", dash="n/a"):
        if v is None:
            return dash
        try:
            return spec.format(v)
        except (ValueError, TypeError):
            return str(v)

    for f in ee.FeatureCollection(col.map(row)).getInfo()["features"]:
        p = f["properties"]
        tid = str(p.get("id") or "?").split("/")[-1]
        tide = p.get("tide_cm")
        print(
            "    {:<50} {}  granule={:>6}  aoi_clear={:>6}  TIDE_F={}".format(
                tid[:50],
                fmt(p.get("system"), "{}"),
                fmt(p.get("granule_cloud"), "{:.2f}"),
                fmt(p.get("aoi_clear_pct"), "{:.1f}") + "%",
                "absent" if tide in (None, -9999) else "{:.2f} m".format(float(tide) / 100.0),
            )
        )


def land_geom(col):
    """Largest contiguous landmasses as a FeatureCollection dict.

    Land is vectorised rather than water: cloud gaps inside the sea turn into
    small detached land blobs that the area filter discards, whereas
    vectorising water shatters the coast into thousands of unusable slivers.
    """
    land = ndwi(col.mosaic()).lte(NDWI_THRESH).rename("land")
    land = land.mask(land)  # drop water pixels so only value-1 regions vectorise
    fc = land.reduceToVectors(
        geometry=aoi(),
        crs=CRS,
        scale=VECTOR_SCALE,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=int(1e13),
        labelProperty="land",
    )
    fc = fc.map(lambda f: f.set("area", f.area(10)))  # this API adds no area prop
    return (fc.filter(ee.Filter.gte("area", MIN_AREA))
            .sort("area", False).limit(60).getInfo())


def water_samples(col, feats, batch=BATCH):
    band = ndwi(col.mosaic())
    w = band.gt(NDWI_THRESH).unmask(-1)
    out = {}
    for i in range(0, len(feats), batch):
        part = feats[i:i + batch]
        fc = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point(f["geometry"]["coordinates"], proj=CRS),
                {"d": f["properties"]["d"], "s": f["properties"]["s"]},
            )
            for f in part
        ])
        got = w.sampleRegions(
            collection=fc, properties=["d", "s"], scale=SCALE,
            geometries=False, tileScale=8
        ).getInfo()["features"]
        for f in got:
            p = f["properties"]
            out[(p["s"], p["d"])] = p["WATER"]
    return out


def principal_line(poly):
    parts = sorted(getattr(poly, "geoms", [poly]), key=lambda g: g.area, reverse=True)
    return parts[0].boundary


def utm_aoi_box():
    corners = [to_utm(Point(x, y)) for x in (AOI_LL[0], AOI_LL[2]) for y in (AOI_LL[1], AOI_LL[3])]
    return box(
        min(c.x for c in corners),
        min(c.y for c in corners),
        max(c.x for c in corners),
        max(c.y for c in corners),
    )


def build_sections(line, water_poly, spacing=SPACING, tangent_window=30.0, keepaway=None, margin=AOI_MARGIN):
    total = line.length
    count = max(int(total // spacing), 1)
    out = []
    for k in range(count + 1):
        d = min(k * spacing, total)
        p = line.interpolate(d)
        a = line.interpolate(max(d - tangent_window, 0.0))
        b = line.interpolate(min(d + tangent_window, total))
        tx, ty = b.x - a.x, b.y - a.y
        m = math.hypot(tx, ty)
        if m < 1e-9:
            continue
        tx, ty = tx / m, ty / m
        nx, ny = -ty, tx
        cand = np.array([nx, ny])
        if not water_poly.contains(Point(p.x + nx * 90, p.y + ny * 90)):
            cand = -cand
        seaward = Point(p.x + cand[0] * 90, p.y + cand[1] * 90)
        inland = Point(p.x - cand[0] * 90, p.y - cand[1] * 90)
        if not (water_poly.contains(seaward) and not water_poly.contains(inland)):
            continue
        if keepaway is not None and keepaway.exterior.distance(p) < margin:
            continue
        out.append((np.array([p.x, p.y]), cand))
    return out


def transect_points(p, n, sid=None):
    feats = []
    for d in np.arange(-INLAND, OFFSHORE + 1e-9, STEP):
        q = p + n * d
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(q[0]), 2),
                                                              round(float(q[1]), 2)]},
                "properties": {"d": round(float(d), 2), "s": sid},
            }
        )
    return feats


def key_of(coord):
    return (round(coord[0], 2), round(coord[1], 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    ee.Initialize()
    ee.data.setDeadline(API_TIMEOUT * 1000)

    cols = {}
    for epoch, date in EPOCHS:
        col = collection(date)
        if not SKIP_REPORT:
            report(date, col)
        if not col.size().getInfo():
            print("  !! no usable scene on {} -- stopping".format(date))
            return
        cols[epoch] = col

    polys, lines = {}, {}
    for epoch, _ in EPOCHS:
        gj = land_geom(cols[epoch])
        feats = gj.get("features") or []
        if not feats:
            print("  !! no land polygon above {:.0f} m2 on {} -- stopping".format(MIN_AREA, epoch))
            return
        polys[epoch] = unary_union([to_utm(shape(f["geometry"])) for f in feats])
        lines[epoch] = principal_line(polys[epoch])
        print("  {}: land polygon area  = {:.1f} km2".format(
            epoch, polys[epoch].area / 1e6))

    stations = build_sections(lines["base"], polys["base"], keepaway=utm_aoi_box())
    if len(stations) > MAX_SECTIONS:
        stride = len(stations) / float(MAX_SECTIONS)
        stations = [stations[int(i * stride)] for i in range(MAX_SECTIONS)]
        print("  (capped to MAX_SECTIONS={})".format(MAX_SECTIONS))
    print("\nsections: {}".format(len(stations)))
    if not stations:
        print("!! no valid sections -- widen the AOI, or lower AOI_MARGIN")
        return

    all_feats = [f for i, (p, n) in enumerate(stations, start=1)
                 for f in transect_points(p, n, i)]
    print("transect sample points: {} ({} per section)".format(
        len(all_feats), len(transect_points(stations[0][0], stations[0][1], 1))))

    vals = {}
    for epoch, col in cols.items():
        vals[epoch] = water_samples(col, all_feats)
        print("{}: {} samples returned".format(epoch, len(vals[epoch])))

    names = [e for e, _ in EPOCHS]
    header = ["Section_ID", "Latitude", "Longitude", "Storm", "Storm_Date"]
    header += [c for _, _, c in PAIRS]
    header += ["MaxRetreat_m", "MeanRetreat_m", "NRate_m_per_yr"]

    rows = []
    out_feats = []
    for idx, (p, n) in enumerate(stations, start=1):
        entry = transect_points(p, n, idx)
        hits = {e: None for e in names}
        for f in entry:
            k = (idx, f["properties"]["d"])
            for epoch in names:
                v = vals[epoch].get(k)
                if v == 0:
                    prev = hits[epoch]
                    hits[epoch] = f["properties"]["d"] if prev is None else max(prev, f["properties"]["d"])
                f["properties"][epoch] = v

        rets = {}
        for a, b, col in PAIRS:
            ha, hb = hits.get(a), hits.get(b)
            rets[col] = None if ha is None or hb is None else round(hb - ha, 2)
        got = [v for v in rets.values() if v is not None]
        wgs = to_wgs(Point(p))
        row = {
            "Section_ID": "S{:03d}".format(idx),
            "Latitude": round(wgs.y, 6),
            "Longitude": round(wgs.x, 6),
            "Storm": "Gaja",
            "Storm_Date": STORM_DATE,
        }
        row.update(rets)
        row["MaxRetreat_m"] = round(max(got), 2) if got else None
        row["MeanRetreat_m"] = round(sum(got) / len(got), 2) if got else None
        row["NRate_m_per_yr"] = (
            round(row["MeanRetreat_m"] / 0.2028, 2)
            if row["MeanRetreat_m"] is not None else None
        )
        rows.append(row)
        out_feats.extend(entry)

    with open(os.path.join(OUT, "transects.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": out_feats}, fh)
    for epoch, _ in EPOCHS:
        with open(os.path.join(OUT, "coast_{}.geojson".format(epoch)), "w") as fh:
            json.dump(mapping(to_wgs(lines[epoch])), fh)

    csv = os.path.join(OUT, "shoreline_retreat.csv")
    with open(csv, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r[k]) for k in header) + "\n")
    print("\nwrote", csv)


if __name__ == "__main__":
    main()