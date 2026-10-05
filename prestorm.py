"""Pre-storm baseline dataset for Cyclone Gaja (Tamil Nadu, 16 Nov 2018).

Extracts every usable Sentinel-2 scene strictly BEFORE landfall and builds a
standalone dataset in out/pre/:

  shoreline_prestorm.csv        one row per transect section, waterline offset
                                per acquisition date plus across-date statistics
  waterline_<date>.geojson      waterline position per acquisition date
  prestorm_baseline.geojson     median pre-storm waterline
  prestorm_noise.csv            pairwise change between every date pair

Offsets are metres seaward of the station point, measured along the seaward
normal defined in coastline.py. All geometry is EPSG:32644.

Note on tile coverage: the AOI straddles MGRS tiles 44PLS and 44PMS, so some
dates cover only part of the AOI and sections outside that part have no value
for that date. N_Dates reports how many dates actually resolved a waterline.
"""

import csv
import json
import os
from statistics import median, stdev

import ee
from shapely.geometry import Point

import coastline as C


def to_lonlat(p):
    c = C.to_wgs(Point(float(p[0]), float(p[1]))).coords[0]
    return c[0], c[1]

STORM_DATE = "2018-11-16"
OUT = "out/pre"

# Every usable pre-storm acquisition. 2018-07-10 (99.6% cloud) is excluded.
# Tiles are noted because coverage is partial for single-tile dates.
PRE_DATES = [
    ("2018-09-13", "44PMS"),
    ("2018-09-08", "44PLS+44PMS"),
    ("2018-03-27", "44PMS"),
    ("2018-02-15", "44PMS"),
    ("2017-12-27", "44PLS"),
    ("2017-12-12", "44PMS"),
    ("2017-11-17", "44PLS+44PMS"),
]
REF_DATE = "2018-09-08"  # defines the sections, same as the storm dataset


def days_before(date):
    return int((ee.Date(STORM_DATE).difference(ee.Date(date), "day")).getInfo())


def main():
    os.makedirs(OUT, exist_ok=True)
    ee.Initialize()
    ee.data.setDeadline(C.API_TIMEOUT * 1000)

    cols = {}
    for date, tiles in PRE_DATES:
        col = C.collection(date)
        if not col.size().getInfo():
            print("  !! no scene on {} -- skipped".format(date))
            continue
        cols[date] = col
    if not cols:
        print("!! no usable pre-storm scenes")
        return
    print("pre-storm dates in use: {}".format(len(cols)))

    # Sections: identical geometry to the storm dataset, from the same
    # reference date, so Section_ID rows can be joined directly.
    gj = C.land_geom(cols[REF_DATE])
    feats = gj.get("features") or []
    if not feats:
        print("!! no land polygon on reference date {}".format(REF_DATE))
        return
    base_poly = C.unary_union([C.to_utm(C.shape(f["geometry"])) for f in feats])
    line = C.principal_line(base_poly)
    # build_sections needs the water polygon, not the land one: passing land
    # points every normal inland and reverses the sign of every change.
    sea = C.sea_geom(cols[REF_DATE])
    if sea is None:
        print("!! no water body above {:.0f} m2 on reference date {}".format(
            C.MIN_SEA_AREA, REF_DATE))
        return
    stations = C.build_sections(line, sea, keepaway=C.utm_aoi_box())
    if len(stations) > C.MAX_SECTIONS:
        stride = len(stations) / float(C.MAX_SECTIONS)
        stations = [stations[int(i * stride)] for i in range(C.MAX_SECTIONS)]
    print("sections: {}".format(len(stations)))

    all_feats = [
        f for i, (p, n) in enumerate(stations, start=1)
        for f in C.transect_points(p, n, i)
    ]
    print("transect sample points: {}".format(len(all_feats)))

    # waterline offset per section per date
    offsets = {}
    for date in cols:
        vals = C.water_samples(cols[date], all_feats)
        hit = {}
        for i, (p, n) in enumerate(stations, start=1):
            best = None
            for f in C.transect_points(p, n, i):
                d = f["properties"]["d"]
                if vals.get((i, d)) == 1:
                    best = d if best is None else max(best, d)
            hit[i] = best
        offsets[date] = hit
        found = sum(1 for v in hit.values() if v is not None)
        print("  {}: waterline found for {}/{} sections".format(date, found, len(stations)))

    # ---------------------------------------------------------------- CSV
    rows = []
    for i, (p, n) in enumerate(stations, start=1):
        lon, lat = to_lonlat(p)
        rec = {"Section_ID": "S{:03d}".format(i),
               "Latitude": round(lat, 6), "Longitude": round(lon, 6)}
        for date in cols:
            v = offsets[date][i]
            rec["d_" + date] = "" if v is None else round(v, 1)
        seen = [offsets[d][i] for d in cols if offsets[d][i] is not None]
        rec["N_Dates"] = len(seen)
        rec["Pre_Storm_Mean_d"] = round(sum(seen) / len(seen), 1) if seen else ""
        rec["Pre_Storm_Median_d"] = round(median(seen), 1) if seen else ""
        rec["Pre_Storm_SD_m"] = round(stdev(seen), 1) if len(seen) > 1 else ""
        rec["Pre_Storm_Range_m"] = round(max(seen) - min(seen), 1) if seen else ""
        rows.append(rec)

    path = os.path.join(OUT, "shoreline_prestorm.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote {}".format(path))

    # -------------------------------------------------- waterline GeoJSON
    def line_feature(offsets_by_section):
        coords = []
        for i, (p, n) in enumerate(stations, start=1):
            d = offsets_by_section.get(i)
            if d is None:
                continue
            q = p + n * d
            coords.append([round(float(q[0]), 2), round(float(q[1]), 2)])
        return {"type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"n_points": len(coords)}}

    for date in cols:
        gj = {"type": "FeatureCollection",
              "crs": {"type": "name", "properties": {"name": C.CRS}},
              "features": [line_feature(offsets[date])]}
        p = os.path.join(OUT, "waterline_{}.geojson".format(date))
        with open(p, "w") as fh:
            json.dump(gj, fh)
    print("wrote {} waterline files".format(len(cols)))

    med = {}
    for i in range(1, len(stations) + 1):
        seen = [offsets[d][i] for d in cols if offsets[d][i] is not None]
        if seen:
            med[i] = median(seen)
    p = os.path.join(OUT, "prestorm_baseline.geojson")
    with open(p, "w") as fh:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": C.CRS}},
                   "features": [line_feature(med)]}, fh)
    print("wrote {}".format(p))

    # ------------------------------------------------------ pairwise noise
    dates = sorted(cols, reverse=True)
    with open(os.path.join(OUT, "prestorm_noise.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Date_A", "Date_B", "Gap_days", "N_sections", "Mean_change_m",
                     "Median_change_m", "RMS_m", "p90_abs_m"])
        for x in range(len(dates)):
            for y in range(x + 1, len(dates)):
                a, b = dates[x], dates[y]
                diffs = [offsets[b][i] - offsets[a][i] for i in range(1, len(stations) + 1)
                         if offsets[a][i] is not None and offsets[b][i] is not None]
                if not diffs:
                    continue
                wr.writerow([a, b, days_before(a) - days_before(b), len(diffs),
                             round(sum(diffs) / len(diffs), 2),
                             round(median(diffs), 2),
                             round((sum(d * d for d in diffs) / len(diffs)) ** 0.5, 2),
                             round(sorted(abs(d) for d in diffs)[int(0.9 * len(diffs))], 2)])
    print("wrote {}".format(os.path.join(OUT, "prestorm_noise.csv")))


if __name__ == "__main__":
    main()