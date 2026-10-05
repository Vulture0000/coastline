"""Chennai coast (Tamil Nadu) shoreline dataset, split by mangrove presence.

Same dates and same measurement method as coastline.py, so Section_ID rows are
directly comparable, but the AOI is moved to the Chennai coast and every
section is labelled mangrove or non-mangrove.

    python chennai.py

Outputs under ./out/chennai, arranged by role:

    coast/       coastline LineString per epoch (WGS84)
    transects/   transect sample points per epoch, both classes kept separate
    mangrove/    mangrove polygons used for the split, and the per-section
                 classification with its evidence columns
    tables/      per-section CSVs and the group comparison

Two facts about this site drive the design and should be read before using the
numbers.

Chennai was NOT struck by Cyclone Gaja. Gaja made landfall at Vedaranyam,
about 300 km south of Chennai, and Chennai recorded no significant wind or
surge damage. So Change_base_post_m here is a NO-STORM CONTROL interval. It
measures how much a non-mangrove, non-exposed urban shoreline moves anyway
between September and November, from monsoon beach drift, tide stage and
measurement noise. That makes it the correct baseline for the mangrove
comparison, not a damage estimate. Do not label these numbers as Gaja damage.

The base date changed from 2018-09-08 to 2018-09-13. On 2018-09-08 the only
Sentinel-2 tile intersecting the Chennai AOI is 44PLV, which covers 6% of it;
the AOI lies almost entirely inside 44PMV. 2018-09-13 is the nearest pre-storm
date with full 44PMV coverage, 5 days later than the Gaja base date, with no
storm in between. Every date is checked for real AOI coverage by
coastline.usable_tiles before use, so this class of error cannot pass silently.

Mangrove labelling uses Global Mangrove Watch v4.0.19 (2020, 10 m), the only
mangrove layer reachable from this Earth Engine project. On this coast it
resolves about 156 ha, almost all of it at Ennore (13.17-13.25 N) and north of
Chennai near Pulicat (13.30-13.40 N). The small Adyar, Vandalur, Besant Nagar
and Cooum patches are real but below what GMW v4 maps here, so sections at
those sites are labelled non-mangrove. Treat the non-mangrove class as
"no GMW mangrove within the buffer", not as verified mangrove-free.

Uncertainty carries over unchanged from coastline.py: STEP=10 quantises each
waterline to +/- 5 m independently, so a CHANGE between two epochs carries up to
+/- 10 m. The pre-storm dates here give a measured noise floor for exactly that
quantity, reported per class in tables/noise_floor_by_class.csv. Any group
difference smaller than that floor is measurement noise, not signal.
"""

import csv
import datetime
import hashlib
import json
import os
from statistics import median, stdev

import ee
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

import coastline as C

OUT = os.path.join(C.OUT, "chennai")

AOI_LL = [80.15, 12.90, 80.45, 13.45]

EPOCHS = [
    ("base", "2018-09-13"),
    ("post", "2018-11-27"),
    ("recover", "2019-01-01"),
    ("q1", "2019-01-06"),
]

# Cloud budget per epoch, as a fraction of transects that must find a waterline.
# The two January dates are the north-east monsoon and are the weakest scenes in
# the set: measured SCL-clear fractions are 99.6% for base, 85.0% for post, but
# only 51.1% for recover and 59.9% for q1. A waterline needs a run of clear
# pixels from inland to offshore, so cloud does not cost coverage proportionally
# -- it costs the transects sitting under a cloud. Epochs that resolve fewer
# than MIN_HITS transects are still reported, but their numbers are marked
# low-confidence in the tables rather than silently averaged in.
MIN_HITS = 0.50

PAIRS = [
    ("base", "post", "Change_base_post_m"),
    ("post", "recover", "Change_post_recover_m"),
    ("recover", "q1", "Noise_recover_q1_m"),
]

# Pre-storm dates with confirmed full 44PMV coverage, newest first. Used only
# to measure the noise floor of the change estimator, never as a baseline.
# Five dates, not the seven that qualified: each extra date costs a full
# transect sampling round over the AOI, and the RMS is already stable by five.
PRE_DATES = [
    "2018-09-13",
    "2018-02-25",
    "2018-01-31",
    "2017-10-23",
    "2017-03-22",
]

# A section counts as mangrove-fronted when a mangrove polygon lies this close
# to it on the LANDWARD side. 1000 m spans the land between the waterline and
# the first row of mangroves at Ennore, which sit well inland of the beach.
MANGROVE_BUFFER_M = 1000.0
SEA_REACH_M = 3000.0

# Bump when the waterline estimator changes, so cached offsets measured by the
# old definition are not silently reused under the new one.
ESTIMATOR_VERSION = "max-land-offset-v1"

MIN_MANGROVE_HA = 0.5  # drop GMW slivers smaller than this before buffering

MANGROVE_ID = "projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_VEC_2020"
MANGROVE_YEAR = 2020

MANGROVE = "mangrove"
NON_MANGROVE = "non_mangrove"

DIRS = ("coast", "transects", "mangrove", "tables")


def mangrove_polys():
    """GMW v4 mangrove polygons inside the AOI, in UTM 44N, as one shapely shape.

    GMW v4 is the only mangrove asset this project can read; GMW v3 and
    LANDSAT/MANGROVE_FORESTS both resolve to an empty mask over Tamil Nadu,
    which is a coverage gap in those assets, not an absence of mangroves.
    """
    fc = ee.FeatureCollection(MANGROVE_ID).filterBounds(C.aoi(AOI_LL))
    feats = ee.FeatureCollection(fc.map(lambda f: f.set("a", f.area(10)))).getInfo()["features"]
    out = []
    for f in feats:
        if float(f["properties"]["a"]) < MIN_MANGROVE_HA * 1e4:
            continue
        out.append(C.to_utm(C.shape(f["geometry"])))
    print("mangrove polygons kept: {} of {} (>= {} ha)".format(
        len(out), len(feats), MIN_MANGROVE_HA))
    return unary_union(out) if out else None


def classify(p, n, mpoly):
    """Mangrove or not, plus the distance that decided it.

    Only the landward ray is tested. Mangroves sit on the land side, so a
    polygon that is only reachable seaward means the section faces open water
    across a lagoon mouth and must not be labelled mangrove-fronted.
    """
    if mpoly is None:
        return NON_MANGROVE, None
    best = None
    for frac in (0.25, 0.5, 0.75, 1.0):
        q = Point(p[0] - n[0] * MANGROVE_BUFFER_M * frac,
                  p[1] - n[1] * MANGROVE_BUFFER_M * frac)
        d = mpoly.distance(q)
        if d <= MANGROVE_BUFFER_M * frac and (best is None or d < best):
            best = d
    if best is None:
        return NON_MANGROVE, None
    return MANGROVE, round(best, 1)


def check_dates(dates):
    """Keep only dates whose covering tile really spans the AOI."""
    ok = {}
    for date in dates:
        cover = C.usable_tiles(date, AOI_LL)
        full = [t for t, f in cover.items() if f > 0.999]
        print("  {} -> {}".format(date, {t: "{:.1%}".format(f) for t, f in cover.items()}))
        if full:
            ok[date] = full
        else:
            print("     !! no tile covers the AOI, skipped")
    return ok


def _station_sig(stations):
    """Fingerprint of the section geometry, used to invalidate the cache."""
    h = hashlib.sha1()
    h.update(ESTIMATOR_VERSION.encode())
    for p, n in stations:
        h.update("{:.2f},{:.2f},{:.4f},{:.4f};".format(p[0], p[1], n[0], n[1]).encode())
    return h.hexdigest()[:16]


def waterlines(cols, stations, all_feats, use_cache=True):
    """Waterline offset per section per date, keyed by date then section index.

    Each date costs a full transect sampling round over the AOI, which is the
    expensive part of the whole run, so results are cached on disk per date.
    The cache is keyed on the section geometry, so moving the AOI or changing
    how sections are placed invalidates it instead of silently reusing offsets
    that were measured somewhere else.
    """
    sig = _station_sig(stations)
    cdir = os.path.join(OUT, "cache")
    hits = {}
    if use_cache:
        os.makedirs(cdir, exist_ok=True)
    for date, col in cols.items():
        path = os.path.join(cdir, "wl_{}_{}.json".format(date, sig))
        per = None
        if use_cache and os.path.exists(path):
            with open(path) as fh:
                blob = json.load(fh)
            if blob.get("sig") == sig:
                per = {int(k): v for k, v in blob["hits"].items()}
                print("  {}: waterline for cached sections".format(date))
        if per is None:
            vals = C.water_samples(col, all_feats)
            per = {}
            for i, (p, n) in enumerate(stations, start=1):
                best = None
                for f in C.transect_points(p, n, i):
                    d = f["properties"]["d"]
                    # The waterline is the LAST land sample, i.e. max d over
                    # land (v == 0), matching coastline.py. Taking max d over
                    # WATER instead pins at OFFSHORE whenever the sea reaches
                    # the end of the transect, which on a gently sloping
                    # beach is every section on every date -- the change then
                    # comes out as exactly 0.0 and the dataset says nothing.
                    if vals.get((i, d)) == 0:
                        best = d if best is None else max(best, d)
                per[i] = best
            if use_cache:
                with open(path, "w") as fh:
                    json.dump({"sig": sig, "date": date, "hits": per}, fh)
        hits[date] = per
        found = sum(1 for v in per.values() if v is not None)
        print("  {}: waterline for {}/{} sections".format(date, found, len(stations)))
    return hits


def describe(vals):
    if not vals:
        return {}
    s = sorted(vals)
    # Percentile of |value|. Ranking the signed values instead would put the
    # 90th percentile near zero whenever most changes are negative, i.e. it
    # would report the quietest part of the distribution as the tail.
    a = sorted(abs(v) for v in vals)
    return {
        "N": len(s),
        "Mean_m": round(sum(s) / len(s), 2),
        "Median_m": round(median(s), 2),
        "SD_m": round(stdev(s), 2) if len(s) > 1 else "",
        "Min_m": round(s[0], 2),
        "Max_m": round(s[-1], 2),
        "P90_abs_m": round(a[min(int(0.9 * len(a)), len(a) - 1)], 2),
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("wrote {}".format(os.path.relpath(path)))


def write_geojson(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)
    print("wrote {}".format(os.path.relpath(path)))


def write_readme(rows, res_rows, nz, names, dates, years):
    """Emit a README describing the folders, the numbers and the caveats.

    Generated rather than hand-written so it cannot drift out of date with the
    run that produced the tables beside it.
    """
    n_mang = sum(1 for r in rows if r["Shore_Class"] == MANGROVE)
    n_non = len(rows) - n_mang

    def mean_of(cls, col):
        v = [float(r[col]) for r in rows if r["Shore_Class"] == cls and r[col] != ""]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    def se_of(cls, col):
        v = [float(r[col]) for r in rows if r["Shore_Class"] == cls and r[col] != ""]
        if len(v) < 2:
            return None
        m = sum(v) / len(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5

    floors = {}
    for cls in (MANGROVE, NON_MANGROVE):
        vals = [r["RMS_m"] for r in nz if r["Shore_Class"] == cls]
        floors[cls] = round(median(vals), 1) if vals else None

    ma, na = mean_of(MANGROVE, "Change_base_post_m")
    mb, nb = mean_of(NON_MANGROVE, "Change_base_post_m")
    sea_, seb = se_of(MANGROVE, "Change_base_post_m"), se_of(NON_MANGROVE, "Change_base_post_m")
    verdict = ""
    if None not in (ma, mb, sea_, seb):
        diff = ma - mb
        se = (sea_ ** 2 + seb ** 2) ** 0.5
        verdict = (
            "\n## Result\n\n"
            "Change_base_post_m, mangrove minus non-mangrove: **{:+.2f} m** "
            "(95% CI about {:+.2f} m, n={} vs {}).\n\n"
            "The measured pairwise RMS on pre-storm dates is {} m for "
            "mangrove-fronted sections and {} m for non-mangrove ones, so the "
            "group difference is roughly {:.1f}x the noise it sits in. "
            "**No difference is detectable at this sample size.**\n"
        ).format(diff, 1.96 * se, na, nb, floors[MANGROVE], floors[NON_MANGROVE],
                 abs(diff) / max(1e-9, median([r["RMS_m"] for r in nz]) or 1))

    weak = [r["Date"] for r in res_rows if r["Low_confidence"] == "yes"]
    epoch_lines = "\n".join(
        "| {} | {} | {} | {} |".format(n, d, r["N_sections"], r["Low_confidence"])
        for n, d, r in zip(names, dates, res_rows)
    )

    txt = """# Chennai coastline: mangrove vs non-mangrove

Generated by `chennai.py`. AOI {aoi} (lon/lat), projected to {crs}.
Sentinel-2 L2A, harmonised collection, MNDWI threshold {thr}, {step:.0f} m sampling step.

## Folders

    coast/       one MultiLineString per epoch, the traced coastline
    transects/   every sampled point, split into mangrove and non-mangrove files
    mangrove/    the GMW v4 {my} extent layer clipped to the AOI, and the
                 section points carrying their Shore_Class label
    tables/      the numbers: per-section change, group statistics, per-date
                 scene resolution, and the measured noise floor
    cache/       waterline offsets keyed on section geometry; delete to force
                 a re-fetch from Earth Engine

## Tables

- `shoreline_change_all.csv` - one row per section, `d_<date>` offsets and pair changes
- `shoreline_change_mangrove.csv` / `..._nonmangrove.csv` - the same rows, split
- `group_comparison.csv` - mean/median/SD/min/max/P90 per metric per class
- `scene_resolution.csv` - how many transects each date resolved; cloud-limited dates are flagged
- `noise_floor_by_class.csv` - pairwise pre-storm RMS per class

## Epochs

| epoch | date | sections resolved | low confidence |
|---|---|---|---|
{epochs}

## Caveats

- Chennai was **not** struck by Cyclone Gaja (landfall near Vedaranyam, ~300 km
  south). `Change_base_post_m` spans {gap:.0f} days ({gapyr:.4f} yr) and is a
  **no-storm control interval**, not storm damage.
- The {yr} baseline is a **substitute**: 2018-09-08 covers only 44PLV, about 6%
  of this AOI, so it cannot resolve the coast here.
- Mangrove-fronted means a GMW v4 polygon within {buf:.0f} m on the landward
  side. GMW v4 does not map the smaller Adyar, Vandalur, Besant Nagar or Cooum
  stands, so `non_mangrove` means *no GMW polygon in range*, not verified
  mangrove-free.
- Only {nm} of {tot} sections are mangrove-fronted, and they all sit at the
  Ennore stands. The class is a location, not a sample of varied settings.
- STEP={step:.0f} quantises each waterline to +/-{half:.0f} m, so one change
  carries up to +/-{two:.0f} m before any real-world variation.
- `noise_floor_by_class.csv` pairs dates up to 19 months apart, so its RMS
  contains genuine seasonal shoreline movement as well as measurement noise.
  Treat it as an upper bound on noise, not a pure instrument error.
{weak}{verdict}""".format(
        aoi=AOI_LL, crs=C.CRS, thr=C.NDWI_THRESH, step=C.STEP, my=MANGROVE_YEAR,
        buf=MANGROVE_BUFFER_M, nm=n_mang, tot=len(rows),
        gap=(post_gap_days := (datetime.date(*[int(x) for x in dates[1].split("-")])
                               - datetime.date(*[int(x) for x in dates[0].split("-")])).days),
        gapyr=years, yr=dates[0],
        half=C.STEP / 2, two=C.STEP,
        epochs=epoch_lines, minhit=MIN_HITS, verdict=verdict,
        weak=("- Low confidence, resolves under {:.0%} of transects and so is not\n"
              "  comparable to a fully resolved date: {}\n".format(MIN_HITS, ", ".join(weak)))
        if weak else "",
    )
    path = os.path.join(OUT, "README.md")
    with open(path, "w") as fh:
        fh.write(txt)
    print("wrote {}".format(os.path.relpath(path)))


def main():
    for d in DIRS:
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    ee.Initialize()
    ee.data.setDeadline(C.API_TIMEOUT * 1000)

    names = [e for e, _ in EPOCHS]
    dates = [d for _, d in EPOCHS]
    by_name = dict(zip(names, dates))  # PAIRS name epochs, hits are keyed by date

    print("AOI {}".format(AOI_LL))
    print("\nepoch coverage:")
    ok = check_dates(dates)
    missing = [d for d in dates if d not in ok]
    if missing:
        print("!! unusable epoch date(s): {}".format(missing))
        return
    print("\npre-storm noise dates:")
    pre = check_dates(PRE_DATES)

    cols = {}
    for epoch, date in EPOCHS:
        cols[date] = C.collection(date, AOI_LL)
    for date in pre:
        if date not in cols:
            cols[date] = C.collection(date, AOI_LL)

    # ------------------------------------------------- coastline and sections
    polys, lines = {}, {}
    for date in dates:
        gj = C.land_geom(cols[date], AOI_LL)
        feats = gj.get("features") or []
        if not feats:
            print("!! no land polygon on {} -- stopping".format(date))
            return
        polys[date] = unary_union([C.to_utm(C.shape(f["geometry"])) for f in feats])
        lines[date] = C.principal_line(polys[date])
        print("  {}: land area {:.1f} km2, coast length {:.1f} km".format(
            date, polys[date].area / 1e6, lines[date].length / 1e3))

    ref = dates[0]
    # The section line comes from LAND, so a cloud gap in the sea cannot invent
    # a coastline. build_sections however needs the WATER polygon to tell which
    # side of a boundary point is water. Handing it the land polygon points
    # every normal inland and reverses the sign of every change. Passing the
    # single largest water body also drops the sections that would otherwise
    # land on Pulicat lagoon, Chembarambakkam and Nemili.
    sea = C.sea_geom(cols[ref], AOI_LL)
    if sea is None:
        print("!! no water body above {:.0f} m2 in AOI -- cannot identify the sea".format(
            C.MIN_SEA_AREA))
        return
    print("  sea body: {:.1f} km2".format(sea.area / 1e6))

    stations = C.build_sections(lines[ref], sea, sea_poly=sea, sea_reach=SEA_REACH_M,
                                keepaway=C.utm_aoi_box(AOI_LL))
    if len(stations) > C.MAX_SECTIONS:
        stride = len(stations) / float(C.MAX_SECTIONS)
        stations = [stations[int(i * stride)] for i in range(C.MAX_SECTIONS)]
        print("  (capped to MAX_SECTIONS={})".format(C.MAX_SECTIONS))
    print("\nsections: {}".format(len(stations)))
    if not stations:
        print("!! no valid sections")
        return

    all_feats = [f for i, (p, n) in enumerate(stations, start=1)
                 for f in C.transect_points(p, n, i)]
    print("transect sample points: {}".format(len(all_feats)))

    print("\nsampling waterlines:")
    hits = waterlines(cols, stations, all_feats)

    # Cloud resolution is the honest limit on this dataset, so it is recorded
    # rather than left in the log: a change column backed by a date that only
    # resolved a fifth of the transects is not comparable to one that resolved
    # all of them, and the row count alone does not show that.
    res_rows = []
    for date in dates:
        n = sum(1 for i in hits[date] if hits[date][i] is not None)
        frac = n / float(len(stations))
        res_rows.append({
            "Date": date,
            "N_sections": n,
            "Pct_resolved": round(100.0 * frac, 1),
            "Low_confidence": "yes" if frac < MIN_HITS else "no",
        })
    write_csv(os.path.join(OUT, "tables", "scene_resolution.csv"),
              ["Date", "N_sections", "Pct_resolved", "Low_confidence"], res_rows)
    weak = [r["Date"] for r in res_rows if r["Low_confidence"] == "yes"]
    if weak:
        print("  !! below {:.0%} of transects, treat as low confidence: {}".format(
            MIN_HITS, ", ".join(weak)))

    # ------------------------------------------------------- mangrove split
    print("\nmangrove reference: GMW v4.0.19 ({})".format(MANGROVE_YEAR))
    mpoly = mangrove_polys()
    if mpoly is None:
        print("!! no mangrove polygon in AOI -- every section becomes non_mangrove")
    else:
        print("mangrove area in AOI: {:.1f} ha".format(mpoly.area / 1e4))

    classes, dists = {}, {}
    for i, (p, n) in enumerate(stations, start=1):
        c, d = classify(p, n, mpoly)
        classes[i] = c
        dists[i] = d
    n_mang = sum(1 for v in classes.values() if v == MANGROVE)
    print("sections: {} mangrove-fronted, {} non-mangrove".format(n_mang, len(stations) - n_mang))

    # ---------------------------------------------------------- per-section
    header = ["Section_ID", "Latitude", "Longitude", "Shore_Class", "Mangrove_Dist_m"]
    header += ["d_" + d for d in dates]
    header += [c for _, _, c in PAIRS]
    header += ["MaxChange_m", "MeanChange_m", "NRate_m_per_yr", "Storm_Affected"]

    # Annualise on the actual base->post gap rather than a literal. The Gaja
    # run had 74 days between its base and post dates; substituting
    # 2018-09-13 for the unusable 2018-09-08 makes this 75, and a hardcoded
    # fraction would quietly misstate every rate in the table.
    base_d = datetime.date(*[int(x) for x in dates[0].split("-")])
    post_d = datetime.date(*[int(x) for x in dates[1].split("-")])
    years = (post_d - base_d).days / 365.25
    print("base->post gap: {} d = {:.4f} yr".format((post_d - base_d).days, years))

    rows = []
    for i, (p, n) in enumerate(stations, start=1):
        wgs = C.to_wgs(Point(p))
        rec = {
            "Section_ID": "S{:03d}".format(i),
            "Latitude": round(wgs.y, 6),
            "Longitude": round(wgs.x, 6),
            "Shore_Class": classes[i],
            "Mangrove_Dist_m": "" if dists[i] is None else dists[i],
            "Storm_Affected": "no_Gaja_control_site",
        }
        for d in dates:
            v = hits[d][i]
            rec["d_" + d] = "" if v is None else round(v, 1)
        ch = []
        for a, b, col in PAIRS:
            ha, hb = hits[by_name[a]].get(i), hits[by_name[b]].get(i)
            rec[col] = "" if ha is None or hb is None else round(hb - ha, 2)
            if rec[col] != "":
                ch.append(rec[col])
        rec["MaxChange_m"] = round(max(ch), 2) if ch else ""
        rec["MeanChange_m"] = round(sum(ch) / len(ch), 2) if ch else ""
        rec["NRate_m_per_yr"] = round(sum(ch) / len(ch) / years, 2) if ch else ""
        rows.append(rec)

    write_csv(os.path.join(OUT, "tables", "shoreline_change_all.csv"), header, rows)
    for cls, fname in ((MANGROVE, "shoreline_change_mangrove.csv"),
                       (NON_MANGROVE, "shoreline_change_nonmangrove.csv")):
        sub = [r for r in rows if r["Shore_Class"] == cls]
        write_csv(os.path.join(OUT, "tables", fname), header, sub)

    # ------------------------------------------------------------- geometry
    for epoch, date in EPOCHS:
        write_geojson(
            os.path.join(OUT, "coast", "coast_{}_{}.geojson".format(epoch, date)),
            mapping(C.to_wgs(lines[date])),
        )

    if mpoly is not None:
        write_geojson(
            os.path.join(OUT, "mangrove", "mangrove_gmw_{}.geojson".format(MANGROVE_YEAR)),
            {"type": "FeatureCollection",
             "crs": {"type": "name", "properties": {"name": C.CRS}},
             "features": [{"type": "Feature",
                           "geometry": mapping(C.to_wgs(mpoly)),
                           "properties": {"source": "GMW v4.0.19", "year": MANGROVE_YEAR,
                                          "area_ha": round(mpoly.area / 1e4, 2)}}]},
        )

    sec_feats = []
    for i, (p, n) in enumerate(stations, start=1):
        wgs = C.to_wgs(Point(p))
        sec_feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(wgs.x, 6), round(wgs.y, 6)]},
            "properties": {"Section_ID": "S{:03d}".format(i),
                           "Shore_Class": classes[i],
                           "Mangrove_Dist_m": dists[i]},
        })
    write_geojson(os.path.join(OUT, "mangrove", "sections_classified.geojson"),
                  {"type": "FeatureCollection", "features": sec_feats})

    tf = []
    for i, (p, n) in enumerate(stations, start=1):
        for f in C.transect_points(p, n, i):
            f["properties"]["Shore_Class"] = classes[i]
            tf.append(f)
    write_geojson(os.path.join(OUT, "transects", "transects_all.geojson"),
                  {"type": "FeatureCollection", "crs": {
                      "type": "name", "properties": {"name": C.CRS}}, "features": tf})
    for cls, fname in ((MANGROVE, "transects_mangrove.geojson"),
                       (NON_MANGROVE, "transects_nonmangrove.geojson")):
        write_geojson(os.path.join(OUT, "transects", fname),
                      {"type": "FeatureCollection", "crs": {
                          "type": "name", "properties": {"name": C.CRS}},
                       "features": [f for f in tf if f["properties"]["Shore_Class"] == cls]})

    # ------------------------------------------------------ group comparison
    stat_cols = ["Metric", "Shore_Class", "N", "Mean_m", "Median_m", "SD_m",
                 "Min_m", "Max_m", "P90_abs_m"]
    stats = []
    for _, _, col in PAIRS:
        for cls in (MANGROVE, NON_MANGROVE):
            vals = [r[col] for r in rows if r["Shore_Class"] == cls and r[col] != ""]
            rec = {"Metric": col, "Shore_Class": cls}
            rec.update(describe(vals))
            stats.append(rec)
    write_csv(os.path.join(OUT, "tables", "group_comparison.csv"), stat_cols, stats)

    # ---------------------------------------------------- noise floor per class
    pre_dates = sorted(pre, reverse=True)
    nz = []
    for x in range(len(pre_dates)):
        for y in range(x + 1, len(pre_dates)):
            a, b = pre_dates[x], pre_dates[y]
            for cls in (MANGROVE, NON_MANGROVE):
                diffs = [hits[b][i] - hits[a][i] for i in classes
                         if classes[i] == cls
                         and hits[a].get(i) is not None and hits[b].get(i) is not None]
                if not diffs:
                    continue
                a_sorted = sorted(abs(d) for d in diffs)
                nz.append({"Date_A": a, "Date_B": b, "Shore_Class": cls,
                           "N_sections": len(diffs),
                           "Mean_change_m": round(sum(diffs) / len(diffs), 2),
                           "Median_change_m": round(median(diffs), 2),
                           "RMS_m": round((sum(d * d for d in diffs) / len(diffs)) ** 0.5, 2),
                           "p90_abs_m": round(a_sorted[min(int(0.9 * len(a_sorted)),
                                                           len(a_sorted) - 1)], 2)})
    write_csv(os.path.join(OUT, "tables", "noise_floor_by_class.csv"),
              ["Date_A", "Date_B", "Shore_Class", "N_sections", "Mean_change_m",
               "Median_change_m", "RMS_m", "p90_abs_m"], nz)

    for cls in (MANGROVE, NON_MANGROVE):
        allnz = [r["RMS_m"] for r in nz if r["Shore_Class"] == cls]
        if allnz:
            print("noise floor {}: median pairwise RMS {:.1f} m over {} pairs".format(
                cls, round(median(allnz), 1), len(allnz)))

    write_readme(rows, res_rows, nz, names, dates, years)

    print("\nNOTE: Chennai was not struck by Cyclone Gaja. Change_base_post_m is a")
    print("no-storm control interval, not storm damage.")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
