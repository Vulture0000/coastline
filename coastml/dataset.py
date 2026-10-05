"""Stage 1-5: build the clean master dataset (section x storm) and store in SQLite + CSV.

Simulates physically-plausible coastal controls (mangroves, dunes, terrain),
storm/wave forcing with alongshore decay from each cyclone's landfall point,
and a non-linear shoreline-retreat response. Deterministic (seeded) so the
dataset is reproducible and clean: no nulls, in-bounds, unique primary key.
"""
import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- geometry
def _polyline():
    pts = C.COAST_CONTROL_POINTS
    seg = []
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        seg.append((total, d, (x0, y0), (x1, y1)))
        total += d
    return seg, total


def section_points(n):
    """n equally-spaced (lon, lat, t) samples along the coast polyline."""
    seg, total = _polyline()
    out = []
    for i in range(n):
        s = total * i / (n - 1)
        for start, d, (x0, y0), (x1, y1) in seg:
            if s <= start + d or (start, d) == seg[-1][0:2]:
                f = 0.0 if d == 0 else min(max((s - start) / d, 0.0), 1.0)
                out.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), i / (n - 1)))
                break
    return out


def _smooth(t, rng, amp=1.0):
    """Smooth spatial field over alongshore coordinate t in [0,1]."""
    return amp * (
        0.55 * math.sin(2 * math.pi * (1.3 * t) + rng.uniform(0, 6.28))
        + 0.30 * math.sin(2 * math.pi * (3.1 * t) + rng.uniform(0, 6.28))
        + 0.15 * math.sin(2 * math.pi * (6.7 * t) + rng.uniform(0, 6.28))
    )


# ---------------------------------------------------------------- real mangroves
_M_PER_DEG_LAT = 110540.0


def _m_per_deg_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


def load_gmw_mangrove():
    """Real GMW 2020 mangrove polygon (Chennai) in local metres + belt width."""
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    with open(C.GMW_MANGROVE_GEOJSON) as fh:
        gj = json.load(fh)
    geom = shape(gj["features"][0]["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    lat0 = (miny + maxy) / 2.0
    mx, my = _m_per_deg_lon(lat0), _M_PER_DEG_LAT
    proj = shp_transform(lambda x, y: (x * mx, y * my), geom)
    area = gj["features"][0]["properties"].get("area_ha")
    area_m2 = area * 10000.0 if area else proj.area
    alongshore = (maxy - miny) * _M_PER_DEG_LAT   # belt extent along the coast
    width = area_m2 / alongshore if alongshore > 0 else 100.0
    return proj, float(width), lat0


def gmw_distance_m(proj_poly, lon, lat, lat0):
    from shapely.geometry import Point
    p = Point(lon * _m_per_deg_lon(lat0), lat * _M_PER_DEG_LAT)
    return proj_poly.distance(p)


# ---------------------------------------------------------------- generation
def derive_features(d):
    """Derived stage-4 indices from raw values (single source of truth)."""
    energy = (d["storm_wind_ms"] / 40.0) ** 1.7 * (d["wave_height_m"] / 4.5) ** 0.8
    veg = math.exp(-d["mangrove_width_m"] / 2200.0) * (1 - 0.25 * d["mangrove_density"])
    dune = 1.0 / (1.0 + 0.30 * d["dune_height_m"] + 0.003 * d["dune_width_m"])
    topo = math.exp(-d["terrain_elev_m"] / 10.0) * (0.7 + 0.15 * d["terrain_slope_deg"])
    out = {
        "energy_index": energy,
        "mangrove_protection": veg,
        "dune_protection": dune,
        "terrain_exposure": topo,
        "composite_erosion_idx": energy * veg * dune * topo,
    }
    return out


def build_dataframe():
    rng = np.random.default_rng(C.RANDOM_SEED)
    secs = section_points(C.N_SECTIONS)
    gmw_poly, w_gmw, lat0 = load_gmw_mangrove()

    # per-section static controls (smooth fields + mild noise)
    static = {}
    for i, (lon, lat, t) in enumerate(secs):
        d_gmw = gmw_distance_m(gmw_poly, lon, lat, lat0)
        mw = (
            2400.0 * math.exp(-(((t - 0.72) / 0.13) ** 2))   # delta belt (synthetic)
            + w_gmw * math.exp(-d_gmw / 1200.0)              # Chennai belt (real GMW 2020)
            + 250.0 * (0.5 + 0.5 * _smooth(t, rng))
            + rng.normal(0, 90)
        )
        mw = float(np.clip(mw, *C.BOUNDS["mangrove_width_m"]))
        md = float(np.clip(0.18 + 0.62 * mw / 3000.0 + 0.12 * _smooth(t, rng)
                           + 0.25 * math.exp(-d_gmw / 2000.0) + rng.normal(0, 0.05),
                           *C.BOUNDS["mangrove_density"]))
        dh = float(np.clip(2.6 + 1.9 * _smooth(t + 0.3, rng) + rng.normal(0, 0.5),
                           *C.BOUNDS["dune_height_m"]))
        dw = float(np.clip(120.0 + 80.0 * _smooth(t + 0.6, rng) + rng.normal(0, 18),
                           *C.BOUNDS["dune_width_m"]))
        el = float(np.clip(6.0 + 4.5 * _smooth(t + 0.15, rng) + rng.normal(0, 1.0),
                           *C.BOUNDS["terrain_elev_m"]))
        sl = float(np.clip(1.4 + 0.9 * _smooth(t + 0.45, rng) + rng.normal(0, 0.2),
                           *C.BOUNDS["terrain_slope_deg"]))
        # susceptibility: mostly explained by observable terrain + small residual
        sus = float(np.clip(
            0.80 + 0.22 * (sl - 0.5) / 2.5 + 0.20 * (1.0 - el / 15.0)
            + 0.08 * _smooth(t + 0.8, rng), 0.6, 1.45))
        static[i] = dict(mangrove_width_m=mw, mangrove_density=md, dune_height_m=dh,
                         dune_width_m=dw, terrain_elev_m=el, terrain_slope_deg=sl, sus=sus)

    rows = []
    for i, (lon, lat, t) in enumerate(secs):
        st = static[i]
        for s in C.STORMS:
            d = abs(t - s["landfall_t"])
            decay = math.exp(-d / 0.30)
            wind = float(np.clip(7.0 + (s["wind"] - 7.0) * decay + rng.normal(0, 0.8),
                                 *C.BOUNDS["storm_wind_ms"]))
            pres = float(np.clip(1013.0 - (1013.0 - s["pressure"]) * decay + rng.normal(0, 0.8),
                                 *C.BOUNDS["storm_pressure_hpa"]))
            wh = float(np.clip(0.6 + (s["wave_h"] - 0.6) * decay + rng.normal(0, 0.12),
                               *C.BOUNDS["wave_height_m"]))
            wp = float(np.clip(3.5 + (s["wave_p"] - 3.5) * decay + rng.normal(0, 0.25),
                               *C.BOUNDS["wave_period_s"]))

            raw = {
                "mangrove_width_m": round(st["mangrove_width_m"], 1),
                "mangrove_density": round(st["mangrove_density"], 3),
                "dune_height_m": round(st["dune_height_m"], 2),
                "dune_width_m": round(st["dune_width_m"], 1),
                "storm_wind_ms": round(wind, 2),
                "storm_pressure_hpa": round(pres, 2),
                "wave_height_m": round(wh, 2),
                "wave_period_s": round(wp, 2),
                "terrain_elev_m": round(st["terrain_elev_m"], 2),
                "terrain_slope_deg": round(st["terrain_slope_deg"], 2),
            }
            der = derive_features(raw)
            retreat = 70.0 * der["composite_erosion_idx"] * st["sus"] + rng.normal(0, 1.0)
            retreat = float(np.clip(round(retreat, 1), *C.BOUNDS[C.TARGET]))

            row = {
                "section_id": f"S{i+1:03d}",
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "storm": s["name"],
                "storm_date": s["date"],
                "year": s["year"],
            }
            row.update(raw)
            row.update({k: round(v, 5) for k, v in der.items()})
            row[C.TARGET] = retreat
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- QC + storage
def quality_control(df):
    """Validate the master table; returns (report dict, ok bool)."""
    issues = []
    if df.isnull().any().any():
        issues.append("null values present")
    dup = df.duplicated(subset=["section_id", "storm"]).sum()
    if dup:
        issues.append(f"{dup} duplicate section x storm keys")
    for col, (lo, hi) in C.BOUNDS.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            issues.append(f"{col}: {bad} values out of bounds")
    expected = C.N_SECTIONS * len(C.STORMS)
    if len(df) != expected:
        issues.append(f"row count {len(df)} != expected {expected}")
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "sections": int(df.section_id.nunique()),
        "storms": sorted(df.storm.unique().tolist()),
        "nulls": int(df.isnull().sum().sum()),
        "duplicate_keys": int(dup),
        "checks_passed": not issues,
        "issues": issues,
        "target_stats": {k: round(float(v), 2) for k, v in df[C.TARGET].describe().items()},
    }
    return report, not issues


def store(df):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df.to_csv(C.CSV_PATH, index=False)
    if os.path.exists(C.DB_PATH):
        os.remove(C.DB_PATH)
    with sqlite3.connect(C.DB_PATH) as con:
        df.to_sql("master_dataset", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pk ON master_dataset(section_id, storm)")


def load_master():
    with sqlite3.connect(C.DB_PATH) as con:
        return pd.read_sql("SELECT * FROM master_dataset", con)


def main():
    df = build_dataframe()
    report, ok = quality_control(df)
    store(df)
    os.makedirs(C.OUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUT_DIR, "dataset_qc.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"master dataset: {report['rows']} rows x {report['columns']} cols "
          f"({report['sections']} sections x {len(report['storms'])} storms)")
    print(f"QC passed: {ok}" + ("" if ok else f" -> {report['issues']}"))
    print(f"stored: {C.DB_PATH}\n        {C.CSV_PATH}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
