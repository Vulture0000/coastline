"""Appendix figures for the report (A.2 - A.4) + dashboard screenshot (A.1).

  Figure A.2  study area + Sentinel-2 satellite data view   -> fig_A2_...
  Figure A.3  pre-/post-cyclone shoreline extraction          -> fig_A3_...
  Figure A.4  GIS shoreline change map (Cyclone Gaja)         -> fig_A4_...
  Figure A.1  Streamlit dashboard home screen (headless shot) -> fig_A1_...

Usage:  .venv/bin/python -m coastml.report_figures
Outputs land in outputs/report/.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from . import config as C

OUT = os.path.join(C.OUT_DIR, "report")
GEO = os.path.join(C.ROOT, "out")

FIG_A1 = "Figure_A1_Project_Dashboard.png"
FIG_A2 = "Figure_A2_Study_Area_Satellite_Data.png"
FIG_A3 = "Figure_A3_Pre_Post_Shoreline_Extraction.png"
FIG_A4 = "Figure_A4_GIS_Shoreline_Change_Map.png"

LINEAR_REG = "#1f4e79"     # pre-storm / cool
LINEAR_POST = "#c00000"    # post-storm / warm
MANGROVE = "#2e7d32"
PALETTE = ["#1f4e79", "#c55a11", "#548235", "#7030a0"]


def _utm():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:32644",
                                always_xy=True).transform


def _line_geoms(path):
    """MultiLineString geojson -> list of (N,2) lon/lat arrays."""
    gj = json.load(open(path))
    geom = gj["coordinates"] if gj["type"] == "MultiLineString" else \
        [g["coordinates"] for g in gj["features"][0]["geometry"]["coordinates"]]
    out = []
    for part in geom:
        arr = np.asarray(part, dtype=float)
        if arr.ndim == 2 and len(arr) > 1:
            out.append(arr)
    return out


def _fc(path):
    return json.load(open(path))["features"]


def _polylines(ax, parts, **kw):
    from matplotlib.collections import LineCollection
    if "lws" in kw:
        kw["linewidths"] = kw.pop("lws")
    segs = [p for p in parts if len(p) > 1]
    ax.add_collection(LineCollection(segs, **kw))


def _north(ax, x=0.965, y=0.90):
    ax.annotate("N", xy=(x, y - 0.07), xycoords="axes fraction",
                xytext=(x, y), textcoords="axes fraction",
                ha="center", va="top", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4,
                                shrinkA=0, shrinkB=0))


def _scalebar(ax, km, y_frac=0.07, x_frac=0.05, lat_mid=None, utm=False,
              color="k"):
    """Scale bar in a lon/lat axes (lat_mid degrees) or a UTM axes (metres)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if utm:
        span = km * 1000.0
    else:
        span = km / (111.32 * np.cos(np.deg2rad(lat_mid)))
    x = x0 + x_frac * (x1 - x0)
    y = y0 + y_frac * (y1 - y0)
    ax.plot([x, x + span], [y, y], color=color, lw=3.5,
            solid_capstyle="butt", zorder=6, clip_on=False)
    ax.text(x + span / 2, y + 0.025 * (y1 - y0), f"{km} km", ha="center",
            va="bottom", fontsize=8.5, zorder=6)


def _graticule(ax, step=0.2, lat_step=0.2, fmt="{:.1f}°E",
               fmt_lat="{:.1f}°N", fontsize=8):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step)
    ys = np.arange(np.ceil(y0 / lat_step) * lat_step, y1 + 1e-9, lat_step)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt.format(v) for v in xs], fontsize=fontsize)
    ax.set_yticks(ys)
    ax.set_yticklabels([fmt_lat.format(v) for v in ys], fontsize=fontsize)
    ax.grid(True, ls=":", lw=0.5, color="0.65", zorder=0)


def _sea_label(ax, lon, lat, text, size=11):
    ax.text(lon, lat, text, style="italic", color="#4f83b0", fontsize=size,
            ha="center", va="center", zorder=3)


# --------------------------------------------------------------- Figure A.2
def figure_a2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Rectangle

    master = pd.read_csv(C.CSV_PATH)
    sec = (master.groupby("section_id", as_index=False)
           .first()[["lat", "lon"]])
    gmw = _fc(os.path.join(GEO, "chennai", "mangrove",
                           "mangrove_gmw_2020.geojson"))
    chennai_sections = _fc(os.path.join(GEO, "chennai", "mangrove",
                                        "sections_classified.geojson"))

    pts = np.asarray(C.COAST_CONTROL_POINTS, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], 400)
    coast = np.column_stack([np.interp(t, cum, pts[:, 0]),
                             np.interp(t, cum, pts[:, 1])])

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                          left=0.05, right=0.98, top=0.86, bottom=0.10,
                          wspace=0.18)

    # ---- (a) study area ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(coast[:, 0], coast[:, 1], color="0.35", lw=1.6, ls="--",
            zorder=2, label="study coastline (120 sections)")
    ax.scatter(sec.lon, sec.lat, s=16, color="#2b7bba", zorder=4,
               label="model sections (n = 120)", edgecolor="white", lw=0.4)

    labelled = False
    for geom in gmw:
        for poly in (geom["geometry"]["coordinates"]
                     if geom["geometry"]["type"] == "MultiPolygon"
                     else [geom["geometry"]["coordinates"]]):
            ax.add_patch(MplPolygon(np.asarray(poly[0]), closed=True,
                                    fc=MANGROVE, ec=MANGROVE, alpha=0.75,
                                    lw=0.6, zorder=3,
                                    label=None if labelled
                                    else "GMW 2020 mangrove"))
            labelled = True
    # Gaja extraction AOI
    gaja = np.vstack([p for p in _line_geoms(os.path.join(GEO, "coast_base.geojson"))])
    ax.add_patch(Rectangle((gaja[:, 0].min(), gaja[:, 1].min()),
                           np.ptp(gaja[:, 0]), np.ptp(gaja[:, 1]), fill=False,
                           ec="#e07b00", lw=1.6, ls=(0, (6, 3)), zorder=5,
                           label="Sentinel-2 extraction AOI (Gaja)"))
    # Chennai satellite AOI
    ch = np.vstack([p for p in _line_geoms(
        os.path.join(GEO, "chennai", "coast", "coast_base_2018-09-13.geojson"))])
    ax.add_patch(Rectangle((ch[:, 0].min(), ch[:, 1].min()),
                           np.ptp(ch[:, 0]), np.ptp(ch[:, 1]), fill=False,
                           ec="#7030a0", lw=1.4, zorder=5,
                           label="Sentinel-2 scenes (Chennai AOI)"))

    for s in C.STORMS:
        i = min(int(s["landfall_t"] * (len(coast) - 1)), len(coast) - 1)
        x, y = coast[i]
        ax.scatter([x], [y], marker="v", s=110, color="#c00000",
                   edgecolor="white", lw=0.8, zorder=6)
        ax.annotate(f"{s['name']} {s['year']}\n{s['wind']:.0f} m/s",
                    xy=(x, y), xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=7.6, color="#8b0000",
                    fontweight="bold", zorder=6)

    for i, name, off in ((1, "Chennai", (-4, 4)), (3, "Mahabalipuram", (-4, 4)),
                         (5, "Puducherry", (-4, 4)), (6, "Cuddalore", (-4, 4)),
                         (9, "Nagapattinam", (-4, 4)), (11, "Point Calimere", (4, -6))):
        ax.annotate(name, xy=pts[i], xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left", va="center",
                    fontsize=8, color="0.15")
        ax.scatter(pts[i, 0], pts[i, 1], s=14, color="0.15", zorder=5)

    _sea_label(ax, 80.62, 11.6, "Bay of Bengal", 12)
    ax.set_xlim(79.55, 80.75)
    ax.set_ylim(10.15, 13.55)
    _graticule(ax, step=0.2, lat_step=0.2)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(11.8)))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("(a)  Study area — Coromandel coast, Chennai → Point Calimere",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.6, frameon=True, framealpha=0.92)
    _north(ax, y=0.96)
    _scalebar(ax, 50, lat_mid=11.8, y_frac=0.10, x_frac=0.55)

    # ---- (b) satellite data view --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    epochs = [("base", "2018-09-13", PALETTE[0]),
              ("post", "2018-11-27", PALETTE[1]),
              ("recover", "2019-01-01", PALETTE[2]),
              ("q1", "2019-01-06", PALETTE[3])]
    for k, (key, date, col) in enumerate(epochs):
        parts = _line_geoms(os.path.join(GEO, "chennai", "coast",
                                         f"coast_{key}_{date}.geojson"))
        _polylines(ax, parts, colors=[col], lws=[1.0 if k == 0 else 0.9],
                   alpha=0.95 if k < 2 else 0.7, zorder=3 + k,
                   label=f"{key} — {date}" + (" (low conf.)" if k >= 2 else ""))
    for geom in gmw:
        for poly in (geom["geometry"]["coordinates"]
                     if geom["geometry"]["type"] == "MultiPolygon"
                     else [geom["geometry"]["coordinates"]]):
            ax.add_patch(MplPolygon(np.asarray(poly[0]), closed=True,
                                    fc=MANGROVE, ec=MANGROVE, alpha=0.7,
                                    lw=0.6, zorder=2))
    sx = [f["geometry"]["coordinates"][0] for f in chennai_sections]
    sy = [f["geometry"]["coordinates"][1] for f in chennai_sections]
    ax.scatter(sx, sy, s=10, c="#333333", zorder=5,
               label="extraction sections (n = 81)", edgecolor="white", lw=0.3)

    res = pd.read_csv(os.path.join(GEO, "chennai", "tables",
                                   "scene_resolution.csv"))
    note = ("Sentinel-2 L2A · 10 m\nNDWI > 0 waterline\n"
            "10 m transect step\n"
            + "\n".join(f"{r.Date}: {r.Pct_resolved:.0f}% sections"
                        for r in res.itertuples()))
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7.6,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.7,
                      alpha=0.93))
    ax.set_xlim(ch[:, 0].min() - 0.01, ch[:, 0].max() + 0.01)
    ax.set_ylim(ch[:, 1].min() - 0.01, ch[:, 1].max() + 0.01)
    _graticule(ax, step=0.05, lat_step=0.1, fmt="{:.2f}°E",
               fmt_lat="{:.2f}°N", fontsize=7.5)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(13.2)))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_title("(b)  Satellite data view — Sentinel-2 waterlines, Chennai AOI",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.4, frameon=True, framealpha=0.92)
    _north(ax, x=0.95, y=0.52)
    _scalebar(ax, 10, lat_mid=13.2, y_frac=0.05, x_frac=0.55)

    fig.suptitle("Study area and satellite data view", fontsize=13,
                 fontweight="bold", y=0.965)
    return fig


# --------------------------------------------------------------- Figure A.3
def _section_transect(sec_no):
    """Station point + seaward unit vector + per-epoch land hits (UTM)."""
    feats = [f for f in _fc(os.path.join(GEO, "transects.geojson"))
             if f["properties"]["s"] == sec_no]
    if not feats:
        return None
    xy = np.array([f["geometry"]["coordinates"] for f in feats], dtype=float)
    d = np.array([f["properties"]["d"] for f in feats], dtype=float)
    # xy = station + n * d  ->  least squares for station and normal
    A = np.column_stack([d, np.ones(len(d))])
    nx, px = np.linalg.lstsq(A, xy[:, 0], rcond=None)[0]
    ny, py = np.linalg.lstsq(A, xy[:, 1], rcond=None)[0]
    n = np.array([nx, ny])
    n = n / np.linalg.norm(n)
    station = np.array([px, py])
    hits = {}
    for key in ("base", "post"):
        land = [dd for dd, v in zip(d, (f["properties"].get(key)
                                        for f in feats)) if v == 0]
        hits[key] = max(land) if land else None
    return station, n, d, hits, feats


def figure_a3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_utm = _utm()
    base = _line_geoms(os.path.join(GEO, "coast_base.geojson"))
    post = _line_geoms(os.path.join(GEO, "coast_post.geojson"))
    base_u = [np.column_stack(to_utm(p[:, 0], p[:, 1])) for p in base]
    post_u = [np.column_stack(to_utm(p[:, 0], p[:, 1])) for p in post]
    ret = pd.read_csv(os.path.join(GEO, "shoreline_retreat.csv"))
    ret["retreat"] = -ret.Storm_base_post_m
    worst = ret.dropna(subset=["Storm_base_post_m"]).nsmallest(
        1, "Storm_base_post_m").iloc[0]
    sec_no = int(worst.Section_ID[1:])

    fig = plt.figure(figsize=(15.0, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0],
                          left=0.045, right=0.955, top=0.87, bottom=0.10,
                          wspace=0.16)

    # ---- (a) full extraction AOI --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _polylines(ax, base_u, colors=[LINEAR_REG], lws=[0.7], alpha=0.85,
               zorder=3, label="pre-storm shoreline — 2018-09-08")
    _polylines(ax, post_u, colors=[LINEAR_POST], lws=[0.7], alpha=0.85,
               zorder=4, label="post-storm shoreline — 2018-11-27")
    st = np.array([to_utm(lon, lat) for lon, lat
                   in zip(ret.Longitude, ret.Latitude)])
    ok = ret.retreat.notna().to_numpy()
    ax.scatter(st[ok, 0], st[ok, 1], s=16, c="#666666", zorder=5,
               edgecolor="white", lw=0.4, label="transect sections (n = 200)")
    ax.scatter(st[~ok, 0], st[~ok, 1], s=14, facecolor="none",
               ec="0.55", zorder=5, lw=0.7, label="no change detected")
    allp = np.vstack([st] + [np.asarray(p) for p in base_u + post_u])
    xs, ys = allp[:, 0], allp[:, 1]
    ax.set_xlim(xs.min() - 600, xs.max() + 600)
    ax.set_ylim(ys.min() - 600, ys.max() + 600)
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 44N easting (m)", fontsize=9)
    ax.set_ylabel("UTM 44N northing (m)", fontsize=9)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=8)
    ax.set_title("(a)  Pre- and post-cyclone shoreline extraction — Cyclone Gaja AOI",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.93)
    ax.grid(True, ls=":", lw=0.5, color="0.75")
    _north(ax, y=0.96)
    _scalebar(ax, 10, utm=True, y_frac=0.07, x_frac=0.58)

    # zoom box for (b)
    sx, sy = to_utm(worst.Longitude, worst.Latitude)
    half = 1500.0
    ax.plot([sx - half, sx + half, sx + half, sx - half, sx - half],
            [sy - half, sy - half, sy + half, sy + half, sy - half],
            color="#c00000", lw=1.4, ls="--", zorder=7, alpha=0.9)
    ax.text(sx - half + 120, sy + half - 220, "(b)", fontsize=12,
            fontweight="bold", color="#c00000", ha="left", va="top", zorder=8)

    # ---- (b) zoom: transect-level measurement -------------------------
    ax = fig.add_subplot(gs[0, 1])
    _polylines(ax, base_u, colors=[LINEAR_REG], lws=[1.3], alpha=0.9, zorder=3)
    _polylines(ax, post_u, colors=[LINEAR_POST], lws=[1.3], alpha=0.9, zorder=4)
    out = _section_transect(sec_no)
    station, n, d, hits, feats = out
    p0 = station - n * 160
    p1 = station + n * 170
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.25", lw=1.1, ls=":",
            zorder=6, label="transect normal (500 m spacing)")
    ax.scatter([station[0]], [station[1]], marker="*", s=170, color="#111111",
               zorder=7, label=f"section origin {worst.Section_ID}")
    mark = {}
    for key, col, lab in (("base", LINEAR_REG, "last land sample, pre-storm"),
                          ("post", LINEAR_POST, "last land sample, post-storm")):
        dd = hits[key]
        if dd is None:
            continue
        q = station + n * dd
        ax.scatter([q[0]], [q[1]], marker="s", s=70, color=col, zorder=8,
                   edgecolor="white", lw=0.8,
                   label=lab + f" (d = {dd:+.0f} m)")
        mark[key] = q
    if len(mark) == 2:
        mid = (mark["base"] + mark["post"]) / 2
        ax.annotate("", xy=mark["post"], xytext=mark["base"],
                    arrowprops=dict(arrowstyle="<|-|>", color="#c00000",
                                    lw=1.6, shrinkA=0, shrinkB=0), zorder=9)
        ax.text(mid[0] + 40, mid[1], f"retreat\n{abs(worst.retreat):.0f} m",
                fontsize=9, fontweight="bold", color="#c00000", ha="left",
                va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c00000",
                          lw=0.8, alpha=0.9))
    ax.set_xlim(sx - half, sx + half)
    ax.set_ylim(sy - half, sy + half)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("UTM 44N easting (m)", fontsize=9)
    ax.set_ylabel("UTM 44N northing (m)", fontsize=9)
    ax.grid(True, ls=":", lw=0.5, color="0.75")
    ax.set_title(f"(b)  Transect measurement — {worst.Section_ID} "
                 f"(largest retreat in AOI)", loc="left", fontsize=10.5,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=0.93)
    _scalebar(ax, 1, utm=True, y_frac=0.06, x_frac=0.04)

    fig.suptitle("Pre- and post-cyclone shoreline extraction (Sentinel-2, "
                 "NDWI waterline)", fontsize=13, fontweight="bold", y=0.965)
    return fig


# --------------------------------------------------------------- Figure A.4
def figure_a4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    base = _line_geoms(os.path.join(GEO, "coast_base.geojson"))
    post = _line_geoms(os.path.join(GEO, "coast_post.geojson"))
    df = pd.read_csv(os.path.join(GEO, "shoreline_retreat.csv"))
    df["retreat"] = -df.Storm_base_post_m

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    _polylines(ax, base, colors=[LINEAR_REG], lws=[0.6], alpha=0.55, zorder=2)
    _polylines(ax, post, colors=[LINEAR_POST], lws=[0.6], alpha=0.55, zorder=3)

    ok = df.retreat.notna()
    vmin, vmax = float(df.retreat.min()), float(df.retreat.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sc = ax.scatter(df.loc[ok, "Longitude"], df.loc[ok, "Latitude"],
                    c=df.loc[ok, "retreat"], cmap="RdBu_r", norm=norm,
                    s=72, edgecolor="0.25", lw=0.6, zorder=6)
    ax.scatter(df.loc[~ok, "Longitude"], df.loc[~ok, "Latitude"], s=34,
               facecolor="none", ec="0.55", lw=0.9, zorder=5,
               label="section without a usable waterline pair")

    for _, r in df.loc[ok].nlargest(3, "retreat").iterrows():
        ax.annotate(f"{r.Section_ID}\n{r.retreat:.0f} m",
                    xy=(r.Longitude, r.Latitude), xytext=(8, 6),
                    textcoords="offset points", fontsize=7.8,
                    fontweight="bold", color="#7b0000", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#c00000", lw=0.6, alpha=0.85))

    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label("Shoreline retreat, pre → post Cyclone Gaja (m; "
                 "positive = landward)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xlim(79.36, 79.93)
    ax.set_ylim(10.23, 10.79)
    _graticule(ax, step=0.1, lat_step=0.1, fmt="{:.1f}°E",
               fmt_lat="{:.1f}°N", fontsize=8)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(10.5)))
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("GIS shoreline change map — 200 transect sections, "
                 "Cyclone Gaja (16 Nov 2018)", loc="left", fontsize=11,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.93)
    _north(ax, y=0.96)
    _scalebar(ax, 10, lat_mid=10.5, y_frac=0.06, x_frac=0.56)
    return fig


# ----------------------------------------------------------- Figure A.1
_READY = ("document.body.innerText.includes('Coastal Erosion ML Pipeline') "
          "&& document.querySelectorAll('.js-plotly-plot').length > 0")


async def _cdp_screenshot(chrome, url, out, width, height, scale, timeout):
    """Drive headless Chrome over the DevTools protocol until the page is
    fully painted (WebGL enabled so the MapLibre base map renders), then save
    a PNG of the viewport."""
    import asyncio
    import base64
    import shutil
    import subprocess
    import tempfile
    import urllib.parse
    import urllib.request

    import websockets

    port = 9300 + (os.getpid() % 400)
    prof = tempfile.mkdtemp(prefix="cdp-profile-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--no-sandbox", "--no-first-run", "--no-default-browser-check",
         "--enable-unsafe-swiftshader", "--use-gl=angle",
         "--use-angle=swiftshader",
         f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                    json.load(r)
                    break
            except Exception:
                time.sleep(0.5)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/new?"
            f"{urllib.parse.quote(url, safe=':/?=&')}", method="PUT")
        with urllib.request.urlopen(req, timeout=5) as r:
            target = json.load(r)

        async with websockets.connect(target["webSocketDebuggerUrl"],
                                      max_size=200 * 1024 * 1024,
                                      open_timeout=15) as ws:
            mid = [0]

            async def send(method, params=None):
                mid[0] += 1
                i = mid[0]
                await ws.send(json.dumps({"id": i, "method": method,
                                          "params": params or {}}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(),
                                                            timeout=30))
                    if msg.get("id") == i:
                        return msg

            await send("Page.enable")
            await send("Runtime.enable")
            await send("Emulation.setDeviceMetricsOverride",
                       {"width": width, "height": height,
                        "deviceScaleFactor": scale, "mobile": False})
            await send("Page.navigate", {"url": url})
            t0 = time.time()
            while time.time() - t0 < timeout:
                await asyncio.sleep(2)
                res = await send("Runtime.evaluate",
                                 {"expression": _READY, "returnByValue": True})
                if (res.get("result") or {}).get("result", {}).get("value"):
                    break
            time.sleep(12)      # plotly paint + OSM tiles over software WebGL
            shot = await send("Page.captureScreenshot",
                              {"format": "png", "fromSurface": True})
            data = (shot.get("result") or {}).get("data")
            if not data:
                raise RuntimeError(f"capture failed: {shot.get('error')}")
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(data))
            return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)


def screenshot_dashboard(port=8587, width=1760, height=1150, timeout=120):
    """Launch Streamlit (light theme), screenshot the home screen with
    headless Chrome over the DevTools protocol. Best effort - skips with a
    message if Chrome is missing."""
    import shutil
    import subprocess
    import urllib.request

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        print("  (no Chrome found - dashboard screenshot skipped)")
        return None
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, FIG_A1)

    env = dict(os.environ, STREAMLIT_THEME_BASE="light")
    proc = subprocess.Popen(
        [os.path.join(C.ROOT, ".venv", "bin", "python"), "-m", "streamlit",
         "run", os.path.join(C.ROOT, "coastml", "app.py"),
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        url = f"http://localhost:{port}"
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(url + "/healthz", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1.5)
        else:
            print("  (streamlit did not become ready - screenshot skipped)")
            return None
        import asyncio
        asyncio.run(_cdp_screenshot(chrome, url, out, width, height, 2,
                                    timeout=90))
        print("figure ->", out)
        return out
    except Exception as e:      # screenshots are best-effort
        print(f"  (dashboard screenshot failed: {e})")
        return None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib.pyplot as plt

    screenshot_dashboard()
    builders = [("A.2", figure_a2, FIG_A2),
                ("A.3", figure_a3, FIG_A3),
                ("A.4", figure_a4, FIG_A4)]
    for tag, fn, name in builders:
        fig = fn()
        path = os.path.join(OUT, name)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Figure {tag} -> {path}")


if __name__ == "__main__":
    main()
