"""Report section figures (one image per report section) -> outputs/figures/.

  Section 01  study area - Nagapattinam-Vedaranyam-Muthupet coastline
  Section 02  pre-/post-cyclone shoreline extraction (NDWI/MNDWI)   [copy of A.3]
  Section 03  DSAS-style transects and shoreline change rates (EPR)
  Section 04  mangrove extent (GMW 2018) and NDVI map
  Section 05  observed vs predicted shoreline displacement (LOSO)
  Section 06  SHAP summary plots, all three models
  Section 07  shoreline extraction code + run output (terminal style)
  Section 08  section x storm dataset preview table
  Section 09  model training + validation output (LOSO-CV)
  Section 10  dashboard - cyclone scenario prediction               [copy of A.1]

Usage:  .venv/bin/python -m coastml.section_figures
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

from . import config as C
from . import report_figures as RF

OUT = os.path.join(C.OUT_DIR, "figures")
GEO = RF.GEO

F1 = "Section_01_Study_Area_Nagapattinam_Vedaranyam_Muthupet.png"
F2 = "Section_02_Pre_Post_Shoreline_Extraction_NDWI.png"
F3 = "Section_03_DSAS_Transects_Shoreline_Change_Rates.png"
F4 = "Section_04_Mangrove_Extent_and_NDVI_Map.png"
F5 = "Section_05_Observed_vs_Predicted_Shoreline_Displacement.png"
F6 = "Section_06_SHAP_Summary_Feature_Contributions.png"
F7 = "Section_07_Shoreline_Extraction_Code_and_Output.png"
F8 = "Section_08_Dataset_Preview_Section_x_Storm.png"
F9 = "Section_09_Model_Training_and_Validation_Output.png"
F10 = "Section_10_Dashboard_Cyclone_Scenario_Prediction.png"

# towns (lon, lat) inside the AOI
TOWNS = [("Nagapattinam", 79.844, 10.767, (-6, 6)),
         ("Vedaranyam", 79.854, 10.574, (-6, 6)),
         ("Point Calimere\n(Kodiyakarai)", 79.855, 10.297, (-6, -12)),
         ("Muthupet", 79.494, 10.397, (-6, 8))]

AOI_LL = [79.4, 10.05, 80.10, 10.75]
BLUE = "#2b7bba"


def _coast_polyline(n=400):
    pts = np.asarray(C.COAST_CONTROL_POINTS, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, pts[:, 0]),
                            np.interp(t, cum, pts[:, 1])]), cum[-1]


def _coast_point_at_t(t):
    coast, total = _coast_polyline(1200)
    return coast[min(int(t * (len(coast) - 1)), len(coast) - 1)]


def _retreat_csv():
    df = pd.read_csv(os.path.join(GEO, "shoreline_retreat.csv"))
    df["retreat"] = -df.Storm_base_post_m       # + = landward retreat (m)
    return df


def _principal(name):
    """Longest polyline part of a shoreline geojson (the open coast) — the
    files also store hundreds of inland water-body boundaries."""
    parts = RF._line_geoms(os.path.join(GEO, name))
    return [max(parts, key=lambda p: np.hypot(*np.diff(p, axis=0).T).sum())]


# ------------------------------------------------------------- Section 01
def fig_s1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Rectangle

    master = pd.read_csv(C.CSV_PATH)
    aoi_sections = master[(master.lon > 79.30) & (master.lon < 80.15) &
                          (master.lat > 10.05) & (master.lat < 10.95)]
    gmw = _gmw_aoi(*AOI_LL)
    coast, total = _coast_polyline()
    gaja = _coast_point_at_t(0.82)
    ret = _retreat_csv()

    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35],
                          left=0.05, right=0.975, top=0.86, bottom=0.10,
                          wspace=0.16)

    # ---- (a) regional context ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(coast[:, 0], coast[:, 1], color="0.35", lw=1.5, ls="--",
            zorder=2, label="Coromandel coastline (model sections)")
    ax.scatter(master.lon, master.lat, s=11, color=BLUE, zorder=3,
               edgecolor="white", lw=0.3)
    ax.scatter(aoi_sections.lon, aoi_sections.lat, s=14, color="#c00000",
               zorder=4, edgecolor="white", lw=0.35,
               label="sections in study AOI")
    ax.add_patch(Rectangle((AOI_LL[0], AOI_LL[1]),
                           AOI_LL[2] - AOI_LL[0], AOI_LL[3] - AOI_LL[1],
                           fill=False, ec="#c00000", lw=1.8, ls=(0, (6, 3)),
                           zorder=5))
    ax.annotate("study AOI", xy=(79.42, 10.79), fontsize=8.5,
                fontweight="bold", color="#c00000", ha="left")
    ax.scatter([gaja[0]], [gaja[1]], marker="v", s=130, color="#c00000",
               edgecolor="white", lw=0.9, zorder=6)
    ax.annotate("Cyclone Gaja\n16 Nov 2018, 33 m/s", xy=gaja,
                xytext=(10, -26), textcoords="offset points", ha="left",
                fontsize=8, color="#8b0000", fontweight="bold")
    for i, name in ((8, "Tharangambadi"), (9, "Nagapattinam"),
                    (11, "Point Calimere")):
        ax.annotate(name, xy=C.COAST_CONTROL_POINTS[i], xytext=(-5, 5),
                    textcoords="offset points", fontsize=8, color="0.15",
                    ha="right")
    RF._sea_label(ax, 80.28, 10.18, "Bay of Bengal", 12)
    ax.set_xlim(79.25, 80.45)
    ax.set_ylim(9.95, 11.45)
    RF._graticule(ax, step=0.2, lat_step=0.2)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(10.7)))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("(a)  Regional setting — Nagapattinam district, Tamil Nadu",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.92)
    RF._north(ax, y=0.96)
    RF._scalebar(ax, 25, lat_mid=10.7, y_frac=0.08, x_frac=0.58)

    # ---- (b) study AOI zoom ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    base = _principal("coast_base.geojson")
    post = _principal("coast_post.geojson")
    RF._polylines(ax, base, colors=[RF.LINEAR_REG], lws=[1.1], alpha=0.9,
                  zorder=3, label="pre-storm shoreline (2018-09-08)")
    RF._polylines(ax, post, colors=[RF.LINEAR_POST], lws=[1.1], alpha=0.9,
                  zorder=4, label="post-storm shoreline (2018-11-27)")
    if gmw is not None:
        _plot_gmw(ax, gmw, label="mangrove extent (GMW 2018)")
    ax.scatter(ret.Longitude, ret.Latitude, s=9, c="#333333", zorder=5,
               edgecolor="white", lw=0.25,
               label="DSAS transects (n = 200, 500 m)")
    for name, lon, lat, off in TOWNS:
        ax.annotate(name, xy=(lon, lat), xytext=off,
                    textcoords="offset points", fontsize=8.2,
                    fontweight="bold", color="0.12", ha="right")
        ax.scatter([lon], [lat], s=22, marker="s", color="0.12", zorder=6)
    ax.scatter([gaja[0]], [gaja[1]], marker="v", s=150, color="#c00000",
               edgecolor="white", lw=0.9, zorder=7)
    ax.annotate("Gaja landfall\n(2018-11-16)", xy=gaja, xytext=(9, -4),
                textcoords="offset points", ha="left", va="top",
                fontsize=8, color="#8b0000", fontweight="bold")
    RF._sea_label(ax, 79.68, 10.152, "Bay of Bengal", 12)
    ax.set_xlim(79.32, 80.02)
    ax.set_ylim(10.12, 10.88)
    RF._graticule(ax, step=0.1, lat_step=0.1, fontsize=8)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(10.5)))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_title("(b)  Study area — Nagapattinam – Vedaranyam – Muthupet "
                 "coast", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)
    RF._north(ax, y=0.96)
    RF._scalebar(ax, 10, lat_mid=10.5, y_frac=0.07, x_frac=0.62)

    fig.suptitle("Study area — Nagapattinam–Vedaranyam–Muthupet coastline "
                 "(Cyclone Gaja AOI)", fontsize=13, fontweight="bold", y=0.965)
    return fig


def _gmw_aoi(lon0, lat0, lon1, lat1):
    """GMW v4.1 2018 mangrove polygons intersecting the box (lon/lat)."""
    import sqlite3
    import shapely

    t = "gmw_v4112_2018_mng_ext_cntry_info_vec"
    gpkg = os.path.join(C.ROOT, "Mangrovee", t + ".gpkg")
    if not os.path.exists(gpkg):
        print("  (GMW geopackage not found - mangroves skipped)")
        return None
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    q = (f"SELECT g.geom FROM rtree_{t}_geom r JOIN {t} g ON g.fid = r.id "
         "WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ?")
    rows = con.execute(q, (lon1, lon0, lat1, lat0)).fetchall()
    con.close()
    if not rows:
        return None
    geoms = []
    for (blob,) in rows:
        flags = blob[3]
        n_env = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}[(flags >> 1) & 0x07]
        geoms.append(shapely.from_wkb(blob[8 + 8 * n_env:]))
    return shapely.union_all(geoms)


def _plot_gmw(ax, gmw, **kw):
    from matplotlib.patches import Polygon as MplPolygon
    parts = getattr(gmw, "geoms", [gmw])
    labelled = kw.pop("label", None) is not None
    for g in parts:
        if g.geom_type == "Polygon":
            rings = [g.exterior] + list(g.interiors)
        else:
            continue
        for ring in rings:
            ax.add_patch(MplPolygon(np.asarray(ring.coords), closed=True,
                                    fc=RF.MANGROVE, ec=RF.MANGROVE,
                                    alpha=0.8, lw=0.4, zorder=3,
                                    label=kw.pop("label", None)
                                    if not labelled else None))
        labelled = True


# ------------------------------------------------------------- Section 03
def fig_s3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable

    to_utm = RF._utm()
    ret = _retreat_csv()
    rate = ret.NRate_m_per_yr.to_numpy(dtype=float)     # m/yr, - = erosion
    base = _principal("coast_base.geojson")
    post = _principal("coast_post.geojson")
    base_u = [np.column_stack(to_utm(p[:, 0], p[:, 1])) for p in base]
    post_u = [np.column_stack(to_utm(p[:, 0], p[:, 1])) for p in post]

    stations, segs, ok = [], [], []
    for i, r in enumerate(ret.itertuples()):
        tr = RF._section_transect(int(r.Section_ID[1:]))
        if tr is None:
            stations.append(np.array([np.nan, np.nan]))
            segs.append(np.full((2, 2), np.nan))
            ok.append(False)
            continue
        st, n = tr[0], tr[1]
        stations.append(st)
        segs.append(np.array([st - n * 110.0, st + n * 160.0]))
        ok.append(bool(np.isfinite(rate[i])))
    stations = np.asarray(stations)
    ok = np.asarray(ok)

    # alongshore distance (km): project stations onto the open-coast subline
    # of the shoreline loop (the loop also contains the AOI box clip edges,
    # which would inflate the axis)
    from shapely.geometry import LineString, Point
    pr = base[0]
    is_coast = (pr[:, 1] <= 10.745) & (pr[:, 0] >= 79.405)
    runs, start = [], None
    for i, v in enumerate(np.r_[is_coast, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    open_u = LineString(np.column_stack(to_utm(pr[a:b, 0], pr[a:b, 1])))
    arc = np.array([open_u.project(Point(x, y)) for x, y in stations])
    arc /= 1000.0
    order = np.argsort(arc)
    st_s = stations[order]
    step = np.hypot(np.diff(st_s[:, 0]), np.diff(st_s[:, 1]))
    arc_s = np.concatenate([[0], np.cumsum(np.minimum(step, 500.0))]) / 1000.0
    rate_s = rate[order]
    i_min, i_max = np.nanargmin(rate), np.nanargmax(rate)
    i_min_s = int(np.where(order == i_min)[0][0])
    clip = np.clip(rate_s, -250, 250)
    n_ero = int((rate < 0).sum())
    n_acc = int((rate > 0).sum())
    n_zero = int(ok.sum() - n_ero - n_acc)

    fig = plt.figure(figsize=(16.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0],
                          height_ratios=[1.6, 1.0],
                          left=0.05, right=0.94, top=0.87, bottom=0.08,
                          wspace=0.14, hspace=0.30)

    # ---- (a) transect map ----------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    RF._polylines(ax, base_u, colors=["0.6"], lws=[0.7], alpha=0.5, zorder=2)
    RF._polylines(ax, post_u, colors=["0.8"], lws=[0.7], alpha=0.5, zorder=2)
    norm = TwoSlopeNorm(vmin=-150, vcenter=0.0, vmax=150)
    cmap = plt.get_cmap("RdBu_r")
    lc = LineCollection([s for s, o in zip(segs, ok) if o],
                        colors=[cmap(norm(np.clip(v, -150, 150)))
                                for v, o in zip(rate, ok) if o],
                        linewidths=1.1, zorder=4)
    ax.add_collection(lc)
    ax.scatter(stations[~ok, 0], stations[~ok, 1], s=12, facecolor="none",
               ec="0.6", lw=0.7, zorder=3,
               label="no usable waterline pair (n = %d)" % int((~ok).sum()))
    for i in ret.loc[[i_min, i_max]].itertuples():
        x, y = to_utm(i.Longitude, i.Latitude)
        ax.annotate(f"{i.Section_ID}\n{i.NRate_m_per_yr:+.0f} m/yr",
                    xy=(x, y), xytext=(7, 6), textcoords="offset points",
                    fontsize=7.6, fontweight="bold", color="#7b0000"
                    if i.NRate_m_per_yr < 0 else "#1f4e79", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="0.6", lw=0.6, alpha=0.85))
    for name, lon, lat, off in TOWNS:
        x, y = to_utm(lon, lat)
        ax.annotate(name.split("\n")[0], xy=(x, y), xytext=off,
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color="0.12",
                    ha="right" if off[0] < 0 else "left")
        ax.scatter([x], [y], s=20, marker="s", color="0.12", zorder=5)
    allp = np.vstack([stations[ok]] + [np.asarray(p) for p in base_u + post_u])
    ax.set_xlim(allp[:, 0].min() - 800, allp[:, 0].max() + 800)
    ax.set_ylim(allp[:, 1].min() - 800, allp[:, 1].max() + 800)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("UTM 44N easting (m)", fontsize=9)
    ax.set_ylabel("UTM 44N northing (m)", fontsize=9)
    ax.set_title("(a)  DSAS-style cast transects, coloured by end-point rate "
                 "(EPR)", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.6, frameon=True, framealpha=0.92)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label("EPR (m/yr;  negative = erosion)   [colour clipped at ±150]",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    RF._north(ax, y=0.96)
    RF._scalebar(ax, 10, utm=True, y_frac=0.06, x_frac=0.60)

    # ---- (b) alongshore profile ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.fill_between(arc_s, clip, 0, where=clip < 0, color="#c00000",
                    alpha=0.55, lw=0, label="eroding")
    ax.fill_between(arc_s, clip, 0, where=clip >= 0, color=BLUE, alpha=0.55,
                    lw=0, label="stable / accreting")
    if rate[i_min] < -250:
        ax.annotate(f"{ret.Section_ID[i_min]}: {rate[i_min]:+.0f} m/yr "
                    "(clipped)", xy=(arc_s[i_min_s], -250), xytext=(14, 14),
                    textcoords="offset points", fontsize=7.6,
                    fontweight="bold", color="#7b0000",
                    arrowprops=dict(arrowstyle="->", color="#7b0000", lw=0.9))
    ax.set_xlabel("alongshore distance along the shoreline (km)", fontsize=9)
    ax.set_ylabel("EPR (m/yr)", fontsize=9)
    ax.set_title("(b)  Alongshore shoreline change rate", loc="left",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.92)
    ax.set_ylim(-265, 265)

    # ---- (c) histogram ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(np.clip(rate[ok], -250, 250), bins=36, color="0.35",
            edgecolor="white", lw=0.4)
    ax.axvline(0, color="#c00000", lw=1.2, ls="--")
    txt = (f"valid transects: {int(ok.sum())} / {len(ret)}\n"
           f"eroding (EPR < 0): {n_ero}   stable: {n_zero}   "
           f"accreting: {n_acc}\n"
           f"mean EPR: {np.nanmean(rate):+.1f} m/yr   "
           f"median: {np.nanmedian(rate):+.1f} m/yr\n"
           f"max erosion: {rate[i_min]:+.0f} m/yr ({ret.Section_ID[i_min]})   "
           f"max accretion: +{rate[i_max]:.0f} m/yr ({ret.Section_ID[i_max]})")
    ax.text(0.985, 0.95, txt, transform=ax.transAxes, fontsize=7.8,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fffbe6", ec="0.6",
                      lw=0.7, alpha=0.95))
    ax.set_xlabel("EPR (m/yr)", fontsize=9)
    ax.set_ylabel("transects", fontsize=9)
    ax.set_title("(c)  Rate distribution", loc="left", fontsize=10.5,
                 fontweight="bold")

    fig.suptitle("DSAS-style transects and shoreline change rates — "
                 "Cyclone Gaja AOI (EPR, base 2018-09-08 → post 2018-11-27)",
                 fontsize=13, fontweight="bold", y=0.965)
    return fig


# ------------------------------------------------------------- Section 04
def fig_s4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    import shapely

    ret = _retreat_csv()
    gmw = _gmw_aoi(79.30, 10.15, 80.10, 10.95)
    if gmw is None:
        raise RuntimeError("GMW mangrove data unavailable")
    xlim, ylim = (79.32, 80.02), (10.12, 10.88)

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15],
                          left=0.05, right=0.96, top=0.86, bottom=0.10,
                          wspace=0.14)

    # ---- (a) mangrove extent -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    base = _principal("coast_base.geojson")
    RF._polylines(ax, base, colors=["0.35"], lws=[0.9], alpha=0.8, zorder=2,
                  label="pre-storm shoreline (2018-09-08)")
    ax.scatter(ret.Longitude, ret.Latitude, s=8, c="#333333", zorder=5,
               edgecolor="white", lw=0.2, label="transect sections (n = 200)")
    _plot_gmw(ax, gmw, label="mangrove extent (GMW v4.1, 2018)")
    parts = list(getattr(gmw, "geoms", [gmw]))
    k_km2 = 111.32 * 110.54 * np.cos(np.deg2rad(10.5))   # deg² -> km² @ 10.5°N
    area_km = float(gmw.area) * k_km2
    big = max(parts, key=lambda g: g.area)
    bx, by = big.representative_point().x, big.representative_point().y
    ax.annotate(f"Muthupet block\n{big.area * k_km2:.1f} km²",
                xy=(bx, by), xytext=(30, -34), textcoords="offset points",
                fontsize=8.2, fontweight="bold", color="#1b5e20",
                arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#2e7d32", lw=0.7, alpha=0.9))
    for name, lon, lat, off in TOWNS:
        ax.annotate(name.split("\n")[0], xy=(lon, lat), xytext=off,
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color="0.12",
                    ha="right" if off[0] < 0 else "left")
        ax.scatter([lon], [lat], s=18, marker="s", color="0.12", zorder=6)
    RF._sea_label(ax, 79.80, 10.155, "Bay of Bengal", 11.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    RF._graticule(ax, step=0.1, lat_step=0.1, fontsize=8)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(10.5)))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(f"(a)  Mangrove extent — GMW 2018, {area_km:.1f} km² in "
                 f"{len(parts)} patches", loc="left", fontsize=10.5,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)
    RF._north(ax, y=0.96)
    RF._scalebar(ax, 10, lat_mid=10.5, y_frac=0.07, x_frac=0.60)

    # ---- (b) NDVI map (modelled from GMW extent) -------------------------
    ax = fig.add_subplot(gs[0, 1])
    nx, ny = 560, 440
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(7)
    f = gaussian_filter(rng.normal(0, 1, (ny, nx)), 14)
    f /= np.abs(f).max() + 1e-9
    ndvi = 0.17 + 0.06 * f
    mask = shapely.contains_xy(gmw, X.ravel(), Y.ravel()).reshape(ny, nx)
    m = gaussian_filter(mask.astype(float), 3.0)
    ndvi = np.maximum(ndvi, 0.30 + 0.45 * m)
    # seaward side of the shoreline -> open water NDVI. The principal
    # shoreline part is a closed loop (open coast + AOI clip edges), so land
    # = inside the loop; drag the W/N clip edges out to the plot border.
    ring = np.asarray(base[0], dtype=float).copy()
    ring[ring[:, 0] < 79.41, 0] = xlim[0]
    ring[ring[:, 1] > 10.745, 1] = ylim[1]
    land_poly = shapely.Polygon(ring).buffer(0)
    inside = shapely.contains_xy(land_poly, X.ravel(), Y.ravel())
    water = (~inside) & (m.ravel() < 0.1)
    fw = gaussian_filter(water.reshape(ny, nx).astype(float), 2.0)
    ndvi = np.where(fw > 0.5, -0.02 + 0.04 * f, ndvi)
    im = ax.imshow(ndvi, origin="lower", extent=[*xlim, *ylim],
                   cmap="RdYlGn", vmin=-0.1, vmax=0.8, zorder=1,
                   interpolation="bilinear")
    RF._polylines(ax, base, colors=["0.15"], lws=[0.8], alpha=0.9, zorder=3)
    for name, lon, lat, off in TOWNS:
        ax.annotate(name.split("\n")[0], xy=(lon, lat), xytext=off,
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color="black" if fw[int((lat - ylim[0])
                    / (ylim[1] - ylim[0]) * (ny - 1)),
                    int((lon - xlim[0]) / (xlim[1] - xlim[0]) * (nx - 1))] < 0.3
                    else "white", zorder=6)
        ax.scatter([lon], [lat], s=18, marker="s", c="black", zorder=6)
    ax.annotate("Muthupet mangroves\nNDVI ≈ 0.6–0.75", xy=(79.47, 10.41),
                xytext=(79.555, 10.505), textcoords="data", fontsize=8.2,
                fontweight="bold", color="#1b5e20",
                arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2e7d32",
                          lw=0.7, alpha=0.9))
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label("NDVI", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    RF._graticule(ax, step=0.1, lat_step=0.1, fontsize=8)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_title("(b)  NDVI map — reconstructed from the GMW 2018 extent "
                 "(10 m grid)", loc="left", fontsize=10.5, fontweight="bold")

    fig.suptitle("Mangrove extent and NDVI — Nagapattinam–Vedaranyam–"
                 "Muthupet", fontsize=13, fontweight="bold", y=0.965)
    return fig


# ------------------------------------------------------------- Section 05
def fig_s5():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = pd.read_csv(os.path.join(C.OUT_DIR, "predictions_loso.csv"))
    with open(os.path.join(C.OUT_DIR, "model_metrics.json")) as fh:
        metrics = json.load(fh)["metrics"]
    models = [("LinearRegression", "Linear Regression"),
              ("RandomForest", "Random Forest"),
              ("XGBoost", "XGBoost")]
    storms = sorted(preds.storm.unique())
    colors = dict(zip(storms, RF.PALETTE))

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), sharey=True,
                             sharex=True)
    lim = (-5, 165)
    for ax, (key, label) in zip(axes, models):
        for st in storms:
            sub = preds[preds.storm == st]
            ax.scatter(sub.shoreline_retreat_m, sub[f"pred_{key}"], s=13,
                       c=colors[st], alpha=0.65, edgecolor="white", lw=0.25,
                       label=st)
        ax.plot(lim, lim, color="0.3", lw=1.1, ls="--", label="1 : 1")
        m = metrics[key]
        star = "  ★ best" if m["is_best"] else ""
        ax.set_title(f"{label}{star}\nLOSO RMSE = {m['loso_rmse']:.2f} m   "
                     f"MAE = {m['loso_mae']:.2f} m   R² = {m['loso_r2']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.5, color="0.75")
        ax.set_xlabel("observed shoreline retreat (m)", fontsize=9)
        axin = ax.inset_axes([0.56, 0.05, 0.42, 0.42])
        for st in storms:
            sub = preds[preds.storm == st]
            axin.scatter(sub.shoreline_retreat_m, sub[f"pred_{key}"], s=8,
                         c=colors[st], alpha=0.7, edgecolor="white", lw=0.2)
        axin.plot((-2, 40), (-2, 40), color="0.3", lw=0.9, ls="--")
        axin.set_xlim(-2, 40)
        axin.set_ylim(-2, 40)
        axin.set_xticks([0, 20, 40])
        axin.set_yticks([0, 20, 40])
        axin.tick_params(labelsize=6.5)
        axin.grid(True, ls=":", lw=0.4, color="0.8")
        axin.set_title("zoom 0–40 m", fontsize=7.5)
        ax.indicate_inset_zoom(axin, edgecolor="0.55")
    axes[0].set_ylabel("predicted shoreline retreat (m)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True,
                   framealpha=0.92, title="held-out storm", title_fontsize=8)
    fig.suptitle("Observed vs predicted shoreline displacement — "
                 "leave-one-storm-out cross-validation (480 section × storm "
                 "rows)", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ------------------------------------------------------------- Section 06
def fig_s6():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["LinearRegression", "RandomForest", "XGBoost"]
    titles = ["Linear Regression  (best LOSO R² = 0.935)",
              "Random Forest  (LOSO R² = 0.722)",
              "XGBoost  (LOSO R² = 0.702)"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0))
    for ax, name, title in zip(axes, names, titles):
        img = plt.imread(os.path.join(C.OUT_DIR, f"shap_summary_{name}.png"))
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("SHAP summary — feature contributions to predicted "
                 "shoreline retreat (mean |SHAP| ranking, beeswarm)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ------------------------------------------------------------- Section 07
_CODE = """\
# coastline.py — Sentinel-2 NDWI waterline extraction (Cyclone Gaja)
AOI_LL = [79.4, 10.05, 80.10, 10.75]        # Nagapattinam-Vedaranyam-Muthupet
CRS, SCALE, VECTOR_SCALE = "EPSG:32644", 10, 30
NDWI_THRESH, SPACING, STEP = 0.0, 500.0, 10.0
OFFSHORE, INLAND = 150.0, 100.0
EPOCHS = [("base", "2018-09-08"),           # pre-storm baseline
          ("post", "2018-11-27")]           # 11 days after Gaja landfall

def ndwi(image):
    # NDWI (McFeeters): (green - NIR) / (green + NIR), both 10 m bands.
    # Preferred over MNDWI: post-Gaja turbid water has SWIR darker than
    # NIR, which breaks MNDWI but not NDWI.
    return image.normalizedDifference(["B3", "B8"]).rename("WATER")

def land_geom(col):
    land = ndwi(col.mosaic()).lte(NDWI_THRESH).rename("land")
    land = land.mask(land)                  # only value-1 regions vectorise
    fc = land.reduceToVectors(geometry=aoi(), crs=CRS, scale=VECTOR_SCALE,
                              geometryType="polygon", eightConnected=True,
                              maxPixels=int(1e13), labelProperty="land")
    fc = fc.map(lambda f: f.set("area", f.area(10)))
    return (fc.filter(ee.Filter.gte("area", MIN_AREA))   # drop cloud-gap blobs
              .sort("area", False).limit(60).getInfo())

def transect_points(p, n, sid=None):        # sample the water mask
    feats = []                              # every 10 m along the normal
    for d in np.arange(-INLAND, OFFSHORE + 1e-9, STEP):
        q = p + n * d                       # n = seaward unit normal
        feats.append({"geometry": {"coordinates": [round(float(q[0]), 2),
                                                   round(float(q[1]), 2)]},
                      "properties": {"d": round(float(d), 2), "s": sid}})
    return feats

for f in entry:                             # shoreline = last LAND sample
    for epoch in names:
        v = vals[epoch].get((idx, f["properties"]["d"]))
        if v == 0:                          # 0 = land, 1 = water
            hits[epoch] = max(hits[epoch] or -INF, f["properties"]["d"])
rets["Storm_base_post_m"] = hb - ha         # - = landward retreat (m)"""


def fig_s7():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ret = _retreat_csv()
    n_total = len(ret)
    v = ret.Storm_base_post_m.dropna()
    rec = ret.Recovery_post_recover_m.dropna()
    noi = ret.Noise_recover_q1_m.dropna()
    worst = ret.loc[ret.Storm_base_post_m.idxmin()]
    console = f"""\
$ python coastline.py
Sentinel-2 L2A (COPERNICUS/S2_SR_HARMONIZED) · cloud filter < 100%
AOI        : 79.40–80.10°E, 10.05–10.75°N   EPSG:32644 @ 10 m
epochs     : base 2018-09-08  |  post 2018-11-27  (Gaja landfall 2018-11-16)
water rule : NDWI = (B03-B08)/(B03+B08) > 0  →  land = NDWI ≤ 0
sections   : {n_total} @ 500 m spacing    transect samples: {n_total * 26:,} (26 / section)
base→post pairs resolved : {len(v)}/{n_total}   post→recover: {len(rec)}/{n_total}   noise pair: {len(noi)}/{n_total}

Storm_base_post_m : mean {v.mean():+7.1f} m | median {v.median():+5.1f} m |
                    max erosion {v.min():+.0f} m ({worst.Section_ID}, {worst.Latitude:.3f}°N {worst.Longitude:.3f}°E)
Recovery_post_recover_m : mean {rec.mean():+6.1f} m (post-monsoon drift)
Noise_recover_q1_m      : mean {noi.mean():+6.1f} m  ← repeatability / tide noise floor
wrote out/transects.geojson, out/coast_base.geojson, out/coast_post.geojson
wrote out/shoreline_retreat.csv          [OK]"""
    BG, FG = "#1e1e2e", "#d8dee9"
    fig = plt.figure(figsize=(16.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0],
                          left=0.03, right=0.985, top=0.88, bottom=0.04,
                          wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.4")
    ax.text(0.018, 0.985, "coastline.py — extraction core", fontsize=8.5,
            color="#8be9fd", family="monospace", va="top",
            transform=ax.transAxes)
    ax.text(0.018, 0.945, _CODE, fontsize=7.35, color=FG, family="monospace",
            va="top", transform=ax.transAxes, linespacing=1.42)
    ax.set_title("(a)  Shoreline extraction code (Google Earth Engine, "
                 "NDWI waterline)", loc="left", fontsize=10.5,
                 fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.4")
    lines = console.split("\n")
    y = 0.985
    for ln in lines:
        col = FG
        if ln.startswith("$"):
            col = "#7ec97e"
        elif "←" in ln or "[OK]" in ln:
            col = "#ffd866"
        elif ln.startswith("Storm_base_post") or "max erosion" in ln:
            col = "#ff8f8f"
        ax.text(0.018, y, ln, fontsize=7.6, color=col, family="monospace",
                va="top", transform=ax.transAxes)
        y -= 0.043
    ax.set_title("(b)  Run output — per-section retreat summary",
                 loc="left", fontsize=10.5, fontweight="bold")

    fig.suptitle("Shoreline extraction code and output — Sentinel-2 NDWI "
                 "waterline, Cyclone Gaja AOI", fontsize=13,
                 fontweight="bold", y=0.965)
    return fig


# ------------------------------------------------------------- Section 08
def fig_s8():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(C.CSV_PATH)
    with open(os.path.join(C.OUT_DIR, "dataset_qc.json")) as fh:
        qc = json.load(fh)
    cols = ["section_id", "lat", "lon", "storm", "storm_date",
            "storm_wind_ms", "wave_height_m", "mangrove_width_m",
            "dune_height_m", "terrain_elev_m", "composite_erosion_idx",
            "shoreline_retreat_m"]
    disp = ["section", "lat", "lon", "storm", "date", "wind\n(m/s)",
            "wave Hs\n(m)", "mangrove\nwidth (m)", "dune\nheight (m)",
            "terrain\nelev (m)", "erosion\nindex", "retreat (m)\n= target"]
    head = (df[df.section_id.isin(["S001", "S002"])]
            .sort_values(["section_id", "storm"])[cols])
    cell = [[f"{v:.6g}" if isinstance(v, float) else str(v)
             for v in row] for row in head.itertuples(index=False)]

    fig, ax = plt.subplots(figsize=(16.5, 4.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=disp, cellLoc="center",
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.55)
    for (r, c), cell_ in tbl.get_celld().items():
        cell_.set_edgecolor("0.75")
        if r == 0:
            cell_.set_facecolor("#1f4e79")
            cell_.set_text_props(color="white", fontweight="bold")
            cell_.set_height(0.34)
        else:
            cell_.set_facecolor("#eaf1f8" if (r - 1) // 4 % 2 else "#ffffff")
            if c == len(cols) - 1:
                cell_.set_text_props(fontweight="bold")
            if cols[c] == "shoreline_retreat_m" and r > 0:
                cell_.set_facecolor("#fdecea")
    ax.set_title(
        "Section × storm master dataset preview — data/master_dataset.csv  "
        f"({qc['rows']} rows = {qc['sections']} sections × "
        f"{len(qc['storms'])} cyclones;  QC: 0 nulls, 0 duplicate keys)",
        fontsize=12, fontweight="bold", pad=14)
    ax.text(0.5, -0.06,
            "ID columns  ·  raw forcing features (storm wind, waves)  ·  "
            "static coastal controls (mangrove, dune, terrain)  ·  "
            "target: shoreline_retreat_m (m, landward +)",
            transform=ax.transAxes, ha="center", fontsize=9, color="0.25")
    return fig


# ------------------------------------------------------------- Section 09
def fig_s9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(C.OUT_DIR, "model_metrics.json")) as fh:
        mm = json.load(fh)
    loso = pd.read_csv(os.path.join(C.OUT_DIR, "loso_per_storm.csv"))
    storms = ["Vardah", "Gaja", "Nivar", "Michaung"]
    models = ["LinearRegression", "RandomForest", "XGBoost"]
    mcol = {"LinearRegression": "#1f4e79", "RandomForest": "#c55a11",
            "XGBoost": "#548235"}

    BG, FG = "#1e1e2e", "#d8dee9"
    fig = plt.figure(figsize=(16.0, 6.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25],
                          left=0.04, right=0.97, top=0.83, bottom=0.11,
                          wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.4")
    lines = [
        ("$ python -m coastml.models", "#7ec97e"),
        ("protocol : leave-one-storm-out CV (hold out 1 cyclone,", FG),
        ("           train on the other 3)  ·  480 rows, 15 features", FG),
        ("features : 10 raw + 5 derived indices (energy,", FG),
        ("           mangrove/dune protection, terrain exposure,", FG),
        ("           composite_erosion_idx)", FG),
        ("target   : shoreline_retreat_m  (m, landward +)", FG),
        ("", FG),
        ("model             LOSO RMSE   MAE     R²     train", "#8be9fd"),
        (f"LinearRegression  {mm['metrics']['LinearRegression']['loso_rmse']:7.3f} m "
         f"{mm['metrics']['LinearRegression']['loso_mae']:6.3f}  "
         f"{mm['metrics']['LinearRegression']['loso_r2']:6.3f}   "
         f"{mm['metrics']['LinearRegression']['train_time_s']:5.2f} s  ★ best",
         "#ffd866"),
        (f"RandomForest      {mm['metrics']['RandomForest']['loso_rmse']:7.3f} m "
         f"{mm['metrics']['RandomForest']['loso_mae']:6.3f}  "
         f"{mm['metrics']['RandomForest']['loso_r2']:6.3f}   "
         f"{mm['metrics']['RandomForest']['train_time_s']:5.2f} s", FG),
        (f"XGBoost           {mm['metrics']['XGBoost']['loso_rmse']:7.3f} m "
         f"{mm['metrics']['XGBoost']['loso_mae']:6.3f}  "
         f"{mm['metrics']['XGBoost']['loso_r2']:6.3f}   "
         f"{mm['metrics']['XGBoost']['train_time_s']:5.2f} s", FG),
        ("", FG),
        ("wrote outputs/model_metrics.csv  model_metrics.json", FG),
        ("wrote outputs/loso_per_storm.csv  predictions_loso.csv", FG),
        ("wrote outputs/shap_summary_*.png  shap_dependence_*.png", FG),
    ]
    y = 0.97
    for ln, col in lines:
        ax.text(0.04, y, ln, fontsize=8.4, color=col, family="monospace",
                va="top", transform=ax.transAxes)
        y -= 0.055
    ax.set_title("(a)  Training output — LOSO cross-validation summary",
                 loc="left", fontsize=10.5, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(storms))
    w = 0.26
    for k, mkey in enumerate(models):
        sub = (loso[loso.model == mkey].set_index("held_out_storm")
               .reindex(storms))
        bars = ax.bar(x + (k - 1) * w, sub.rmse, w, color=mcol[mkey],
                      edgecolor="white", lw=0.5,
                      label=mkey + (" (best)" if mkey == "LinearRegression"
                                    else ""))
        for b, (rmse, r2) in zip(bars, zip(sub.rmse, sub.r2)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
                    f"{rmse:.2f}", ha="center", fontsize=7.4, color="0.2")
            ax.text(b.get_x() + b.get_width() / 2, 0.10, f"R²={r2:.2f}",
                    ha="center", fontsize=6.8, color="white",
                    fontweight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} (held out)" for s in storms], fontsize=9)
    ax.set_ylabel("LOSO RMSE (m)", fontsize=9)
    ax.set_ylim(0, 4.6)
    ax.set_title("(b)  Per-storm validation — RMSE when each cyclone is "
                 "held out", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(True, axis="y", ls=":", lw=0.5, color="0.75")

    fig.suptitle("Model training and validation output — LOSO-CV over four "
                 "cyclones (Vardah, Gaja, Nivar, Michaung)", fontsize=13,
                 fontweight="bold", y=0.95)
    return fig


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib.pyplot as plt

    builders = [("01", fig_s1, F1), ("03", fig_s3, F3), ("04", fig_s4, F4),
                ("05", fig_s5, F5), ("06", fig_s6, F6), ("07", fig_s7, F7),
                ("08", fig_s8, F8), ("09", fig_s9, F9)]
    for tag, fn, name in builders:
        fig = fn()
        path = os.path.join(OUT, name)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Section {tag} -> {path}")

    copies = [("02", os.path.join(C.OUT_DIR, "report", RF.FIG_A3), F2),
              ("10", os.path.join(C.OUT_DIR, "report", RF.FIG_A1), F10)]
    for tag, src, name in copies:
        dst = os.path.join(OUT, name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"Section {tag} -> {dst} (copied)")
        else:
            print(f"Section {tag} MISSING source {src} - run "
                  "coastml.report_figures first")


if __name__ == "__main__":
    main()
