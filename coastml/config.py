"""Central configuration: study area, storms, feature schema, paths."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
DB_PATH = os.path.join(DATA_DIR, "master.db")
CSV_PATH = os.path.join(DATA_DIR, "master_dataset.csv")

RANDOM_SEED = 42

# --- Study area: Coromandel Coast, Chennai -> Point Calimere (Tamil Nadu) ---
# Control points (lon, lat) tracing the coastline, north to south.
COAST_CONTROL_POINTS = [
    (80.348, 13.312),  # north Chennai (GMW mangrove belt)
    (80.320, 13.100),  # Chennai
    (80.180, 12.830),  # south of Chennai
    (80.130, 12.620),  # Mahabalipuram
    (79.860, 12.250),  # Puducherry north
    (79.830, 11.940),  # Puducherry
    (79.770, 11.720),  # Cuddalore
    (79.830, 11.470),  # Pichavaram / delta
    (79.850, 11.100),  # Tharangambadi
    (79.840, 10.770),  # Nagapattinam
    (79.860, 10.500),  # Vedaranyam
    (79.900, 10.300),  # Point Calimere
]

N_SECTIONS = 120          # coastal sections (~500 m-2 km spacing along shore)
SECTION_SPACING_LABEL = "~2 km alongshore"

# --- Cyclone events (real NIO cyclones + design storms) ---
# landfall_t: position of landfall along coast (0 = north/Chennai, 1 = south/Point Calimere)
# wind: max sustained wind (m/s), pressure: min central pressure (hPa)
# wave_h: offshore significant wave height (m), wave_p: peak period (s)
STORMS = [
    {"name": "Vardah",     "year": 2016, "date": "2016-12-12", "landfall_t": 0.10, "wind": 42.0, "pressure": 978.0, "wave_h": 5.2, "wave_p": 10.0},
    {"name": "Gaja",       "year": 2018, "date": "2018-11-16", "landfall_t": 0.82, "wind": 33.0, "pressure": 988.0, "wave_h": 4.1, "wave_p": 9.0},
    {"name": "Nivar",      "year": 2020, "date": "2020-11-27", "landfall_t": 0.42, "wind": 30.0, "pressure": 990.0, "wave_h": 3.6, "wave_p": 8.5},
    {"name": "Michaung",   "year": 2023, "date": "2023-12-04", "landfall_t": 0.05, "wind": 28.0, "pressure": 992.0, "wave_h": 3.4, "wave_p": 8.0},
]

# Real mangrove data (Global Mangrove Watch 2020, Chennai) used to anchor the
# mangrove feature field; see out/chennai/README.md.
GMW_MANGROVE_GEOJSON = os.path.join(
    ROOT, "out", "chennai", "mangrove", "mangrove_gmw_2020.geojson")
GMW_SECTIONS_GEOJSON = os.path.join(
    ROOT, "out", "chennai", "mangrove", "sections_classified.geojson")

# --- Feature schema (matches architecture diagram, stage 4) ---
RAW_FEATURES = [
    "mangrove_width_m",   # Mangrove: width
    "mangrove_density",   # Mangrove: density
    "dune_height_m",      # Dune: height
    "dune_width_m",       # Dune: width
    "storm_wind_ms",      # Storm: wind
    "storm_pressure_hpa", # Storm: pressure
    "wave_height_m",      # Wave: height
    "wave_period_s",      # Wave: period
    "terrain_elev_m",     # Terrain: elevation
    "terrain_slope_deg",  # Terrain: slope
]
# Derived indices (stage 4 feature engineering) - dimensionless, literature-style
DERIVED_FEATURES = [
    "energy_index",          # wind^1.7 * wave^0.8 forcing
    "mangrove_protection",   # exp(-width/2200) * (1 - 0.25*density)
    "dune_protection",       # 1 / (1 + 0.3*height + 0.003*width)
    "terrain_exposure",      # exp(-elev/10) * (0.7 + 0.15*slope)
    "composite_erosion_idx", # energy * mangrove * dune * terrain
]
FEATURES = RAW_FEATURES + DERIVED_FEATURES
TARGET = "shoreline_retreat_m"

ID_COLS = ["section_id", "lat", "lon", "storm", "storm_date", "year"]

# Physically-plausible bounds used both to generate and to QC the dataset.
BOUNDS = {
    "mangrove_width_m":   (0.0, 3000.0),
    "mangrove_density":   (0.0, 1.0),
    "dune_height_m":      (0.2, 8.0),
    "dune_width_m":       (5.0, 300.0),
    "storm_wind_ms":      (5.0, 50.0),
    "storm_pressure_hpa": (960.0, 1013.0),
    "wave_height_m":      (0.3, 7.0),
    "wave_period_s":      (3.0, 14.0),
    "terrain_elev_m":     (0.5, 25.0),
    "terrain_slope_deg":  (0.1, 5.0),
    "energy_index":          (0.0, 2.0),
    "mangrove_protection":   (0.0, 1.2),
    "dune_protection":       (0.0, 1.0),
    "terrain_exposure":      (0.0, 1.2),
    "composite_erosion_idx": (0.0, 1.5),
    TARGET:               (-5.0, 160.0),
}
