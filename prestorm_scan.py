import ee
ee.Initialize()
import coastline as C

PRE_END = "2018-11-16"
col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
       .filterBounds(C.aoi())
       .filterDate("2017-11-01", PRE_END))

print("total pre-storm scenes:", col.size().getInfo(), flush=True)

by = col.aggregate_array("system:time_start").distinct().sort()
dates = by.getInfo()
print("distinct pre-storm dates:", len(dates), flush=True)

rows = []
for t in dates:
    d = ee.Date(t).format("YYYY-MM-dd").getInfo()
    c = col.filterDate(d, ee.Date(d).advance(1, "day"))
    n = c.size().getInfo()
    if not n:
        continue
    mn = c.aggregate_min("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    tiles = sorted(set(c.aggregate_array("MGRS_TILE").getInfo()))
    rows.append((d, n, mn, ",".join(tiles)))
for d, n, mn, tiles in rows:
    print(f"  {d}  n={n}  cloud_min={mn:5.1f}%  tiles={tiles}", flush=True)
