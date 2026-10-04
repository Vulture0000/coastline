"""Quota-cheap scene inventory for the 4 epochs. Metadata + one small
reduceRegion per scene. No reduceToVectors, no sampleRegions, no getInfo on
geometry. Run this before coastline.py to confirm which scenes will be used.
"""

import ee

import coastline as C

ee.Initialize()
print("project:", ee.data.get_persistent_credentials().quota_project_id)
print("AOI:", C.AOI_LL, "| CLOUD_MAX:", C.CLOUD_MAX)

total = 0
for epoch, date in C.EPOCHS:
    col = C.collection(date)
    n = col.size().getInfo()
    total += n
    C.report(date, col)
    print("  epoch {} -> {} scene(s)".format(epoch, n))

print("\ntotal scenes to read:", total)
print("bands per scene: B03 B04 B08 B11 SCL")