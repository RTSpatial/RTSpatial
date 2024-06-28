#!/usr/bin/env python3
import geopandas
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) != 3:
    exit(1)
df = geopandas.read_file(in_file)
df = df.to_crs(epsg=4326)

with open(out_file, 'w') as fo:
    for original_id, row in df.iterrows():
        wkt = row.geometry.wkt
        fo.write(wkt + "\n")
