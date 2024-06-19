#!/usr/bin/env bash

for count in 1000 10000 100000 1000000 10000000; do
  # Generate bounding boxes
  for dist in uniform gaussian; do
    ./generator.py distribution=$dist \
      cardinality=$count \
      dimensions=2 \
      geometry=box \
      polysize=0.01 \
      maxseg=3 \
      format=wkt \
      maxsize=0.005,0.005 >"box/${dist}_${count}.wkt"

    ./generator.py distribution=$dist \
      cardinality=$count \
      dimensions=2 \
      geometry=point \
      format=wkt >"point/${dist}_${count}.wkt"
  done
done
