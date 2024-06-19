#!/usr/bin/env bash

BASE=1000000
for dist in uniform gaussian; do
  for query in 1000 10000 100000 1000000; do
    /local/storage/liang/.clion/RTSpatial/cmake-build-release-dl190/bin/rtspatial \
      -box /local/storage/liang/.clion/RTPointIndex/dataset/box/${dist}_${BASE}.wkt \
      -query /local/storage/liang/.clion/RTPointIndex/dataset/box/${dist}_${query}.wkt | tee "${dist}_${BASE}_${query}.log"
  done
done

