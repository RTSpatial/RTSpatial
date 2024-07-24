#ifndef RTSPATIAL_FLAGS_H
#define RTSPATIAL_FLAGS_H

#include <gflags/gflags.h>

#include "flags.h"
DECLARE_string(box);
DECLARE_string(point_query);
DECLARE_string(box_query);
DECLARE_double(load_factor);
DECLARE_string(predicate);
DECLARE_int32(limit_box);
DECLARE_int32(limit_query);
DECLARE_int32(parallelism);
DECLARE_string(serialize);
#endif  // RTSPATIAL_FLAGS_H
