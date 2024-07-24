#include "flags.h"

DEFINE_string(box, "", "path of the box dataset (wkt, box)");
DEFINE_string(point_query, "", "path of the query dataset (wkt, box, point)");
DEFINE_string(box_query, "", "path of the query dataset (wkt, box, point)");
DEFINE_double(load_factor,0.1, "");
DEFINE_string(predicate, "", "predicate, contains, intersects");
DEFINE_int32(limit_box, -1, "");
DEFINE_int32(limit_query, -1, "");
DEFINE_int32(parallelism, 1, "number of BVHs and rays");
DEFINE_string(serialize, "", "a directory to store serialized wkt file");