#ifndef RTSPATIAL_GEOM_PREDICATES_H
#define RTSPATIAL_GEOM_PREDICATES_H
#include "rtspatial/utils/array_view.h"

namespace rtspatial {

enum class Predicate {
  kIntersects,
  kContains,
};

template <typename QUERY_T>
class Queries {
 public:
  Queries(Predicate p, ArrayView<QUERY_T> queries) : p_(p), queries_(queries) {}

 private:
  Predicate p_;
  ArrayView<QUERY_T> queries_;
};

}  // namespace rtspatial

#endif  // RTSPATIAL_GEOM_PREDICATES_H
