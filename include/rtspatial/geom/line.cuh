#ifndef RTSPATIAL_GEOM_LINE_H
#define RTSPATIAL_GEOM_LINE_H
#include <optix_types.h>

#include "rtspatial/details/config.h"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/helpers.h"

namespace rtspatial {
template <typename POINT_T>
class Line {
  using point_t = POINT_T;
  using coord_t = typename point_t::coord_t;

 public:
  Line() = default;

  Line(const point_t& p1, const point_t& p2) : p1_(p1), p2_(p2) {}

  DEV_HOST_INLINE const point_t& get_p1() const { return p1_; }

  DEV_HOST_INLINE const point_t& get_p2() const { return p2_; }

//  DEV_HOST_INLINE bool Intersects(const Envelope<POINT_T>& envelope) const {
//    return other.min_.get_x() <= max_.get_x() &&
//           other.max_.get_x() >= min_.get_x() &&
//           other.min_.get_y() <= max_.get_y() &&
//           other.max_.get_y() >= min_.get_y();
//  }

 private:
  point_t p1_, p2_;
};

}  // namespace rtspatial
#endif  // RTSPATIAL_GEOM_LINE_H
