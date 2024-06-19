#ifndef RTSPATIAL_GEOM_ENVELOPE_H
#define RTSPATIAL_GEOM_ENVELOPE_H
#include "rtspatial/geom/point.cuh"

namespace rtspatial {
template <typename POINT_T>
class Envelope {
  using point_t = POINT_T;
  using coord_t = typename point_t::coord_t;

 public:
  Envelope() = default;

  DEV_HOST Envelope(const point_t& min, const point_t& max)
      : min_(min), max_(max) {}

  DEV_HOST_INLINE const point_t& get_min() const { return min_; }

  DEV_HOST_INLINE const point_t& get_max() const { return max_; }

  DEV_HOST_INLINE bool Contains(const point_t& p) const {
    return p.get_x() >= min_.get_x() && p.get_x() <= max_.get_x() &&
           p.get_y() >= min_.get_y() && p.get_y() <= max_.get_y();
  }

  DEV_HOST_INLINE bool Contains(const Envelope<POINT_T>& other) const {
    return other.get_min().get_x() >= min_.get_x() &&
           other.get_max().get_x() <= max_.get_x() &&
           other.get_min().get_y() >= min_.get_y() &&
           other.get_max().get_y() <= max_.get_y();
  }

  DEV_HOST_INLINE bool Intersects(const Envelope<POINT_T>& other) const {
    return other.min_.get_x() <= max_.get_x() &&
           other.max_.get_x() >= min_.get_x() &&
           other.min_.get_y() <= max_.get_y() &&
           other.max_.get_y() >= min_.get_y();
  }

  DEV_HOST_INLINE point_t Center() const {
    point_t p((min_.get_x() + max_.get_x()) / (coord_t) 2.0,
              (min_.get_y() + max_.get_y()) / (coord_t) 2.0);
    return p;
  }

 private:
  point_t min_, max_;
};

}  // namespace rtspatial
#endif  // RTSPATIAL_GEOM_ENVELOPE_H
