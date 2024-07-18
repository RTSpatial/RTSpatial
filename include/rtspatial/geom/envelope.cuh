#ifndef RTSPATIAL_GEOM_ENVELOPE_H
#define RTSPATIAL_GEOM_ENVELOPE_H
#include <optix_types.h>

#include "rtspatial/details/config.h"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/helpers.h"
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

namespace details {

template <typename COORD_T, int N_DIMS>
DEV_HOST_INLINE OptixAabb
EnvelopeToOptixAabb(const Envelope<Point<COORD_T, N_DIMS>>& envelope) {}

template <>
DEV_HOST_INLINE OptixAabb
EnvelopeToOptixAabb<float, 2>(const Envelope<Point<float, 2>>& envelope) {
  OptixAabb aabb;
  const auto& min_point = envelope.get_min();
  const auto& max_point = envelope.get_max();
  aabb.minX = min_point.get_x();
  aabb.maxX = max_point.get_x();
  aabb.minY = min_point.get_y();
  aabb.maxY = max_point.get_y();
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

template <>
DEV_HOST_INLINE OptixAabb
EnvelopeToOptixAabb<double, 2>(const Envelope<Point<double, 2>>& envelope) {
  OptixAabb aabb;
  const auto& min_point = envelope.get_min();
  const auto& max_point = envelope.get_max();
  aabb.minX = next_float_from_double(min_point.get_x(), -1, 2);
  aabb.maxX = next_float_from_double(max_point.get_x(), 1, 2);
  aabb.minY = next_float_from_double(min_point.get_y(), -1, 2);
  aabb.maxY = next_float_from_double(max_point.get_y(), 1, 2);
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

template <typename COORD_T, int N_DIMS>
DEV_HOST_INLINE OptixAabb EnvelopeToOptixAabb(
    const Envelope<Point<COORD_T, N_DIMS>>& envelope, int layer) {}

template <>
DEV_HOST_INLINE OptixAabb EnvelopeToOptixAabb<float, 2>(
    const Envelope<Point<float, 2>>& envelope, int layer) {
  OptixAabb aabb;
  const auto& min_point = envelope.get_min();
  const auto& max_point = envelope.get_max();
  aabb.minX = min_point.get_x();
  aabb.maxX = max_point.get_x();
  aabb.minY = min_point.get_y();
  aabb.maxY = max_point.get_y();
  aabb.minZ = aabb.maxZ = layer;
  return aabb;
}

template <>
DEV_HOST_INLINE OptixAabb EnvelopeToOptixAabb<double, 2>(
    const Envelope<Point<double, 2>>& envelope, int layer) {
  OptixAabb aabb;
  const auto& min_point = envelope.get_min();
  const auto& max_point = envelope.get_max();
  aabb.minX = next_float_from_double(min_point.get_x(), -1, 2);
  aabb.maxX = next_float_from_double(max_point.get_x(), 1, 2);
  aabb.minY = next_float_from_double(min_point.get_y(), -1, 2);
  aabb.maxY = next_float_from_double(max_point.get_y(), 1, 2);
  aabb.minZ = aabb.maxZ = layer;
  return aabb;
}
}  // namespace details
}  // namespace rtspatial
#endif  // RTSPATIAL_GEOM_ENVELOPE_H
