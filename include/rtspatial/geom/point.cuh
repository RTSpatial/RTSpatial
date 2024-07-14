#ifndef RTSPATIAL_GEOM_POINT_H
#define RTSPATIAL_GEOM_POINT_H
#include <vector_types.h>
#include "rtspatial/utils/util.h"
namespace rtspatial {

template <typename COORD_T, int N_DIMS>
class Point {};

template <>
class Point<float, 2> {
 public:
  using coord_t = float;
  static constexpr int dim = 2;
  Point() = default;

  DEV_HOST Point(const float2& p) : p_(p) {}

  DEV_HOST Point(float x, float y) {
    p_.x = x;
    p_.y = y;
  }

  DEV_HOST_INLINE float get_x() const { return p_.x; }

  DEV_HOST_INLINE void set_x(float x) { p_.x = x; }

  DEV_HOST_INLINE float get_y() const { return p_.y; }

  DEV_HOST_INLINE void set_y(float y) { p_.y = y; }

 private:
  float2 p_;
};

template <>
class Point<double, 2> {
 public:
  using coord_t = double;
  static constexpr int dim = 2;
  Point() = default;

  DEV_HOST Point(const double2& p) : p_(p) {}

  DEV_HOST Point(double x, double y) {
    p_.x = x;
    p_.y = y;
  }

  DEV_HOST_INLINE double get_x() const { return p_.x; }

  DEV_HOST_INLINE void set_x(double x) { p_.x = x; }

  DEV_HOST_INLINE double get_y() const { return p_.y; }

  DEV_HOST_INLINE void set_y(double y) { p_.y = y; }

 private:
  double2 p_;
};

}  // namespace rtspatial
#endif  // RTSPATIAL_GEOM_POINT_H
