/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

This file is part of pbrt.

  Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  - Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

  - Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#ifndef RTSPATIAL_DETAILS_RAY_PARAMS_H
#define RTSPATIAL_DETAILS_RAY_PARAMS_H
#include <optix.h>
#include <thrust/swap.h>
#include <cfloat>

#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/line.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/util.h"
#define FLT_GAMMA(N) (((N) * FLT_EPSILON) / (1 - (N) * FLT_EPSILON))
#define DBL_GAMMA(N) (((N) * DBL_EPSILON) / (1 - (N) * DBL_EPSILON))
namespace rtspatial {
namespace details {

template <typename COORD_T, int N_DIMS>
struct RayParams {
  DEV_HOST_INLINE void Compute(const OptixAabb& aabb, bool inverse) {}

  DEV_HOST_INLINE void Compute(const Envelope<Point<COORD_T, N_DIMS>>& envelope,
                               bool inverse) {}
};

template <>
struct RayParams<float, 2> {
  float2 o;  // ray origin
  float2 d;  // ray direction


  DEV_HOST_INLINE void Compute(const Line<Point<float, 2>>& line) {
    float2 p1{line.get_p1().get_x(), line.get_p1().get_y()};
    float2 p2{line.get_p2().get_x(), line.get_p2().get_y()};

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void Compute(const Envelope<Point<float, 2>>& envelope,
                               bool diagonal) {
    const auto& min_corner = envelope.get_min();
    const auto& max_corner = envelope.get_max();

    float2 p1{min_corner.get_x(), min_corner.get_y()};
    float2 p2{max_corner.get_x(), max_corner.get_y()};

    if (diagonal) {
      p1.x = max_corner.get_x();
      p1.y = min_corner.get_y();
      p2.x = min_corner.get_x();
      p2.y = max_corner.get_y();
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void Compute(const OptixAabb& aabb, bool diagonal) {
    float2 p1{aabb.minX, aabb.minY};
    float2 p2{aabb.maxX, aabb.maxY};

    if (diagonal) {
      p1.x = aabb.maxX;
      p1.y = aabb.minY;
      p2.x = aabb.minX;
      p2.y = aabb.maxY;
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void PrintParams(const char* prefix) const {
    printf("%s, o: (%.6f, %.6f), d: (%.6f, %.6f)\n", prefix, o.x, o.y, d.x,
           d.y);
  }

  // Hacked from PBRTv3
  DEV_HOST_INLINE bool HitAABB(const OptixAabb& aabb) const {
    // FIXME: a little greater than 1.0 as a workaround
    float t0 = 0, t1 = nextafterf(1.0, FLT_MAX);
    const auto* pMin = reinterpret_cast<const float*>(&aabb.minX);
    const auto* pMax = reinterpret_cast<const float*>(&aabb.maxX);

    for (int i = 0; i < 2; ++i) {
      // Update interval for _i_th bounding box slab
      float invRayDir = 1 / reinterpret_cast<const float*>(&d)[i];
      float tNear =
          (pMin[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;
      float tFar =
          (pMax[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) {
        thrust::swap(tNear, tFar);
      }

      // Update _tFar_ to ensure robust ray--bounds intersection
      tFar *= 1 + 2 * FLT_GAMMA(3);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;

      if (t0 > t1)
        return false;
    }
    return true;
  }

  DEV_HOST_INLINE bool IsHit(const Envelope<Point<float, 2>>& envelope) const {
    // FIXME: a little greater than 1.0 as a workaround
    float t0 = 0, t1 = nextafterf(1.0, FLT_MAX);
    const auto* pMin = reinterpret_cast<const float*>(&envelope.get_min());
    const auto* pMax = reinterpret_cast<const float*>(&envelope.get_max());

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      // Update interval for _i_th bounding box slab
      float invRayDir = 1 / reinterpret_cast<const float*>(&d)[i];
      float tNear =
          (pMin[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;
      float tFar =
          (pMax[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) {
        thrust::swap(tNear, tFar);
      }

      // Update _tFar_ to ensure robust ray--bounds intersection
      tFar *= 1 + 2 * FLT_GAMMA(3);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;

      if (t0 > t1)
        return false;
    }
    return true;
  }
};

template <>
struct RayParams<double, 2> {
  double2 o;  // ray origin
  double2 d;  // ray direction

  DEV_HOST_INLINE void Compute(const Line<Point<double, 2>>& line) {
    auto min_x = std::min(line.get_p1().get_x(), line.get_p2().get_x());
    auto min_y = std::min(line.get_p1().get_y(), line.get_p2().get_y());

    auto max_x = std::max(line.get_p1().get_x(), line.get_p2().get_x());
    auto max_y = std::max(line.get_p1().get_y(), line.get_p2().get_y());

    double2 p1;
    double2 p2;

    p1.x = next_float_from_double(min_x, -1, 2);
    p1.y = next_float_from_double(min_y, -1, 2);
    p2.x = next_float_from_double(max_x, 1, 2);
    p2.y = next_float_from_double(max_y, 1, 2);

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void Compute(const Envelope<Point<double, 2>>& envelope,
                               bool inverse) {
    const auto& min_corner = envelope.get_min();
    const auto& max_corner = envelope.get_max();

    double2 p1{min_corner.get_x(), min_corner.get_y()};
    double2 p2{max_corner.get_x(), max_corner.get_y()};

    if (inverse) {
      p1.x = max_corner.get_x();
      p1.y = min_corner.get_y();
      p2.x = min_corner.get_x();
      p2.y = max_corner.get_y();
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void Compute(const OptixAabb& aabb, bool inverse) {
    double2 p1{aabb.minX, aabb.minY};
    double2 p2{aabb.maxX, aabb.maxY};

    if (inverse) {
      p1.x = aabb.maxX;
      p1.y = aabb.minY;
      p2.x = aabb.minX;
      p2.y = aabb.maxY;
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void PrintParams(const char* prefix) const {
    printf("%s, o: (%.6f, %.6f), d: (%.6f, %.6f)\n", prefix, o.x, o.y, d.x,
           d.y);
  }

  // Hacked from PBRTv3
  DEV_HOST_INLINE bool HitAABB(const OptixAabb& aabb) const {
    // FIXME: a little greater than 1.0 as a workaround
    double t0 = 0, t1 = nextafterf(1.0, DBL_MAX);
    const auto* pMin = reinterpret_cast<const float*>(&aabb.minX);
    const auto* pMax = reinterpret_cast<const float*>(&aabb.maxX);

    for (int i = 0; i < 2; ++i) {
      // Update interval for _i_th bounding box slab
      auto invRayDir = 1 / reinterpret_cast<const double*>(&d)[i];
      auto tNear =
          (pMin[i] - reinterpret_cast<const double*>(&o)[i]) * invRayDir;
      auto tFar =
          (pMax[i] - reinterpret_cast<const double*>(&o)[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) {
        thrust::swap(tNear, tFar);
      }

      // Update _tFar_ to ensure robust ray--bounds intersection
      tFar *= 1 + 2 * FLT_GAMMA(3);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;

      if (t0 > t1)
        return false;
    }
    return true;
  }

  DEV_HOST_INLINE bool IsHit(const Envelope<Point<double, 2>>& envelope) const {
    // FIXME: a little greater than 1.0 as a workaround
    double t0 = 0, t1 = nextafterf(1.0, DBL_MAX);
    const auto* pMin = reinterpret_cast<const double*>(&envelope.get_min());
    const auto* pMax = reinterpret_cast<const double*>(&envelope.get_max());

    for (int i = 0; i < 2; ++i) {
      // Update interval for _i_th bounding box slab
      auto invRayDir = 1 / reinterpret_cast<const double*>(&d)[i];
      auto tNear =
          (pMin[i] - reinterpret_cast<const double*>(&o)[i]) * invRayDir;
      auto tFar =
          (pMax[i] - reinterpret_cast<const double*>(&o)[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) {
        thrust::swap(tNear, tFar);
      }

      // Update _tFar_ to ensure robust ray--bounds intersection
      tFar *= 1 + 2 * DBL_GAMMA(3);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;

      if (t0 > t1)
        return false;
    }
    return true;
  }
};
}  // namespace details
}  // namespace rtspatial
#endif  // RTSPATIAL_DETAILS_RAY_PARAMS_H
