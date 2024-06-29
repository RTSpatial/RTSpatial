#include <cuda.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <optix.h>
#include <optix_device.h>

#include "rtspatial/details/launch_parameters.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    rtspatial::details::LaunchParamsContainsPoint<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void __anyhit__contains_point_query_2d_triangle() {
  auto triangle_id = optixGetPrimitiveIndex();
  auto envelope_id = triangle_id / 2;
  auto query_id = optixGetPayload_0();
  const auto& envelope = params.envelopes[envelope_id];
  const auto& query = params.queries[query_id];
  const auto& min_corner = envelope.get_min();
  const auto& max_corner = envelope.get_max();

#ifndef NDEBUG
  OptixTraversableHandle gas = optixGetGASTraversableHandle();
  int mid_tri = triangle_id / 2;
  unsigned int primIdx1 = mid_tri * 2;
  unsigned int primIdx2 = primIdx1 + 1;
  unsigned int sbtIdx = optixGetSbtGASIndex();
  float time = optixGetRayTime();
  float3 data1[3], data2[3];

  optixGetTriangleVertexData(gas, primIdx1, sbtIdx, time, data1);
  optixGetTriangleVertexData(gas, primIdx2, sbtIdx, time, data2);

  float2 min_tri, max_tri;

  min_tri.x = std::numeric_limits<FLOAT_TYPE>::max();
  min_tri.y = std::numeric_limits<FLOAT_TYPE>::max();
  max_tri.x = std::numeric_limits<FLOAT_TYPE>::min();
  max_tri.y = std::numeric_limits<FLOAT_TYPE>::min();

  for (int i = 0; i < 3; i++) {
    min_tri.x = std::min(min_tri.x, data1[i].x);
    min_tri.y = std::min(min_tri.y, data1[i].y);
    max_tri.x = std::max(max_tri.x, data1[i].x);
    max_tri.y = std::max(max_tri.y, data1[i].y);
  }

  for (int i = 0; i < 3; i++) {
    min_tri.x = std::min(min_tri.x, data2[i].x);
    min_tri.y = std::min(min_tri.y, data2[i].y);
    max_tri.x = std::max(max_tri.x, data2[i].x);
    max_tri.y = std::max(max_tri.y, data2[i].y);
  }

  assert(min_tri.x == min_corner.get_x());
  assert(max_tri.x == max_corner.get_x());
  assert(min_tri.y == min_corner.get_y());
  assert(max_tri.y == max_corner.get_y());

//  printf("tri x [%f, %f] - y [%f, %f] , box: x [%f, %f] - y [%f, %f]\n",
//         min_tri.x, max_tri.x, min_tri.y, max_tri.y, min_corner.get_x(),
//         max_corner.get_x(), min_corner.get_y(), max_corner.get_y());

  //  printf("tri1 (%f, %f), (%f, %f), (%f, %f); tri2 (%f, %f), (%f, %f), (%f,
  //  %f), box: x [%f, %f] - y [%f, %f]\n",
  //         data1[0].x, data1[0].y, data1[1].x, data1[1].y, data1[2].x,
  //         data1[2].y, data2[0].x, data2[0].y, data2[1].x, data2[1].y,
  //         data2[2].x, data2[2].y, min_corner.get_x(), max_corner.get_x(),
  //         min_corner.get_y(), max_corner.get_y()
  //         );

#endif

//  if (envelope.Contains(query)) {
    params.result.Append(thrust::make_pair(envelope_id, query_id));
//  }
  optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__contains_point_query_2d_triangle() {
  const auto& queries = params.queries;
  float tmin = 0;
  float tmax = 1;

  for (auto i = optixGetLaunchIndex().x; i < queries.size();
       i += optixGetLaunchDimensions().x) {
    const auto& p = queries[i];
    float3 origin;
    origin.x = p.get_x();
    origin.y = p.get_y();
    origin.z = -0.1;

    float3 dir = {0, 0, 1};

    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               i);
  }
}
