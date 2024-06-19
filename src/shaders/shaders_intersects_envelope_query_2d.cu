#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rtspatial/details/launch_parameters.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    rtspatial::details::LaunchParamsIntersectsEnvelope<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void __intersection__intersects_envelope_query_2d() {
  auto envelope_id = optixGetPrimitiveIndex();
  auto query_id = optixGetPayload_0();
  //  const auto& envelope = params.envelopes[envelope_id];
  //  const auto& query = params.queries[query_id];
  // This filter is not necessary anymore
  //      if (envelope.Intersects(query)) ...
  const auto& aabb = params.aabbs[envelope_id];
  bool query_hit = params.ray_params_queries[query_id].HitAABB(aabb);

  if (query_hit) {
    //
    //    printf("xsect %d %u %u, envelope x [%.4f, %.4f] y [%.4f, %.4f],
    //    query
    //    x
    //    [%.4f, %.4f] y [%.4f, %.4f]\n",
    //           x,
    //           envelope_id, query_id,
    //           envelope.get_min().get_x(),
    //           envelope.get_max().get_x(),
    //           envelope.get_min().get_y(),
    //           envelope.get_max().get_y(),
    //           query.get_min().get_x(),
    //           query.get_max().get_x(),
    //           query.get_min().get_y(),
    //           query.get_max().get_y()
    //    );

    if (params.inverse) {
      const auto& aabb_query = params.aabbs_queries[query_id];
      bool box_hit = params.ray_params[envelope_id].HitAABB(aabb_query);

      if (!box_hit) {
        params.result.AppendWarp(thrust::make_pair(query_id, envelope_id));
      }
    } else {
      params.result.AppendWarp(thrust::make_pair(envelope_id, query_id));
    }
  }
}

extern "C" __global__ void __raygen__intersects_envelope_query_2d() {
  const auto& queries = params.queries;

  for (auto i = optixGetLaunchIndex().x; i < queries.size();
       i += optixGetLaunchDimensions().x) {
    const auto& ray_params = params.ray_params_queries[i];

    float3 origin;
    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = 0;

    float3 dir = {0, 0, 0};
    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;

    float tmin = 0;
    float tmax = 1;

    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               i);
  }
}
