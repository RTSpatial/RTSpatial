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
  auto primitive_idx = optixGetPrimitiveIndex();
  size_t envelope_id;
  auto query_id = optixGetPayload_0();

  // This filter is not necessary anymore
  //      if (envelope.Intersects(query)) ...

  if (params.inverse) {
    envelope_id = primitive_idx;
  } else {
    auto inst_id = optixGetInstanceId();
    envelope_id = params.prefix_sum[inst_id] + primitive_idx;
  }
  const auto& envelope = params.envelopes[envelope_id];
  const auto& aabb = params.aabbs[envelope_id];
  const auto& query = params.queries[query_id];

  rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;

  ray_params.Compute(query, !params.inverse);
  bool query_hit = ray_params.HitAABB(aabb);
  // bool query_hit = params.ray_params_queries[query_id].HitAABB(aabb);

  if (query_hit) {
    if (params.inverse) {
      const auto& aabb_query = params.aabbs_queries[query_id];

      ray_params.Compute(envelope, params.inverse);

      bool box_hit = ray_params.HitAABB(aabb_query);

      //      bool box_hit = params.ray_params[envelope_id].HitAABB(aabb_query);

      if (!box_hit) {
        params.result.AppendWarp(thrust::make_pair(query_id, envelope_id));
      }
    } else {
      params.result.AppendWarp(thrust::make_pair(envelope_id, query_id));
    }
  }
}

extern "C" __global__ void __raygen__intersects_envelope_query_2d() {
  using float2d_t = typename cuda_vec<FLOAT_TYPE>::type_2d;
  const auto& queries = params.queries;

  for (auto i = optixGetLaunchIndex().x; i < queries.size();
       i += optixGetLaunchDimensions().x) {
    const auto& query = queries[i];
    rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;
    float3 origin, dir;

    ray_params.Compute(query, !params.inverse);
    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = 0;

    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;
    dir.z = 0;

    //        const auto& ray_params = params.ray_params_queries[i];
    //
    //        float3 origin;
    //        origin.x = ray_params.o.x;
    //        origin.y = ray_params.o.y;
    //        origin.z = 0;
    //
    //        float3 dir = {0, 0, 0};
    //        dir.x = ray_params.d.x;
    //        dir.y = ray_params.d.y;

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
