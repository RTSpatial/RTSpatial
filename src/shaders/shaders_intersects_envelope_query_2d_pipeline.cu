#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rtspatial/details/config.h"
#include "rtspatial/details/launch_parameters.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    rtspatial::details::LaunchParamsIntersectsEnvelopeNew<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void
__intersection__intersects_envelope_query_2d_forward() {
  auto primitive_idx = optixGetPrimitiveIndex();
  auto query_id = optixGetPayload_0();

  // This filter is not necessary anymore
  //      if (envelope.Intersects(query)) ...
  auto inst_id = optixGetInstanceId();
  auto envelope_id = params.prefix_sum[inst_id] + primitive_idx;
  const auto& envelope = params.geoms[envelope_id];
  const auto& query = params.queries[query_id];

  rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;

  ray_params.Compute(query, true);

  bool query_hit = ray_params.IsHit(envelope);

  if (query_hit) {
    params.result.AppendWarp(thrust::make_pair(envelope_id, query_id));
    atomicAdd(&params.n_hits[query_id], 1);
  }
}

extern "C" __global__ void __raygen__intersects_envelope_query_2d_forward() {
  using float2d_t = typename cuda_vec<FLOAT_TYPE>::type_2d;
  const auto& queries = params.queries;

  // TODO: Split queries into multiple BVHs
  // Cast multiple rays to the BVHs

  for (auto i = optixGetLaunchIndex().x; i < params.queries.size();
       i += optixGetLaunchDimensions().x) {
    //    auto query_id = params.begin_query + i;
    const auto& query = queries[i];
    rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;
    float3 origin, dir;

    ray_params.Compute(query, true);
    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = 0;

    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;
    dir.z = 0;

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

extern "C" __global__ void
__intersection__intersects_envelope_query_2d_backward() {
  auto query_id = optixGetPrimitiveIndex();
  auto geom_id = optixGetPayload_0();

  const auto& envelope = params.geoms[geom_id];
  const auto& query = params.queries[query_id];

  rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;

  ray_params.Compute(envelope, false);  // anti-diagonal
  bool query_hit = ray_params.IsHit(query);

  if (query_hit) {
    ray_params.Compute(query, true);

    bool box_hit = ray_params.IsHit(envelope);

    if (!box_hit) {
      params.result.AppendWarp(thrust::make_pair(geom_id, query_id));
      atomicAdd(&params.n_hits[query_id], 1);
    }
  }
}

extern "C" __global__ void __raygen__intersects_envelope_query_2d_backward() {
  for (auto geom_id = optixGetLaunchIndex().x; geom_id < params.geoms.size();
       geom_id += optixGetLaunchDimensions().x) {
    unsigned layer = optixGetLaunchIndex().y;
    const auto& geom = params.geoms[geom_id];
    rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;
    float3 origin, dir;

    ray_params.Compute(geom, false);  // anti-diagonal
    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = layer * AABB_Z_SCALE;

    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;
    dir.z = 0.0f;

    float tmin = 0;
    float tmax = 1;

    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               geom_id);
  }
}