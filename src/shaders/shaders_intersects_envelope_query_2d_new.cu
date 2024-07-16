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

extern "C" __global__ void
__intersection__intersects_envelope_query_2d_forward() {
  auto primitive_idx = optixGetPrimitiveIndex();
  auto query_id = optixGetPayload_0();

  // This filter is not necessary anymore
  //      if (envelope.Intersects(query)) ...
  auto inst_id = optixGetInstanceId();
  auto envelope_id = params.prefix_sum[inst_id] + primitive_idx;
  const auto& envelope = params.envelopes[envelope_id];
  const auto& aabb = params.aabbs[envelope_id];
  const auto& query = params.queries[query_id];

  rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;

  ray_params.Compute(query, true);
  bool query_hit = ray_params.HitAABB(aabb);

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
  auto primitive_idx = optixGetPrimitiveIndex();
  auto query_id = optixGetPayload_0();
  auto handle_id = optixGetPayload_1();
  auto envelope_id = params.prefix_sum[handle_id] + primitive_idx;

  const auto& envelope = params.envelopes[envelope_id];
  const auto& aabb = params.aabbs[envelope_id];
  const auto& query = params.queries[query_id];

  rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;

  ray_params.Compute(query, false);
  bool query_hit = ray_params.HitAABB(aabb);

  if (query_hit) {
    const auto& aabb_query = params.aabbs_queries[query_id];

    ray_params.Compute(envelope, true);

    bool box_hit = ray_params.HitAABB(aabb_query);

    if (!box_hit) {
      params.result.AppendWarp(thrust::make_pair(query_id, envelope_id));
      atomicAdd(&params.n_hits[query_id], 1);
    }
  }
}

extern "C" __global__ void __raygen__intersects_envelope_query_2d_backward() {
  for (auto query_id = optixGetLaunchIndex().x;
       query_id < params.queries.size();
       query_id += optixGetLaunchDimensions().x) {
    unsigned handle_idx = optixGetLaunchIndex().y;
    const auto& query = params.queries[query_id];
    rtspatial::details::RayParams<FLOAT_TYPE, 2> ray_params;
    float3 origin, dir;

    ray_params.Compute(query, false);  // anti-diagonal
    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = 0;

    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;
    dir.z = 0;

    float tmin = 0;
    float tmax = 1;

    optixTrace(params.backward_handles[handle_idx], origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               query_id, handle_idx);
  }
}