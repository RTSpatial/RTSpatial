#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rtspatial/details/launch_parameters.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    rtspatial::details::LaunchParamsContainsPoint<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void __intersection__contains_point_query_2d() {
  auto primitive_idx = optixGetPrimitiveIndex();
  auto inst_id = optixGetInstanceId();
  auto geom_id = params.prefix_sum[inst_id] + primitive_idx;
  auto query_id = optixGetPayload_0();
  const auto& envelope = params.envelopes[geom_id];
  const auto& query = params.queries[query_id];

  if (envelope.Contains(query)) {
    rtspatial_handle_point_contains(geom_id, query_id, params.arg);
  }
}

extern "C" __global__ void __raygen__contains_point_query_2d() {
  const auto& queries = params.queries;
  float tmin = 0;
  float tmax = FLT_MIN;

  for (auto i = optixGetLaunchIndex().x; i < queries.size();
       i += optixGetLaunchDimensions().x) {
    const auto& p = queries[i];
    float3 origin;
    origin.x = p.get_x();
    origin.y = p.get_y();
    origin.z = 0;
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
