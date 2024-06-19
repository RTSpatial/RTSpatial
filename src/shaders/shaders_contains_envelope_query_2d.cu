#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rtspatial/details/launch_parameters.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    rtspatial::details::LaunchParamsContainsEnvelope<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void __intersection__contains_envelope_query_2d() {
  auto envelope_id = optixGetPrimitiveIndex();
  auto query_id = optixGetPayload_0();
  const auto& envelope = params.envelopes[envelope_id];
  const auto& query = params.queries[query_id];

  if (envelope.Contains(query)) {
    params.result.Append(thrust::make_pair(envelope_id, query_id));
  }
}

extern "C" __global__ void __raygen__contains_envelope_query_2d() {
  const auto& queries = params.queries;
  float tmin = 0;
  float tmax = FLT_MIN;

  for (auto i = optixGetLaunchIndex().x; i < queries.size();
       i += optixGetLaunchDimensions().x) {
    const auto& query = queries[i];
    auto center = query.Center();
    float3 origin;
    origin.x = center.get_x();
    origin.y = center.get_y();
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
