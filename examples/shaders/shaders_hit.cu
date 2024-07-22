#include <optix.h>
#include <thrust/pair.h>

#include <cstdint>

#include "rtspatial/utils/queue.h"

extern "C" __device__ void __direct_callable__rtspatial_hit(uint32_t geom_id,
                                                            uint32_t query_id,
                                                            void* arg) {
  auto* queue =
      static_cast<rtspatial::dev::Queue<thrust::pair<uint32_t, uint32_t>>*>(
          arg);
  queue->Append(thrust::make_pair(geom_id, query_id));
}

