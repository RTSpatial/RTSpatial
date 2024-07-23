#ifndef RTSPATIAL_DEFAULT_HANDLERS_COUNTING_HANDLERS_H
#define RTSPATIAL_DEFAULT_HANDLERS_COUNTING_HANDLERS_H
#include <optix.h>
#include <cstdint>

extern "C" __device__ void rtspatial_handle_point_contains(uint32_t geom_id,
                                                           uint32_t query_id,
                                                           void* arg) {
  atomicAdd(static_cast<unsigned long long int*>(arg), 1);
}

extern "C" __device__ void rtspatial_handle_envelope_contains(uint32_t geom_id,
                                                              uint32_t query_id,
                                                              void* arg) {
  atomicAdd(static_cast<unsigned long long int*>(arg), 1);
}

extern "C" __device__ void rtspatial_handle_envelope_intersects(
    uint32_t geom_id, uint32_t query_id, void* arg) {
  atomicAdd(static_cast<unsigned long long int*>(arg), 1);
}

#endif