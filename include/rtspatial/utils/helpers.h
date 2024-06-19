#ifndef RTSPATIAL_UTILS_HELPERS_H
#define RTSPATIAL_UTILS_HELPERS_H
#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_types.h>

namespace rtspatial {
// Adapted from
// http://stackoverflow.com/questions/466204/rounding-up-to-nearest-power-of-2
template <typename UnsignedType>
UnsignedType next_power_2(UnsignedType v) {
  static_assert(std::is_unsigned<UnsignedType>::value,
                "Only works for unsigned types");
  --v;
  for (int i = 1; i < sizeof(v) * CHAR_BIT; i *= 2) {
    v |= v >> i;
  }
  return ++v;
}

/**
 * Convert double to float with rounding
 *
 * @v double
 * @dir dir = -1, round down; dir = 1 round up
 * @iter number of calling nextafter
 */
__forceinline__ __device__ __host__ float next_float_from_double(double v,
                                                                 int dir,
                                                                 int iter = 1) {
  assert(dir == 1 || dir == -1);
  auto fv = static_cast<float>(v);  // pos number
  if (fv == 0) {
    return 0.0f;
  }
  float to = v * dir < 0 ? 0 : dir * std::numeric_limits<float>::infinity();

  for (int i = 0; i < iter; i++) {
    fv = std::nextafter(fv, to);
  }

  return fv;
};
}  // namespace rtspatial

#endif  // RTSPATIAL_UTILS_HELPERS_H
