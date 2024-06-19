#ifndef RTSPATIAL_UTILS_TYPE_TRAITS_H
#define RTSPATIAL_UTILS_TYPE_TRAITS_H
#include <vector_types.h>

struct int128_2 {
  __int128 x, y;
};

struct int128_3 {
  __int128 x, y, z;
};

template <class COORD_T>
struct cuda_vec {};

template <>
struct cuda_vec<int> {
  using type_2d = int2;
  using type_3d = int3;
};

template <>
struct cuda_vec<unsigned int> {
  using type_2d = uint2;
  using type_3d = uint3;
};

template <>
struct cuda_vec<long> {
  using type_2d = long2;
  using type_3d = long3;
};

template <>
struct cuda_vec<long long> {
  using type_2d = longlong2;
  using type_3d = longlong2;
};

template <>
struct cuda_vec<unsigned long int> {
  using type_2d = ulong2;
  using type_3d = ulong3;
};

template <>
struct cuda_vec<unsigned long long int> {
  using type_2d = ulonglong2;
  using type_3d = ulonglong3;
};

template <>
struct cuda_vec<float> {
  using type_2d = float2;
  using type_3d = float3;
};

template <>
struct cuda_vec<double> {
  using type_2d = double2;
  using type_3d = double3;
};

template <>
struct cuda_vec<__int128> {
  using type_2d = int128_2;
  using type_3d = int128_3;
};

#endif
