
#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "rtspatial/details/ray_params.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/type_traits.h"
namespace rtspatial {

namespace details {
template <typename COORD_T, int N_DIMS>
struct LaunchParamsContainsPoint {
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;
  ArrayView<size_t> prefix_sum;
  ArrayView<point_t> queries;
  ArrayView<envelope_t> envelopes;
  dev::Queue<thrust::pair<size_t, size_t>> result;
  OptixTraversableHandle handle;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsContainsEnvelope {
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;
  ArrayView<size_t> prefix_sum;
  ArrayView<envelope_t> queries;
  ArrayView<envelope_t> envelopes;
  dev::Queue<thrust::pair<size_t, size_t>> result;
  OptixTraversableHandle handle;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsIntersectsEnvelope {
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;
  using ray_params_t = RayParams<COORD_T, N_DIMS>;

  ArrayView<size_t> prefix_sum;
  ArrayView<envelope_t> envelopes;
  ArrayView<envelope_t> queries;
  ArrayView<ray_params_t> ray_params;
  ArrayView<ray_params_t> ray_params_queries;
  ArrayView<OptixAabb> aabbs;
  ArrayView<OptixAabb> aabbs_queries;
  bool inverse;
  dev::Queue<thrust::pair<size_t, size_t>> result;
  OptixTraversableHandle handle;
};
}  // namespace details

}  // namespace rtspatial
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
