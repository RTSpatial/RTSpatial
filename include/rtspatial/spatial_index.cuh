#ifndef RTSPATIAL_SPATIAL_INDEX_H
#define RTSPATIAL_SPATIAL_INDEX_H
#include <thrust/async/transform.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include "rtspatial/details/ray_params.h"
#include "rtspatial/details/rt_engine.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/helpers.h"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/stopwatch.h"

namespace rtspatial {
namespace details {
template <typename COORD_T, int N_DIMS>
inline void FillAABBs(cudaStream_t cuda_stream,
                      ArrayView<Envelope<Point<COORD_T, N_DIMS>>> envelopes,
                      device_uvector<OptixAabb>& aabbs, bool append) {}

template <>
inline void FillAABBs<float, 2>(cudaStream_t cuda_stream,
                                ArrayView<Envelope<Point<float, 2>>> envelopes,
                                device_uvector<OptixAabb>& aabbs, bool append) {
  size_t prev_size = aabbs.size();

  if (append) {
    aabbs.resize(aabbs.size() + envelopes.size());
  } else {
    aabbs.resize(envelopes.size());
  }

  thrust::transform(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                    envelopes.end(), aabbs.begin() + prev_size,
                    [] __device__(const Envelope<Point<float, 2>>& envelope) {
                      OptixAabb aabb;
                      const auto& min_point = envelope.get_min();
                      const auto& max_point = envelope.get_max();
                      aabb.minX = min_point.get_x();
                      aabb.maxX = max_point.get_x();
                      aabb.minY = min_point.get_y();
                      aabb.maxY = max_point.get_y();
                      aabb.minZ = aabb.maxZ = 0;
                      return aabb;
                    });
}

template <>
inline void FillAABBs<double, 2>(
    cudaStream_t cuda_stream, ArrayView<Envelope<Point<double, 2>>> envelopes,
    device_uvector<OptixAabb>& aabbs, bool append) {
  size_t prev_size = aabbs.size();

  if (append) {
    aabbs.resize(aabbs.size() + envelopes.size());
  } else {
    aabbs.resize(envelopes.size());
  }

  thrust::transform(
      thrust::cuda::par.on(cuda_stream), envelopes.begin(), envelopes.end(),
      aabbs.begin() + prev_size,
      [] __device__(const Envelope<Point<double, 2>>& envelope) {
        OptixAabb aabb;
        const auto& min_point = envelope.get_min();
        const auto& max_point = envelope.get_max();
        aabb.minX = next_float_from_double(min_point.get_x(), -1, 2);
        aabb.maxX = next_float_from_double(max_point.get_x(), 1, 2);
        aabb.minY = next_float_from_double(min_point.get_y(), -1, 2);
        aabb.maxY = next_float_from_double(max_point.get_y(), 1, 2);
        aabb.minZ = aabb.maxZ = 0;
        return aabb;
      });
}

template <typename COORD_T, int N_DIMS>
inline void AppendTriangles(
    cudaStream_t cuda_stream,
    ArrayView<Envelope<Point<COORD_T, N_DIMS>>> envelopes,
    device_uvector<float3>& vertices, device_uvector<uint3>& indices) {}

template <>
inline void AppendTriangles<float, 2>(
    cudaStream_t cuda_stream, ArrayView<Envelope<Point<float, 2>>> envelopes,
    device_uvector<float3>& vertices, device_uvector<uint3>& indices) {
  size_t prev_vertices_size = vertices.size();
  size_t prev_indices_size = indices.size();

  vertices.resize(vertices.size() + envelopes.size() * 4);
  indices.resize(indices.size() + envelopes.size() * 2);

  float3* d_vertices =
      thrust::raw_pointer_cast(vertices.data() + prev_vertices_size);
  uint3* d_indices =
      thrust::raw_pointer_cast(indices.data() + prev_indices_size);

  thrust::for_each(
      thrust::cuda::par.on(cuda_stream),
      thrust::make_counting_iterator<uint32_t>(0),
      thrust::make_counting_iterator<uint32_t>(envelopes.size()),
      [=] __device__(uint32_t i) mutable {
        const auto& envelope = envelopes[i];
        const auto& min_corner = envelope.get_min();
        const auto& max_corner = envelope.get_max();

        d_vertices[i * 4] = float3{min_corner.get_x(), min_corner.get_y(), 0};
        d_vertices[i * 4 + 1] =
            float3{max_corner.get_x(), min_corner.get_y(), 0};
        d_vertices[i * 4 + 2] =
            float3{max_corner.get_x(), max_corner.get_y(), 0};
        d_vertices[i * 4 + 3] =
            float3{min_corner.get_x(), max_corner.get_y(), 0};
        d_indices[i * 2] = uint3{i * 4, i * 4 + 1, i * 4 + 3};
        d_indices[i * 2 + 1] = uint3{i * 4 + 2, i * 4 + 1, i * 4 + 3};
      });
}
}  // namespace details

// Ref: https://arc2r.github.io/book/Spatial_Predicates.html
template <typename COORD_T, int N_DIMS, bool USE_TRIANGLE = false>
class SpatialIndex {
  static_assert(std::is_floating_point<COORD_T>::value,
                "Unsupported COORD_T type");
  using ray_params_t = details::RayParams<COORD_T, N_DIMS>;

 public:
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;
  using result_queue_t = Queue<thrust::pair<size_t, size_t>>;

  void Init(const std::string& exec_root) {
    details::RTConfig config = details::get_default_rt_config(exec_root);

    rt_engine_.Init(config);
    Clear();
  }

  void Reserve(size_t capacity) {
    envelopes_.reserve(capacity);
    aabbs_.reserve(capacity);
  }

  void Clear() {
    envelopes_.clear();
    aabbs_.clear();
    vertices_.clear();
    indices_.clear();
    ray_params_.clear();
    as_buf_.clear();
    gas_handles_.clear();
    gas_handles_tri_.clear();
    aabbs_queries_.clear();
    gas_buf_queries_.clear();
    ray_params_queries_.clear();
    h_prefix_sum_.clear();
    h_prefix_sum_.push_back(0);
  }

  void Insert(ArrayView<envelope_t> envelopes,
              cudaStream_t cuda_stream = nullptr) {
    if (envelopes.empty()) {
      return;
    }
    size_t curr_geom_size = envelopes.size();
    size_t prev_geom_size = envelopes_.size();

    envelopes_.resize(envelopes_.size() + curr_geom_size);
    ray_params_.resize(ray_params_.size() + curr_geom_size);

    thrust::copy(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                 envelopes.end(), envelopes_.begin() + prev_geom_size);

    thrust::transform(
        thrust::cuda::par.on(cuda_stream), envelopes.begin(), envelopes.end(),
        ray_params_.begin() + prev_geom_size,
        [] __device__(const Envelope<Point<COORD_T, N_DIMS>>& envelope) {
          details::RayParams<COORD_T, N_DIMS> params;
          params.Compute(envelope, false /* inverse */);
          return params;
        });

    OptixTraversableHandle handle;
    thrust::device_vector<unsigned char> buf;

    if (USE_TRIANGLE) {
      size_t prev_vertices_size = vertices_.size();
      size_t prev_indices_size = indices_.size();
      details::AppendTriangles(cuda_stream, envelopes, vertices_, indices_);

      handle = rt_engine_.BuildAccelTriangle(
          cuda_stream,
          ArrayView<float3>(
              thrust::raw_pointer_cast(vertices_.data()) + prev_vertices_size,
              vertices_.size() - prev_vertices_size),
          ArrayView<uint3>(
              thrust::raw_pointer_cast(indices_.data()) + prev_indices_size,
              indices_.size() - prev_indices_size),
          buf);
      as_buf_[handle] = std::move(buf);

      gas_handles_tri_.push_back(handle);
      ias_handle_tri_ =
          rt_engine_.BuildInstanceAccel(cuda_stream, gas_handles_tri_, buf);

      if (as_buf_.find(ias_handle_tri_) != as_buf_.end()) {
        as_buf_.erase(ias_handle_tri_);
      }

      as_buf_[ias_handle_tri_] = std::move(buf);
    }

    details::FillAABBs(cuda_stream, envelopes, aabbs_, true /* append */);
    handle = rt_engine_.BuildAccelCustom(
        cuda_stream,
        ArrayView<OptixAabb>(
            thrust::raw_pointer_cast(aabbs_.data()) + prev_geom_size,
            curr_geom_size),
        buf);
    as_buf_[handle] = std::move(buf);
    gas_handles_.push_back(handle);

    if (as_buf_.find(ias_handle_) != as_buf_.end()) {
      as_buf_.erase(ias_handle_);
    }

    ias_handle_ = rt_engine_.BuildInstanceAccel(cuda_stream, gas_handles_, buf);
    as_buf_[ias_handle_] = std::move(buf);
    h_prefix_sum_.push_back(h_prefix_sum_.back() + curr_geom_size);
    d_prefix_sum_ = h_prefix_sum_;
  }

  void ContainsWhatQuery(ArrayView<point_t> queries, result_queue_t& result,
                         cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    details::LaunchParamsContainsPoint<COORD_T, N_DIMS> params;

    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.queries = queries;
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.result = result.DeviceObject();
    if (USE_TRIANGLE) {
      params.handle = ias_handle_tri_;
    } else {
      params.handle = ias_handle_;
    }

    rt_engine_.CopyLaunchParams(cuda_stream, params);
    details::ModuleIdentifier id =
        details::ModuleIdentifier::NUM_MODULE_IDENTIFIERS;

    if (std::is_same<COORD_T, float>::value) {
      if (USE_TRIANGLE) {
        id = details::ModuleIdentifier::
            MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D_TRIANGLE;
      } else {
        id = details::ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D;
      }
    } else if (std::is_same<COORD_T, double>::value) {
      if (USE_TRIANGLE) {
        id = details::ModuleIdentifier::
            MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D_TRIANGLE;
      } else {
        id =
            details::ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D;
      }
    }

    dim3 dims;

    dims.x = queries.size();
    dims.y = 1;
    dims.z = 1;

    rt_engine_.Render(cuda_stream, id, dims);
  }

  void ContainsWhatQuery(ArrayView<envelope_t> queries, result_queue_t& result,
                         cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    details::LaunchParamsContainsEnvelope<COORD_T, N_DIMS> params;

    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.queries = queries;
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.result = result.DeviceObject();
    params.handle = ias_handle_;

    rt_engine_.CopyLaunchParams(cuda_stream, params);
    details::ModuleIdentifier id;
    if (std::is_same<COORD_T, float>::value) {
      id =
          details::ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_ENVELOPE_QUERY_2D;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_DOUBLE_CONTAINS_ENVELOPE_QUERY_2D;
    }

    dim3 dims;

    dims.x = queries.size();
    dims.y = 1;
    dims.z = 1;

    rt_engine_.Render(cuda_stream, id, dims);
  }

  void IntersectsWhatQuery(const ArrayView<envelope_t> queries,
                           result_queue_t& result,
                           cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }

    ArrayView<envelope_t> d_envelopes(envelopes_);
    ArrayView<ray_params_t> ray_params(ray_params_);
    Stopwatch sw;
    double t_prepare_queries, t_build_bvh, t_forward_trace, t_backward_trace;

    sw.start();
    ray_params_queries_.resize(queries.size());
    details::FillAABBs(cuda_stream, queries, aabbs_queries_,
                       false /* append */);
    thrust::transform(
        thrust::cuda::par.on(cuda_stream), queries.begin(), queries.end(),
        ray_params_queries_.begin(),
        [] __device__(const Envelope<Point<COORD_T, N_DIMS>>& envelope) {
          details::RayParams<COORD_T, N_DIMS> params;
          params.Compute(envelope, true /* inverse */);
          return params;
        });

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_prepare_queries = sw.ms();

    sw.start();
    OptixTraversableHandle handle_queries;
    handle_queries = rt_engine_.BuildAccelCustom(
        cuda_stream, ArrayView<OptixAabb>(aabbs_queries_), gas_buf_queries_,
        true /* prefer_fast_build */);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_build_bvh = sw.ms();

    details::LaunchParamsIntersectsEnvelope<COORD_T, N_DIMS> params;

    // Query cast rays
    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.queries = queries;
    params.ray_params = ArrayView<ray_params_t>(ray_params_);
    params.ray_params_queries = ArrayView<ray_params_t>(ray_params_queries_);
    params.aabbs = ArrayView<OptixAabb>(aabbs_);
    params.aabbs_queries = ArrayView<OptixAabb>(aabbs_queries_);
    params.result = result.DeviceObject();
    params.handle = ias_handle_;
    params.inverse = false;

    details::ModuleIdentifier id;
    if (std::is_same<COORD_T, float>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D;
    }

    dim3 dims;

    dims.x = queries.size();
    dims.y = 1;
    dims.z = 1;

    sw.start();
    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_forward_trace = sw.ms();

    std::cout << "First batch size " << result.size(cuda_stream) << "\n";

    //      // Ray tracing from base envelopes
    params.envelopes.Swap(params.queries);
    params.ray_params.Swap(params.ray_params_queries);
    params.aabbs.Swap(params.aabbs_queries);
    params.handle = handle_queries;
    params.inverse = true;

    dims.x = envelopes_.size();

    sw.start();
    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_backward_trace = sw.ms();

    std::cout << "Second batch size " << result.size(cuda_stream) << "\n";
    std::cout << "Prepare params " << t_prepare_queries << " ms, query BVH "
              << t_build_bvh << " ms, forward " << t_forward_trace
              << " ms, backward " << t_backward_trace << " ms\n";
  }

  void Delete(const ArrayView<size_t> envelope_ids) {}

  void Update(const ArrayView<thrust::pair<size_t, envelope_t>> updates) {}

 private:
  details::RTEngine rt_engine_;
  // for base geometries
  device_uvector<envelope_t> envelopes_;
  pinned_vector<size_t> h_prefix_sum_;
  thrust::device_vector<size_t> d_prefix_sum_;  // prefix sum for each insertion
  // Customized AABBs
  device_uvector<OptixAabb> aabbs_;
  // Builtin triangles, contains only
  device_uvector<float3> vertices_;
  device_uvector<uint3> indices_;
  // Intersection only
  device_uvector<ray_params_t> ray_params_;
  std::map<OptixTraversableHandle, thrust::device_vector<unsigned char>>
      as_buf_;
  std::vector<OptixTraversableHandle> gas_handles_;
  std::vector<OptixTraversableHandle> gas_handles_tri_;
  OptixTraversableHandle ias_handle_;
  OptixTraversableHandle ias_handle_tri_;
  // for queries, updated for every batch of queries
  device_uvector<OptixAabb> aabbs_queries_;
  thrust::device_vector<unsigned char> gas_buf_queries_;
  device_uvector<ray_params_t> ray_params_queries_;
};

}  // namespace rtspatial

#endif  // RTSPATIAL_SPATIAL_INDEX_H
