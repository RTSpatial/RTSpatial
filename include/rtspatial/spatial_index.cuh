#ifndef RTSPATIAL_SPATIAL_INDEX_H
#define RTSPATIAL_SPATIAL_INDEX_H
#include <thrust/async/transform.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/unique.h>

#include "rtspatial/details/rt_engine.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/bitset.h"
#include "rtspatial/utils/helpers.h"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/stopwatch.h"

namespace rtspatial {
namespace details {

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
      thrust::raw_pointer_cast(vertices.data()) + prev_vertices_size;
  uint3* d_indices =
      thrust::raw_pointer_cast(indices.data()) + prev_indices_size;

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
    as_buf_.clear();
    gas_handles_.clear();
    gas_handles_tri_.clear();
    aabbs_queries_.clear();
    gas_buf_queries_.clear();
    h_prefix_sum_.clear();
    h_prefix_sum_.push_back(0);
  }

  void Insert(ArrayView<envelope_t> envelopes,
              cudaStream_t cuda_stream = nullptr) {
    if (envelopes.empty()) {
      return;
    }
    size_t curr_size = envelopes.size();
    size_t prev_size = envelopes_.size();

    envelopes_.resize(envelopes_.size() + curr_size);
    aabbs_.resize(aabbs_.size() + curr_size);

    thrust::copy(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                 envelopes.end(), envelopes_.begin() + prev_size);

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

    thrust::transform(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                      envelopes.end(), aabbs_.begin() + prev_size,
                      [] __device__(const envelope_t& envelope) {
                        return details::EnvelopeToOptixAabb(envelope);
                      });

    handle = rt_engine_.BuildAccelCustom(
        cuda_stream,
        ArrayView<OptixAabb>(
            thrust::raw_pointer_cast(aabbs_.data()) + prev_size, curr_size),
        buf, false /*prefer_fast_build*/);
    as_buf_[handle] = std::move(buf);
    gas_handles_.push_back(handle);

    if (as_buf_.find(ias_handle_) != as_buf_.end()) {
      as_buf_.erase(ias_handle_);
    }

    ias_handle_ = rt_engine_.BuildInstanceAccel(cuda_stream, gas_handles_, buf);
    as_buf_[ias_handle_] = std::move(buf);
    h_prefix_sum_.push_back(h_prefix_sum_.back() + curr_size);
    d_prefix_sum_ = h_prefix_sum_;
  }

  void Delete(const ArrayView<size_t> envelope_ids, cudaStream_t stream) {}

  void Update(const ArrayView<thrust::pair<size_t, envelope_t>> updates,
              cudaStream_t stream) {
    touched_batch_ids_.Init(d_prefix_sum_.size() - 1);
    ArrayView<envelope_t> v_envelopes(envelopes_);
    ArrayView<OptixAabb> v_aabbs(aabbs_);
    ArrayView<size_t> v_prefix_sum(d_prefix_sum_);
    auto v_touched_batch_ids = touched_batch_ids_.DeviceObject();
    size_t max_id = h_prefix_sum_.back();

    thrust::for_each(
        thrust::cuda::par.on(stream), updates.begin(), updates.end(),
        [=] __device__(const thrust::pair<size_t, envelope_t>& pair) mutable {
          size_t idx = pair.first;
          const auto& envelope = pair.second;

          assert(idx < max_id);
          v_aabbs[idx] = details::EnvelopeToOptixAabb(envelope);
          v_envelopes[idx] = envelope;

          auto it = thrust::upper_bound(thrust::seq, v_prefix_sum.begin(),
                                        v_prefix_sum.end(), idx);
          assert(it != v_prefix_sum.end());
          auto batch_id = v_prefix_sum.end() - it - 1;
          assert(batch_id >= 0 && batch_id < v_prefix_sum.size());
          v_touched_batch_ids.set_bit_atomic(batch_id);
        });

    auto touched_batch_ids = touched_batch_ids_.DumpPositives(stream);

    for (auto batch_id : touched_batch_ids) {
      auto handle = gas_handles_[batch_id];
      auto& buf = as_buf_[handle];
      auto begin = h_prefix_sum_[batch_id];
      auto size = h_prefix_sum_[batch_id + 1] - begin;

      auto new_handle = rt_engine_.UpdateAccelCustom(
          stream, handle,
          ArrayView<OptixAabb>(thrust::raw_pointer_cast(aabbs_.data()) + begin,
                               size),
          buf, false /*prefer_fast_build*/);
      assert(new_handle == handle);
    }

    // Rebuild IAS
    as_buf_.erase(ias_handle_);
    thrust::device_vector<unsigned char> buf;
    ias_handle_ = rt_engine_.BuildInstanceAccel(stream, gas_handles_, buf);
    as_buf_[ias_handle_] = std::move(buf);
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
    Stopwatch sw;
    double t_prepare_queries, t_build_bvh, t_forward_trace, t_backward_trace;

    sw.start();
    aabbs_queries_.resize(queries.size());
    thrust::transform(thrust::cuda::par.on(cuda_stream), queries.begin(),
                      queries.end(), aabbs_queries_.begin(),
                      [] __device__(const envelope_t& envelope) {
                        return details::EnvelopeToOptixAabb(envelope);
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

    n_hits.resize(queries.size(), 0);

    details::LaunchParamsIntersectsEnvelope<COORD_T, N_DIMS> params;

    // Query cast rays
    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.queries = queries;
    params.aabbs = ArrayView<OptixAabb>(aabbs_);
    params.aabbs_queries = ArrayView<OptixAabb>(aabbs_queries_);
    params.result = result.DeviceObject();
    params.handle = ias_handle_;
    params.inverse = false;
    params.n_hits = thrust::raw_pointer_cast(n_hits.data());

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
    {
      pinned_vector<uint32_t> h_n_hits = n_hits;
      uint32_t total_hits = 0, max_hits = 0;

      for (int i = 0; i < h_n_hits.size(); i++) {
        max_hits = std::max(max_hits, h_n_hits[i]);
        total_hits += h_n_hits[i];
      }

      std::cout << "First batch rays " << dims.x << " results size "
                << result.size(cuda_stream) << " total hits " << total_hits
                << " max hits " << max_hits << "\n";
    }

    n_hits.resize(envelopes_.size(), 0);

    // Ray tracing from base envelopes
    params.envelopes.Swap(params.queries);
    params.aabbs.Swap(params.aabbs_queries);
    params.handle = handle_queries;
    params.inverse = true;
    params.n_hits = thrust::raw_pointer_cast(n_hits.data());

    sw.start();
    dims.x = params.queries.size();
    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_backward_trace = sw.ms();

    // TODO: Cut a long ray to many short rays for load balancing
    {
      pinned_vector<uint32_t> h_n_hits = n_hits;
      uint32_t total_hits = 0, max_hits = 0;

      for (int i = 0; i < h_n_hits.size(); i++) {
        max_hits = std::max(max_hits, h_n_hits[i]);
        total_hits += h_n_hits[i];
      }

      std::cout << "Second batch rays " << dims.x << " results size "
                << result.size(cuda_stream) << " total hits " << total_hits
                << " max hits " << max_hits << "\n";
    }

    std::cout << "Prepare params " << t_prepare_queries << " ms, build BVH "
              << t_build_bvh << " ms, forward " << t_forward_trace
              << " ms, backward " << t_backward_trace << " ms\n";
  }

  void IntersectsWhatQueryLB(const ArrayView<envelope_t> queries,
                             result_queue_t& result, int parallelism,
                             cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }

    ArrayView<envelope_t> d_envelopes(envelopes_);
    Stopwatch sw;
    double t_prepare_queries, t_build_bvh, t_forward_trace, t_backward_trace;

    sw.start();
    aabbs_queries_.resize(queries.size());
    thrust::transform(thrust::cuda::par.on(cuda_stream), queries.begin(),
                      queries.end(), aabbs_queries_.begin(),
                      [] __device__(const envelope_t& envelope) {
                        return details::EnvelopeToOptixAabb(envelope);
                      });
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_prepare_queries = sw.ms();

    n_hits.resize(queries.size(), 0);

    details::LaunchParamsIntersectsEnvelope<COORD_T, N_DIMS> params;

    // Query cast rays
    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.queries = queries;
    params.aabbs = ArrayView<OptixAabb>(aabbs_);
    params.aabbs_queries = ArrayView<OptixAabb>(aabbs_queries_);
    params.result = result.DeviceObject();
    params.handle = ias_handle_;
    params.n_hits = thrust::raw_pointer_cast(n_hits.data());

    details::ModuleIdentifier id;
    if (std::is_same<COORD_T, float>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD;
    }

    dim3 dims;

    dims.x = queries.size();
    dims.y = 1;
    dims.z = 1;

    n_hits.resize(params.queries.size(), 0);

    sw.start();
    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_forward_trace = sw.ms();

    {
      pinned_vector<uint32_t> h_n_hits = n_hits;
      uint32_t total_hits = 0, max_hits = 0;

      for (int i = 0; i < h_n_hits.size(); i++) {
        max_hits = std::max(max_hits, h_n_hits[i]);
        total_hits += h_n_hits[i];
      }

      std::cout << "First batch rays " << dims.x << " results size "
                << result.size(cuda_stream) << " total hits " << total_hits
                << " max hits " << max_hits << "\n";
    }
    n_hits.resize(envelopes_.size(), 0);

    int num_handles = parallelism;
    size_t num_queries_per_bvh =
        (aabbs_queries_.size() + num_handles - 1) / num_handles;

    h_backward_prefix_sum_.resize(num_handles + 1, 0);
    backward_gas_buf_.resize(num_handles);
    h_backward_gas_handles_.resize(num_handles);

    Stopwatch sw1;
    sw.start();
    for (int handle_id = 0; handle_id < num_handles; handle_id++) {
      auto begin = handle_id * num_queries_per_bvh;
      auto end = std::min(begin + num_queries_per_bvh, aabbs_queries_.size());
      auto size = end - begin;
      sw1.start();
      h_backward_gas_handles_[handle_id] = rt_engine_.BuildAccelCustom(
          cuda_stream,
          ArrayView<OptixAabb>(
              thrust::raw_pointer_cast(aabbs_queries_.data()) + begin, size),
          backward_gas_buf_[handle_id], true /* prefer_fast_build */);
      sw1.stop();
      h_backward_prefix_sum_[handle_id + 1] =
          h_backward_prefix_sum_[handle_id] + size;
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_build_bvh = sw.ms();

    d_backward_gas_handles_ = h_backward_gas_handles_;
    d_backward_prefix_sum_ = h_backward_prefix_sum_;

    // Ray tracing from base envelopes
    params.envelopes.Swap(params.queries);
    params.aabbs.Swap(params.aabbs_queries);
    params.backward_handles =
        ArrayView<OptixTraversableHandle>(d_backward_gas_handles_);
    params.prefix_sum = ArrayView<size_t>(d_backward_prefix_sum_);
    params.n_hits = thrust::raw_pointer_cast(n_hits.data());
    params.backward_handles =
        ArrayView<OptixTraversableHandle>(d_backward_gas_handles_);

    dims.x = params.queries.size();
    dims.y = num_handles;

    if (std::is_same<COORD_T, float>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD;
    }

    sw.start();
    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_backward_trace = sw.ms();

    {
      pinned_vector<uint32_t> h_n_hits = n_hits;
      uint32_t total_hits = 0, max_hits = 0;

      for (int i = 0; i < h_n_hits.size(); i++) {
        max_hits = std::max(max_hits, h_n_hits[i]);
        total_hits += h_n_hits[i];
      }

      std::cout << "Second batch rays " << dims.x << " results size "
                << result.size(cuda_stream) << " total hits " << total_hits
                << " max hits " << max_hits << "\n";
    }
    std::cout << "Prepare params " << t_prepare_queries << " ms, build BVH "
              << t_build_bvh << " ms, forward " << t_forward_trace
              << " ms, backward " << t_backward_trace << " ms\n";
    //    sw.start();
    //    thrust::sort(thrust::cuda::par.on(cuda_stream), result.data(),
    //                 result.data() + result.size(cuda_stream));
    //    auto end = thrust::unique(thrust::cuda::par.on(cuda_stream),
    //    result.data(),
    //                              result.data() + result.size(cuda_stream));
    //    result.set_size(end - result.data());
    //    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    //    sw.stop();
    //    std::cout << "sort time " << sw.ms() << std::endl;
  }

 private:
  details::RTEngine rt_engine_;
  // for base geometries
  device_uvector<envelope_t> envelopes_;
  pinned_vector<size_t> h_prefix_sum_;
  thrust::device_vector<size_t> d_prefix_sum_;  // prefix sum for each insertion
  Bitset<uint32_t> touched_batch_ids_;
  // Customized AABBs
  device_uvector<OptixAabb> aabbs_;
  // Builtin triangles, contains only
  device_uvector<float3> vertices_;
  device_uvector<uint3> indices_;
  // Intersection only
  std::map<OptixTraversableHandle, thrust::device_vector<unsigned char>>
      as_buf_;
  std::vector<OptixTraversableHandle> gas_handles_;
  std::vector<OptixTraversableHandle> gas_handles_tri_;
  OptixTraversableHandle ias_handle_;
  OptixTraversableHandle ias_handle_tri_;
  // for queries, updated for every batch of queries
  device_uvector<OptixAabb> aabbs_queries_;
  thrust::device_vector<unsigned char> gas_buf_queries_;
  thrust::device_vector<uint32_t> n_hits;

  // backward
  pinned_vector<size_t> h_backward_prefix_sum_;
  thrust::device_vector<size_t> d_backward_prefix_sum_;
  pinned_vector<OptixTraversableHandle> h_backward_gas_handles_;
  thrust::device_vector<OptixTraversableHandle> d_backward_gas_handles_;
  std::vector<thrust::device_vector<unsigned char>> backward_gas_buf_;
};

}  // namespace rtspatial

#endif  // RTSPATIAL_SPATIAL_INDEX_H
