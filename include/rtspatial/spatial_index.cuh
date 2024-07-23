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
#include "rtspatial/utils/event_pool.h"
#include "rtspatial/utils/helpers.h"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/stopwatch.h"

namespace rtspatial {

struct Config {
  std::string ptx_root;
  size_t max_geometries = 1000 * 1000 * 10;
  size_t max_queries = 1000 * 1000;
  bool preallocate = false;
  bool prefer_fast_build_geom = false;
  bool prefer_fast_build_query = true;
};

// Ref: https://arc2r.github.io/book/Spatial_Predicates.html
template <typename COORD_T, int N_DIMS, bool USE_TRIANGLE = false>
class SpatialIndex {
  static_assert(std::is_floating_point<COORD_T>::value,
                "Unsupported COORD_T type");

 public:
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;
  using result_queue_t = Queue<thrust::pair<uint32_t, uint32_t>>;

  void Init(const Config& config) {
    config_ = config;
    details::RTConfig rt_config =
        details::get_default_rt_config(config.ptx_root);

    rt_engine_.Init(rt_config);
    ev_pool_.CacheEvents(500);

    size_t buf_size = 0;

    buf_size += rt_engine_.EstimateMemoryUsageForAABB(config.max_geometries);
    if (USE_TRIANGLE) {
      buf_size +
          rt_engine_.EstimateMemoryUsageForTriangle(config.max_geometries);
    }
    buf_size += rt_engine_.EstimateMemoryUsageForAABB(config.max_queries);
    buf_size += sizeof(OptixAabb) * config.max_queries;
    reuse_buf_.Init(buf_size);
    if (config.preallocate) {
      envelopes_.reserve(config.max_geometries);
      aabbs_.reserve(config.max_geometries);
    }
    Clear();
  }

  void Clear() {
    reuse_buf_.Clear();
    ias_handle_ = 0;
    ias_handle_tri_ = 0;
    handle_to_as_buf_.clear();
    gas_handles_.clear();
    gas_handles_tri_.clear();
    envelopes_.clear();
    h_prefix_sum_.clear();
    h_prefix_sum_.push_back(0);
    d_prefix_sum_.clear();
    touched_batch_ids_.Clear();
    aabbs_.clear();
    vertices_.clear();
    indices_.clear();
  }

  void Insert(ArrayView<envelope_t> envelopes,
              cudaStream_t cuda_stream = nullptr) {
    if (envelopes.empty()) {
      return;
    }
    size_t prev_size = envelopes_.size();

    envelopes_.resize(envelopes_.size() + envelopes.size());
    aabbs_.resize(envelopes_.size());

    thrust::copy(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                 envelopes.end(), envelopes_.begin() + prev_size);

    thrust::transform(thrust::cuda::par.on(cuda_stream), envelopes.begin(),
                      envelopes.end(), aabbs_.begin() + prev_size,
                      [] __device__(const envelope_t& envelope) {
                        return details::EnvelopeToOptixAabb(envelope);
                      });
    // Pop up IAS buffers before building any GAS buffer
    auto it_as_buf = handle_to_as_buf_.find(ias_handle_tri_);

    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    it_as_buf = handle_to_as_buf_.find(ias_handle_);
    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    OptixTraversableHandle handle;
    char* gas_buf;
    size_t as_buf_size;

    if (USE_TRIANGLE) {
      size_t prev_vertices_size = vertices_.size();
      size_t prev_indices_size = indices_.size();

      // 1 aabb consists of two triangles, four vertices
      vertices_.resize(vertices_.size() + envelopes.size() * 4);
      indices_.resize(indices_.size() + envelopes.size() * 2);

      float3* vertices =
          thrust::raw_pointer_cast(vertices_.data()) + prev_vertices_size;
      uint3* indices =
          thrust::raw_pointer_cast(indices_.data()) + prev_indices_size;

      thrust::for_each(thrust::cuda::par.on(cuda_stream),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>(envelopes.size()),
                       [=] __device__(size_t i) mutable {
                         const auto& envelope = envelopes[i];
                         auto aabb = details::EnvelopeToOptixAabb(envelope);
                         details::OptixAabbToTriangles<N_DIMS>(
                             aabb, i, vertices, indices);
                       });

      gas_buf = reuse_buf_.GetDataTail();
      as_buf_size = reuse_buf_.GetOccupiedSize();

      handle = rt_engine_.BuildAccelTriangle(
          cuda_stream,
          ArrayView<float3>(vertices, vertices_.size() - prev_vertices_size),
          ArrayView<uint3>(indices, indices_.size() - prev_indices_size),
          reuse_buf_, config_.prefer_fast_build_geom);

      as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
      handle_to_as_buf_[handle] = std::make_pair(gas_buf, as_buf_size);
      gas_handles_tri_.push_back(handle);
    }

    // Build GAS
    gas_buf = reuse_buf_.GetDataTail();
    as_buf_size = reuse_buf_.GetOccupiedSize();

    handle = rt_engine_.BuildAccelCustom(
        cuda_stream,
        ArrayView<OptixAabb>(
            thrust::raw_pointer_cast(aabbs_.data()) + prev_size,
            envelopes.size()),
        reuse_buf_, config_.prefer_fast_build_geom);

    as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
    handle_to_as_buf_[handle] = std::make_pair(gas_buf, as_buf_size);
    gas_handles_.push_back(handle);

    // Build IAS of the above GAS
    gas_buf = reuse_buf_.GetDataTail();
    as_buf_size = reuse_buf_.GetOccupiedSize();
    ias_handle_ = rt_engine_.BuildInstanceAccel(
        cuda_stream, gas_handles_, reuse_buf_, config_.prefer_fast_build_geom);
    as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
    handle_to_as_buf_[ias_handle_] = std::make_pair(gas_buf, as_buf_size);

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    // Build IAS of the triangles
    if (USE_TRIANGLE) {
      gas_buf = reuse_buf_.GetDataTail();
      as_buf_size = reuse_buf_.GetOccupiedSize();
      ias_handle_tri_ = rt_engine_.BuildInstanceAccel(
          cuda_stream, gas_handles_tri_, reuse_buf_,
          config_.prefer_fast_build_geom);
      as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
      handle_to_as_buf_[ias_handle_tri_] = std::make_pair(gas_buf, as_buf_size);
    }

    h_prefix_sum_.push_back(h_prefix_sum_.back() + envelopes.size());
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
      auto& buf_size = handle_to_as_buf_.at(handle);
      auto begin = h_prefix_sum_[batch_id];
      auto size = h_prefix_sum_[batch_id + 1] - begin;
      size_t offset = buf_size.first - reuse_buf_.GetData();

      auto new_handle = rt_engine_.UpdateAccelCustom(
          stream, handle,
          ArrayView<OptixAabb>(thrust::raw_pointer_cast(aabbs_.data()) + begin,
                               size),
          reuse_buf_, offset, config_.prefer_fast_build_geom);
      // Updating does not change the handle
      assert(new_handle == handle);
    }

    // Update triangle vertices
    if (USE_TRIANGLE) {
      auto* vertices = thrust::raw_pointer_cast(vertices_.data());
      auto* indices = thrust::raw_pointer_cast(indices_.data());

      thrust::for_each(
          thrust::cuda::par.on(stream), updates.begin(), updates.end(),
          [=] __device__(const thrust::pair<size_t, envelope_t>& pair) mutable {
            size_t idx = pair.first;
            auto& aabb = v_aabbs[idx];

            details::OptixAabbToTriangles<N_DIMS>(aabb, idx, vertices, indices);
          });

      for (auto batch_id : touched_batch_ids) {
        auto handle = gas_handles_tri_[batch_id];
        auto& buf_size = handle_to_as_buf_.at(handle);
        auto begin = h_prefix_sum_[batch_id];
        auto size = h_prefix_sum_[batch_id + 1] - begin;
        size_t buf_offset = buf_size.first - reuse_buf_.GetData();

        auto new_handle = rt_engine_.UpdateAccelTriangle(
            stream, ArrayView<float3>(vertices + begin * 4, size * 4),
            ArrayView<uint3>(indices + begin * 2, size * 2), reuse_buf_,
            buf_offset, config_.prefer_fast_build_geom);
        // Updating does not change the handle
        assert(new_handle == handle);
      }
    }

    // Rebuild IAS
    // Pop up IAS buffers before building any GAS buffer
    auto it_as_buf = handle_to_as_buf_.find(ias_handle_tri_);

    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    it_as_buf = handle_to_as_buf_.find(ias_handle_);
    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    char* gas_buf;
    size_t as_buf_size;

    as_buf_size = reuse_buf_.GetOccupiedSize();
    gas_buf = reuse_buf_.GetDataTail();
    ias_handle_ = rt_engine_.BuildInstanceAccel(
        stream, gas_handles_, reuse_buf_, config_.prefer_fast_build_geom);
    as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
    handle_to_as_buf_[ias_handle_] = std::make_pair(gas_buf, as_buf_size);

    if (USE_TRIANGLE) {
      gas_buf = reuse_buf_.GetDataTail();
      as_buf_size = reuse_buf_.GetOccupiedSize();
      ias_handle_tri_ = rt_engine_.BuildInstanceAccel(
          stream, gas_handles_tri_, reuse_buf_, config_.prefer_fast_build_geom);
      as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
      handle_to_as_buf_[ias_handle_tri_] = std::make_pair(gas_buf, as_buf_size);
    }
  }

  void ContainsWhatQuery(ArrayView<point_t> queries, void* arg,
                         cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    details::LaunchParamsContainsPoint<COORD_T, N_DIMS> params;

    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.queries = queries;
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.arg = arg;
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

  void ContainsWhatQuery(ArrayView<envelope_t> queries, void* arg,
                         cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    details::LaunchParamsContainsEnvelope<COORD_T, N_DIMS> params;

    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.queries = queries;
    params.envelopes = ArrayView<envelope_t>(envelopes_);
    params.arg = arg;
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

  void IntersectsWhatQuery(const ArrayView<envelope_t> queries, void* arg,
                           cudaStream_t cuda_stream = nullptr,
                           int parallelism = 500) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }

    ArrayView<envelope_t> d_envelopes(envelopes_);
    details::LaunchParamsIntersectsEnvelope<COORD_T, N_DIMS> params;

    // Query cast rays
    params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
    params.geoms = ArrayView<envelope_t>(envelopes_);
    params.queries = queries;
    params.arg = arg;
    params.handle = ias_handle_;

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

    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);

    size_t curr_buf_size = reuse_buf_.GetOccupiedSize();
    size_t query_aabbs_size = sizeof(OptixAabb) * queries.size();
    ArrayView<OptixAabb> aabbs_queries(
        reinterpret_cast<OptixAabb*>(reuse_buf_.Acquire(query_aabbs_size)),
        queries.size());

    thrust::transform(thrust::cuda::par.on(cuda_stream),
                      thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(queries.size()),
                      aabbs_queries.begin(), [=] __device__(size_t i) mutable {
                        size_t layer = i % parallelism;
                        return details::EnvelopeToOptixAabb(queries[i], layer);
                      });

    OptixTraversableHandle handle = rt_engine_.BuildAccelCustom(
        cuda_stream, aabbs_queries, reuse_buf_, true /* prefer_fast_build */);

    // Ray tracing from base envelopes
    params.handle = handle;
    dims.x = params.geoms.size();
    dims.y = parallelism;

    if (std::is_same<COORD_T, float>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::
          MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD;
    }

    rt_engine_.CopyLaunchParams(cuda_stream, params);
    rt_engine_.Render(cuda_stream, id, dims);
    reuse_buf_.SetTail(curr_buf_size);
  }

 private:
  Config config_;
  details::RTEngine rt_engine_;
  // Data structures for geometries
  /*
   * Buffer structure is like a stack
   * GAS1 GAS2 ... GASn IAS IASTri [Query GAS]
   */
  ReusableBuffer reuse_buf_;
  OptixTraversableHandle ias_handle_, ias_handle_tri_;
  std::map<OptixTraversableHandle, std::pair<char*, size_t>> handle_to_as_buf_;
  std::vector<OptixTraversableHandle> gas_handles_, gas_handles_tri_;

  EventPool ev_pool_;
  device_uvector<envelope_t> envelopes_;
  pinned_vector<size_t> h_prefix_sum_;
  thrust::device_vector<size_t> d_prefix_sum_;  // prefix sum for each insertion
  Bitset<uint32_t> touched_batch_ids_;
  // Customized AABBs
  device_uvector<OptixAabb> aabbs_;
  // Builtin triangles, contains only
  device_uvector<float3> vertices_;
  device_uvector<uint3> indices_;
};

}  // namespace rtspatial

#endif  // RTSPATIAL_SPATIAL_INDEX_H
