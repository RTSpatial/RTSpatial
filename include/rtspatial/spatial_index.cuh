#ifndef RTSPATIAL_SPATIAL_INDEX_H
#define RTSPATIAL_SPATIAL_INDEX_H

#include <thrust/async/transform.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/unique.h>

#include "rtspatial/details/rt_engine.h"
#include "rtspatial/details/sampler.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/bitset.h"
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
  bool prefer_fast_build_query = false;
  // Parallelism prediction
  float geom_sample_rate = 0.2;
  float query_sample_rate = 0.2;
  float intersect_cost_weight = 0.9;
  uint32_t max_geom_samples = 10000;
  uint32_t max_query_samples = 1000;
  uint32_t max_parallelism = 512;
};

namespace details {

template <typename ENVELOPE_T>
__global__ void CalculateNumIntersects(ArrayView<ENVELOPE_T> geoms,
                                       ArrayView<ENVELOPE_T> queries,
                                       uint32_t* n_intersects) {
  auto warp_id = TID_1D / WARP_SIZE;
  auto n_warps = TOTAL_THREADS_1D / WARP_SIZE;
  auto lane_id = threadIdx.x % WARP_SIZE;

  for (uint32_t geom_id = warp_id; geom_id < geoms.size(); geom_id += n_warps) {
    for (uint32_t query_id = lane_id; query_id < queries.size();
         query_id += WARP_SIZE) {
      if (geoms[geom_id].Intersects(queries[query_id])) {
        atomicAdd(n_intersects, 1);
      }
    }
  }
}

}  // namespace details

// Ref: https://arc2r.github.io/book/Spatial_Predicates.html
template <typename COORD_T, int N_DIMS>
class SpatialIndex {
  static_assert(std::is_floating_point<COORD_T>::value,
                "Unsupported COORD_T type");

 public:
  using point_t = Point<COORD_T, N_DIMS>;
  using envelope_t = Envelope<point_t>;

  void Init(const Config& config) {
    config_ = config;
    details::RTConfig rt_config =
        details::get_default_rt_config(config.ptx_root);

    rt_config.n_pipelines = 10;

    rt_engine_.Init(rt_config);

    size_t buf_size = 0;

    buf_size += rt_engine_.EstimateMemoryUsageForAABB(
        config.max_geometries, config.prefer_fast_build_geom);
    buf_size += rt_engine_.EstimateMemoryUsageForAABB(
        config.max_queries, config.prefer_fast_build_query);
    // FIXME: Reserve space to IAS, implement EstimateMemoryUsageIAS
    buf_size *= 1.1;
    buf_size += sizeof(OptixAabb) * config.max_queries;

    reuse_buf_.Init(buf_size);
    if (config.preallocate) {
      envelopes_.reserve(config.max_geometries);
      aabbs_.reserve(config.max_geometries);
    }

    geom_samples_.resize(config.max_geom_samples);
    query_samples_.resize(config.max_query_samples);
    sampler_.Init(std::max(config.max_geom_samples, config.max_query_samples));
    Clear();
  }

  void Clear() {
    reuse_buf_.Clear();
    ias_handle_ = 0;
    handle_to_as_buf_.clear();
    gas_handles_.clear();
    envelopes_.clear();
    h_prefix_sum_.clear();
    h_prefix_sum_.push_back(0);
    d_prefix_sum_.clear();
    touched_batch_ids_.Clear();
    aabbs_.clear();
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

    auto it_as_buf = handle_to_as_buf_.find(ias_handle_);
    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    OptixTraversableHandle handle;
    char* gas_buf;
    size_t as_buf_size;

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

    h_prefix_sum_.push_back(h_prefix_sum_.back() + envelopes.size());
    d_prefix_sum_ = h_prefix_sum_;
  }

  void Delete(const ArrayView<size_t> envelope_ids, cudaStream_t stream) {
    touched_batch_ids_.Init(d_prefix_sum_.size() - 1);
    ArrayView<envelope_t> v_envelopes(envelopes_);
    ArrayView<OptixAabb> v_aabbs(aabbs_);
    ArrayView<size_t> v_prefix_sum(d_prefix_sum_);
    auto v_touched_batch_ids = touched_batch_ids_.DeviceObject();
    size_t max_id = h_prefix_sum_.back();

    thrust::for_each(thrust::cuda::par.on(stream), envelope_ids.begin(),
                     envelope_ids.end(), [=] __device__(size_t idx) mutable {
                       auto& envelope = v_envelopes[idx];

                       assert(idx < max_id);
                       envelope.Invalid();  // Turn into the degenerate case
                       v_aabbs[idx] = details::EnvelopeToOptixAabb(envelope);

                       auto it = thrust::upper_bound(thrust::seq,
                                                     v_prefix_sum.begin(),
                                                     v_prefix_sum.end(), idx);
                       assert(it != v_prefix_sum.end());
                       auto batch_id = v_prefix_sum.end() - it - 1;
                       assert(batch_id >= 0 && batch_id < v_prefix_sum.size());
                       v_touched_batch_ids.set_bit_atomic(batch_id);
                     });

    auto touched_batch_ids = touched_batch_ids_.DumpPositives(stream);

    // Update GASs
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

    // Update IAS
    auto it_as_buf = handle_to_as_buf_.find(ias_handle_);

    if (it_as_buf != handle_to_as_buf_.end()) {
      auto& buf_size = it_as_buf->second;
      size_t offset = buf_size.first - reuse_buf_.GetData();

      // Handle should not be changed
      auto new_handle = rt_engine_.UpdateInstanceAccel(
          stream, gas_handles_, reuse_buf_, offset,
          config_.prefer_fast_build_geom);
      assert(new_handle == ias_handle_);
    }
  }

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

    // Update IAS
    auto it_as_buf = handle_to_as_buf_.find(ias_handle_);

    if (it_as_buf != handle_to_as_buf_.end()) {
      auto& buf_size = it_as_buf->second;
      size_t offset = buf_size.first - reuse_buf_.GetData();

      // Handle should not be changed
      auto new_handle = rt_engine_.UpdateInstanceAccel(
          stream, gas_handles_, reuse_buf_, offset,
          config_.prefer_fast_build_geom);
      assert(new_handle == ias_handle_);
    }
  }

  /**
   * Return the geometries in the index that contains the query points
   * @param queries Query points
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
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
    params.handle = ias_handle_;

    rt_engine_.CopyLaunchParams(cuda_stream, params);
    details::ModuleIdentifier id =
        details::ModuleIdentifier::NUM_MODULE_IDENTIFIERS;

    if (std::is_same<COORD_T, float>::value) {
      id = details::ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D;
    } else if (std::is_same<COORD_T, double>::value) {
      id = details::ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D;
    }

    dim3 dims;

    dims.x = queries.size();
    dims.y = 1;
    dims.z = 1;

    rt_engine_.Render(cuda_stream, id, dims);
  }

  /**
   * Return the geometries in the index that contains the query envelopes
   * @param queries Query envelopes
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
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

  /**
   * Return the geometries in the index that intersects the query envelopes
   * @param queries Query envelopes
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
  void IntersectsWhatQuery(ArrayView<envelope_t> queries, void* arg,
                           cudaStream_t cuda_stream = nullptr,
                           int parallelism = 32) {
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

    OptixTraversableHandle handle =
        rt_engine_.BuildAccelCustom(cuda_stream, aabbs_queries, reuse_buf_,
                                    config_.prefer_fast_build_query);

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

  /**
   * Return the geometries in the index that intersects the query envelopes
   * @param queries Query envelopes
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
  void IntersectsWhatQueryProfiling(
      ArrayView<envelope_t> queries,
      dev::Queue<thrust::pair<uint32_t, uint32_t>>* arg,
      cudaStream_t cuda_stream = nullptr, int parallelism = 32) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }

    Stopwatch sw;
    pinned_vector<uint32_t> h_size(1);
    uint32_t* p_size = thrust::raw_pointer_cast(h_size.data());
    size_t n_results_forward, n_results_backward;
    double t_forward_ms, t_bvh_ms, t_backward_ms;
    sw.start();

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
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_forward_ms = sw.ms();

    thrust::for_each(
        thrust::device, arg, arg + 1,
        [=] __device__(dev::Queue<thrust::pair<uint32_t, uint32_t>> & result) {
          *p_size = result.size();
        });

    n_results_forward = *p_size;

    sw.start();
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

    OptixTraversableHandle handle =
        rt_engine_.BuildAccelCustom(cuda_stream, aabbs_queries, reuse_buf_,
                                    config_.prefer_fast_build_query);

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();
    t_bvh_ms = sw.ms();

    sw.start();
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
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    sw.stop();

    t_backward_ms = sw.ms();

    thrust::for_each(
        thrust::device, arg, arg + 1,
        [=] __device__(dev::Queue<thrust::pair<uint32_t, uint32_t>> & result) {
          *p_size = result.size();
        });

    n_results_backward = *p_size - n_results_forward;

    std::cout << "Forward pass " << t_forward_ms << " ms, results "
              << n_results_forward << " BVH " << t_bvh_ms
              << " ms, Backward pass " << t_backward_ms << " ms, results "
              << n_results_backward << std::endl;
  }

  int CalculateBestParallelism(ArrayView<envelope_t> queries,
                               cudaStream_t cuda_stream = nullptr) {
    int best_parallelism = 1;
    auto n_geoms = envelopes_.size();
    auto n_queries = queries.size();

    if (n_geoms == 0 || n_queries == 0) {
      return best_parallelism;
    }

    ArrayView<envelope_t> v_envelopes(envelopes_);

    uint32_t geom_sample_size =
        std::min(config_.max_geom_samples,
                 std::max(1u, (uint32_t) (n_geoms * config_.geom_sample_rate)));
    uint32_t query_sample_size = std::min(
        config_.max_query_samples,
        std::max(1u, (uint32_t) (n_queries * config_.query_sample_rate)));

    sampler_.Sample(cuda_stream, v_envelopes, geom_sample_size, geom_samples_);
    sampler_.Sample(cuda_stream, queries, query_sample_size, query_samples_);

    ArrayView<envelope_t> v_geom_samples(geom_samples_);
    ArrayView<envelope_t> v_query_samples(query_samples_);

    int grid_size, block_size;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &grid_size, &block_size, details::CalculateNumIntersects<envelope_t>, 0,
        reinterpret_cast<int>(MAX_BLOCK_SIZE)));

    // Outer loop takes the larger size for high parallelism
    if (v_query_samples.size() > v_geom_samples.size()) {
      std::swap(v_geom_samples, v_query_samples);
    }

    sampled_n_intersects_.set(cuda_stream, 0);

    details::CalculateNumIntersects<<<grid_size, block_size, 0, cuda_stream>>>(
        v_geom_samples, v_query_samples, sampled_n_intersects_.data());

    auto geom_sample_rate = (double) geom_sample_size / n_geoms;
    auto query_sample_rate = (double) query_sample_size / n_queries;
    auto predicated_n_intersects = sampled_n_intersects_.get(cuda_stream) /
                                   geom_sample_rate / query_sample_rate;
    auto selectivity = (double) predicated_n_intersects / (n_geoms * n_queries);

    auto min_cost = std::numeric_limits<double>::max();
    int parallelism = 1;

    while (parallelism < config_.max_parallelism) {
      double per_ray_search_costs = log10(n_queries);
      double cast_rays_costs = n_geoms * parallelism * per_ray_search_costs;
      double intersect_costs = n_geoms * n_queries * selectivity / parallelism;
      double cost = (1 - config_.intersect_cost_weight) * cast_rays_costs +
                    config_.intersect_cost_weight * intersect_costs;

      if (cost < min_cost) {
        min_cost = cost;
        best_parallelism = parallelism;
      }
      parallelism *= 2;
    }

    return best_parallelism;
  }

 private:
  Stream extra_stream_;
  Config config_;
  details::RTEngine rt_engine_;
  // Data structures for geometries
  /*
   * Buffer structure is like a stack
   * GAS1 GAS2 ... GASn IAS [Query GAS]
   */
  ReusableBuffer reuse_buf_;
  OptixTraversableHandle ias_handle_;
  std::map<OptixTraversableHandle, std::pair<char*, size_t>> handle_to_as_buf_;
  std::vector<OptixTraversableHandle> gas_handles_;

  device_uvector<envelope_t> envelopes_;
  pinned_vector<size_t> h_prefix_sum_;
  thrust::device_vector<size_t> d_prefix_sum_;  // prefix sum for each insertion
  Bitset<uint32_t> touched_batch_ids_;
  // Customized AABBs
  device_uvector<OptixAabb> aabbs_;

  // Cost estimation
  Sampler sampler_;
  thrust::device_vector<envelope_t> geom_samples_;
  thrust::device_vector<envelope_t> query_samples_;
  SharedValue<uint32_t> sampled_n_intersects_;
};

}  // namespace rtspatial

#endif  // RTSPATIAL_SPATIAL_INDEX_H
