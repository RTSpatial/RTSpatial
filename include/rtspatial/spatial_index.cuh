#ifndef RTSPATIAL_SPATIAL_INDEX_H
#define RTSPATIAL_SPATIAL_INDEX_H

#include <thrust/async/transform.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/unique.h>

#include "rtspatial/details/rt_engine.h"
#include "rtspatial/details/sampler.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/line.cuh"
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
  bool compact = false;
  // Parallelism prediction
  float geom_sample_rate = 0.2;
  float query_sample_rate = 0.2;
  float intersect_cost_weight = 0.95;
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
  using line_t = Line<point_t>;

  void Init(const Config& config) {
    config_ = config;
    details::RTConfig rt_config =
        details::get_default_rt_config(config.ptx_root);

    rt_config.n_pipelines = 10;

    rt_engine_.Init(rt_config);

    size_t buf_size = 0;

    buf_size += rt_engine_.EstimateMemoryUsageForAABB(
        config.max_geometries, config.prefer_fast_build_geom, config.compact);
    buf_size += rt_engine_.EstimateMemoryUsageForAABB(
        config.max_queries, config.prefer_fast_build_query, false);
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
        reuse_buf_, config_.prefer_fast_build_geom, config_.compact);

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

    thrust::for_each(
        thrust::cuda::par.on(stream), envelope_ids.begin(), envelope_ids.end(),
        [=] __device__(size_t idx) mutable {
          if (idx < v_envelopes.size()) {
            auto& envelope = v_envelopes[idx];

            assert(idx < max_id);
            envelope.Invalid();  // Turn into the degenerate case
            v_aabbs[idx] = details::EnvelopeToOptixAabb(envelope);

            auto it = thrust::upper_bound(thrust::seq, v_prefix_sum.begin(),
                                          v_prefix_sum.end(), idx);
            assert(it != v_prefix_sum.end());
            auto batch_id = v_prefix_sum.end() - it - 1;
            assert(batch_id >= 0 && batch_id < v_prefix_sum.size());
            v_touched_batch_ids.set_bit_atomic(batch_id);
          }
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

    if (!touched_batch_ids.empty()) {
      // Update IAS
      auto as_buf = handle_to_as_buf_.at(ias_handle_);
      auto& buf_size = as_buf.second;
      size_t offset = as_buf.first - reuse_buf_.GetData();

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

      std::cout << "Update begin\n";
      auto new_handle = rt_engine_.UpdateAccelCustom(
          stream, handle,
          ArrayView<OptixAabb>(thrust::raw_pointer_cast(aabbs_.data()) + begin,
                               size),
          reuse_buf_, offset, config_.prefer_fast_build_geom, config_.compact);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::cout << "Update end\n";
      // Updating does not change the handle
      assert(new_handle == handle);
    }

    // Update IAS
    if (!touched_batch_ids.empty()) {
      // Update IAS
      auto as_buf = handle_to_as_buf_.at(ias_handle_);
      auto& buf_size = as_buf.second;
      size_t offset = as_buf.first - reuse_buf_.GetData();

      // Handle should not be changed
      auto new_handle = rt_engine_.UpdateInstanceAccel(
          stream, gas_handles_, reuse_buf_, offset,
          config_.prefer_fast_build_geom);
      assert(new_handle == ias_handle_);
    }
  }

  /**
   * Return the geometries in the index that satisfy p with the queries
   * @param p predicate
   * @param queries query points
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
  void Query(Predicate p, ArrayView<point_t> queries, void* arg,
             cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    switch (p) {
    case Predicate::kContains: {
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
        id =
            details::ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D;
      }

      dim3 dims;

      dims.x = queries.size();
      dims.y = 1;
      dims.z = 1;

      rt_engine_.Render(cuda_stream, id, dims);
      break;
    }
    default:
      std::cerr << "Invalid predicate" << std::endl;
      abort();
    }
  }

  /**
   * Return the geometries in the index that satisfy p with the queries
   * @param p predicate
   * @param queries Query envelopes
   * @param arg argument passing into the callback handler
   * @param cuda_stream CUDA stream
   */
  void Query(Predicate p, ArrayView<envelope_t> queries, void* arg,
             cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }

    switch (p) {
    case Predicate::kContains: {
      details::LaunchParamsContainsEnvelope<COORD_T, N_DIMS> params;

      params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
      params.queries = queries;
      params.envelopes = ArrayView<envelope_t>(envelopes_);
      params.arg = arg;
      params.handle = ias_handle_;

      rt_engine_.CopyLaunchParams(cuda_stream, params);
      details::ModuleIdentifier id;
      if (std::is_same<COORD_T, float>::value) {
        id = details::ModuleIdentifier::
            MODULE_ID_FLOAT_CONTAINS_ENVELOPE_QUERY_2D;
      } else if (std::is_same<COORD_T, double>::value) {
        id = details::ModuleIdentifier::
            MODULE_ID_DOUBLE_CONTAINS_ENVELOPE_QUERY_2D;
      }

      dim3 dims;

      dims.x = queries.size();
      dims.y = 1;
      dims.z = 1;

      rt_engine_.Render(cuda_stream, id, dims);
      break;
    }

    case Predicate::kIntersects: {
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
      int parallelism = CalculateBestParallelism(queries, cuda_stream);
      size_t curr_buf_size = reuse_buf_.GetOccupiedSize();
      size_t query_aabbs_size = sizeof(OptixAabb) * queries.size();
      ArrayView<OptixAabb> aabbs_queries(
          reinterpret_cast<OptixAabb*>(reuse_buf_.Acquire(query_aabbs_size)),
          queries.size());

      thrust::transform(
          thrust::cuda::par.on(cuda_stream),
          thrust::make_counting_iterator<size_t>(0),
          thrust::make_counting_iterator<size_t>(queries.size()),
          aabbs_queries.begin(), [=] __device__(size_t i) mutable {
            size_t layer = i % parallelism;
            return details::EnvelopeToOptixAabb(queries[i], layer);
          });

      OptixTraversableHandle handle =
          rt_engine_.BuildAccelCustom(cuda_stream, aabbs_queries, reuse_buf_,
                                      config_.prefer_fast_build_query, false);

      // Ray tracing from base envelopes
      params.handle = handle;
      dims.x = params.geoms.size();
      dims.y = parallelism;
      uint32_t max_size = 1 << 30;

      if (dims.x * dims.y > max_size) {
        dims.x = max_size / dims.y;
      }

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
      break;
    }
    }
  }

  void Query(Predicate p, ArrayView<line_t> queries, void* arg,
             cudaStream_t cuda_stream = nullptr) {
    if (queries.empty() || envelopes_.empty()) {
      return;
    }
    ArrayView<envelope_t> d_envelopes(envelopes_);

    switch (p) {
    case Predicate::kContains: {
      break;
    }
    case Predicate::kIntersects: {
      details::LaunchParamsIntersectsLine<COORD_T, N_DIMS> params;

      // Query cast rays
      params.prefix_sum = ArrayView<size_t>(d_prefix_sum_);
      params.geoms = d_envelopes;
      params.queries = queries;
      params.arg = arg;
      params.handle = ias_handle_;

      details::ModuleIdentifier id;
      if (std::is_same<COORD_T, float>::value) {
        id =
            details::ModuleIdentifier::MODULE_ID_FLOAT_INTERSECTS_LINE_QUERY_2D;
      } else if (std::is_same<COORD_T, double>::value) {
        id = details::ModuleIdentifier::
            MODULE_ID_DOUBLE_INTERSECTS_LINE_QUERY_2D;
      }

      dim3 dims;

      dims.x = queries.size();
      dims.y = 1;
      dims.z = 1;

      rt_engine_.CopyLaunchParams(cuda_stream, params);
      rt_engine_.Render(cuda_stream, id, dims);
      break;
    }
    }
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
                                    config_.prefer_fast_build_query, false);

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
//    std::cout << "predicated selectivity " << selectivity << std::endl;

    while (parallelism < config_.max_parallelism) {
      double per_ray_search_costs = log10(n_queries);
      double cast_rays_costs = n_geoms * parallelism * per_ray_search_costs;
      double intersect_costs = n_geoms * n_queries * selectivity / parallelism;
      double cost = (1 - config_.intersect_cost_weight) * cast_rays_costs +
                    config_.intersect_cost_weight * intersect_costs;

//      std::cout << "curr cost " << min_cost << " cost " << cost
//                << " parallelsim " << parallelism << " cast cost " << cast_rays_costs << " intersect cost " << intersect_costs << std::endl;
      if (cost < min_cost) {
        min_cost = cost;
        best_parallelism = parallelism;
      }
      parallelism *= 2;
    }

    return best_parallelism;
  }

  void Optimize(cudaStream_t cuda_stream = nullptr) {
    auto gas_num = gas_handles_.size();
    if (gas_num == 0) {
      return;
    }

    size_t total_geoms = h_prefix_sum_.back();
    size_t avg_geoms = (total_geoms + gas_num - 1) / gas_num;
    size_t merge_threshold = avg_geoms * 1.5;
    auto partitions = Optimize(0, gas_num, merge_threshold);

    auto size_list = std::get<0>(partitions);
    auto buf_list = std::get<1>(partitions);
    auto handle_list = std::get<2>(partitions);
    std::vector<size_t> prefix_sum;
    std::vector<OptixTraversableHandle> gas_handles;
    size_t curr_geoms = 0;

    prefix_sum.push_back(curr_geoms);

    for (size_t i = 0; i < size_list.size(); i++) {
      auto n_geoms = size_list[i];
      auto handle = handle_list[i];
      // Need rebuild
      if (handle == 0) {
        const auto& as_buf = buf_list[i];
        handle = rt_engine_.BuildAccelCustom(
            cuda_stream,
            ArrayView<OptixAabb>(
                thrust::raw_pointer_cast(aabbs_.data()) + curr_geoms, n_geoms),
            as_buf.first, as_buf.second, reuse_buf_,
            config_.prefer_fast_build_geom);
        handle_to_as_buf_[handle] = as_buf;
      }

      gas_handles.push_back(handle);
      curr_geoms += size_list[i];
      prefix_sum.push_back(curr_geoms);
    }

    h_prefix_sum_ = prefix_sum;
    d_prefix_sum_ = h_prefix_sum_;
    gas_handles_ = gas_handles;

    // Pop up IAS buffers before building any GAS buffer
    auto it_as_buf = handle_to_as_buf_.find(ias_handle_);
    if (it_as_buf != handle_to_as_buf_.end()) {
      reuse_buf_.Release(it_as_buf->second.second);
      handle_to_as_buf_.erase(it_as_buf);
    }

    // Build IAS of the above GAS
    auto gas_buf = reuse_buf_.GetDataTail();
    auto as_buf_size = reuse_buf_.GetOccupiedSize();
    ias_handle_ = rt_engine_.BuildInstanceAccel(
        cuda_stream, gas_handles_, reuse_buf_, config_.prefer_fast_build_geom);
    as_buf_size = reuse_buf_.GetOccupiedSize() - as_buf_size;
    handle_to_as_buf_[ias_handle_] = std::make_pair(gas_buf, as_buf_size);
  }

  std::tuple<std::vector<size_t>, std::vector<std::pair<char*, size_t>>,
             std::vector<OptixTraversableHandle>>
  Optimize(size_t gas_begin, size_t gas_end, size_t merge_threshold) {
    if (gas_begin >= gas_end) {
      return std::make_tuple(std::vector<size_t>{},
                             std::vector<std::pair<char*, size_t>>{},
                             std::vector<OptixTraversableHandle>{});
    }

    auto gas_num = gas_end - gas_begin;

    if (gas_num == 1) {
      auto n_geoms1 = h_prefix_sum_[gas_begin + 1] - h_prefix_sum_[gas_begin];
      auto handle = gas_handles_[gas_begin];
      auto buf = handle_to_as_buf_.at(handle);

      return std::make_tuple(std::vector<size_t>{n_geoms1},
                             std::vector<std::pair<char*, size_t>>{buf},
                             std::vector<OptixTraversableHandle>{handle});
    } else if (gas_num == 2) {
      auto n_geoms1 = h_prefix_sum_[gas_begin + 1] - h_prefix_sum_[gas_begin];
      auto n_geoms2 =
          h_prefix_sum_[gas_begin + 2] - h_prefix_sum_[gas_begin + 1];
      auto handle1 = gas_handles_[gas_begin];
      auto handle2 = gas_handles_[gas_begin + 1];
      auto as_buf1 = handle_to_as_buf_.at(handle1);
      auto as_buf2 = handle_to_as_buf_.at(handle2);

      if (n_geoms1 <= merge_threshold || n_geoms2 <= merge_threshold) {
        auto as_buf =
            std::make_pair(as_buf1.first, as_buf1.second + as_buf2.second);
        handle_to_as_buf_.erase(handle1);
        handle_to_as_buf_.erase(handle2);

        return std::make_tuple(std::vector<size_t>{n_geoms1 + n_geoms2},
                               std::vector<std::pair<char*, size_t>>{as_buf},
                               std::vector<OptixTraversableHandle>{0});
      } else {
        return std::make_tuple(
            std::vector<size_t>{n_geoms1, n_geoms2},
            std::vector<std::pair<char*, size_t>>{as_buf1, as_buf2},
            std::vector<OptixTraversableHandle>{handle1, handle2});
      }
    }

    auto mid = (gas_begin + gas_end) / 2;
    auto left = Optimize(gas_begin, mid, merge_threshold);
    auto right = Optimize(mid, gas_end, merge_threshold);

    std::vector<size_t> size_list = std::get<0>(left);
    size_list.insert(size_list.end(), std::get<0>(right).begin(),
                     std::get<0>(right).end());

    std::vector<std::pair<char*, size_t>> buffer_list = std::get<1>(left);
    buffer_list.insert(buffer_list.end(), std::get<1>(right).begin(),
                       std::get<1>(right).end());

    std::vector<OptixTraversableHandle> handle_list = std::get<2>(left);
    handle_list.insert(handle_list.end(), std::get<2>(right).begin(),
                       std::get<2>(right).end());

    return std::make_tuple(size_list, buffer_list, handle_list);
  }

  void PrintMemoryUsage() {
    uint32_t bvh_buf_mb = reuse_buf_.GetCapacity() / 1024 / 1024;
    uint32_t geometries_mb =
        envelopes_.size() * sizeof(envelope_t) / 1024 / 1024;
    uint32_t aabbs_mb = aabbs_.size() * sizeof(OptixAabb) / 1024 / 1024;
    uint32_t total_mb = bvh_buf_mb + geometries_mb + aabbs_mb;
    std::cout << "BVH " << bvh_buf_mb << " MB, "
              << "Geometries " << geometries_mb << " MB, "
              << "AABBs " << aabbs_mb << " MB, "
              << "Total " << total_mb << " MB" << std::endl;
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
