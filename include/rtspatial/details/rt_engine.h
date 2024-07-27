#ifndef RTSPATIAL_DETAILS_RT_ENGINE_H
#define RTSPATIAL_DETAILS_RT_ENGINE_H
#include <cuda.h>
#include <optix_types.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtspatial/details/launch_parameters.h"
#include "rtspatial/details/reusable_buffer.h"
#include "rtspatial/details/sbt_record.h"
#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/shared_value.h"

#define MODULE_ENABLE_MISS (1 << 0)
#define MODULE_ENABLE_CH (1 << 1)
#define MODULE_ENABLE_AH (1 << 2)
#define MODULE_ENABLE_IS (1 << 3)

namespace rtspatial {
namespace details {
enum ModuleIdentifier {
  MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D,
  MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D,
  MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D_TRIANGLE,
  MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D_TRIANGLE,
  MODULE_ID_FLOAT_CONTAINS_ENVELOPE_QUERY_2D,
  MODULE_ID_DOUBLE_CONTAINS_ENVELOPE_QUERY_2D,
  MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD,
  MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD,
  MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD,
  MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD,
  NUM_MODULE_IDENTIFIERS
};
static const float IDENTICAL_TRANSFORMATION_MTX[12] = {
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

class Module {
 public:
  Module()
      : enabled_module_(0), n_payload_(0), n_attribute_(0) {}

  explicit Module(ModuleIdentifier id)
      : id_(id), enabled_module_(0), n_payload_(0), n_attribute_(0) {}

  void EnableMiss() { enabled_module_ |= MODULE_ENABLE_MISS; }

  void EnableClosestHit() { enabled_module_ |= MODULE_ENABLE_CH; }

  void EnableAnyHit() { enabled_module_ |= MODULE_ENABLE_AH; }

  void EnableIsIntersection() { enabled_module_ |= MODULE_ENABLE_IS; }

  bool IsMissEnable() const { return enabled_module_ & MODULE_ENABLE_MISS; }

  bool IsClosestHitEnable() const { return enabled_module_ & MODULE_ENABLE_CH; }

  bool IsAnyHitEnable() const { return enabled_module_ & MODULE_ENABLE_AH; }

  bool IsIsIntersectionEnabled() const {
    return enabled_module_ & MODULE_ENABLE_IS;
  }

  void set_id(ModuleIdentifier id) { id_ = id; }

  void set_program_path(const std::string& program_path) {
    program_path_ = program_path;
  }
  const std::string& get_program_path() const { return program_path_; }

  void set_function_suffix(const std::string& function_suffix) {
    function_suffix_ = function_suffix;
  }
  const std::string& get_function_suffix() const { return function_suffix_; }

  void set_n_payload(int n_payload) { n_payload_ = n_payload; }

  int get_n_payload() const { return n_payload_; }

  void set_n_attribute(int n_attribute) { n_attribute_ = n_attribute; }

  int get_n_attribute() const { return n_attribute_; }

  ModuleIdentifier get_id() const { return id_; }

 private:
  ModuleIdentifier id_;
  std::string program_path_;
  std::string function_suffix_;
  int enabled_module_;

  int n_payload_;
  int n_attribute_;
};

struct RTConfig {
  RTConfig()
      : max_reg_count(0),
        max_traversable_depth(2),  // IAS+GAS
        max_trace_depth(2),
        logCallbackLevel(1),
        opt_level(OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        dbg_level(OPTIX_COMPILE_DEBUG_LEVEL_NONE),
        n_pipelines(1) {}

  void AddModule(const Module& mod) {
    if (access(mod.get_program_path().c_str(), R_OK) != 0) {
      std::cerr << "Error: cannot open " << mod.get_program_path() << std::endl;
      abort();
    }

    modules[mod.get_id()] = mod;
  }

  int max_reg_count;
  int max_traversable_depth;
  int max_trace_depth;
  int logCallbackLevel;
  OptixCompileOptimizationLevel opt_level;
  OptixCompileDebugLevel dbg_level;
  std::map<ModuleIdentifier, Module> modules;
  int n_pipelines;
};

RTConfig get_default_rt_config(const std::string& ptx_root);

class RTEngine {
 public:
  RTEngine() = default;

  void Init(const RTConfig& config) {
    initOptix(config);
    createContext();
    createModule(config);
    createRaygenPrograms(config);
    createMissPrograms(config);
    createHitgroupPrograms(config);
    createPipeline(config);
    buildSBT(config);
  }

  OptixTraversableHandle BuildAccelCustom(cudaStream_t cuda_stream,
                                          ArrayView<OptixAabb> aabbs,
                                          ReusableBuffer& buf,
                                          bool prefer_fast_build = false) {
    return buildAccel(cuda_stream, aabbs, buf, prefer_fast_build);
  }

  OptixTraversableHandle UpdateAccelCustom(cudaStream_t cuda_stream,
                                           OptixTraversableHandle handle,
                                           ArrayView<OptixAabb> aabbs,
                                           ReusableBuffer& buf,
                                           size_t buf_offset,
                                           bool prefer_fast_build = false) {
    return updateAccel(cuda_stream, handle, aabbs, buf, buf_offset,
                       prefer_fast_build);
  }

  OptixTraversableHandle BuildAccelTriangle(cudaStream_t cuda_stream,
                                            ArrayView<float3> vertices,
                                            ArrayView<uint3> indices,
                                            ReusableBuffer& buf,
                                            bool prefer_fast_build = false) {
    return buildAccelTriangle(cuda_stream, vertices, indices, buf,
                              prefer_fast_build);
  }

  OptixTraversableHandle UpdateAccelTriangle(cudaStream_t cuda_stream,
                                             ArrayView<float3> vertices,
                                             ArrayView<uint3> indices,
                                             ReusableBuffer& buf,
                                             size_t buf_offset,
                                             bool prefer_fast_build = false) {
    return updateAccelTriangle(cuda_stream, vertices, indices, buf, buf_offset,
                               prefer_fast_build);
  }

  OptixTraversableHandle BuildInstanceAccel(
      cudaStream_t cuda_stream, std::vector<OptixTraversableHandle>& handles,
      ReusableBuffer& buf, bool prefer_fast_build = false) {
    tmp_h_instances_.resize(handles.size());
    tmp_instances_.resize(handles.size());

    for (size_t i = 0; i < handles.size(); i++) {
      tmp_h_instances_[i].instanceId = i;
      memcpy(tmp_h_instances_[i].transform, IDENTICAL_TRANSFORMATION_MTX,
             sizeof(float) * 12);
      tmp_h_instances_[i].traversableHandle = handles[i];
      tmp_h_instances_[i].sbtOffset = 0;
      tmp_h_instances_[i].visibilityMask = 255;
      tmp_h_instances_[i].flags = OPTIX_INSTANCE_FLAG_NONE;
    }

    thrust::copy(thrust::cuda::par.on(cuda_stream), tmp_h_instances_.begin(),
                 tmp_h_instances_.end(), tmp_instances_.begin());

    return buildInstanceAccel(cuda_stream,
                              ArrayView<OptixInstance>(tmp_instances_), buf,
                              prefer_fast_build);
  }

  OptixTraversableHandle UpdateInstanceAccel(
      cudaStream_t cuda_stream, std::vector<OptixTraversableHandle>& handles,
      ReusableBuffer& buf, size_t buf_offset, bool prefer_fast_build = false) {
    tmp_h_instances_.resize(handles.size());
    tmp_instances_.resize(handles.size());

    for (size_t i = 0; i < handles.size(); i++) {
      tmp_h_instances_[i].instanceId = i;
      memcpy(tmp_h_instances_[i].transform, IDENTICAL_TRANSFORMATION_MTX,
             sizeof(float) * 12);
      tmp_h_instances_[i].traversableHandle = handles[i];
      tmp_h_instances_[i].sbtOffset = 0;
      tmp_h_instances_[i].visibilityMask = 255;
      tmp_h_instances_[i].flags = OPTIX_INSTANCE_FLAG_NONE;
    }
    thrust::copy(thrust::cuda::par.on(cuda_stream), tmp_h_instances_.begin(),
                 tmp_h_instances_.end(), tmp_instances_.begin());

    return updateInstanceAccel(cuda_stream,
                               ArrayView<OptixInstance>(tmp_instances_), buf,
                               buf_offset, prefer_fast_build);
  }

  void Render(cudaStream_t cuda_stream, ModuleIdentifier mod, dim3 dim);

  template <typename T>
  void CopyLaunchParams(cudaStream_t cuda_stream, const T& params) {
    params_size_ = sizeof(params);
    if (params_size_ > h_launch_params_.size()) {
      h_launch_params_.resize(params_size_);
      launch_params_.resize(h_launch_params_.size());
    }
    *reinterpret_cast<T*>(thrust::raw_pointer_cast(h_launch_params_.data())) =
        params;
    CUDA_CHECK(
        cudaMemcpyAsync(thrust::raw_pointer_cast(launch_params_.data()),
                        thrust::raw_pointer_cast(h_launch_params_.data()),
                        params_size_, cudaMemcpyHostToDevice, cuda_stream));
  }

  OptixDeviceContext get_context() const { return optix_context_; }

  size_t EstimateMemoryUsageForAABB(size_t num_aabbs,
                                    bool prefer_fast_build);

  size_t EstimateMemoryUsageForTriangle(size_t num_aabbs,
                                        bool prefer_fast_build);

 private:
  void initOptix(const RTConfig& config);

  void createContext();

  void createModule(const RTConfig& config);


  void createRaygenPrograms(const RTConfig& config);

  void createMissPrograms(const RTConfig& config);

  void createHitgroupPrograms(const RTConfig& config);

  void createPipeline(const RTConfig& config);

  void buildSBT(const RTConfig& config);

  OptixTraversableHandle buildAccel(cudaStream_t cuda_stream,
                                    ArrayView<OptixAabb> aabbs,
                                    ReusableBuffer& buf,
                                    bool prefer_fast_build);

  OptixTraversableHandle updateAccel(cudaStream_t cuda_stream,
                                     OptixTraversableHandle handle,
                                     ArrayView<OptixAabb> aabbs,
                                     ReusableBuffer& buf, size_t buf_offset,
                                     bool prefer_fast_build);

  OptixTraversableHandle buildAccelTriangle(cudaStream_t cuda_stream,
                                            ArrayView<float3> vertices,
                                            ArrayView<uint3> indices,
                                            ReusableBuffer& buf,
                                            bool prefer_fast_build);

  OptixTraversableHandle updateAccelTriangle(cudaStream_t cuda_stream,
                                             ArrayView<float3> vertices,
                                             ArrayView<uint3> indices,
                                             ReusableBuffer& buf,
                                             size_t buf_offset,
                                             bool prefer_fast_build);

  OptixTraversableHandle buildInstanceAccel(cudaStream_t cuda_stream,
                                            ArrayView<OptixInstance> instances,
                                            ReusableBuffer& buf,
                                            bool prefer_fast_build);

  OptixTraversableHandle updateInstanceAccel(cudaStream_t cuda_stream,
                                             ArrayView<OptixInstance> instances,
                                             ReusableBuffer& buf,
                                             size_t buf_offset,
                                             bool prefer_fast_build);

  static size_t getAccelAlignedSize(size_t size) {
    if (size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0) {
      return size;
    }

    return size - size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT +
           OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
  }

  std::vector<char> readData(const std::string& filename) {
    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail()) {
      std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
      return {};
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail()) {
      std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
      return {};
    }

    return data;
  }

  CUcontext cuda_context_;
  OptixDeviceContext optix_context_;

  // modules that contains device program
  std::vector<OptixModule> modules_;
  OptixModuleCompileOptions module_compile_options_ = {};

  std::vector<OptixPipeline> pipelines_;
  std::vector<OptixPipelineCompileOptions> pipeline_compile_options_;
  OptixPipelineLinkOptions pipeline_link_options_ = {};

  std::vector<OptixProgramGroup> raygen_pgs_;
  std::vector<thrust::device_vector<RaygenRecord>> raygen_records_;

  std::vector<OptixProgramGroup> miss_pgs_;
  std::vector<thrust::device_vector<MissRecord>> miss_records_;

  std::vector<OptixProgramGroup> hitgroup_pgs_;
  std::vector<thrust::device_vector<HitgroupRecord>> hitgroup_records_;
  std::vector<OptixShaderBindingTable> sbts_;
  uint32_t params_size_;

  // device data
  pinned_vector<OptixInstance> tmp_h_instances_;
  thrust::device_vector<OptixInstance> tmp_instances_;

  pinned_vector<char> h_launch_params_;
  thrust::device_vector<char> launch_params_;
  SharedValue<uint64_t> compacted_size_;
};
}  // namespace details

}  // namespace rtspatial

#endif  // RTSPATIAL_DETAILS_RT_ENGINE_H
