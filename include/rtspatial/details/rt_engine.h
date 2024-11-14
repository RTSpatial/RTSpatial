#ifndef RTSPATIAL_DETAILS_RT_ENGINE_H
#define RTSPATIAL_DETAILS_RT_ENGINE_H
#include <cuda.h>
// #include <optix_function_table_definition.h>  // for g_optixFunctionTable
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
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
  MODULE_ID_FLOAT_CONTAINS_ENVELOPE_QUERY_2D,
  MODULE_ID_DOUBLE_CONTAINS_ENVELOPE_QUERY_2D,
  MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD,
  MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD,
  MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD,
  MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD,
  MODULE_ID_FLOAT_INTERSECTS_LINE_QUERY_2D,
  MODULE_ID_DOUBLE_INTERSECTS_LINE_QUERY_2D,
  NUM_MODULE_IDENTIFIERS
};

static const float IDENTICAL_TRANSFORMATION_MTX[12] = {
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

class Module {
 public:
  Module() : enabled_module_(0), n_payload_(0), n_attribute_(0) {}

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

inline RTConfig get_default_rt_config(const std::string& ptx_root) {
  RTConfig config;

  {
    Module mod;

    mod.set_id(ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/float_shaders_contains_point_query_2d.ptx");
    mod.set_function_suffix("contains_point_query_2d");
    mod.EnableIsIntersection();
    mod.set_n_payload(1);

    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/double_shaders_contains_point_query_2d.ptx");
    config.AddModule(mod);
  }

  {
    Module mod;

    mod.set_id(ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_ENVELOPE_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/float_shaders_contains_envelope_query_2d.ptx");
    mod.set_function_suffix("contains_envelope_query_2d");
    mod.EnableIsIntersection();
    mod.set_n_payload(1);

    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_ENVELOPE_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/double_shaders_contains_envelope_query_2d.ptx");
    config.AddModule(mod);
  }

  {
    Module mod;

    mod.set_id(
        ModuleIdentifier::MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD);
    mod.set_program_path(ptx_root +
                         "/float_shaders_intersects_envelope_query_2d.ptx");
    mod.set_function_suffix("intersects_envelope_query_2d_forward");
    mod.EnableIsIntersection();
    mod.set_n_payload(2);

    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::
                   MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_FORWARD);
    mod.set_program_path(ptx_root +
                         "/double_shaders_intersects_envelope_query_2d.ptx");
    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::
                   MODULE_ID_FLOAT_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD);
    mod.set_program_path(ptx_root +
                         "/float_shaders_intersects_envelope_query_2d.ptx");
    mod.set_function_suffix("intersects_envelope_query_2d_backward");
    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::
                   MODULE_ID_DOUBLE_INTERSECTS_ENVELOPE_QUERY_2D_BACKWARD);
    mod.set_program_path(ptx_root +
                         "/double_shaders_intersects_envelope_query_2d.ptx");
    config.AddModule(mod);
  }

  {
    Module mod;

    mod.set_id(ModuleIdentifier::MODULE_ID_FLOAT_INTERSECTS_LINE_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/float_shaders_intersects_line_query_2d.ptx");
    mod.set_function_suffix("intersects_line_query_2d");
    mod.EnableIsIntersection();
    mod.set_n_payload(1);

    config.AddModule(mod);

    mod.set_id(ModuleIdentifier::MODULE_ID_DOUBLE_INTERSECTS_LINE_QUERY_2D);
    mod.set_program_path(ptx_root +
                         "/double_shaders_intersects_line_query_2d.ptx");
    config.AddModule(mod);
  }

#ifndef NDEBUG
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

  return config;
}

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
                                          bool prefer_fast_build = false,
                                          bool compact = false) {
    return buildAccel(cuda_stream, aabbs, buf, prefer_fast_build, compact);
  }

  OptixTraversableHandle BuildAccelCustom(cudaStream_t cuda_stream,
                                          ArrayView<OptixAabb> aabbs,
                                          char* buffer, size_t buffer_size,
                                          ReusableBuffer& buf,
                                          bool prefer_fast_build = false) {
    return buildAccel(cuda_stream, aabbs, buffer, buffer_size, buf,
                      prefer_fast_build, false);
  }

  OptixTraversableHandle UpdateAccelCustom(
      cudaStream_t cuda_stream, OptixTraversableHandle handle,
      ArrayView<OptixAabb> aabbs, ReusableBuffer& buf, size_t buf_offset,
      bool prefer_fast_build = false, bool compact = false) {
    return updateAccel(cuda_stream, handle, aabbs, buf, buf_offset,
                       prefer_fast_build, compact);
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

  void Render(cudaStream_t cuda_stream, ModuleIdentifier mod, dim3 dim) {
    void* launch_params = thrust::raw_pointer_cast(launch_params_.data());

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipelines_[mod], cuda_stream,
                            /*! parameters and SBT */
                            reinterpret_cast<CUdeviceptr>(launch_params),
                            params_size_, &sbts_[mod],
                            /*! dimensions of the launch: */
                            dim.x, dim.y, dim.z));
  }

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

  size_t EstimateMemoryUsageForAABB(size_t num_aabbs, bool prefer_fast_build,
                                    bool compact) {
    OptixBuildInput build_input = {};
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = nullptr;
    build_input.customPrimitiveArray.flags = build_input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.numPrimitives = num_aabbs;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
        sizeof(uint32_t);
    build_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    if (compact) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));
    return blas_buffer_sizes.outputSizeInBytes +
           blas_buffer_sizes.tempSizeInBytes;
  }

 private:
  void initOptix(const RTConfig& config) {
    // https://stackoverflow.com/questions/10415204/how-to-create-a-cuda-context
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
    h_launch_params_.resize(1024);
    launch_params_.resize(1024);
  }

  static void context_log_cb(unsigned int level, const char* tag,
                             const char* message, void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
  }

  void createContext() {
    CUresult cu_res = cuCtxGetCurrent(&cuda_context_);
    if (cu_res != CUDA_SUCCESS)
      fprintf(stderr, "Error querying current context: error code %d\n",
              cu_res);
    OptixDeviceContextOptions options;
    options.logCallbackFunction = context_log_cb;
    options.logCallbackData = nullptr;

#ifndef NDEBUG
    options.logCallbackLevel = 4;
    options.validationMode = OptixDeviceContextValidationMode::
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.logCallbackLevel = 2;
#endif
    OPTIX_CHECK(
        optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
  }

  void createModule(const RTConfig& config) {
    module_compile_options_.maxRegisterCount = config.max_reg_count;
    module_compile_options_.optLevel = config.opt_level;
    module_compile_options_.debugLevel = config.dbg_level;
    pipeline_compile_options_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    pipeline_link_options_.maxTraceDepth = config.max_trace_depth;

    modules_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : config.modules) {
      std::vector<char> programData = readData(pair.second.get_program_path());
      auto& pipeline_compile_options = pipeline_compile_options_[pair.first];

      //    pipeline_compile_options.traversableGraphFlags =
      //        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      pipeline_compile_options.traversableGraphFlags =
          OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

      pipeline_compile_options.usesMotionBlur = false;
      pipeline_compile_options.numPayloadValues = pair.second.get_n_payload();
      pipeline_compile_options.numAttributeValues =
          pair.second.get_n_attribute();
      pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
      pipeline_compile_options.pipelineLaunchParamsVariableName =
          RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME;
      //    pipeline_compile_options.usesPrimitiveTypeFlags =
      //        OPTIX_PRIMITIVE_TYPE_CUSTOM | OPTIX_PRIMITIVE_TYPE_SPHERE;

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixModuleCreate(optix_context_, &module_compile_options_,
                                    &pipeline_compile_options,
                                    programData.data(), programData.size(), log,
                                    &sizeof_log, &modules_[pair.first]));
#ifndef NDEBUG
      if (sizeof_log > 1) {
        std::cout << log << std::endl;
      }
#endif
    }
  }

  void createRaygenPrograms(const RTConfig& config) {
    const auto& conf_modules = config.modules;
    raygen_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : conf_modules) {
      auto f_name = "__raygen__" + pair.second.get_function_suffix();
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module = modules_[pair.first];
      pgDesc.raygen.entryFunctionName = f_name.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &raygen_pgs_[pair.first]));
#ifndef NDEBUG
      if (sizeof_log > 1) {
        std::cout << log << std::endl;
      }
#endif
    }
  }

  void createMissPrograms(const RTConfig& config) {
    const auto& conf_modules = config.modules;
    miss_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : conf_modules) {
      auto& mod = pair.second;
      auto f_name = "__miss__" + mod.get_function_suffix();
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

      pgDesc.miss.module = nullptr;
      pgDesc.miss.entryFunctionName = nullptr;

      if (mod.IsMissEnable()) {
        pgDesc.miss.module = modules_[pair.first];
        pgDesc.miss.entryFunctionName = f_name.c_str();
      }

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &miss_pgs_[pair.first]));
#ifndef NDEBUG
      if (sizeof_log > 1) {
        std::cout << log << std::endl;
      }
#endif
    }
  }

  void createHitgroupPrograms(const RTConfig& config) {
    auto& conf_modules = config.modules;
    hitgroup_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : conf_modules) {
      const auto& conf_mod = pair.second;
      auto f_name_anythit = "__anyhit__" + conf_mod.get_function_suffix();
      auto f_name_intersect =
          "__intersection__" + conf_mod.get_function_suffix();
      auto f_name_closesthit =
          "__closesthit__" + conf_mod.get_function_suffix();
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pg_desc = {};

      pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

      pg_desc.hitgroup.moduleIS = nullptr;
      pg_desc.hitgroup.entryFunctionNameIS = nullptr;
      pg_desc.hitgroup.moduleAH = nullptr;
      pg_desc.hitgroup.entryFunctionNameAH = nullptr;
      pg_desc.hitgroup.moduleCH = nullptr;
      pg_desc.hitgroup.entryFunctionNameCH = nullptr;

      if (conf_mod.IsIsIntersectionEnabled()) {
        pg_desc.hitgroup.moduleIS = modules_[pair.first];
        pg_desc.hitgroup.entryFunctionNameIS = f_name_intersect.c_str();
      }

      if (conf_mod.IsAnyHitEnable()) {
        pg_desc.hitgroup.moduleAH = modules_[pair.first];
        pg_desc.hitgroup.entryFunctionNameAH = f_name_anythit.c_str();
      }

      if (conf_mod.IsClosestHitEnable()) {
        pg_desc.hitgroup.moduleCH = modules_[pair.first];
        pg_desc.hitgroup.entryFunctionNameCH = f_name_closesthit.c_str();
      }

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pg_desc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &hitgroup_pgs_[pair.first]));
#ifndef NDEBUG
      if (sizeof_log > 1) {
        std::cout << log << std::endl;
      }
#endif
    }
  }

  void createPipeline(const RTConfig& config) {
    pipelines_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : config.modules) {
      std::vector<OptixProgramGroup> program_groups;
      program_groups.push_back(raygen_pgs_[pair.first]);
      program_groups.push_back(miss_pgs_[pair.first]);
      program_groups.push_back(hitgroup_pgs_[pair.first]);

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixPipelineCreate(
          optix_context_, &pipeline_compile_options_[pair.first],
          &pipeline_link_options_, program_groups.data(),
          (int) program_groups.size(), log, &sizeof_log,
          &pipelines_[pair.first]));
#ifndef NDEBUG
      if (sizeof_log > 1) {
        std::cout << log << std::endl;
      }
#endif
      OptixStackSizes stack_sizes = {};
      for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes,
                                                  pipelines_[pair.first]));
      }

      uint32_t direct_callable_stack_size_from_traversal;
      uint32_t direct_callable_stack_size_from_state;
      uint32_t continuation_stack_size;

      OPTIX_CHECK(optixUtilComputeStackSizes(
          &stack_sizes, config.max_trace_depth,
          0,  // maxCCDepth
          0,  // maxDCDepth
          &direct_callable_stack_size_from_traversal,
          &direct_callable_stack_size_from_state, &continuation_stack_size));
      OPTIX_CHECK(optixPipelineSetStackSize(
          pipelines_[pair.first], direct_callable_stack_size_from_traversal,
          direct_callable_stack_size_from_state, continuation_stack_size,
          config.max_traversable_depth  // maxTraversableDepth
          ));
    }
  }

  void buildSBT(const RTConfig& config) {
    sbts_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
    raygen_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
    miss_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
    hitgroup_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

    for (auto& pair : config.modules) {
      auto& sbt = sbts_[pair.first];
      std::vector<RaygenRecord> raygenRecords;
      {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pgs_[pair.first], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
      }
      raygen_records_[pair.first] = raygenRecords;
      sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
          thrust::raw_pointer_cast(raygen_records_[pair.first].data()));

      std::vector<MissRecord> missRecords;
      {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pgs_[pair.first], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
      }

      miss_records_[pair.first] = missRecords;
      sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
          thrust::raw_pointer_cast(miss_records_[pair.first].data()));
      sbt.missRecordStrideInBytes = sizeof(MissRecord);
      sbt.missRecordCount = (int) missRecords.size();
      sbt.callablesRecordBase = 0;

      std::vector<HitgroupRecord> hitgroupRecords;
      {
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pgs_[pair.first], &rec));
        rec.data = nullptr;
        hitgroupRecords.push_back(rec);
      }
      hitgroup_records_[pair.first] = hitgroupRecords;
      sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
          thrust::raw_pointer_cast(hitgroup_records_[pair.first].data()));
      sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
      sbt.hitgroupRecordCount = (int) hitgroupRecords.size();
    }
  }

  OptixTraversableHandle buildAccel(cudaStream_t cuda_stream,
                                    ArrayView<OptixAabb> aabbs,
                                    ReusableBuffer& buf, bool prefer_fast_build,
                                    bool compact = false) {
    OptixTraversableHandle traversable;
    OptixBuildInput build_input = {};
    CUdeviceptr d_aabb = THRUST_TO_CUPTR(aabbs.data());
    // Setup AABB build input. Don't disable AH.
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    uint32_t num_prims = aabbs.size();

    assert(reinterpret_cast<uint64_t>(aabbs.data()) %
               OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ==
           0);

    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.flags = build_input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.numPrimitives = num_prims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
        sizeof(uint32_t);
    build_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    if (compact) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }

    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));

    // Alignment
    buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = reinterpret_cast<CUdeviceptr>(compacted_size_.data());

    if (compact) {
      // Layout: |Out Buf|Temp Buf|Compact Buf|
      char* compressed_buf = buf.Acquire(blas_buffer_sizes.outputSizeInBytes);
      char* out_buf = buf.Acquire(blas_buffer_sizes.outputSizeInBytes);
      char* temp_buf = buf.Acquire(blas_buffer_sizes.tempSizeInBytes);

      OPTIX_CHECK(optixAccelBuild(
          optix_context_, cuda_stream, &accelOptions, &build_input, 1,
          reinterpret_cast<CUdeviceptr>(temp_buf),
          blas_buffer_sizes.tempSizeInBytes,
          reinterpret_cast<CUdeviceptr>(out_buf),
          blas_buffer_sizes.outputSizeInBytes, &traversable, &emitDesc, 1));

      auto compacted_size = compacted_size_.get(cuda_stream);

      OPTIX_CHECK(
          optixAccelCompact(optix_context_, cuda_stream, traversable,
                            reinterpret_cast<CUdeviceptr>(compressed_buf),
                            compacted_size, &traversable));

      buf.Release(blas_buffer_sizes.outputSizeInBytes +
                  blas_buffer_sizes.tempSizeInBytes);
    } else {
      char* out_buf = buf.Acquire(blas_buffer_sizes.outputSizeInBytes);
      char* temp_buf = buf.Acquire(blas_buffer_sizes.tempSizeInBytes);

      OPTIX_CHECK(optixAccelBuild(
          optix_context_, cuda_stream, &accelOptions, &build_input, 1,
          reinterpret_cast<CUdeviceptr>(temp_buf),
          blas_buffer_sizes.tempSizeInBytes,
          reinterpret_cast<CUdeviceptr>(out_buf),
          blas_buffer_sizes.outputSizeInBytes, &traversable, nullptr, 0));

      buf.Release(blas_buffer_sizes.tempSizeInBytes);
    }

    return traversable;
  }

  OptixTraversableHandle buildAccel(cudaStream_t cuda_stream,
                                    ArrayView<OptixAabb> aabbs, char* buffer,
                                    size_t buffer_size, ReusableBuffer& buf,
                                    bool prefer_fast_build,
                                    bool compact = false) {
    OptixTraversableHandle traversable;
    OptixBuildInput build_input = {};
    CUdeviceptr d_aabb = THRUST_TO_CUPTR(aabbs.data());
    // Setup AABB build input. Don't disable AH.
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    uint32_t num_prims = aabbs.size();

    assert(reinterpret_cast<uint64_t>(aabbs.data()) %
               OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ==
           0);

    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.flags = build_input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.numPrimitives = num_prims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
        sizeof(uint32_t);
    build_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));

    // Alignment
    auto align_size = getAccelAlignedSize(reinterpret_cast<size_t>(buffer)) -
                      reinterpret_cast<size_t>(buffer);

    if (align_size + blas_buffer_sizes.outputSizeInBytes > buffer_size) {
      std::cerr << "Buffer use up" << std::endl;
      abort();
    }

    char* out_buf = buffer + align_size;
    char* temp_buf = buf.Acquire(blas_buffer_sizes.tempSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        reinterpret_cast<CUdeviceptr>(temp_buf),
        blas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(out_buf),
        blas_buffer_sizes.outputSizeInBytes, &traversable, nullptr, 0));

    buf.Release(blas_buffer_sizes.tempSizeInBytes);
    // TODO: compact
    return traversable;
  }

  OptixTraversableHandle updateAccel(cudaStream_t cuda_stream,
                                     OptixTraversableHandle handle,
                                     ArrayView<OptixAabb> aabbs,
                                     ReusableBuffer& buf, size_t buf_offset,
                                     bool prefer_fast_build, bool compact) {
    OptixBuildInput build_input = {};
    CUdeviceptr d_aabb = THRUST_TO_CUPTR(aabbs.data());
    // Setup AABB build input. Don't disable AH.
    // OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    uint32_t num_prims = aabbs.size();

    assert(reinterpret_cast<uint64_t>(aabbs.data()) %
               OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ==
           0);

    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.flags = build_input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.numPrimitives = num_prims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
        sizeof(uint32_t);
    build_input.customPrimitiveArray.primitiveIndexOffset = 0;

    // ==================================================================
    // Bottom-level acceleration structure (BLAS) setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    if (compact) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));
    char* out_buf = buf.GetData() + buf_offset;
    size_t tail = buf.GetTail();
    // Alignment
    buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
    char* temp_buf = buf.Acquire(blas_buffer_sizes.tempUpdateSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        THRUST_TO_CUPTR(temp_buf), blas_buffer_sizes.tempUpdateSizeInBytes,
        THRUST_TO_CUPTR(out_buf), blas_buffer_sizes.outputSizeInBytes, &handle,
        nullptr, 0));
    buf.SetTail(tail);
    return handle;
  }

  OptixTraversableHandle buildInstanceAccel(cudaStream_t cuda_stream,
                                            ArrayView<OptixInstance> instances,
                                            ReusableBuffer& buf,
                                            bool prefer_fast_build) {
    OptixTraversableHandle traversable;
    OptixBuildInput build_input = {};

    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances =
        reinterpret_cast<CUdeviceptr>(instances.data());
    build_input.instanceArray.numInstances = instances.size();

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));
    // Alignment
    buf.Acquire(getInstanceAlignedSize(buf.GetTail()) - buf.GetTail());
    char* out_buf = buf.Acquire(blas_buffer_sizes.outputSizeInBytes);
    char* temp_buf = buf.Acquire(blas_buffer_sizes.tempSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        THRUST_TO_CUPTR(temp_buf), blas_buffer_sizes.tempSizeInBytes,
        THRUST_TO_CUPTR(out_buf), blas_buffer_sizes.outputSizeInBytes,
        &traversable, nullptr, 0));

    buf.Release(blas_buffer_sizes.tempSizeInBytes);
    return traversable;
  }

  OptixTraversableHandle updateInstanceAccel(cudaStream_t cuda_stream,
                                             ArrayView<OptixInstance> instances,
                                             ReusableBuffer& buf,
                                             size_t buf_offset,
                                             bool prefer_fast_build) {
    OptixTraversableHandle traversable;
    OptixBuildInput build_input = {};

    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances =
        reinterpret_cast<CUdeviceptr>(instances.data());
    build_input.instanceArray.numInstances = instances.size();

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    if (prefer_fast_build) {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
      accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                             &build_input,
                                             1,  // num_build_inputs
                                             &blas_buffer_sizes));

    char* out_buf = buf.GetData() + buf_offset;
    size_t tail = buf.GetTail();
    // Alignment
    buf.Acquire(getInstanceAlignedSize(buf.GetTail()) - buf.GetTail());
    char* temp_buf = buf.Acquire(blas_buffer_sizes.tempUpdateSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        THRUST_TO_CUPTR(temp_buf), blas_buffer_sizes.tempSizeInBytes,
        THRUST_TO_CUPTR(out_buf), blas_buffer_sizes.outputSizeInBytes,
        &traversable, nullptr, 0));

    buf.SetTail(tail);

    return traversable;
  }

  static size_t getAccelAlignedSize(size_t size) {
    if (size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0) {
      return size;
    }

    return size - size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT +
           OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
  }

  static size_t getInstanceAlignedSize(size_t size) {
    if (size % OPTIX_INSTANCE_BYTE_ALIGNMENT == 0) {
      return size;
    }

    return size - size % OPTIX_INSTANCE_BYTE_ALIGNMENT +
           OPTIX_INSTANCE_BYTE_ALIGNMENT;
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
