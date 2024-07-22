#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>  // for g_optixFunctionTable
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <iostream>
#include <stdexcept>

#include "rtspatial/details/rt_engine.h"
#include "rtspatial/details/sbt_record.h"
#include "rtspatial/utils/exception.h"
#include "rtspatial/utils/util.h"

namespace rtspatial {
namespace details {
RTConfig get_default_rt_config(const std::string& ptx_root) {
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

    mod.set_id(
        ModuleIdentifier::MODULE_ID_FLOAT_CONTAINS_POINT_QUERY_2D_TRIANGLE);
    mod.set_program_path(ptx_root +
                         "/float_shaders_contains_point_query_2d_triangle.ptx");
    mod.set_function_suffix("contains_point_query_2d_triangle");
    mod.EnableAnyHit();
    mod.set_n_payload(1);

    config.AddModule(mod);

    mod.set_id(
        ModuleIdentifier::MODULE_ID_DOUBLE_CONTAINS_POINT_QUERY_2D_TRIANGLE);
    mod.set_program_path(
        ptx_root + "/double_shaders_contains_point_query_2d_triangle.ptx");
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

void RTEngine::initOptix(const RTConfig& config) {
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

void RTEngine::createContext() {
  CUresult cu_res = cuCtxGetCurrent(&cuda_context_);
  if (cu_res != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cu_res);
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

void RTEngine::createModule(const RTConfig& config) {
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
    pipeline_compile_options.numAttributeValues = pair.second.get_n_attribute();
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName =
        RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME;
    //    pipeline_compile_options.usesPrimitiveTypeFlags =
    //        OPTIX_PRIMITIVE_TYPE_CUSTOM | OPTIX_PRIMITIVE_TYPE_SPHERE;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(optix_context_, &module_compile_options_,
                                  &pipeline_compile_options, programData.data(),
                                  programData.size(), log, &sizeof_log,
                                  &modules_[pair.first]));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      std::cout << log << std::endl;
    }
#endif
  }

  // External Module
  std::vector<char> programData =
      readData(config.external_module.get_program_path());
  auto& pipeline_compile_options =
      pipeline_compile_options_[ModuleIdentifier::MODULE_ID_EXTERNAL];

  //    pipeline_compile_options.traversableGraphFlags =
  //        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

  pipeline_compile_options.usesMotionBlur = false;
  pipeline_compile_options.numPayloadValues = 0;
  pipeline_compile_options.numAttributeValues = 0;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName =
      RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreate(
      optix_context_, &module_compile_options_, &pipeline_compile_options,
      programData.data(), programData.size(), log, &sizeof_log,
      &modules_[ModuleIdentifier::MODULE_ID_EXTERNAL]));
#ifndef NDEBUG
  if (sizeof_log > 1) {
    std::cout << log << std::endl;
  }
#endif
}

void RTEngine::createExternalPrograms(const RTConfig& config) {
  OptixProgramGroupDesc pgd;
  OptixProgramGroupOptions pgOptions = {};

  auto func_name = std::string("__direct_callable__" +
                               config.external_module.get_function_suffix());

  pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd.callables.moduleDC = modules_[ModuleIdentifier::MODULE_ID_EXTERNAL];
  pgd.callables.entryFunctionNameDC = func_name.c_str();
  pgd.callables.moduleCC = nullptr;
  pgd.callables.entryFunctionNameCC = nullptr;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgd, 1, &pgOptions, log,
                                      &sizeof_log, &external_pg_));
  if (sizeof_log > 1) {
    std::cout << log << std::endl;
  }
}

void RTEngine::createRaygenPrograms(const RTConfig& config) {
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
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &raygen_pgs_[pair.first]));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      std::cout << log << std::endl;
    }
#endif
  }
}

/*! does all setup for the miss program(s) we are going to use */
void RTEngine::createMissPrograms(const RTConfig& config) {
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
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &miss_pgs_[pair.first]));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      std::cout << log << std::endl;
    }
#endif
  }
}

/*! does all setup for the hitgroup program(s) we are going to use */
void RTEngine::createHitgroupPrograms(const RTConfig& config) {
  auto& conf_modules = config.modules;
  hitgroup_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : conf_modules) {
    const auto& conf_mod = pair.second;
    auto f_name_anythit = "__anyhit__" + conf_mod.get_function_suffix();
    auto f_name_intersect = "__intersection__" + conf_mod.get_function_suffix();
    auto f_name_closesthit = "__closesthit__" + conf_mod.get_function_suffix();
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
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pg_desc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &hitgroup_pgs_[pair.first]));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      std::cout << log << std::endl;
    }
#endif
  }
}

/*! assembles the full pipeline of all programs */
void RTEngine::createPipeline(const RTConfig& config) {
  pipelines_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : config.modules) {
    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back(raygen_pgs_[pair.first]);
    program_groups.push_back(miss_pgs_[pair.first]);
    program_groups.push_back(hitgroup_pgs_[pair.first]);
    program_groups.push_back(external_pg_);

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

/*! constructs the shader binding table */
void RTEngine::buildSBT(const RTConfig& config) {
  sbts_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  raygen_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  miss_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  hitgroup_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  callable_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

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

    std::vector<CallableRecord> callableRecords;
    {
      CallableRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(external_pg_, &rec));
      rec.data = nullptr;
      callableRecords.push_back(rec);
    }
    callable_records_[pair.first] = callableRecords;
    sbt.callablesRecordBase = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(callable_records_[pair.first].data()));
    sbt.callablesRecordStrideInBytes = sizeof(CallableRecord);
    sbt.callablesRecordCount = (int) callableRecords.size();
  }
}

OptixTraversableHandle RTEngine::buildAccel(cudaStream_t cuda_stream,
                                            ArrayView<OptixAabb> aabbs,
                                            ReusableBuffer& buf,
                                            bool prefer_fast_build) {
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
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  build_input.customPrimitiveArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  // FIXME: prefer_fast_build is consistent with updateAccel
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
  buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
  char* out_buf = buf.Acquire(blas_buffer_sizes.outputSizeInBytes);
  char* temp_buf = buf.Acquire(blas_buffer_sizes.tempSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(
      optix_context_, cuda_stream, &accelOptions, &build_input, 1,
      reinterpret_cast<CUdeviceptr>(temp_buf),
      blas_buffer_sizes.tempSizeInBytes, reinterpret_cast<CUdeviceptr>(out_buf),
      blas_buffer_sizes.outputSizeInBytes, &traversable, nullptr, 0));

  buf.Release(blas_buffer_sizes.tempSizeInBytes);
  return traversable;
}

OptixTraversableHandle RTEngine::updateAccel(cudaStream_t cuda_stream,
                                             OptixTraversableHandle handle,
                                             ArrayView<OptixAabb> aabbs,
                                             ReusableBuffer& buf,
                                             size_t buf_offset,
                                             bool prefer_fast_build) {
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
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
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

OptixTraversableHandle RTEngine::buildAccelTriangle(cudaStream_t cuda_stream,
                                                    ArrayView<float3> vertices,
                                                    ArrayView<uint3> indices,
                                                    ReusableBuffer& buf,
                                                    bool prefer_fast_build) {
  OptixTraversableHandle traversable;
  OptixBuildInput build_input = {};
  auto d_indices = reinterpret_cast<CUdeviceptr>(indices.data());
  auto d_vertices = reinterpret_cast<CUdeviceptr>(vertices.data());
  // Setup AABB build input. Don't disable AH.
  uint32_t build_input_flags[1] = {
      OPTIX_GEOMETRY_FLAG_NONE |
      OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};

  memset(&build_input.triangleArray, 0, sizeof(build_input.triangleArray));

  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  build_input.triangleArray.numVertices = vertices.size();
  build_input.triangleArray.vertexBuffers = &d_vertices;

  if (indices.empty()) {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
  } else {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = indices.size();
    build_input.triangleArray.indexBuffer = d_indices;
  }
  build_input.triangleArray.flags = build_input_flags;
  build_input.triangleArray.numSbtRecords = 1;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.triangleArray.sbtIndexOffsetBuffer = 0;
  build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  build_input.triangleArray.primitiveIndexOffset = 0;

  // ==================================================================
  // Bottom-level acceleration structure (BLAS) setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  if (prefer_fast_build) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  } else {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
#ifndef NDEBUG
  accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
#endif
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                           &build_input,
                                           1,  // num_build_inputs
                                           &blas_buffer_sizes));
  // Alignment
  buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
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

OptixTraversableHandle RTEngine::updateAccelTriangle(cudaStream_t cuda_stream,
                                                     ArrayView<float3> vertices,
                                                     ArrayView<uint3> indices,
                                                     ReusableBuffer& buf,
                                                     size_t buf_offset,
                                                     bool prefer_fast_build) {
  OptixTraversableHandle traversable;
  OptixBuildInput build_input = {};
  auto d_indices = reinterpret_cast<CUdeviceptr>(indices.data());
  auto d_vertices = reinterpret_cast<CUdeviceptr>(vertices.data());
  // Setup AABB build input. Don't disable AH.
  uint32_t build_input_flags[1] = {
      OPTIX_GEOMETRY_FLAG_NONE |
      OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};

  memset(&build_input.triangleArray, 0, sizeof(build_input.triangleArray));

  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  build_input.triangleArray.numVertices = vertices.size();
  build_input.triangleArray.vertexBuffers = &d_vertices;

  if (indices.empty()) {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
  } else {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = indices.size();
    build_input.triangleArray.indexBuffer = d_indices;
  }
  build_input.triangleArray.flags = build_input_flags;
  build_input.triangleArray.numSbtRecords = 1;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.triangleArray.sbtIndexOffsetBuffer = 0;
  build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  build_input.triangleArray.primitiveIndexOffset = 0;

  // ==================================================================
  // Bottom-level acceleration structure (BLAS) setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  if (prefer_fast_build) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  } else {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
#ifndef NDEBUG
  accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
#endif
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
  buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
  char* temp_buf = buf.Acquire(blas_buffer_sizes.tempUpdateSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(
      optix_context_, cuda_stream, &accelOptions, &build_input, 1,
      THRUST_TO_CUPTR(temp_buf), blas_buffer_sizes.tempSizeInBytes,
      THRUST_TO_CUPTR(out_buf), blas_buffer_sizes.outputSizeInBytes,
      &traversable, nullptr, 0));

  buf.SetTail(tail);

  return traversable;
}

OptixTraversableHandle RTEngine::buildInstanceAccel(
    cudaStream_t cuda_stream, ArrayView<OptixInstance> instances,
    ReusableBuffer& buf, bool prefer_fast_build) {
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
  buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
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

OptixTraversableHandle RTEngine::updateInstanceAccel(
    cudaStream_t cuda_stream, ArrayView<OptixInstance> instances,
    ReusableBuffer& buf, size_t buf_offset, bool prefer_fast_build) {
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

  char* out_buf = buf.GetData() + buf_offset;
  size_t tail = buf.GetTail();
  // Alignment
  buf.Acquire(getAccelAlignedSize(buf.GetTail()) - buf.GetTail());
  char* temp_buf = buf.Acquire(blas_buffer_sizes.tempUpdateSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(
      optix_context_, cuda_stream, &accelOptions, &build_input, 1,
      THRUST_TO_CUPTR(temp_buf), blas_buffer_sizes.tempSizeInBytes,
      THRUST_TO_CUPTR(out_buf), blas_buffer_sizes.outputSizeInBytes,
      &traversable, nullptr, 0));

  buf.SetTail(tail);

  return traversable;
}

void RTEngine::Render(cudaStream_t cuda_stream, ModuleIdentifier mod,
                      dim3 dim) {
  void* launch_params = thrust::raw_pointer_cast(launch_params_.data());

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipelines_[mod], cuda_stream,
                          /*! parameters and SBT */
                          reinterpret_cast<CUdeviceptr>(launch_params),
                          params_size_, &sbts_[mod],
                          /*! dimensions of the launch: */
                          dim.x, dim.y, dim.z));
}

size_t RTEngine::EstimateMemoryUsageForAABB(size_t num_aabbs) {
  OptixBuildInput build_input = {};
  uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers = nullptr;
  build_input.customPrimitiveArray.flags = build_input_flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;
  build_input.customPrimitiveArray.numPrimitives = num_aabbs;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  build_input.customPrimitiveArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
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

size_t RTEngine::EstimateMemoryUsageForTriangle(size_t num_aabb) {
  uint32_t build_input_flags[1] = {
      OPTIX_GEOMETRY_FLAG_NONE |
      OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
  OptixBuildInput build_input = {};
  memset(&build_input.triangleArray, 0, sizeof(build_input.triangleArray));

  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  build_input.triangleArray.numVertices = 4 * num_aabb;

  build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
  build_input.triangleArray.numIndexTriplets = 2 * num_aabb;
  build_input.triangleArray.flags = build_input_flags;
  build_input.triangleArray.numSbtRecords = 1;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.triangleArray.sbtIndexOffsetBuffer = 0;
  build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  build_input.triangleArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
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
}  // namespace details
}  // namespace rtspatial