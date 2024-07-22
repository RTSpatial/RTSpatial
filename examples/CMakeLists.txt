include("cmake/FindGFlags.cmake")
if (NOT GFLAGS_FOUND)
    message(STATUS "gflags not found, build without gflags")
else ()
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()
include_directories(${OptiX_INCLUDE})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/examples")

###############################################################################
# Experiments

set(SHADERS "${PROJECT_SOURCE_DIR}/examples/shaders/shaders_hit.cu")

NVCUDA_COMPILE_MODULE(
        SOURCES ${SHADERS}
        DEPENDENCIES ${SHADERS_HEADERS}
        TARGET_PATH "${MODULE_TARGET_DIR}/ptx"
        EXTENSION ".ptx"
        GENERATED_FILES PROGRAM_MODULES
        NVCC_OPTIONS "--ptx"
        "--relocatable-device-code=true"
        "--expt-relaxed-constexpr"
        "-Wno-deprecated-gpu-targets"
        "-I${OptiX_INCLUDE}"
        "-I${PROJECT_SOURCE_DIR}/include"
)

message("This ${PROGRAM_MODULES}")
add_executable(rtspatial_exec spatial_index.cu flags.cpp ${PROGRAM_MODULES})

target_link_libraries(rtspatial_exec rtspatial ${GFLAGS_LIBRARIES})
target_compile_options(rtspatial_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rtspatial_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)
