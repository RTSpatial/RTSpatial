include_directories(${OptiX_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/src)

set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")
set(ENABLED_ARCHS "80")

set(SHADERS
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_point_query_2d_triangle.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_point_query_2d.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_envelope_query_2d.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_intersects_envelope_query_2d.cu"
)

#set(SHADERS_HEADERS "${PROJECT_SOURCE_DIR}/include/details/launch_parameters.h")

set(ALL_PROGRAM_MODULES "")
set(FLOAT_TYPES "float;double")
foreach (FLOAT_TYPE IN LISTS FLOAT_TYPES)
    NVCUDA_COMPILE_MODULE(
            SOURCES ${SHADERS}
            DEPENDENCIES ${SHADERS_HEADERS}
            TARGET_PATH "${MODULE_TARGET_DIR}/ptx"
            PREFIX "${FLOAT_TYPE}_"
            EXTENSION "${OPTIX_MODULE_EXTENSION}"
            GENERATED_FILES PROGRAM_MODULES
            NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
            "--gpu-architecture=compute_${ENABLED_ARCHS}"
            "--use_fast_math"
            "--relocatable-device-code=true"
            "--expt-relaxed-constexpr"
            "-Wno-deprecated-gpu-targets"
            "-I${OptiX_INCLUDE}"
            "-I${PROJECT_SOURCE_DIR}/include"
            "-DFLOAT_TYPE=${FLOAT_TYPE}"
    )
    list(APPEND ALL_PROGRAM_MODULES ${PROGRAM_MODULES})
endforeach ()

message("All modules ${ALL_PROGRAM_MODULES}")


#set(RTSPATIAL_HDR_DIR "${PROJECT_SOURCE_DIR}/include/rtspatial")
#set(RTPSATIAL_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
#set(RTPSATIAL_CUH "")
#set(RTPSATIAL_CU "")
#
#set(RTPSATIAL_BASE_CUH
#        "${RTSPATIAL_HDR_DIR}/geom/point.cuh"
#        "${RTSPATIAL_HDR_DIR}/geom/envelope.cuh"
#        "${RTSPATIAL_HDR_DIR}/rtspatial.h"
#        "${RTSPATIAL_HDR_DIR}/spatial_index.cuh"
#)
#
#list(APPEND RTPSATIAL_CUH ${RTPSATIAL_BASE_CUH})
#
#
#set(RTPSATIAL_BASE_CU
#        "${RTPSATIAL_SRC_DIR}/spatial_index.cu"
#        "${RTPSATIAL_SRC_DIR}/rt_engine.cu"
#)
#
#list(APPEND RTPSATIAL_CU ${RTPSATIAL_BASE_CU})
#
#set(RTSPATIAL_SOURCES ${RTPSATIAL_CU} ${RTPSATIAL_CUH})

include_directories("${PROJECT_SOURCE_DIR}/include")

###############################################################################
# Targets installation

#install(TARGETS rtspatial
#        EXPORT libspatialindexTargets
#        RUNTIME DESTINATION ${SIDX_BIN_DIR}
#        LIBRARY DESTINATION ${SIDX_LIB_DIR}
#        ARCHIVE DESTINATION ${SIDX_LIB_DIR})
#
#export(
#        TARGETS
#        rtspatial
#        NAMESPACE
#        libspatialindex::
#        FILE
#        "${SIDX_LIB_DIR}/libspatialindexTargets.cmake")
#
#install(
#        EXPORT
#        libspatialindexTargets
#        DESTINATION
#        "${SIDX_LIB_DIR}/cmake/libspatialindex")
#
#target_include_directories(spatialindex_c
#        INTERFACE
#        $<INSTALL_INTERFACE:include>)
#target_include_directories(spatialindex
#        INTERFACE
#        $<INSTALL_INTERFACE:include>)
#
#install(DIRECTORY ${SIDX_HEADERS_DIR}
#        DESTINATION include/spatialindex
#        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


###############################################################################
# Tests
find_package(Boost REQUIRED)
add_executable(rtspatial_tests tests/rtspatial_tests.cu rt_engine.cu ${ALL_PROGRAM_MODULES})

target_link_libraries(rtspatial_tests
        ${GTEST_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        cuda # CUcontext
)
target_compile_options(rtspatial_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rtspatial_tests PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)


###############################################################################
# Demos


add_executable(rtspatial
        spatial_index.cu
        rt_engine.cu
        flags.cpp
        ${ALL_PROGRAM_MODULES}
)

target_link_libraries(rtspatial
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        cuda # CUcontext
        ${GFLAGS_LIBRARIES}
)
target_compile_options(rtspatial PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rtspatial PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)

