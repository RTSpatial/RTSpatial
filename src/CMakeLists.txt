include_directories(${OptiX_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/src)

message("OptiX_INCLUDE: ${OptiX_INCLUDE}")

set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

set(SHADERS
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_point_query_2d_triangle.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_point_query_2d.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_contains_envelope_query_2d.cu"
        "${PROJECT_SOURCE_DIR}/src/shaders/shaders_intersects_envelope_query_2d.cu"
)

#set(SHADERS_HEADERS "${PROJECT_SOURCE_DIR}/include/details/launch_parameters.h")

message("MODULE_TARGET_DIR: ${MODULE_TARGET_DIR}")

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


set(RTSPATIAL_HDR_DIR "${PROJECT_SOURCE_DIR}/include/rtspatial/")
set(RTPSATIAL_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(RTPSATIAL_CUH "")
set(RTPSATIAL_CU "")

set(RTPSATIAL_BASE_CUH
        "${RTSPATIAL_HDR_DIR}/geom/point.cuh"
        "${RTSPATIAL_HDR_DIR}/geom/envelope.cuh"
        "${RTSPATIAL_HDR_DIR}/rtspatial.h"
        "${RTSPATIAL_HDR_DIR}/spatial_index.cuh"
)

list(APPEND RTPSATIAL_CUH ${RTPSATIAL_BASE_CUH})


set(RTPSATIAL_BASE_CU
        "${RTPSATIAL_SRC_DIR}/rt_engine.cu"
)

list(APPEND RTPSATIAL_CU ${RTPSATIAL_BASE_CU})
list(APPEND RTSPATIAL_CU ${ALL_PROGRAM_MODULES})

set(RTSPATIAL_SOURCES ${RTPSATIAL_CU} ${RTPSATIAL_CUH})

include_directories("${PROJECT_SOURCE_DIR}/include")

###############################################################################
# Library
add_library(rtspatial SHARED ${RTPSATIAL_CU} ${ALL_PROGRAM_MODULES})
target_link_libraries(rtspatial
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        cuda # CUcontext
)
target_compile_options(rtspatial PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr >)
set_target_properties(rtspatial PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)


###############################################################################
# Targets installation
set(RTSPATIAL_BIN_DIR bin)
set(RTSPATIAL_LIB_DIR lib)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Includes
install(DIRECTORY ${RTSPATIAL_HDR_DIR}
        DESTINATION include/rtspatial
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")

# Ptx
install(DIRECTORY "${MODULE_TARGET_DIR}/ptx"
        DESTINATION "${RTSPATIAL_LIB_DIR}")

install(FILES "${PROJECT_SOURCE_DIR}/cmake/FindOptiX.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial")


install(TARGETS rtspatial
        EXPORT rtspatial-targets
        RUNTIME DESTINATION ${RTSPATIAL_BIN_DIR}
        LIBRARY DESTINATION ${RTSPATIAL_LIB_DIR}
        ARCHIVE DESTINATION ${RTSPATIAL_LIB_DIR})

install(EXPORT rtspatial-targets
        FILE rtspatial-targets.cmake
        NAMESPACE rtspatial::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial")

write_basic_package_version_file(
        "rtspatial-config-version.cmake"
        VERSION ${RTSPATIAL_VERSION}
        COMPATIBILITY AnyNewerVersion)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/rtspatial CACHE PATH "Location of header files")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Location of library files")

configure_package_config_file("${PROJECT_SOURCE_DIR}/rtspatial-config.in.cmake"
        "${PROJECT_BINARY_DIR}/rtspatial-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial
        PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

install(FILES "${PROJECT_BINARY_DIR}/rtspatial-config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/rtspatial-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial")

target_include_directories(rtspatial
        INTERFACE
        $<INSTALL_INTERFACE:include>)

