@PACKAGE_INIT@

set_and_check(RTSPATIAL_INCLUDE_DIR "@PACKAGE_INCLUDE_INSTALL_DIR@")
set_and_check(RTSPATIAL_INCLUDE_DIRS "${RTSPATIAL_INCLUDE_DIR}")
set_and_check(RTSPATIAL_LIBRARY_DIR "@PACKAGE_LIB_INSTALL_DIR@")
set_and_check(RTSPATIAL_SHADER_DIR "${RTSPATIAL_INCLUDE_DIR}/rtspatial/shaders")

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

message("OptiX_INSTALL_DIR: ${OptiX_INSTALL_DIR}")

include("${RTSPATIAL_LIBRARY_DIR}/cmake/rtspatial/FindOptiX.cmake")
include("${RTSPATIAL_LIBRARY_DIR}/cmake/rtspatial/nvcuda_compile_module.cmake")

FUNCTION(RTSPATIAL_COMPILE_SHADERS CALLBACK_INCLUDE_DIR CALLBACK_HEADER OUTPUT_DIR GENERATED_FILES)
    set(FLOAT_TYPES "float;double")
    set(OPTIX_MODULE_EXTENSION ".ptx")
    set(OPTIX_PROGRAM_TARGET "--ptx")

    file(GLOB SHADERS "${RTSPATIAL_SHADER_DIR}/*.cu")

    foreach (FLOAT_TYPE IN LISTS FLOAT_TYPES)
        message("-- Defining shaders (FLOAT_TYPE: ${FLOAT_TYPE}, CALLBACK_INCLUDE_DIR: ${CALLBACK_INCLUDE_DIR}, CALLBACK_HEADER: ${CALLBACK_HEADER})")
        NVCUDA_COMPILE_MODULE(
                SOURCES ${SHADERS}
                DEPENDENCIES ${SHADERS_HEADERS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${FLOAT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--use_fast_math"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
                "-I${OptiX_INCLUDE}"
                "-I${RTSPATIAL_INCLUDE_DIR}"
                "-I${CALLBACK_INCLUDE_DIR}"
                "-DFLOAT_TYPE=${FLOAT_TYPE}"
                -include "${CALLBACK_HEADER}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})
    endforeach ()
    set(${GENERATED_FILES} ${ALL_GENERATED_FILES} PARENT_SCOPE)
ENDFUNCTION()


include_directories(${OptiX_INCLUDE})

include(FindPackageMessage)
find_package_message(rtspatial
"Found rtspatial: ${CMAKE_CURRENT_LIST_FILE}"
"Version \"@RTSPATIAL_VERSION@\""
)
