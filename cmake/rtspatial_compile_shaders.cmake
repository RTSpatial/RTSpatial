
FUNCTION(RTSPATIAL_COMPILE_SHADERS CALLBACK_INCLUDE_DIR CALLBACK_HEADER OUTPUT_DIR GENERATED_FILES)
    set(FLOAT_TYPES "float;double")

    set(OPTIX_MODULE_EXTENSION ".ptx")
    set(OPTIX_PROGRAM_TARGET "--ptx")

    file(GLOB SHADERS "${PROJECT_SOURCE_DIR}/src/shaders/*.cu")

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
                "--gpu-architecture=compute_${ENABLED_ARCHS}"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
                "-I${OptiX_INCLUDE}"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-I${CALLBACK_INCLUDE_DIR}"
                "-DFLOAT_TYPE=${FLOAT_TYPE}"
                -include "${CALLBACK_HEADER}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})
    endforeach ()
    set(${GENERATED_FILES} ${ALL_GENERATED_FILES} PARENT_SCOPE)
ENDFUNCTION()
