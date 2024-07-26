find_package(Boost 1.76 REQUIRED COMPONENTS serialization)
include("cmake/FindGFlags.cmake")
if (NOT GFLAGS_FOUND)
    message(STATUS "gflags not found, build without gflags")
else ()
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()
include_directories(${OptiX_INCLUDE})
include_directories("${PROJECT_SOURCE_DIR}/include")

RTSPATIAL_COMPILE_SHADERS("${PROJECT_SOURCE_DIR}/include/rtspatial/default_handlers" "collecting_handlers.h"
        "${MODULE_TARGET_DIR}/examples_ptx" PROGRAM_MODULES)

###############################################################################
# Experiments
add_executable(rtspatial_exec spatial_index.cu flags.cpp ${PROGRAM_MODULES})

target_link_libraries(rtspatial_exec rtspatial ${GFLAGS_LIBRARIES} ${Boost_LIBRARIES})
target_compile_options(rtspatial_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
target_compile_options(rtspatial_exec PRIVATE -DPTX_ROOT="${MODULE_TARGET_DIR}/examples_ptx")
set_target_properties(rtspatial_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)
