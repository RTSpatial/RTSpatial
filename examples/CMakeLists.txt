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
add_executable(expr expr.cu flags.cpp)

target_link_libraries(expr rtspatial ${GFLAGS_LIBRARIES})
target_compile_options(expr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(expr PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)

###############################################################################
# Demos
add_executable(rtspatial_exec spatial_index.cu flags.cpp)

target_link_libraries(rtspatial_exec rtspatial ${GFLAGS_LIBRARIES})
target_compile_options(rtspatial_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rtspatial_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)
