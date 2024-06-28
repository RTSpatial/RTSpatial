find_package(Boost REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OptiX_INCLUDE})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/test")

add_executable(rtspatial_tests rtspatial_tests.cu)
target_link_libraries(rtspatial_tests ${GTEST_LIBRARIES} rtspatial)
target_compile_options(rtspatial_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rtspatial_tests PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)
