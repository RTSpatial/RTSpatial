find_package(Boost REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OptiX_INCLUDE})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include")

RTSPATIAL_COMPILE_SHADERS("${PROJECT_SOURCE_DIR}/include/rtspatial/default_handlers" "counting_handlers.h"
        "${MODULE_TARGET_DIR}/tests_ptx" PROGRAM_MODULES)

add_executable(rtspatial_tests rtspatial_tests.cu ${PROGRAM_MODULES})
target_link_libraries(rtspatial_tests cuda ${GTEST_LIBRARIES})
target_compile_options(rtspatial_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
target_compile_options(rtspatial_tests PRIVATE -DPTX_ROOT="${MODULE_TARGET_DIR}/tests_ptx")
set_target_properties(rtspatial_tests PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON)
