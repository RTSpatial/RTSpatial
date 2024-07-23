include_directories(${OptiX_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories("${PROJECT_SOURCE_DIR}/include")

set(RTSPATIAL_HDR_DIR "${PROJECT_SOURCE_DIR}/include/rtspatial/")
set(RTPSATIAL_SRC_DIR "${PROJECT_SOURCE_DIR}/src")


###############################################################################
# Library
add_library(rtspatial SHARED "${RTPSATIAL_SRC_DIR}/rt_engine.cu")

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

install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/shaders"
        DESTINATION "${RTSPATIAL_LIB_DIR}")


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

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header files")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Location of library files")

configure_package_config_file("${PROJECT_SOURCE_DIR}/rtspatial-config.in.cmake"
        "${PROJECT_BINARY_DIR}/rtspatial-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial
        PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

install(FILES "${PROJECT_BINARY_DIR}/rtspatial-config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/rtspatial-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial")


install(FILES "${PROJECT_SOURCE_DIR}/cmake/FindOptiX.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/nvcuda_compile_module.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/rtspatial_compile_shaders.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rtspatial")


target_include_directories(rtspatial
        INTERFACE
        $<INSTALL_INTERFACE:include>)

