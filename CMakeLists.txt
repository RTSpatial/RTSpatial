cmake_minimum_required(VERSION 3.27)
project(RTSpatial LANGUAGES CUDA C CXX)

set(RTSPATIAL_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(RTSPATIAL_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(RTSPATIAL_VERSION ${RTSPATIAL_VERSION_MAJOR}.${RTSPATIAL_VERSION_MINOR} CACHE STRING "version" FORCE)


set(RTSPATIAL_VERSION_MAJOR "1")
set(RTSPATIAL_VERSION_MINOR "0")
set(RTSPATIAL_VERSION_STRING
        "${RTSPATIAL_VERSION_MAJOR}.${RTSPATIAL_VERSION_MINOR}")
set(RTSPATIAL_LIB_VERSION "1.0")

set(ENABLED_ARCHS "80")


if (NOT WIN32)
    # visual studio doesn't like these (not need them):
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17")
    set(CUDA_PROPAGATE_HOST_FLAGS ON CACHE STRING "Propagate C/CXX_FLAGS and friends to the host compiler via -Xcompile")
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if (WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    # Set the base folder where the per-project "core" folders with the *.ptx or *.optixir files get created.
    set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
    # Enable multi-processor build on all Visual Studio versions.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
    # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif ()

# Some useful macros
macro(ADD_TARGET_PROPERTIES _target _name)
    set(_properties)
    foreach (_prop ${ARGN})
        set(_properties "${_properties} ${_prop}")
    endforeach (_prop)
    get_target_property(_old_properties ${_target} ${_name})
    if (NOT _old_properties)
        # In case it's NOTFOUND
        set(_old_properties)
    endif (NOT _old_properties)
    set_target_properties(${_target} PROPERTIES ${_name} "${_old_properties} ${_properties}")
endmacro(ADD_TARGET_PROPERTIES)

macro(TARGET_INCLUDE_SYMBOL target symbol)
    if (WIN32)
        if (LOCAL_ARCH STREQUAL "amd64")
            add_target_properties(${target} LINK_FLAGS /include:${symbol})
        endif ()
    endif ()
    if (UNIX)
        add_target_properties(${target} LINK_FLAGS "-Wl,--undefined=${symbol}")
    endif ()
endmacro()

find_package(CUDAToolkit REQUIRED)

include(cmake/configure_build_type.cmake)
include(cmake/FindOptiX.cmake)
include(cmake/nvcuda_compile_module.cmake)
include_directories(${PROJECT_SOURCE_DIR})
# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(test)