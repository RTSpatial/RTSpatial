@PACKAGE_INIT@

set_and_check(RTSPATIAL_INCLUDE_DIR "@PACKAGE_INCLUDE_INSTALL_DIR@")
set_and_check(RTSPATIAL_INCLUDE_DIRS "${RTSPATIAL_INCLUDE_DIR}")
set_and_check(RTSPATIAL_LIBRARY_DIR "@PACKAGE_LIB_INSTALL_DIR@")
set_and_check(RTSPATIAL_PTX_DIR "${RTSPATIAL_LIBRARY_DIR}/ptx")
set(RTSPATIAL_LIBRARIES rtspatial)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

message("OptiX_INSTALL_DIR: ${OptiX_INSTALL_DIR}")

include("${RTSPATIAL_LIBRARY_DIR}/cmake/rtspatial/FindOptiX.cmake")
include_directories(${OptiX_INCLUDE})

message("OptiX_INCLUDE ${OptiX_INCLUDE}")


include(FindPackageMessage)
find_package_message(${RTSPATIAL_LIBRARIES}
        "Found rtspatial: ${CMAKE_CURRENT_LIST_FILE}"
        "Version \"@RTSPATIAL_VERSION@\""
)
