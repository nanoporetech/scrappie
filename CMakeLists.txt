cmake_minimum_required (VERSION 2.6)
project (scrappie)

include (InstallRequiredSystemLibraries)
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Scrappie is the technology demonstration platform for the Research algorithms' group.")
set (CPACK_PACKAGE_VENDOR "Oxford Nanopore Technologies")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set (CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENCE")
set (CPACK_PACKAGE_VERSION_MAJOR 0)
set (CPACK_PACKAGE_VERSION_MINOR 1)
set (CPACK_PACKAGE_VERSION_PATCH 1)
set (CI_VERSION_PATCH $ENV{CI_VERSION_PATCH})
set (CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}-${CI_VERSION_PATCH}")
set (CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
set (CPACK_PACKAGE_NAME "ont-${PROJECT_NAME}")

set (CPACK_DEBIAN_PACKAGE_MAINTAINER "Tim Massingham <tim.massingham@nanoporetech.com>")
set (CPACK_DEBIAN_PACKAGE_SECTION "base")
set (CPACK_DEBIAN_PACKAGE_DEPENDS "libopenblas-base, libhdf5-7")
set (CPACK_DEBIAN_BUILD_DEPENDS "libopenblas-base, libopenblas-dev, libhdf5-7, libhdf5-dev, cmake")
set (CPACK_PACKAGING_INSTALL_PREFIX "/opt/scrappie")

set (CPACK_GENERATOR "TGZ;DEB")
include (CPack)


configure_file (
    "${PROJECT_SOURCE_DIR}/src/version.h.in"
    "${PROJECT_BINARY_DIR}/include/version.h"
)

include_directories ("${PROJECT_BINARY_DIR}/include")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -O3 -fopenmp -march=core2 -ffast-math -std=c99 -DUSE_SSE2 -DNDEBUG")

add_executable (scrappie src/scrappie.c src/decode.c src/features.c src/layers.c src/read_events.c src/util.c)
target_link_libraries (scrappie blas hdf5 m)
install (TARGETS scrappie DESTINATION bin)
