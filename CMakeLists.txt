cmake_minimum_required (VERSION 2.6)
project (Scrappie)

set (Scrappie_VERSION_MAJOR 0)
set (Scrappie_VERSION_MINOR 1)
set (Scrappie_VERSION_SUB 0)
set (Scrappie_VERSION_PATCH 0)
configure_file (
    "${PROJECT_SOURCE_DIR}/src/version.h.in"
    "${PROJECT_BINARY_DIR}/include/version.h"
)
configure_file (
    "${PROJECT_SOURCE_DIR}/DEBIAN/control"
    "${PROJECT_BINARY_DIR}/tmp/control"
)

include_directories ("${PROJECT_BINARY_DIR}/include")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -O3 -fopenmp -march=core2 -ffast-math -std=c99 -DUSE_SSE2 -DNDEBUG")

add_executable (scrappie src/basecall.c src/decode.c src/features.c src/layers.c src/read_events.c src/util.c)
target_link_libraries (scrappie blas hdf5 m)
install (TARGETS scrappie DESTINATION bin)
