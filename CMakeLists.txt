cmake_minimum_required (VERSION 2.6)
project (scrappie C)

include (InstallRequiredSystemLibraries)
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Scrappie is the technology demonstration platform for the Research algorithms' group.")
set (CPACK_PACKAGE_VENDOR "Oxford Nanopore Technologies")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set (CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENCE")
set (CPACK_PACKAGE_VERSION_MAJOR 0)
set (CPACK_PACKAGE_VERSION_MINOR 3)
set (CPACK_PACKAGE_VERSION_PATCH 0)
# Get the latest abbreviated commit hash of the working branch
execute_process(
	COMMAND git log -1 --format=%h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE GIT_COMMIT_HASH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

set (CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}-${GIT_COMMIT_HASH}")
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

add_library (scrappie_lib src/scrappie_raw.c src/decode.c src/nnfeatures.c src/layers.c src/fast5_interface.c src/util.c src/networks.c)
add_executable (scrappie src/scrappie.c)
add_executable (scrappie_raw src/scrappie_raw.c)

include_directories ("${PROJECT_BINARY_DIR}/include")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -O3 -march=native -ffast-math -std=c99 -DUSE_SSE2 -DNDEBUG")
# Check for OpenMP support in compiler
include (CheckCCompilerFlag)
check_c_compiler_flag ("-fopenmp" HAS_OPENMP)
if (HAS_OPENMP)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif (HAS_OPENMP)

# Find right hdf5 file
include (CheckIncludeFile)
check_include_file ("hdf5.h" HDF5_STANDARD)
if (HDF5_STANDARD)
	set (HDF5 "hdf5")
else (HDF5_STANDARD)
	check_include_file ("hdf5/serial/hdf5.h" HDF5_SERIAL)
	if (HDF5_SERIAL)
		set (HDF5 "hdf5_serial")
		target_include_directories (scrappie_lib PUBLIC "/usr/include/hdf5/serial")
		#target_include_directories (scrappie_raw PUBLIC "/usr/include/hdf5/serial")
	endif (HDF5_SERIAL)
endif (HDF5_STANDARD)



target_link_libraries (scrappie scrappie_lib blas ${HDF5} m)
target_link_libraries (scrappie_raw scrappie_lib blas ${HDF5} m)
if (APPLE)
	target_link_libraries (scrappie argp)
	target_link_libraries (scrappie_raw argp)
endif (APPLE)

install (TARGETS scrappie DESTINATION bin)
install (TARGETS scrappie_raw DESTINATION bin)
