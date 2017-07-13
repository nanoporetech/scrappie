cmake_minimum_required (VERSION 2.6)
project (scrappie C)

option (BUILD_SHARED_LIB "Build a shared library" OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
if(NOT CMAKE_BUILD_TYPE)
	message("Defaulting to release build")
	set(CMAKE_BUILD_TYPE Release)
endif()

include (InstallRequiredSystemLibraries)
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Scrappie is the technology demonstration platform for the Research algorithms' group.")
set (CPACK_PACKAGE_VENDOR "Oxford Nanopore Technologies")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set (CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE.md")
set (CPACK_PACKAGE_VERSION_MAJOR 1)
set (CPACK_PACKAGE_VERSION_MINOR 0)
set (CPACK_PACKAGE_VERSION_PATCH 1)

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


##
#   Set up include directories
##
include_directories ("${PROJECT_BINARY_DIR}/include")
if (OPENBLAS_ROOT)
	include_directories ("${OPENBLAS_ROOT}/include")
	link_directories ("${OPENBLAS_ROOT}/lib")
	set (BLAS "openblas")
else ()
	set (BLAS "blas")
endif ()
if (HDF5_ROOT)
	include_directories ("${HDF5_ROOT}/include")
	link_directories ("${HDF5_ROOT}/lib")
	set (CMAKE_REQUIRED_INCLUDES ${HDF5_ROOT}/include)
endif()


##
#   Set up what is to be built
##
add_library (scrappie_objects OBJECT src/decode.c src/event_detection.c src/layers.c src/networks.c src/nnfeatures.c src/scrappie_common.c src/scrappie_matrix.c src/util.c)
set_property(TARGET scrappie_objects PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library (scrappie_static STATIC $<TARGET_OBJECTS:scrappie_objects>)
set_target_properties(scrappie_static PROPERTIES OUTPUT_NAME scrappie CLEAN_DIRECT_OUTPUT 1)
add_executable (scrappie src/scrappie.c src/scrappie_raw.c src/scrappie_events.c src/scrappie_subcommands.c src/scrappie_help.c src/fast5_interface.c)

if (BUILD_SHARED_LIB)
	if (APPLE)
		message (SEND_ERROR "Building shared library on OSX not yet supported")
	endif (APPLE)
	add_library (scrappie_shared SHARED $<TARGET_OBJECTS:scrappie_objects>)
	set_target_properties(scrappie_shared PROPERTIES OUTPUT_NAME scrappie CLEAN_DIRECT_OUTPUT 1)
	install (TARGETS scrappie_shared LIBRARY DESTINATION lib)
endif (BUILD_SHARED_LIB)


##
#  Check whether compiler supports openmp.
##
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -fgnu89-inline -O3 -march=native -std=c99 -DUSE_SSE2 -DNDEBUG")
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -Wall -DABORT_ON_NULL -Wno-unused-function -fgnu89-inline -g -march=native -std=c99 -DUSE_SSE2")
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
		target_include_directories (scrappie_objects PUBLIC "/usr/include/hdf5/serial")
		target_include_directories (scrappie PUBLIC "/usr/include/hdf5/serial")
	endif (HDF5_SERIAL)
endif (HDF5_STANDARD)

target_link_libraries (scrappie scrappie_static ${BLAS} ${HDF5} m)
if (APPLE)
	target_link_libraries (scrappie argp)
endif (APPLE)

install (TARGETS scrappie scrappie_static RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)


enable_testing()
add_executable(scrappie_unittest src/test/scrappie_test_runner.c src/test/test_scrappie_matrix_util.c src/test/scrappie_matrix_util.c src/test/test_scrappie_convolution.c src/test/test_skeleton.c)
target_include_directories(scrappie_unittest PUBLIC "src/test" "src")
target_link_libraries(scrappie_unittest scrappie_static ${BLAS} ${HDF5} m cunit)

set (READSDIR ${PROJECT_SOURCE_DIR}/reads)
set (ENV{OPENBLAS_NUM_THREADS} 1)

add_test(NAME unittest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/ COMMAND scrappie_unittest)
add_test(test_events_call scrappie events --threads 1 ${READSDIR})
add_test(test_raw_call scrappie raw --threads 1 ${READSDIR})
add_test(test_licence scrappie licence)
add_test(test_licence scrappie license)
add_test(test_help scrappie help)
add_test(test_help_events scrappie help events)
add_test(test_help_raw scrappie help raw)
add_test(test_version scrappie version)

add_custom_target(test-verbose COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
