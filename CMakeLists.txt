cmake_minimum_required (VERSION 2.8.12)
project (scrappie C)

option (BUILD_SHARED_LIB "Build a shared library" OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Chaos;Release")
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
set (CPACK_PACKAGE_VERSION_MINOR 4)
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
set (CPACK_DEBIAN_PACKAGE_DEPENDS "libopenblas-base, libhdf5-7, libcunit1")
set (CPACK_DEBIAN_BUILD_DEPENDS "libopenblas-base, libopenblas-dev, libhdf5-7, libhdf5-dev, cmake, libcunit1-dev")
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


##
#   Set up what is to be built
##
add_library (scrappie_objects OBJECT src/decode.c src/event_detection.c src/layers.c src/networks.c src/nnfeatures.c src/scrappie_common.c src/scrappie_matrix.c src/scrappie_seq_helpers.c src/util.c src/homopolymer.c)
set_property(TARGET scrappie_objects PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library (scrappie_static STATIC $<TARGET_OBJECTS:scrappie_objects>)
set_target_properties(scrappie_static PROPERTIES OUTPUT_NAME scrappie CLEAN_DIRECT_OUTPUT 1)
add_executable (scrappie_squiggle src/scrappie.c src/scrappie_squiggle.c src/scrappie_subcommands.c src/scrappie_help.c )

if (BUILD_SHARED_LIB)
	if (APPLE)
		message (SEND_ERROR "Building shared library on OSX not yet supported")
	endif (APPLE)
	add_library (scrappie_shared SHARED $<TARGET_OBJECTS:scrappie_objects>)
	set_target_properties(scrappie_shared PROPERTIES OUTPUT_NAME scrappie CLEAN_DIRECT_OUTPUT 1)
	install (TARGETS scrappie_shared LIBRARY DESTINATION lib)
endif (BUILD_SHARED_LIB)

#if ((CMAKE_BUILD_VERSION strequal "Chaos") AND (not defined CHAOSMONKEY))
#    set(CHAOSMONEY 0.1)
#endif ((CMAKE_BUILD_VERSION strequal "Chaos") AND (not defined CHAOSMONKEY))
if (NOT DEFINED CHAOSMONKEY)
    set(CHAOSMONKEY 0.1)
endif ()


##
#  Check whether compiler supports openmp.
##
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -fstack-protector-all -fgnu89-inline -O3 -march=native -std=c99 -DUSE_SSE2 -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG")
set (CMAKE_C_FLAGS_CHAOS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -fstack-protector-all -fgnu89-inline -g -march=native -std=c99 -DUSE_SSE2 -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG -DCHAOSMONKEY=${CHAOSMONKEY}")
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -Werror -Wall -Wno-cpp -DABORT_ON_NULL -Wno-unused-function -fstack-protector-all -fgnu89-inline -g -march=native -std=c99 -DUSE_SSE2 -D__USE_MISC -D_POSIX_SOURCE")
# Check for OpenMP support in compiler
include (CheckCCompilerFlag)
check_c_compiler_flag ("-fopenmp" HAS_OPENMP)
if (HAS_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set (USE_THREADS "--threads" "1")
endif (HAS_OPENMP)


target_link_libraries (scrappie_squiggle scrappie_static ${BLAS} m)
if (APPLE)
	target_link_libraries (scrappie_squiggle argp)
endif (APPLE)

install (TARGETS scrappie_squiggle scrappie_static RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)

