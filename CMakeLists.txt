# cmake compatibility issues
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# project name
PROJECT(MultiViewNative CXX)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
 
# version number
SET (MULTIVIEWNATIVE_NAME "CUDA Multi-View Deconvolution (Native) Library")
SET (MULTIVIEWNATIVE_CODENAME "${PROJECT_NAME}")
SET (MULTIVIEWNATIVE_COPYRIGHT_YEARS "2014")
SET (MULTIVIEWNATIVE_VERSION_MAJOR 0)
SET (MULTIVIEWNATIVE_VERSION_MINOR 0)
SET (MULTIVIEWNATIVE_VERSION_PATCH 0)
SET (MULTIVIEWNATIVE_VERSION "${MULTIVIEWNATIVE_VERSION_MAJOR}.${MULTIVIEWNATIVE_VERSION_MINOR}.${MULTIVIEWNATIVE_VERSION_PATCH}")
SET (MULTIVIEWNATIVE_VENDOR_ID "mpi cbg")
SET (MULTIVIEWNATIVE_VENDOR_NAME "Max Planck Institute of Molecular Cell Biology and Genetics ")
SET (MULTIVIEWNATIVE_VENDOR_URL "www.mpi-cbg.de")
SET (MULTIVIEWNATIVE_ID "${MULTIVIEWNATIVE_VENDOR_ID}.${PROJECT_NAME}")

# trying to setup paths so this package can be picked up
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

# shared path is architecture independent for now, TODO extend this to lib/bin/include
IF(UNIX)
  IF(APPLE)
    set(INSTALL_SHARE_DIR ${PROJECT_NAME}.app/Contents/Resources/ CACHE PATH "Installation directory for shared files")
    #the following was tested with OSX 10.8.5 and Xcode 5.0.2
    #seems to me that under apple the rpath is not stripped automatically when doing the install
    #under linux it is
    SET(CMAKE_SKIP_RPATH ON)
  ELSE(APPLE)
    set(INSTALL_SHARE_DIR lib/CMake/${PROJECT_NAME} CACHE PATH "Installation directory for shared files")
  ENDIF(APPLE)
ELSE(UNIX)
  IF(WIN32 AND NOT CYGWIN)
    set(INSTALL_SHARE_DIR CMake CACHE PATH "Installation directory for shared files")
  ELSE(WIN32)
    MESSAGE(FATAL_ERROR ">> UNKNOWN ARCHITECTURE .. unable to set share dir")
  ENDIF(WIN32)
ENDIF(UNIX)
 
# Make relative paths absolute (needed later on)
foreach(p LIB INCLUDE SHARE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

include_directories("${PROJECT_SOURCE_DIR}/inc")

# set project build type
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")

#C/C++ specific flags
IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -ggdb ")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -ggdb ")
ENDIF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
 
IF("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -ggdb ")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -ggdb ")
ENDIF()

check_cxx_compiler_flag(-fvectorize HAS_CLANG_VECTORIZE_COMPILERFLAG)
check_cxx_compiler_flag(-ftree-vectorize HAS_GCC_TREE_VECTORIZE_COMPILERFLAG)
check_cxx_compiler_flag(-march=native HAS_MARCH_COMPILERFLAG)
check_cxx_compiler_flag(-ffast-math HAS_FAST_MATH_COMPILERFLAG)

SET(HOST_COMPILER_RELEASE_FLAGS "-Wall -O3")
IF(HAS_CLANG_VECTORIZE_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -fvectorize")
ENDIF()

IF(HAS_GCC_TREE_VECTORIZE_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -ftree-vectorize")
ENDIF()

IF(HAS_MARCH_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -march=native")
ENDIF()

IF(HAS_FAST_MATH_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -ffast-math")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${HOST_COMPILER_RELEASE_FLAGS} ")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${HOST_COMPILER_RELEASE_FLAGS} ")
MESSAGE(">> using release build flags: ${HOST_COMPILER_RELEASE_FLAGS}")
#########################################################################################################
## CUDA related
# project options
OPTION(INCLUDE_CUDA "Set to OFF to not search for CUDA" ON)
# find project dependencies
# find cuda
IF(INCLUDE_CUDA)
  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    IF("${CUDA_LIBRARIES}" MATCHES ".*libcuda.so.*")
      message(STATUS "libcuda.so found by FindCUDA.cmake")
    ELSE()
      message(STATUS "libcuda.so not found by FindCUDA.cmake")
      find_library(LIBCUDA_LOCATION libcuda.so PATHS /usr/lib64 /usr/lib PATH_SUFFIXES /nvidia-current /nvidia-bumblebee)
      find_path(LIBCUDA_PATH libcuda.so PATHS /usr/lib64 /usr/lib PATH_SUFFIXES /nvidia-current /nvidia-bumblebee)
      SET(CUDA_LIBRARIES "${CUDA_LIBRARIES};${LIBCUDA_LOCATION}")
      IF("${CUDA_LIBRARIES}" MATCHES ".*libcuda.so.*")
	message(STATUS "manually added libcuda.so to CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-L${LIBCUDA_PATH} -lcuda")
      ELSE()
	message(WARNING "unable to find libcuda.so, cross fingers to continue")
      ENDIF()
    ENDIF()
    SET(CUDA_VERBOSE_BUILD ON)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_10,code=sm_10;-gencode arch=compute_20,code=sm_20)
    SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    IF(APPLE)
      IF(${CUDA_HOST_COMPILER} MATCHES "/usr/bin/.*cc" OR EXISTS "/usr/bin/llvm-g++")
	MESSAGE(">> adapting CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) to match a CUDA supported compiler (/usr/bin/llvm-g++-4.2)")
	SET(CUDA_HOST_COMPILER "/usr/bin/llvm-g++")
	SET(CMAKE_CXX_COMPILER ${CUDA_HOST_COMPILER})
	SET(CMAKE_C_COMPILER "/usr/bin/llvm-gcc")
	#gcc-4.2 is quite old and usually does not support 
	STRING(REPLACE "${HOST_COMPILER_RELEASE_FLAGS}" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
	STRING(REPLACE "${HOST_COMPILER_RELEASE_FLAGS}" "" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
      ELSE()
	MESSAGE(WARNING ">> unknown CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) or /usr/bin/llvm-g++-4.2 does not exist, cuda host compiler remains set to default")
      ENDIF()
    ENDIF(APPLE)
    IF("${CUDA_VERSION}" VERSION_GREATER "5" OR "${CUDA_VERSION}" VERSION_EQUAL "5")
      MESSAGE(">> compiling for Compute Capability 2.x, 3.0 and 3.5 only ")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30;-gencode arch=compute_35,code=sm_35")
    ELSE("${CUDA_VERSION_MAJOR}" STREQUAL "5")
      MESSAGE(">> CUDA less than version 5.0 detected, compiling for Compute Capability 2.x only ")
    ENDIF()
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O2;--use_fast_math)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-g;-G)
  ELSE(CUDA_FOUND)
    MESSAGE(FATAL_ERROR ">> CUDA not found. Exiting ...")
  ENDIF(CUDA_FOUND)
ENDIF(INCLUDE_CUDA)
 

# add subdirectories
ADD_SUBDIRECTORY(src)

#play around with tests
ADD_SUBDIRECTORY(tests)
enable_testing()
include("CTestLists.txt")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
#              v = binary    v = library

export(PACKAGE ${PROJECT_NAME})

# Create the multiviewnative-config.cmake and multiviewnative-config-version files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_SHARE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(multiviewnative-config.cmake.in
  "${PROJECT_BINARY_DIR}/multiviewnative-config.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${MULTIVIEWNATIVE}/${REL_INCLUDE_DIR}")
configure_file(multiviewnative-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/multiviewnative-config.cmake" @ONLY)
# ... for both
configure_file(multiviewnative-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/multiviewnative-config-version.cmake" @ONLY)

# Install the multiviewnative-config.cmake and multiviewnative-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/multiviewnative-config.cmake"
  "${PROJECT_BINARY_DIR}/multiviewnative-config-version.cmake"
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree 
install(EXPORT multiviewnative-targets 
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)





