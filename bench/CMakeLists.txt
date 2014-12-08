INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

#FIXME: OS dependent
#check if the string does end on '/' (if not append it)
string(REGEX MATCH ".*/$"
       TEST_LMVN_TEST_IMAGE_PATH ${LMVN_TEST_IMAGE_PATH})

IF(NOT TEST_LMVN_TEST_IMAGE_PATH)
SET(LMVN_TEST_IMAGE_PATH "${LMVN_TEST_IMAGE_PATH}/")
ENDIF()



IF(LMVN_TEST_IMAGE_PATH)
FILE(GLOB TEST_TIFF_FILES_AVAILABLE "${LMVN_TEST_IMAGE_PATH}/*.tif")
message(">> tests: using test images at ${LMVN_TEST_IMAGE_PATH}")

configure_file( "${PROJECT_SOURCE_DIR}/tests/tests_config.h.in"
  "${PROJECT_SOURCE_DIR}/tests/tests_config.h"
  )
ELSE()
warning(">> unable to configure (some) tests! LMVN_TEST_IMAGE_PATH not given")
ENDIF()

#not sure if this line is correct (appending versus setting)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(FFTW)
IF(FFTW_FOUND)
SET(FFTW_LIBS_TO_USE ${FFTWF_THREADS_LIB} ${FFTWF_LIB} m)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${FFTW_INCLUDES}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${FFTW_INCLUDES}")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${FFTW_INCLUDES}")
ENDIF()

IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(bench_gpu_deconvolve bench_gpu_deconvolve.cu)
target_link_libraries(bench_gpu_deconvolve boost_program_options boost_regex boost_system boost_timer ${PROJECT_NAME})
set_target_properties(bench_gpu_deconvolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

ENDIF()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 
    message(">> tests: added ${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS} to compiler flags")

IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_omp.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_omp.*")
  SET(FFTW_LIBS_TO_USE ${FFTWF_OMP_LIB} ${FFTWF_LIB} 
    m)
endif()
else(OPENMP_FOUND)
  warning(">> OpenMP not found, expect related error/warning messages")
endif(OPENMP_FOUND)

IF(FFTW_FOUND)
MESSAGE(">> Linking against ${FFTW_LIBS_TO_USE}")
ENDIF()


FIND_PACKAGE(TIFF QUIET)


IF(TIFF_FOUND AND TEST_TIFF_FILES_AVAILABLE)

add_executable(bench_cpu_deconvolve bench_cpu_deconvolve.cpp)
target_link_libraries(bench_cpu_deconvolve boost_system boost_filesystem boost_timer boost_unit_test_framework tiff ${PROJECT_NAME})
set_target_properties(bench_cpu_deconvolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

ELSE()
MESSAGE(WARNING ">> libtiff not found, skipping tests test_view_from_disk_fixture, test_tiff_stack_fixture, test_iteration_data_fixture, test_reference_data_fixture, test_cpu_deconvolve, test_cpu_asymm_convolve_from_tiffs, bench_cpu_deconvolve")
ENDIF()


