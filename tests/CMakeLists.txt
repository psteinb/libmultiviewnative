INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED      ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

IF(LMVN_TEST_IMAGE_PATH)
message(">> tests: using test images at ${LMVN_TEST_IMAGE_PATH}")
FILE(GLOB TEST_TIFF_FILES_AVAILABLE ${LMVN_TEST_IMAGE_PATH} "*.tif")
configure_file( "${PROJECT_SOURCE_DIR}/tests/tests_config.h.in"
  "${PROJECT_SOURCE_DIR}/tests/tests_config.h"
  )
ELSE()
warning(" unable to configure tests! ")
ENDIF()

add_executable(test_multi_array_fixture test_multi_array_fixture.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_multi_array_fixture boost_system boost_filesystem boost_unit_test_framework)

add_executable(test_image_stack test_image_stack.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_image_stack boost_system boost_filesystem boost_unit_test_framework)


#not sure if this line is correct (appending versus setting)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(FFTW)
IF(FFTW_FOUND)
SET(FFTW_LIBS_TO_USE ${FFTWF_THREADS_LIB} ${FFTWF_LIB} # ${FFTW_THREADS_LIB} ${FFTW_LIB}
  m)
ENDIF()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
    set (CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    set (CMAKE_EXE_LINKER_FLAGS ${OpenMP_EXE_LINKER_FLAGS}) 
    message(">> tests: added ${OpenMP_C_FLAGS} and ${OpenMP_CXX_FLAGS} to compiler flags")

IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_omp.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_omp.*")
  SET(FFTW_LIBS_TO_USE ${FFTWF_OMP_LIB} ${FFTWF_LIB} # ${FFTW_OMP_LIB} ${FFTW_LIB}
    m)
endif()
else(OPENMP_FOUND)
  warning(">> OpenMP not found, expect related error/warning messages")
endif(OPENMP_FOUND)

IF(FFTW_FOUND)
MESSAGE(">> Linking against ${FFTW_LIBS_TO_USE}")
ENDIF()


add_executable(test_fftw_convolve test_fftw_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_fftw_convolve boost_system boost_filesystem boost_unit_test_framework ${FFTW_LIBS_TO_USE})

add_executable(test_discrete_convolve test_discrete_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_discrete_convolve boost_system boost_filesystem boost_unit_test_framework)

add_executable(test_cpu_symm_convolve test_cpu_symm_convolve.cpp)
target_link_libraries(test_cpu_symm_convolve boost_system boost_filesystem boost_unit_test_framework ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})

add_executable(test_cpu_asymm_convolve test_cpu_asymm_convolve.cpp)
target_link_libraries(test_cpu_asymm_convolve boost_system boost_filesystem boost_unit_test_framework ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})

FIND_PACKAGE(TIFF QUIET)

IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp)
target_link_libraries(test_gpu_convolve boost_system boost_filesystem boost_unit_test_framework ${PROJECT_NAME})

CUDA_ADD_EXECUTABLE(test_cuda_memory test_cuda_memory.cu)
target_link_libraries(test_cuda_memory boost_system boost_filesystem boost_unit_test_framework )

IF(TIFF_FOUND AND TEST_TIFF_FILES_AVAILABLE)
CUDA_ADD_EXECUTABLE(test_gpu_deconvolve test_gpu_deconvolve.cpp)
target_link_libraries(test_gpu_deconvolve boost_system boost_filesystem boost_unit_test_framework tiff ${PROJECT_NAME})

CUDA_ADD_EXECUTABLE(test_gpu_deconvolve_single_stepped test_gpu_deconvolve_single_stepped.cu)
target_link_libraries(test_gpu_deconvolve_single_stepped boost_system boost_filesystem boost_unit_test_framework tiff ${PROJECT_NAME})
ELSE()
warning(">> libtiff not found, skipping tests test_gpu_deconvolve and test_gpu_deconvolve_single_stepped")
ENDIF()

ELSE(CUDA_FOUND)
WARNING("Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)



IF(TIFF_FOUND AND TEST_TIFF_FILES_AVAILABLE)
add_executable(test_view_from_disk_fixture test_view_from_disk_fixture.cpp)
target_link_libraries(test_view_from_disk_fixture boost_system boost_filesystem boost_unit_test_framework tiff)

add_executable(test_tiff_stack_fixture test_tiff_stack_fixture.cpp)
target_link_libraries(test_tiff_stack_fixture boost_system boost_filesystem boost_unit_test_framework tiff)

add_executable(test_iteration_data_fixture test_iteration_data_fixture.cpp)
target_link_libraries(test_iteration_data_fixture boost_system boost_filesystem boost_unit_test_framework tiff)

add_executable(test_reference_data_fixture test_reference_data_fixture.cpp)
target_link_libraries(test_reference_data_fixture boost_system boost_filesystem boost_unit_test_framework tiff)

add_executable(test_cpu_deconvolve test_cpu_deconvolve.cpp)
target_link_libraries(test_cpu_deconvolve boost_system boost_filesystem boost_unit_test_framework tiff ${PROJECT_NAME})
find_package(OpenMP)
if (OPENMP_FOUND)
  set_target_properties(test_cpu_deconvolve PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif (OPENMP_FOUND)

add_executable(test_cpu_asymm_convolve_from_tiffs test_cpu_asymm_convolve_from_tiffs.cpp)
target_link_libraries(test_cpu_asymm_convolve_from_tiffs boost_system boost_filesystem boost_unit_test_framework ${FFTW_LIBS_TO_USE} tiff ${PROJECT_NAME})

add_executable(bench_cpu_deconvolve bench_cpu_deconvolve.cpp)
target_link_libraries(bench_cpu_deconvolve boost_system boost_filesystem boost_timer boost_unit_test_framework tiff ${PROJECT_NAME})


ELSE()
WARNING(">> libtiff not found, skipping tests test_view_from_disk_fixture, test_tiff_stack_fixture, test_iteration_data_fixture, test_reference_data_fixture, test_cpu_deconvolve, test_cpu_asymm_convolve_from_tiffs, bench_cpu_deconvolve\!")
ENDIF()


