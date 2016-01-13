INCLUDE_DIRECTORIES(.)
include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_SOURCE_DIR}/src)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

#FIXME: OS dependent
IF(DEFINED ${BOOST_ROOT} AND ${BOOST_ROOT} MATCHES ".*/.*")
  SET(BOOST_LIB_DIR ${BOOST_ROOT}/lib)
ELSE()
  SET(BOOST_LIB_DIR ${Boost_LIBRARY_DIRS})
ENDIF()

add_library( ext_boost_system  SHARED IMPORTED )
set_target_properties( ext_boost_system PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_system.so )

add_library( ext_boost_filesystem  SHARED IMPORTED )
set_target_properties( ext_boost_filesystem PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_filesystem.so )

add_library( ext_boost_unit_test_framework  SHARED IMPORTED )
set_target_properties( ext_boost_unit_test_framework PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_unit_test_framework.so )

#FIXME: OS dependent
#check if the string does end on '/' (if not append it)
string(REGEX MATCH ".*/$"
       TEST_LMVN_TEST_IMAGE_PATH ${LMVN_TEST_IMAGE_PATH})

IF(NOT TEST_LMVN_TEST_IMAGE_PATH)
SET(LMVN_TEST_IMAGE_PATH "${LMVN_TEST_IMAGE_PATH}/")
ENDIF()



IF(LMVN_TEST_IMAGE_PATH)
FILE(GLOB TEST_TIFF_FILES_AVAILABLE "${LMVN_TEST_IMAGE_PATH}/*.tif")
message(">> [tests] using test images at ${LMVN_TEST_IMAGE_PATH}")

configure_file( "${PROJECT_SOURCE_DIR}/tests/tests_config.h.in"
  "${PROJECT_SOURCE_DIR}/tests/tests_config.h"
  )
ELSE()
warning(">> [tests] unable to configure (some) tests! LMVN_TEST_IMAGE_PATH not given")
ENDIF()

add_executable(test_multi_array_fixture test_multi_array_fixture.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_multi_array_fixture ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework)

add_executable(test_image_stack test_image_stack.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_image_stack ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework)

add_executable(test_padd_utils test_padd_utils.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_padd_utils ext_boost_unit_test_framework )

#not sure if this line is correct (appending versus setting)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

## FINDING FFTW
SET(FFTW_USE_STATIC_LIBS TRUE)
FIND_PACKAGE(FFTW QUIET)
IF(NOT FFTW_FOUND)
 message(">> [src] Static FFTW libs not found, trying dynamic libraries")
 SET(FFTW_USE_STATIC_LIBS FALSE)
 FIND_PACKAGE(FFTW REQUIRED)
ENDIF()

IF(FFTW_FOUND)
  SET(FFTW_LIBS_TO_USE ${FFTW_FOUND_SERIAL_LIBS} m)
  INCLUDE_DIRECTORIES(${FFTW_INCLUDES})
  LINK_DIRECTORIES(${FFTW_ROOT}/lib) 
ENDIF()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_omp.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_omp.*")
  SET(FFTW_LIBS_TO_USE ${FFTW_LIBS_TO_USE} ${FFTW_FOUND_OPENMP_LIBS} m)
endif()
else(OPENMP_FOUND)

  IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_threads.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_threads.*")
    SET(FFTW_LIBS_TO_USE ${FFTW_LIBS_TO_USE} ${FFTW_FOUND_THREADS_LIBS} m)
  endif()
  
endif(OPENMP_FOUND)

## FINDING FFTW END


IF(FFTW_FOUND)
MESSAGE(">> [tests] Linking against ${FFTW_LIBS_TO_USE}")
ENDIF()

add_executable(test_point test_point.cpp)
target_link_libraries(test_point ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE})

add_executable(test_plan_store test_plan_store.cpp)
target_link_libraries(test_plan_store ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE})

add_executable(test_fftw_convolve test_fftw_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_fftw_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE})

add_executable(test_discrete_convolve test_discrete_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_discrete_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework)

add_executable(test_cpu_symm_convolve test_cpu_symm_convolve.cpp)
target_link_libraries(test_cpu_symm_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})

add_executable(test_cpu_convolve_api test_cpu_convolve_api.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_cpu_convolve_api ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE} )

add_executable(test_cpu_asymm_convolve test_cpu_asymm_convolve.cpp)
target_link_libraries(test_cpu_asymm_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})

FIND_PACKAGE(TIFF QUIET)

IF(CUDA_FOUND)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-I${PROJECT_SOURCE_DIR}/src)
#include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp)
target_link_libraries(test_gpu_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${PROJECT_NAME})

CUDA_ADD_EXECUTABLE(test_gpu_convolve_impl test_gpu_convolve_impl.cu)
CUDA_ADD_CUFFT_TO_TARGET(test_gpu_convolve_impl)
target_link_libraries(test_gpu_convolve_impl ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework)


CUDA_ADD_EXECUTABLE(test_gpu_batched_fft test_gpu_batched_fft.cu)
CUDA_ADD_CUFFT_TO_TARGET(test_gpu_batched_fft)
target_link_libraries(test_gpu_batched_fft ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework)
set_target_properties(test_gpu_batched_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/src -I${PROJECT_SOURCE_DIR}/tests -I${PROJECT_SOURCE_DIR}/inc")

CUDA_ADD_EXECUTABLE(test_cuda_memory test_cuda_memory.cu)
target_link_libraries(test_cuda_memory ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework )

IF(TEST_TIFF_FILES_AVAILABLE)
  IF(TIFF_FOUND )
    CUDA_ADD_EXECUTABLE(test_gpu_deconvolve test_gpu_deconvolve.cpp)
    target_link_libraries(test_gpu_deconvolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff ${PROJECT_NAME})

    CUDA_ADD_EXECUTABLE(test_gpu_deconvolve_impl test_gpu_deconvolve_impl.cu)
    CUDA_ADD_CUFFT_TO_TARGET(test_gpu_deconvolve_impl)
    target_link_libraries(test_gpu_deconvolve_impl ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff ${PROJECT_NAME})
    set_target_properties(test_gpu_deconvolve_impl PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/src -I${PROJECT_SOURCE_DIR}/tests")

    CUDA_ADD_EXECUTABLE(test_gpu_deconvolve_single_stepped test_gpu_deconvolve_single_stepped.cu)
    target_link_libraries(test_gpu_deconvolve_single_stepped ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff ${PROJECT_NAME})


    CUDA_ADD_EXECUTABLE(test_gpu_plan_store test_plan_store.cu)
    CUDA_ADD_CUFFT_TO_TARGET(test_gpu_plan_store)
    target_link_libraries(test_gpu_plan_store ext_boost_unit_test_framework)

  ELSE()
    MESSAGE(WARNING ">> [tests] libtiff not found, skipping tests test_gpu_deconvolve and test_gpu_deconvolve_single_stepped")
  ENDIF()
ELSE()
  MESSAGE(WARNING ">> [tests] tiff test samples not found")
ENDIF()





ELSE(CUDA_FOUND)
MESSAGE(WARNING ">> [tests] Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)


IF(TEST_TIFF_FILES_AVAILABLE)
IF(TIFF_FOUND)
add_executable(test_view_from_disk_fixture test_view_from_disk_fixture.cpp)
target_link_libraries(test_view_from_disk_fixture ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff)

add_executable(test_tiff_stack_fixture test_tiff_stack_fixture.cpp)
target_link_libraries(test_tiff_stack_fixture ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff)

add_executable(test_iteration_data_fixture test_iteration_data_fixture.cpp)
target_link_libraries(test_iteration_data_fixture ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff)

add_executable(test_reference_data_fixture test_reference_data_fixture.cpp)
target_link_libraries(test_reference_data_fixture ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff)

add_executable(test_cpu_deconvolve test_cpu_deconvolve.cpp)
target_link_libraries(test_cpu_deconvolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework tiff ${PROJECT_NAME})
find_package(OpenMP)
if (OPENMP_FOUND)
  set_target_properties(test_cpu_deconvolve PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif (OPENMP_FOUND)

add_executable(test_cpu_asymm_convolve_from_tiffs test_cpu_asymm_convolve_from_tiffs.cpp)
target_link_libraries(test_cpu_asymm_convolve_from_tiffs ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${FFTW_LIBS_TO_USE} tiff ${PROJECT_NAME})


ELSE()
MESSAGE(WARNING ">> [tests] libtiff not found, skipping tests test_view_from_disk_fixture, test_tiff_stack_fixture, test_iteration_data_fixture, test_reference_data_fixture, test_cpu_deconvolve, test_cpu_asymm_convolve_from_tiffs, bench_cpu_deconvolve")
ENDIF()

ELSE()
MESSAGE(WARNING ">> [tests] tiff test samples not found")
ENDIF()


