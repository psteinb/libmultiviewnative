INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED      ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

SET(CMAKE_BUILD_TYPE Debug)

add_executable(test_multi_array_fixture test_multi_array_fixture.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_multi_array_fixture boost_system boost_filesystem boost_unit_test_framework)

#not sure if this line is correct (appending versus setting)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
FIND_PACKAGE(FFTW)

add_executable(test_fftw_convolve test_fftw_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_fftw_convolve boost_system boost_filesystem boost_unit_test_framework fftw3f)

add_executable(test_discrete_convolve test_discrete_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_discrete_convolve boost_system boost_filesystem boost_unit_test_framework)

add_executable(test_cpu_symm_convolve test_cpu_symm_convolve.cpp)
target_link_libraries(test_cpu_symm_convolve boost_system boost_filesystem boost_unit_test_framework fftw3f ${PROJECT_NAME})

add_executable(test_cpu_asymm_convolve test_cpu_asymm_convolve.cpp)
target_link_libraries(test_cpu_asymm_convolve boost_system boost_filesystem boost_unit_test_framework fftw3f tiff ${PROJECT_NAME})

IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp)
target_link_libraries(test_gpu_convolve boost_system boost_filesystem boost_unit_test_framework ${PROJECT_NAME})

CUDA_ADD_EXECUTABLE(test_cuda_memory test_cuda_memory.cu)
target_link_libraries(test_cuda_memory boost_system boost_filesystem boost_unit_test_framework )

ELSE(CUDA_FOUND)
WARNING("Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)

FIND_PACKAGE(TIFF REQUIRED)

IF(TIFF_FOUND)
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

add_executable(test_cpu_asymm_convolve_from_tiffs test_cpu_asymm_convolve_from_tiffs.cpp)
target_link_libraries(test_cpu_asymm_convolve_from_tiffs boost_system boost_filesystem boost_unit_test_framework fftw3f tiff ${PROJECT_NAME})

ELSE(TIFF_FOUND)
WARNING("Skipping tests based on reference images, libtiff not found\!")
ENDIF(TIFF_FOUND)


