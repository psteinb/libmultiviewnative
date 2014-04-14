INCLUDE_DIRECTORIES(.)

IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

SET(CMAKE_BUILD_TYPE Debug)

add_executable(test_multi_array_fixture test_multi_array_fixture.cpp)
target_link_libraries(test_multi_array_fixture boost_system boost_filesystem boost_unit_test_framework)

#not sure if this line is correct (appending versus setting)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
FIND_PACKAGE(FFTW)

add_executable(test_fftw_convolve test_fftw_convolve.cpp ${PROJECT_SOURCE_DIR}/src/image_stack_utils.cpp)
target_link_libraries(test_fftw_convolve boost_system boost_filesystem boost_unit_test_framework fftw3f)

add_executable(test_cpu_convolve test_cpu_convolve.cpp)
target_link_libraries(test_cpu_convolve boost_system boost_filesystem boost_unit_test_framework fftw3f ${PROJECT_NAME})


IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp)
target_link_libraries(test_gpu_convolve boost_system boost_filesystem boost_unit_test_framework ${PROJECT_NAME})
ELSE(CUDA_FOUND)
WARNING("Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)




