INCLUDE_DIRECTORIES(.)

FIND_PACKAGE (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 


SET(CMAKE_BUILD_TYPE Debug)

add_executable(multi_array_fixture test_multi_array_fixture.cpp)
target_link_libraries(multi_array_fixture boost_system boost_filesystem boost_unit_test_framework)

IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(gpu_convolve test_gpu_convolve.cpp)
target_link_libraries(gpu_convolve boost_system boost_filesystem boost_unit_test_framework ${PROJECT_NAME})

ELSE(CUDA_FOUND)
WARNING("Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)




