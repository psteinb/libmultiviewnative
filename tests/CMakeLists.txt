# include directories
# src directory
INCLUDE_DIRECTORIES(.)

FIND_PACKAGE (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 


SET(CMAKE_BUILD_TYPE Debug)

add_executable(multi_array_fixture test_multi_array_fixture.cpp)
target_link_libraries(multi_array_fixture boost_system boost_filesystem boost_unit_test_framework)




# build and link
# CUDA_ADD_LIBRARY(${PROJECT_NAME} multiviewnative.cu SHARED)
# CUDA_ADD_CUFFT_TO_TARGET(${PROJECT_NAME})
# TARGET_LINK_LIBRARIES (${PROJECT_NAME} cuda )

# SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/inc/multiviewnative.h") 
# INSTALL(TARGETS ${PROJECT_NAME}
#   EXPORT multiviewnative-targets 
#   LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
#   PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT inc)




