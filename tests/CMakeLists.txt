# include directories
# src directory
INCLUDE_DIRECTORIES(.)

FIND_PACKAGE (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 

add_executable(Independent test_Independent.cpp)
target_link_libraries(Independent system filesystem unit_test_framework)
MESSAGE("LINKING TO INDEPENDENT .. ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}")

# build and link
# CUDA_ADD_LIBRARY(${PROJECT_NAME} multiviewnative.cu SHARED)
# CUDA_ADD_CUFFT_TO_TARGET(${PROJECT_NAME})
# TARGET_LINK_LIBRARIES (${PROJECT_NAME} cuda )

# SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/inc/multiviewnative.h") 
# INSTALL(TARGETS ${PROJECT_NAME}
#   EXPORT multiviewnative-targets 
#   LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
#   PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT inc)




