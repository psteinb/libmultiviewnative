# include directories
# src directory
INCLUDE_DIRECTORIES(.)

FIND_PACKAGE (Boost 1.40 COMPONENTS "thread" REQUIRED)

# build and link
CUDA_ADD_LIBRARY(${PROJECT_NAME} multiviewnative.cu multiviewnative.cpp SHARED)
CUDA_ADD_CUFFT_TO_TARGET(${PROJECT_NAME})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
FIND_PACKAGE(FFTW)

target_link_libraries(${PROJECT_NAME} fftw3f_threads fftw3f fftw3_threads fftw3 boost_thread-mt)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/inc/multiviewnative.h") 
INSTALL(TARGETS ${PROJECT_NAME}
  EXPORT multiviewnative-targets 
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT inc)




