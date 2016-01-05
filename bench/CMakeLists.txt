INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework chrono thread timer REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

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

add_library( ext_boost_chrono  SHARED IMPORTED )
set_target_properties( ext_boost_chrono PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_chrono.so )

add_library( ext_boost_timer  SHARED IMPORTED )
set_target_properties( ext_boost_timer PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_timer.so )

add_library( ext_boost_thread  SHARED IMPORTED )
set_target_properties( ext_boost_thread PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_thread.so )


add_library( ext_boost_program_options  SHARED IMPORTED )
set_target_properties( ext_boost_program_options PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_program_options.so )

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



IF(CUDA_FOUND)

#TODO: CHECK IF tests is in ${CUDA_NVCC_FLAGS}
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-I${PROJECT_SOURCE_DIR}/tests;-I${PROJECT_SOURCE_DIR}/src)


CUDA_ADD_EXECUTABLE(bench_gpu_deconvolve bench_gpu_deconvolve.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_deconvolve)
target_link_libraries(bench_gpu_deconvolve ext_boost_system ext_boost_filesystem ext_boost_timer ext_boost_chrono ext_boost_unit_test_framework tiff ${PROJECT_NAME})

CUDA_ADD_EXECUTABLE(bench_gpu_deconvolve_synthetic bench_gpu_deconvolve_synthetic.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_deconvolve_synthetic)
target_link_libraries(bench_gpu_deconvolve_synthetic ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_timer ext_boost_chrono ext_boost_unit_test_framework ${PROJECT_NAME})

#set_target_properties(bench_gpu_deconvolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")


CUDA_ADD_EXECUTABLE(bench_gpu_nd_fft bench_gpu_nd_fft.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_nd_fft)
target_link_libraries(bench_gpu_nd_fft ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono)
#set_target_properties(bench_gpu_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

CUDA_ADD_EXECUTABLE(bench_gpu_many_nd_fft bench_gpu_many_nd_fft.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_many_nd_fft)
target_link_libraries(bench_gpu_many_nd_fft ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono)
#set_target_properties(bench_gpu_many_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

CUDA_ADD_EXECUTABLE(bench_gpu_many_nd_folds bench_gpu_many_nd_folds.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_many_nd_folds)
target_link_libraries(bench_gpu_many_nd_folds ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono ${PROJECT_NAME})
#set_target_properties(bench_gpu_many_nd_folds PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests -I${PROJECT_SOURCE_DIR}/inc")


ENDIF()

find_package(OpenMP)
if (OPENMP_FOUND)

# IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_omp.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_omp.*")
#   IF(FFTW_OMP_LIB AND FFTWF_OMP_LIB)
#   SET(FFTW_LIBS_TO_USE ${FFTWF_OMP_LIB} ${FFTW_OMP_LIB} ${FFTW_LIBS_TO_USE})
#   ENDIF() 

# ENDIF()

  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 
  message(">> [BENCH] added ${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS} to compiler flags")

else(OPENMP_FOUND)
  warning(">> OpenMP not found, expect related error/warning messages")
endif(OPENMP_FOUND)

IF(FFTW_FOUND)
MESSAGE(">> [BENCH] Linking against ${FFTW_LIBS_TO_USE}")
ENDIF()

ADD_EXECUTABLE(bench_cpu_nd_fft bench_cpu_nd_fft.cpp)
target_link_libraries(bench_cpu_nd_fft ext_boost_program_options ext_boost_filesystem ext_boost_system ext_boost_thread ext_boost_chrono ${FFTW_LIBS_TO_USE})
set_target_properties(bench_cpu_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

ADD_EXECUTABLE(bench_cpu_deconvolve_synthetic bench_cpu_deconvolve_synthetic.cpp)
target_link_libraries(bench_cpu_deconvolve_synthetic ext_boost_program_options ext_boost_filesystem ext_boost_system ext_boost_thread ext_boost_chrono ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})
set_target_properties(bench_cpu_deconvolve_synthetic PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")


ADD_EXECUTABLE(bench_cpu_many_nd_fft bench_cpu_many_nd_fft.cpp)
target_link_libraries(bench_cpu_many_nd_fft ext_boost_program_options ext_boost_filesystem ext_boost_system ext_boost_thread ext_boost_chrono ${FFTW_LIBS_TO_USE})
set_target_properties(bench_cpu_many_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

ADD_EXECUTABLE(bench_cpu_many_nd_folds bench_cpu_many_nd_folds.cpp)
target_link_libraries(bench_cpu_many_nd_folds ext_boost_program_options ext_boost_filesystem ext_boost_system ext_boost_thread ext_boost_chrono ${FFTW_LIBS_TO_USE} ${PROJECT_NAME})
set_target_properties(bench_cpu_many_nd_folds PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests -I${PROJECT_SOURCE_DIR}/inc")


FIND_PACKAGE(TIFF QUIET)


IF(TIFF_FOUND)

add_executable(bench_cpu_deconvolve bench_cpu_deconvolve.cpp)
target_link_libraries(bench_cpu_deconvolve ext_boost_system ext_boost_filesystem ext_boost_timer ext_boost_chrono ext_boost_unit_test_framework tiff ${PROJECT_NAME})
set_target_properties(bench_cpu_deconvolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

ELSE()
MESSAGE(WARNING ">> libtiff not found, skipping bench_cpu_deconvolve")
ENDIF()


