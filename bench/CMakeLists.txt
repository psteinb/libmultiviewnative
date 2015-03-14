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

FIND_PACKAGE(FFTW)
IF(FFTW_FOUND)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${FFTW_INCLUDES} -L${FFTW_ROOT}lib")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${FFTW_INCLUDES} -L${FFTW_ROOT}/lib")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${FFTW_INCLUDES} -L${FFTW_ROOT}/lib")

SET(FFTW_LIBS_TO_USE ${FFTWF_LIB} ${FFTW_LIB} m)

IF("${FFTW_LIBRARIES}" MATCHES ".*fftw3f_omp.*" AND "${FFTW_LIBRARIES}" MATCHES ".*fftw3_omp.*")
  IF(FFTW_OMP_LIB AND FFTWF_OMP_LIB)
  SET(FFTW_LIBS_TO_USE ${FFTWF_OMP_LIB} ${FFTW_OMP_LIB} ${FFTW_LIBS_TO_USE})
  ENDIF() 
ELSE()
  IF(FFTWF_THREADS_LIB AND FFTW_THREADS_LIB)
    SET(FFTW_LIBS_TO_USE ${FFTWF_THREADS_LIB} ${FFTW_THREADS_LIB} ${FFTW_LIBS_TO_USE} )
  ENDIF()
ENDIF()


ENDIF()

IF(CUDA_FOUND)
# CUDA_ADD_EXECUTABLE(bench_gpu_deconvolve bench_gpu_deconvolve.cu)
# target_link_libraries(bench_gpu_deconvolve boost_program_options boost_regex boost_system boost_timer ${PROJECT_NAME})
# set_target_properties(bench_gpu_deconvolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

CUDA_ADD_EXECUTABLE(bench_gpu_nd_fft bench_gpu_nd_fft.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_nd_fft)
target_link_libraries(bench_gpu_nd_fft ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono)
set_target_properties(bench_gpu_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

CUDA_ADD_EXECUTABLE(bench_gpu_many_nd_fft bench_gpu_many_nd_fft.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_many_nd_fft)
target_link_libraries(bench_gpu_many_nd_fft ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono)
set_target_properties(bench_gpu_many_nd_fft PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests")

CUDA_ADD_EXECUTABLE(bench_gpu_many_nd_folds bench_gpu_many_nd_folds.cu)
CUDA_ADD_CUFFT_TO_TARGET(bench_gpu_many_nd_folds)
target_link_libraries(bench_gpu_many_nd_folds ext_boost_program_options ext_boost_system ext_boost_filesystem ext_boost_chrono ${PROJECT_NAME})
set_target_properties(bench_gpu_many_nd_folds PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/tests -I${PROJECT_SOURCE_DIR}/inc")


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


