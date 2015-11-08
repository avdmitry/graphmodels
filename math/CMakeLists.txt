cmake_minimum_required(VERSION 2.8)

project(math)

include(../cmake/utils.cmake)

option(BUILD_CUDA "build cuda math library" OFF)
option(BUILD_BLAS "build blas math library" OFF)
option(BUILD_CPU "build cpu math library" ON)

find_package(BLAS)
if (${BLAS_FOUND})
  message(STATUS "blas library: " ${BLAS_LIBRARIES})
else()
  set(BUILD_BLAS OFF CACHE BOOL "build blas math library")
endif()

find_package(CUDA 6.5)
if (${CUDA_FOUND})
  message(STATUS "cuda library: " ${CUDA_LIBRARIES})
  message(STATUS "cuda include dir: " ${CUDA_INCLUDE_DIRS})
else()
  set(BUILD_CUDA OFF CACHE BOOL "build cuda math library")
endif()

if (BUILD_CUDA)
add_definitions(-DBUILD_CUDA)
endif()
if (BUILD_BLAS)
add_definitions(-DBUILD_BLAS)
endif()
if (BUILD_CPU)
add_definitions(-DBUILD_CPU)
endif()

message(STATUS "BUILD CUDA: " ${BUILD_CUDA})
message(STATUS "BUILD BLAS: " ${BUILD_BLAS})
message(STATUS "BUILD CPU: " ${BUILD_CPU})

#OFF
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_FLAGS "-O3 -ffast-math -std=c++11")

list(APPEND CUDA_NVCC_FLAGS "-O3 --use_fast_math -std=c++11")
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

file(GLOB_RECURSE srcs ./*.cc)
file(GLOB_RECURSE scus ./*.cu*)
file(GLOB_RECURSE hdrs ./*.h*)

CUDA_ADD_LIBRARY(math
  ${scus} ${srcs} ${hdrs}
  SHARED
  OPTIONS
  RELEASE -DNDEBUG
  DEBUG -g -DDEBUG
  )

default_properties(math)

#/usr/lib/libopenblas.so.0
#/usr/lib/libatlas.so
set(CUDNN_PATH "/usr/local/cuda/lib64")

target_link_libraries(math /usr/local/cuda/lib64/libcublas.so /usr/lib/libopenblas.so.0 ${CUDNN_PATH}/libcudnn.so)

message(STATUS " Build type:       ${CMAKE_BUILD_TYPE}")
message("")

