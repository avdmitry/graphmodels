cmake_minimum_required(VERSION 2.8.8)

project(graphmodels CXX C)

include(cmake/utils.cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB_RECURSE srcs src/*.cc)
file(GLOB_RECURSE test test/*.cc)
file(GLOB_RECURSE hdrs include/*.h*)

include_directories(include)
add_library(graphmodels ${srcs} ${test} ${hdrs})
default_properties(graphmodels)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")

add_subdirectory(math)

add_applications()

message(STATUS " Build type:       ${CMAKE_BUILD_TYPE}")
message("")

