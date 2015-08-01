cmake_minimum_required(VERSION 2.8.8)

project(graphmodels CXX C)

include(cmake/utils.cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB_RECURSE srcs src/*.cc)
file(GLOB_RECURSE hdrs include/*.h*)

include_directories(include)
add_library(graphmodels ${srcs} ${hdrs})
default_properties(graphmodels)
set(CMAKE_CXX_FLAGS "-std=c++11")

add_applications()

message(STATUS " Build type:       ${CMAKE_BUILD_TYPE}")
message("")

