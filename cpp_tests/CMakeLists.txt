cmake_minimum_required(VERSION 2.8)

project(pycg_descent)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
ADD_DEFINITIONS(-O3)
ADD_DEFINITIONS(-lm)
ADD_DEFINITIONS(-std=c++0x)
ADD_DEFINITIONS(-Wall)

#cmake_policy(SET CMP0015 NEW)

# set the cgd include directory
set(cgd_include /home/sm958/Work/PyCG_DESCENT/source)
include_directories(${cgd_include})
message("cgd include directory ${cgd_include}")

# build the cgd_descent library
FILE(GLOB cgd_sources ${cgd_include}/*.c)
add_library(cgd_lib ${cgd_sources})

# set the pele include directory
set(pele_include /home/sm958/Work/pele/source)
include_directories(${pele_include})
message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.cpp)
add_library(pele_lib ${pele_sources})

## get all the source files
#FILE(GLOB sources *.cpp)

add_executable(test_main bench_minimization.cpp)#${sources}
target_link_libraries(test_main cgd_lib pele_lib)
