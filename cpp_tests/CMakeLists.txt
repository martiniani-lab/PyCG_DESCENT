cmake_minimum_required(VERSION 2.8)

project(pycg_descent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
ADD_DEFINITIONS(-O3)
ADD_DEFINITIONS(-lm)
ADD_DEFINITIONS(-Wall)
ADD_DEFINITIONS(-g)

#cmake_policy(SET CMP0015 NEW)

# set the cgd include directory
set(cgd_include ../source/CG_DESCENT)
include_directories(${cgd_include})
message("cgd include directory ${cgd_include}")

# build the cgd_descent library
FILE(GLOB cgd_sources ${cgd_include}/*.c*)
add_library(cgd_lib ${cgd_sources})

set(PELE_DIR ../../pele)

# set the pele include directory
set(pele_include ${PELE_DIR}/source CACHE STRING "the pele c++ source directory")

if(EXISTS ${pele_include}/pele/array.hpp)
	message("pele include directory: ${pele_include}")
else()
	message(FATAL_ERROR "pele include directory is not correct: ${pele_include} : use ccmake to set it")
endif()

set(PELE_EXTERN_DIR ${PELE_DIR}/extern/install)
set(EXTERN_INCLUDE_DIR ${PELE_EXTERN_DIR}/include/) # FOR EIGEN, CVODE etc
set(LAPACKE_INCLUDE_DIR ${PELE_EXTERN_DIR}/include/Eigen/src/misc)

include_directories(${pele_include})
include_directories(../source)
include_directories(${EXTERN_INCLUDE_DIR})
include_directories(${LAPACKE_INCLUDE_DIR})

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib ${pele_sources})

## get all the source files
#FILE(GLOB sources *.cpp)

add_executable(test_main bench_minimization.cpp)#${sources}
target_link_libraries(test_main cgd_lib pele_lib)
