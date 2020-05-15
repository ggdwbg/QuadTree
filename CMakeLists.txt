cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(quadtree VERSION 0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O0 -march=native -lcudart")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "")

add_executable(quadtree_run_tests
        driver.cxx
        converter.cpp
        converter.h
        CUDA/quadtree.cpp
        CUDA/quadtree.h
        CUDA/cuda_mul.cu
        )
target_compile_options(quadtree_run_tests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-g -G --gpu-architecture=sm_50>)

set_target_properties(quadtree_run_tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

# dummy