cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(quadtree VERSION 0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(quadtree_run_tests driver.cxx sparse_matrix.cpp sparse_matrix.h slow_mul.cpp slow_mul.h fast_cuda_mul.cpp fast_cuda_mul.h)