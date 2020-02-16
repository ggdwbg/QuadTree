cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(quadtree VERSION 0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(quadtree_run_tests driver.cxx binary_sparse_matrix.cpp binary_sparse_matrix.h slow_mul.cpp slow_mul.h CUDA/fast_cuda_mul.cpp CUDA/fast_cuda_mul.h CUDA/quadtree_mul.cu)

set_target_properties(quadtree_run_tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# dummy