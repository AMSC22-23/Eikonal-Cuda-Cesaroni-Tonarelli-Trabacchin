cmake_minimum_required(VERSION 3.20)
project(Eikonal-Cuda-Cesaroni-Tonarelli-Trabacchin CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

include_directories(src)
add_executable(main test/main.cu)

set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)







#include_directories("/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3")





set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
