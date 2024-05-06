cmake_minimum_required(VERSION 3.20)
project(Eikonal-Cuda-Cesaroni-Tonarelli-Trabacchin CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CUDA_ARCHITECTURES 61)


include_directories(src)
include_directories(test)
include_directories("localProblem_alt2/include")
include_directories("/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3")

add_executable(test test/test_local_solver.cu src/Kernels.cu src/Solver.cuh)

add_executable(main test/test_solver.cu src/Solver.cuh src/Mesh.cuh)

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        >)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        >)

