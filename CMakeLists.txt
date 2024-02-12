cmake_minimum_required(VERSION 3.20)
project(Eikonal-Cuda-Cesaroni-Tonarelli-Trabacchin CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(main main.cu)

set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)