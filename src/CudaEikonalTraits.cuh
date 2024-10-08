#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_CUDAEIKONALTRAITS_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_CUDAEIKONALTRAITS_CUH
#include <Eigen/Core>
template <typename Float, size_t D>
struct CudaEikonalTraits {

    using VectorExt = typename Eigen::Matrix<Float, D, 1>;
    using VectorV = typename Eigen::Matrix<Float, D + 1, 1>;
    using Matrix = typename Eigen::Matrix<Float, D, D>;

};
#endif