//
// Created by Melanie Tonarelli on 16/02/24.
//


#include "../localProblem_alt2/include/Phi.hpp"
#include "../localProblem_alt2/include/solveEikonalLocalProblem.hpp"
#include <iostream>
#include "../src/LocalSolver.cuh"

int main() {
    constexpr int D = 3;
    constexpr int N = 4;
    using VectorExt = typename Eikonal::Eikonal_traits<D, N - 2>::VectorExt;
    using VectorV = typename Eigen::Matrix<double,4,1>;
    VectorV values;
    //values << 0.26718, 0.405334, 0.275872, 0.000000;
    std::array<VectorExt, N> coordinates;



    /*coordinates[3] = {0.25,0.511862,1};
    coordinates[2] = {1, 0.684215, 0.813472};
    coordinates[1] = {1, 1.2, 1};
    coordinates[0] = {1, 0.506082 , 0.867624};
     */

    values << 0.275872, 0.696969, 0.26718, 0.405334;
    coordinates[3] = {1, 1.2, 1};
    coordinates[2] = {1, 0.506082 , 0.867624};
    coordinates[1] = {0.25,0.511862,1};
    coordinates[0] = {1, 0.684215, 0.813472};

    typename Eikonal::Eikonal_traits<D,N - 2>::AnisotropyM velocity;
    velocity << 1,0,0,
            0,1,0,
            0,0,1;

    VectorExt e12 = coordinates[1] - coordinates[0];
    VectorExt e13 = coordinates[2] - coordinates[0];
    VectorExt e23 = coordinates[2] - coordinates[1];
    VectorExt e14 = coordinates[3] - coordinates[0];
    VectorExt e24 = coordinates[3] - coordinates[1];
    VectorExt e34 = coordinates[3] - coordinates[2];
    double M[6];
    M[0] = e12.transpose() * velocity * e12;
    M[1] = e13.transpose() * velocity * e13;
    M[2] = e23.transpose() * velocity * e23;
    M[3] = e14.transpose() * velocity * e14;
    M[4] = e24.transpose() * velocity * e24;
    M[5] = e34.transpose() * velocity * e34;
    Eikonal::SimplexData<D, N> simplex{coordinates, velocity};
    //Eikonal::solveEikonalLocalProblem<N, D> localSolver{simplex,values};
    //auto sol_prof = localSolver();
    auto [sol_our, lambda1, lambda2] = LocalSolver<D, double>::solve(coordinates, values, M, velocity, 2);

    //std::cout << "sol prof = " << sol_prof.value << std::endl;
    //std::cout << "lambda prof = " << sol_prof.lambda << std::endl;
    std::cout << "lambda our = " << lambda1 << " " << lambda2 << std::endl;
    std::cout << "sol our = " << sol_our << std::endl;


    VectorExt alpha;
    VectorExt beta;
    VectorExt gamma;
    VectorExt sol;
    sol << lambda1, lambda2, 1.0 - lambda1 - lambda2;

    alpha << e13.transpose() * velocity * e13, e23.transpose() * velocity * e13, e34.transpose() * velocity * e13;
    beta << e13.transpose() * velocity * e23, e23.transpose() * velocity * e23, e34.transpose() * velocity * e23;
    gamma << e13.transpose() * velocity * e34, e23.transpose() * velocity * e34, e34.transpose() * velocity * e34;
    typename Eikonal::Eikonal_traits<D,N - 2>::AnisotropyM m;
    m << alpha[0], beta[0], gamma[0],
            alpha[1], beta[1], gamma[1],
            alpha[2], beta[2], gamma[2];

    double lhs1 = (values[2]-values[0])*sqrt(sol.transpose()*m*sol);
    double rhs1 = sol.transpose() * alpha;

    double lhs2 = (values[2] - values[1]) * sol.transpose() * alpha;
    double rhs2 = (values[2] - values[0]) * sol.transpose() * beta;

    std::cout << "lhs1 = " << lhs1 << std::endl;
    std::cout << "rhs1 = " << rhs1 << std::endl;
    std::cout << "lhs2 = " << lhs2 << std::endl;
    std::cout << "rhs2 = " << rhs2 << std::endl;


    auto [rotate, sign] = LocalSolver<D, double>::rotate(LocalSolver<D, double>::getGrayCode(1, 2), 2);
    std::cout << rotate << " " << sign << std::endl;
}
