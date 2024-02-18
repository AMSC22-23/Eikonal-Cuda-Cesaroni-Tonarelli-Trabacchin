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
    VectorExt values;
    values << 5, 1.5, 7;
    //std::array<std::array<double, D>, N> coordinates;
    std::array<VectorExt, N> coordinates;
    coordinates[3] = {0,1,0};
    coordinates[2] = {1, 0, 0};
    coordinates[1] = {1.5, 1, 0};
    coordinates[0] = {2, 0 ,1};

    typename Eikonal::Eikonal_traits<D,N - 2>::AnisotropyM velocity;
    velocity << 1,0,0,
            0,1,0,
            0,0,1;
    Eikonal::SimplexData<D, N> simplex{coordinates, velocity};
    Eikonal::solveEikonalLocalProblem<N, D> localSolver{simplex,values};
    auto sol = localSolver();






    double alpha1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
    double alpha2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[0]);
    double alpha3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[0]);

    double beta1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[1]);
    double beta2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
    double beta3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[1]);

    double gamma1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[2]);
    double gamma2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[3] - coordinates[2]);
    double gamma3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[3] - coordinates[2]);

    // prima faccia 2d: x = 0 e  y = 1
    double delta1 = (coordinates[1] - coordinates[0]).transpose() * velocity * (coordinates[1] - coordinates[0]);
    double delta2 = (coordinates[3] - coordinates[0]).transpose() * velocity * (coordinates[1] - coordinates[0]);
    double delta3 = (coordinates[3] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[0]);


    // seconda faccia 2d: x = 1 e  y = 2
    double theta1 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
    double theta2 = (coordinates[3] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
    double theta3 = (coordinates[3] - coordinates[1]).transpose() * velocity * (coordinates[3] - coordinates[1]);


    // terza faccia 2d: x = 0 e  y = 2
    double eps1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
    double eps2 = (coordinates[3] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
    double eps3 = (coordinates[3] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[0]);




    double lambda11, lambda21, lambda12, lambda22;
    LocalSolver<double>::solve3D(values[2] - values[0], values[2] - values[1], alpha1, alpha2, alpha3, beta1, beta2, beta3,
                               gamma1, gamma2, gamma3, &lambda11,&lambda21, &lambda12, &lambda22);
    std::cout << "iterative solver solution status " << sol.status << std::endl;
    std::cout << "iterative solver solution: " <<  sol.value << std::endl;
    std::cout << "iterative solver lambda1, lambda2 " << sol.lambda[0] << " " << sol.lambda[1] << std::endl;
    VectorExt solPos = lambda11 * coordinates[0] + lambda21 * coordinates[1] + (1 - lambda11 - lambda21) * coordinates[2];
    double sol_direct = lambda11 * values[0] + lambda21 * values[1] + (1 - lambda11 - lambda21) * values[2] + sqrt((coordinates[3] - solPos).transpose()*velocity*(coordinates[3] - solPos));
    std::cout << "direct solver solution 1: " << sol_direct << std::endl;
    std::cout << "direct solver lambda1, lambda2: " << lambda11 << " " << lambda21 << std::endl;
    VectorExt solPos2 = lambda12 * coordinates[0] + lambda22 * coordinates[1] + (1 - lambda12 - lambda22) * coordinates[2];
    double sol_direct2 = lambda12 * values[0] + lambda22 * values[1] + (1 - lambda12 - lambda22) * values[2] + sqrt((coordinates[3] - solPos2).transpose()*velocity*(coordinates[3] - solPos2));
    std::cout << "direct solver solution 2: " << sol_direct2 << std::endl;
    std::cout << "direct solver lambda1, lambda2: " << lambda12 << " " << lambda22 << std::endl;

    //assert(sol.status == 0);
    VectorExt v1 = {lambda11, lambda21, 1-lambda11-lambda21};
    VectorExt v2 = {lambda12, lambda22, 1-lambda12-lambda22};
    typename Eikonal::Eikonal_traits<D,N - 2>::AnisotropyM m;
    m << alpha1, beta1, gamma1,
            alpha2, beta2, gamma2,
            alpha3, beta3, gamma3;

    VectorExt  alpha = {alpha1,alpha2, alpha3};
    VectorExt beta = {beta1, beta2, beta3};
    double lhs1 = (values[2] - values[1]) *  v1.transpose() *  alpha;
    double rhs1 = (values[2] - values[0]) *  v1.transpose() *  beta;
    std::cout << lhs1 << " = " << rhs1 << std::endl;

    double lhs2 = (values[2] - values[1]) *  v2.transpose() *  alpha;
    double rhs2 = (values[2] - values[0]) *  v2.transpose() *  beta;
    std::cout << lhs2 << " = " << rhs2 << std::endl;

    double lhs3 = (values[2] - values[0]) * std::sqrt(v1.transpose() * m * v1);
    double rhs3 = v1.transpose() * alpha;
    std::cout << lhs3 << " = " << rhs3 << std::endl;

    double lhs4 = (values[2] - values[0]) * std::sqrt(v2.transpose() * m * v2);
    double rhs4 = v2.transpose() * alpha;
    std::cout << lhs4 << " = " << rhs4 << std::endl;


    VectorExt v3 = {sol.lambda[0], sol.lambda[1], 1-sol.lambda[0]-sol.lambda[1]};
    double lhs5 = (values[2] - values[0]) * std::sqrt(v3.transpose() * m * v3);
    double rhs5 = v3.transpose() * alpha;
    std::cout << lhs5 << " = " << rhs5 << std::endl;

    double lhs6 = (values[2] - values[1]) *  v3.transpose() *  alpha;
    double rhs6 = (values[2] - values[0]) *  v3.transpose() *  beta;
    std::cout << lhs6 << " = " << rhs6 << std::endl;


    std::cout << "x = 0, y = 1" << std::endl;
    LocalSolver<double>::solve2D(values[1] - values[0], delta1, delta2, delta3, &lambda11, &lambda21);
    std::cout << "lambda 1 = " << lambda11 << " lambda 2 = " << lambda21 << std::endl;

    std::cout << "x = 1, y = 2" << std::endl;
    LocalSolver<double>::solve2D(values[2] - values[1], theta1, theta2, theta3, &lambda11, &lambda21);
    std::cout << "lambda 1 = " << lambda11 << " lambda 2 = " << lambda21 << std::endl;

    std::cout << "x = 0, y = 2" << std::endl;
    LocalSolver<double>::solve2D(values[2] - values[0], eps1, eps2, eps3, &lambda11, &lambda21);
    std::cout << "lambda 1 = " << lambda11 << " lambda 2 = " << lambda21 << std::endl;
}
