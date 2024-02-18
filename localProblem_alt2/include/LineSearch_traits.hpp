/*
 * LineSearch_traits.hpp
 *
 *  Created on: Dec 27, 2020
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_LINESEARCH_LINESEARCH_TRAITS_HPP_
#define EXAMPLES_SRC_LINESEARCH_LINESEARCH_TRAITS_HPP_
#include "Eigen/Core"
#include <functional>
namespace apsc
{


template<int N>
struct LineSearch_traits
{
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, N, 1>;
    using Matrix = Eigen::Matrix<Scalar, N, N>;
    using CostFunction = std::function<Scalar(Vector const &)>;
    using Gradient = std::function<Vector(Vector const &)>;
    using Hessian  = std::function<Matrix(Vector const &)>;
};

} // namespace apsc

#endif /* EXAMPLES_SRC_LINESEARCH_LINESEARCH_TRAITS_HPP_ */
