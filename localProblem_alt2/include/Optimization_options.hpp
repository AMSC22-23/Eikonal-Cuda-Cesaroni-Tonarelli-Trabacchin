/*
 * Optimization_options.hpp
 *
 *  Created on: Dec 27, 2020
 *      Author: forma
 */
#ifndef EXAMPLES_SRC_LINESEARCH_OPTIMIZATION_OPTIONS_HPP_
#define EXAMPLES_SRC_LINESEARCH_OPTIMIZATION_OPTIONS_HPP_
#include "LineSearch_traits.hpp"
#include <exception>
#include <vector>
namespace apsc
{
/*!
 * It holds all options for the line search algorithm
 */
template<int N>
struct OptimizationOptions
{
  using Scalar = typename apsc::LineSearch_traits<N>::Scalar;
  Scalar       relTol = 1.e-5; //!< relative tolerance
  Scalar       absTol = 1.e-5; //!< absolute tolerance
  unsigned int maxIter = 500;  //!< max n. of Iteration
   //@todo method to read from file
};

/*!
 * It holds the main data for the optimization algorithm
 */
template<int N>
struct OptimizationData
{
  typename apsc::LineSearch_traits<N>::CostFunction costFunction; //!< The cost function.
  typename apsc::LineSearch_traits<N>::Gradient
              gradient;              //!< The gradient of the cost function.
  //! The Hessian: by default an empty matrix
  typename apsc::LineSearch_traits<N>::Hessian hessian=[](typename apsc::LineSearch_traits<N>::Vector const &){return typename apsc::LineSearch_traits<N>::Matrix();};
  std::size_t NumberOfVariables = 0; //! The number of variables of the problem.
  bool        bounded = false;       //! We may have bound constraints
  std::vector<double> lowerBounds;   //! Uses if bounded=true
  std::vector<double> upperBounds;   //! Uses if bounded=true
};
template<int N>
inline void
setBounds(OptimizationData<N> &optimizationData, std::vector<double> const &lo,
          std::vector<double> const &up)
{
  if(optimizationData.NumberOfVariables > lo.size() and
     optimizationData.NumberOfVariables > up.size())
    {
      throw std::runtime_error("Wrong bound sizes");
    }
  optimizationData.bounded = true;
  optimizationData.lowerBounds = lo;
  optimizationData.upperBounds = up;
}

/*!
 * A structure used to hold the current values.
 */
 template<int N>
struct OptimizationCurrentValues
{
  typename apsc::LineSearch_traits<N>::Scalar currentCostValue; //!< current cost.
  typename apsc::LineSearch_traits<N>::Vector currentPoint;     //!< current point.
  typename apsc::LineSearch_traits<N>::Vector currentGradient;  //!< current gradient.
  typename apsc::LineSearch_traits<N>::Matrix currentHessian;  //!< current Hessian.
  bool        bounded = false;       //! We may have bound constraints
  std::vector<double> lowerBounds;   //! Uses if bounded=true
  std::vector<double> upperBounds;   //! Uses if bounded=true
};
} // namespace apsc

#endif /* EXAMPLES_SRC_LINESEARCH_OPTIMIZATION_OPTIONS_HPP_ */
