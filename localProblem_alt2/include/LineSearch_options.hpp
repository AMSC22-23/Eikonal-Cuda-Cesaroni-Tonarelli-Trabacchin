/*
 * LineSearch_options.hpp
 *
 *  Created on: Dec 27, 2020
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_LINESEARCH_LINESEARCH_OPTIONS_HPP_
#define EXAMPLES_SRC_LINESEARCH_LINESEARCH_OPTIONS_HPP_
#include "LineSearch_traits.hpp"
#include <string>
namespace apsc {
template<int N>
struct LineSearchOptions
{
  using Scalar = typename apsc::LineSearch_traits<N>::Scalar;
  Scalar sufficientDecreaseCoefficient =
    1.e-2; //!< The coefficient for sufficient decrease
  Scalar stepSizeDecrementFactor = 0.5; //!< How much to decrement alpha
  Scalar secondWolfConditionFactor =
    0.9;                          //!< Second Wolfe condition factor (not used)
  Scalar       initialStep = 1.0; //!< the initial alpha value.
  unsigned int maxIter =
    40; //!< max number of iterations in the backtracking algorithm
};

} // namespace apsc

#endif /* EXAMPLES_SRC_LINESEARCH_LINESEARCH_OPTIONS_HPP_ */
