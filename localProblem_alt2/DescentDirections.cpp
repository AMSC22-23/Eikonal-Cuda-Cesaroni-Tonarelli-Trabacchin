/*
 * DescentDirections.cpp
 *
 *  Created on: Dec 28, 2020
 *      Author: forma
 */
#include "DescentDirections.hpp"
#include "Eigen/Dense"

template<int N>
typename apsc::LineSearch_traits<N>::Vector
apsc::BFGSDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
{
  // First time is a gradient step
  if(firstTime)
    {
      auto const n = values.currentPoint.size();
      H = Eigen::MatrixXd::Identity(n, n);
      firstTime = false;
      this->previousValues = values;
      return -values.currentGradient;
    }

  typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector sk =
    values.currentPoint - this->previousValues.currentPoint;
  auto const yks = yk.dot(sk);
  // Correct approximate Hessian only if we maintain sdp property if not keep
  // the old one
  if(yks > this->smallNumber * sk.norm() * yk.norm())
    {
      typename apsc::LineSearch_traits<N>::Vector Hs;
      Hs = H * sk;
      H += (yk * yk.transpose()) / yks - (Hs * Hs.transpose()) / (sk.dot(Hs));
    }
  this->previousValues = values;
  typename apsc::LineSearch_traits<N>::Vector d = H.fullPivLu().solve(-g);
  return d;
}

template<int N>
void
apsc::BFGSDirection<N>::reset()
{
  this->firstTime = true;
}

/*                   BFGS with approximate inverse */
template<int N>
typename apsc::LineSearch_traits<N>::Vector
apsc::BFGSIDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
{
  // First time is a gradient step
  if(firstTime)
    {
      auto n = values.currentPoint.size();
      H = Eigen::MatrixXd::Identity(n, n);
      firstTime = false;
      this->previousValues = values;
      return -values.currentGradient;
    }

  typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector sk =
    values.currentPoint - this->previousValues.currentPoint;
  auto const yks = yk.dot(sk);
  // Correct approximate Hessian only if we maintain sdp property if not keep
  // the old one
  if(yks > this->smallNumber * sk.norm() * yk.norm())
    {
      H += sk * sk.transpose() * (yks + yk.transpose() * H * yk) / (yks * yks) -
           (H * yk * sk.transpose() + sk * yk.transpose() * H) / yks;
    }
  this->previousValues = values;
  typename apsc::LineSearch_traits<N>::Vector d = -H * g;
  return d;
}

template<int N>
void
apsc::BFGSIDirection<N>::reset()
{
  this->firstTime = true;
}

/*
 * Barzilai-Borwain
 */
template<int N>
typename apsc::LineSearch_traits<N>::Vector
apsc::BBDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
{
  // First time is a gradient step
  if(firstTime)
    {
      firstTime = false;
      this->previousValues = values;
      return -values.currentGradient;
    }

  typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
  typename apsc::LineSearch_traits<N>::Vector sk =
    values.currentPoint - this->previousValues.currentPoint;
  auto yks = yk.dot(sk);
  auto ykk = yk.dot(yk);
  this->previousValues = values;
  // Correct approximate Hessian only if we maintain sdp property if not keep
  // the old one
  if(yks > this->smallNumber * sk.norm() * yk.norm() and ykk > smallNumber)
    {
      // I use a mix of the two possible strategies
      return -0.5 * (yks / ykk + sk.dot(sk) / yks) * g;
      // Strategy 1
      // return -(yks/ykk)*g;
      // Strategy 2
      // return -(sk.dot(sk)/yks)*g;
    }
  else
    {
      return -g;
    }
}

/*
 * Non-linear CG with Polak-Ribier formula
 */
template<int N>
typename apsc::LineSearch_traits<N>::Vector
apsc::CGDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
{
  // First time is a gradient step
  if(firstTime)
    {
      firstTime = false;
      this->prevDk = -values.currentGradient;
    }
  else
    {
      typename apsc::LineSearch_traits<N>::Vector const &gk1 = values.currentGradient;
      typename apsc::LineSearch_traits<N>::Vector const &gk =
        this->previousValues.currentGradient;
      typename apsc::LineSearch_traits<N>::Vector const &dk = this->prevDk;
      // Polak Ribiere formula
      this->prevDk =
        -gk1 + (gk1.dot(gk1 - gk) / (gk.dot(gk))) * dk; // store for next time
      if(prevDk.dot(gk1) > 0)
        prevDk =
          -gk1; // check if direction is a descent if not go back to gradient
    }
  this->previousValues = values; // store for next time
  return this->prevDk;
}
