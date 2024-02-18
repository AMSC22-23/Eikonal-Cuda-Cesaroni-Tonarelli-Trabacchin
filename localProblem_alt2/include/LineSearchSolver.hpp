/*
 * LineSearchSolver.hpp
 *
 *  Created on: Dec 28, 2020
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_LINESEARCH_LINESEARCHSOLVER_HPP_
#define EXAMPLES_SRC_LINESEARCH_LINESEARCHSOLVER_HPP_
#include "DescentDirectionBase.hpp"
#include "LineSearch.hpp"
#include <memory>
#include <tuple>
#include <exception>
#include <iostream>
#include <limits>
#include <algorithm>
namespace apsc
{
/*!
 * The class that implements a line search technique for unconstrained
 * minimization of a function.
 */
template<int N>
class LinearSearchSolver
{
public:
  LinearSearchSolver() = default;
  /*!
   * You may pass all needed parameters in the constructors
   *
   * @param data The structure containing the cost function and the gradient
   * function
   * @param ddptr a unique pointer to the concrete callable object that computes
   * the search direction
   * @param opt The options for the line search iterations.
   * @param lsOpt The options for the backtracking function.
   * @note The object containing the definition of the desscent direction is
   * MOVED into the class. So you have to be sure that the argument is movable.
   * I have not implemented the clone method for DescentDirectionBase, so this
   * class is NOT copy constructible, not copy-assigneable.
   */
  LinearSearchSolver(
    apsc::OptimizationData<N> const &                data,
    std::unique_ptr<apsc::DescentDirectionBase<N>> &&ddptr,
    apsc::OptimizationOptions<N> opt = apsc::OptimizationOptions<N>(),
    apsc::LineSearchOptions<N>   lsOpt = apsc::LineSearchOptions<N>())
    : optimizationData(data), options(opt), lineSearchOptions(lsOpt),
      descentDirectionFinderPtr(std::move(ddptr))
  {
      if constexpr (N == 2) {
          R3<<  -std::sqrt(2.)/2., -std::sqrt(2.)/2.,
                  std::sqrt(2.)/2., -std::sqrt(2.)/2.;
      }
  }

  /*!
   * You need to set the intitial point for the search.
   * @param initialPoint The initial point fo the search.
   */
  void setInitialPoint(typename apsc::LineSearch_traits<N>::Vector initialPoint);

  std::tuple<apsc::OptimizationCurrentValues<N>, std::size_t, int> solve();

  /*!
   * You may change the rule to compute the descent direction.
   * @note It is moved into the class.
   * @param descentDirectionFinderPtr The new object that computed the descent
   * direction
   */
  void
  setDescentDirectionFinderPtr(
    std::unique_ptr<apsc::DescentDirectionBase<N>> &&descentDirectionFinderPtr)
  {
    this->descentDirectionFinderPtr = std::move(descentDirectionFinderPtr);
  }

  //! The structure holding the cost function and the gradient
  apsc::OptimizationData<N> optimizationData;
  //! The structure containing the options for the line search algorithm
  apsc::OptimizationOptions<N> options;
  //! The structure containing the options for the backtracking.
  apsc::LineSearchOptions<N> lineSearchOptions;

private:
  /*!
   * Implements a simple backtracking
   * @param searchDirection The search directions
   * @return The new point, the new cost value and a status. status=1 ->
   * direction is not a descent direction. status=2 -> number of attempts
   * exceeded
   */
  std::tuple<typename apsc::LineSearch_traits<N>::Vector, typename apsc::LineSearch_traits<N>::Scalar,
             int>
  backtrack(typename apsc::LineSearch_traits<N>::Vector  &searchDirection) const;
  apsc::OptimizationCurrentValues<N>             currentValues;
  std::unique_ptr<apsc::DescentDirectionBase<N>> descentDirectionFinderPtr;
  std::tuple<typename apsc::LineSearch_traits<N>::Vector, bool>
  project(typename apsc::LineSearch_traits<N>::Vector const &newPoint) const;
  void projectDirection(typename apsc::LineSearch_traits<N>::Vector & searchDirection) const;
  typename apsc::LineSearch_traits<N>::Matrix R3;
};











    template<int N>
    void
    apsc::LinearSearchSolver<N>::setInitialPoint(
            typename apsc::LineSearch_traits<N>::Vector initialPoint)
    {
        if(!this->optimizationData.costFunction || !this->optimizationData.gradient)
        {
            throw std::runtime_error("You cannot set an initial point before setting "
                                     "cost function and gradient\n");
        }
        this->descentDirectionFinderPtr
                ->reset(); // reset the descent direction computation (when needed)
        this->currentValues.currentPoint = initialPoint;
        this->currentValues.currentGradient =
                this->optimizationData.gradient(initialPoint);
        this->currentValues.currentCostValue =
                this->optimizationData.costFunction(initialPoint);
        this->currentValues.currentHessian =
                this->optimizationData.hessian(initialPoint);
        // This is for bounded problems
        this->currentValues.bounded = this->optimizationData.bounded;
        this->currentValues.lowerBounds = this->optimizationData.lowerBounds;
        this->currentValues.upperBounds = this->optimizationData.upperBounds;

        std::size_t n = initialPoint.size();
        if(n != this->optimizationData.NumberOfVariables)
        {
            std::cerr << "Number of variables indicated in the data ("
                      << this->optimizationData.NumberOfVariables
                      << ") does not correspond to the size of the initial point("
                      << n << ").\n  I am fixing it, hoping for the best.\n";
            this->optimizationData.NumberOfVariables = n;
        }
        if(optimizationData.bounded)
        {
            auto good = true;
            for(auto i = 0u; i < optimizationData.NumberOfVariables; ++i)
            {
                good = good and initialPoint[i] >= optimizationData.lowerBounds[i] and
                       initialPoint[i] <= optimizationData.upperBounds[i];
            }
            if(not good)
                throw std::runtime_error("Initial point must be within given bounds");
        }
    }
    template<int N>
    std::tuple<apsc::OptimizationCurrentValues<N>, std::size_t, int>
    apsc::LinearSearchSolver<N>::solve()
    {
        auto const &relTol = this->options.relTol;
        auto const &absTol = this->options.absTol;
        auto const &maxIter = this->options.maxIter;
        auto &      currentPoint = this->currentValues.currentPoint;
        auto &      currentValue = this->currentValues.currentCostValue;
        auto &      currentGradient = this->currentValues.currentGradient;
        auto &      currentHessian = this->currentValues.currentHessian;
        auto        gradientNorm = currentGradient.norm();
        // the relative tolerance is respect initial gradient norm.
        auto const  testValue = relTol * gradientNorm;
        std::size_t iter = 0;
//  bool const &bounded = this->optimizationData.bounded;
        auto        stepLength = 2 * absTol;
        auto        valTol =absTol + relTol*std::abs(currentValue);
        auto        valChange  = 2 * valTol;

#ifdef VERBOSE
        std::clog << "Initial values.\t Grad=" << currentGradient.transpose()
                <<"\t Point " << currentPoint.transpose() << std::endl;
#endif
        int status=0;
        while(gradientNorm > (testValue + absTol) and stepLength > absTol and
              valChange >valTol and iter < maxIter and status==0)
        {
            // get descent direction
            auto newPoint=currentPoint;
            auto newValue=currentValue;
            typename apsc::LineSearch_traits<N>::Vector    dd =
                    this->descentDirectionFinderPtr->operator()(this->currentValues);
            if (dd.norm()>absTol)
            { // compute
                std::tie(newPoint, newValue, status) = this->backtrack(dd);
            }
            // check what happened
            if(status != 0)
            {
                newPoint=currentPoint;
                newValue=currentValue;
                if(status == 1)
                    std::cerr<<"Error in LinearSearchSolver: I have found a non-descent direction.";
                //else // if(not bounded) // this test is disabled for bounded problems
                    //std::cerr<<"Error in LinearSearchSolver: I cannot satisfy the sufficient decrease condition.";
            }
            else
            {
                stepLength = (newPoint - currentPoint).norm();
                currentPoint = newPoint;
                valChange=std::abs(currentValue-newValue);
                currentValue = newValue;
                currentGradient = this->optimizationData.gradient(newPoint);
                currentHessian = this->optimizationData.hessian(newPoint);
                gradientNorm = currentGradient.norm();
#ifdef VERBOSE
                std::clog << "iter=" << iter << "\t Grad=" << currentGradient.transpose()
                << "\t StepL/Abstol=" << stepLength / absTol << " Point "
                << " " << currentPoint.transpose() << "\t DD "<<dd.transpose()<<"\t Value "<<currentValue<<std::endl;
#endif
                ++iter;
            }
        }
        if (status==0)status = iter < maxIter ? 0 : 3;
        return {this->currentValues, iter, status};
    }

// A simple backtracking. I implement only sufficient decrease condition.
    template<int N>
    std::tuple<typename apsc::LineSearch_traits<N>::Vector, typename apsc::LineSearch_traits<N>::Scalar,
            int>
    apsc::LinearSearchSolver<N>::backtrack(
            typename apsc::LineSearch_traits<N>::Vector &searchDirection) const
    {
        apsc::OptimizationData<N> const &         data = this->optimizationData;
        apsc::OptimizationCurrentValues<N> const &currentValues = this->currentValues;
        apsc::LineSearchOptions<N> const &        options = this->lineSearchOptions;
        using Scalar = typename apsc::LineSearch_traits<N>::Scalar;
        using Vector = typename apsc::LineSearch_traits<N>::Vector;
        using CostFunction = typename apsc::LineSearch_traits<N>::CostFunction;
        // Check if we are stuck
        if(searchDirection.norm() < this->options.absTol)
            return {currentValues.currentPoint, currentValues.currentCostValue, 0};
        // Check is direction is descent direction
        Scalar gradstep = currentValues.currentGradient.transpose() * searchDirection;
        if(gradstep >= 0.)
        {
            std::cerr << gradstep << " not valid. Reverted to gradient\n";
            searchDirection = -currentValues.currentGradient;
            gradstep = -searchDirection.squaredNorm();
        }

        CostFunction const &            f = data.costFunction;
        Vector                          currentPoint = currentValues.currentPoint;
        auto const &                    maxIter = options.maxIter;
        auto                            alpha = options.initialStep;
        unsigned int                    iter = 0u;
        typename apsc::LineSearch_traits<N>::Vector nextPoint;
        nextPoint = currentPoint + alpha * searchDirection;
        bool bumped = false;
        //if(this->optimizationData.bounded)
        std::tie(nextPoint, bumped) = project(nextPoint);
        Scalar nextValue = f(nextPoint);
        // iterate until sufficient decrease condition is met.
        // Some code repetition to avoid an if into a tight loop
        alpha=std::min(1.0,1./searchDirection.norm());
//  if(this->optimizationData.bounded)
//    {
        // if we are on the boundary I deactivactivate the sufficient decrease
        //
        while((nextValue >=
               currentValues.currentCostValue +
               options.sufficientDecreaseCoefficient * alpha * gradstep) and
              (iter < maxIter))
        {
            ++iter;
            alpha *= options.stepSizeDecrementFactor;
            nextPoint = currentPoint + alpha * searchDirection;
            std::tie(nextPoint, bumped) = project(nextPoint);
            nextValue = f(nextPoint);
        }
/*    }
  else
    {
      while((nextValue >
             currentValues.currentCostValue +
               options.sufficientDecreaseCoefficient * alpha * gradstep) and
            (iter < maxIter))
        {
          ++iter;
          alpha *= options.stepSizeDecrementFactor;
          nextPoint = currentPoint + alpha * searchDirection;
          nextValue = f(nextPoint);
        }
    }
*/
        int status = iter < maxIter ? 0 : 2;
        return {nextPoint, nextValue, status};
    }

    template<int N>
    std::tuple<typename apsc::LineSearch_traits<N>::Vector, bool>
    apsc::LinearSearchSolver<N>::project(
            typename apsc::LineSearch_traits<N>::Vector const &newPoint) const
    {
        std::size_t                     i = 0u;
        typename apsc::LineSearch_traits<N>::Vector res = newPoint;
        bool                            bumped = false;
        if constexpr (N == 2) {
            double avg=std::min(1.0,res[0]+res[1]);
            double jmp=res[0]-res[1];
            res.coeffRef(0)=(avg+jmp)/2.;
            res.coeffRef(1)=(avg-jmp)/2.;
            /*
             *
            constexpr double bumppoint=-std::sqrt(2.)/2.;
            res=R3*res;
            res.coeffRef(0)=std::max(bumppoint,res(0));
            res=R3.transpose()*res;
            */
            bumped = std::abs(res[0]+res[1]-1.0)<=100.*std::numeric_limits<double>::epsilon();
        }
        for(auto &x : res)
        {
            x = std::clamp(x, this->optimizationData.lowerBounds[i],
                           this->optimizationData.upperBounds[i]);
            bumped = bumped or (x == this->optimizationData.lowerBounds[i] or
                                x == this->optimizationData.upperBounds[i]);
            ++i;
        }
        return {res, bumped};
    }

    template<int N>
    void
    apsc::LinearSearchSolver<N>::projectDirection(
            typename apsc::LineSearch_traits<N>::Vector &searchDirection) const
    {
        for(auto i = 0u; i < this->optimizationData.lowerBounds.size(); ++i)
        {
            if(this->currentValues.currentPoint[i] == optimizationData.lowerBounds[i])
                searchDirection[i] = std::max(searchDirection[i], 0.);
            if(this->currentValues.currentPoint[i] == optimizationData.upperBounds[i])
                searchDirection[i] = std::min(searchDirection[i], 0.);
        }
    }



















} // namespace apsc


#endif /* EXAMPLES_SRC_LINESEARCH_LINESEARCHSOLVER_HPP_ */
