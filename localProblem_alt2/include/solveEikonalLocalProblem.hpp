#ifndef HH_SOLVEEIKONAL__HH
#define HH_SOLVEEIKONAL__HH
#include "DescentDirectionFactory.hpp"
#include "DescentDirections.hpp"
#include "GradientFiniteDifference.hpp"
#include "LineSearch.hpp"
#include "LineSearchSolver.hpp"
#include "Phi.hpp"
#include <cmath>
#include <iostream>
#include <memory>
namespace Eikonal
{
/*!
 * @brief Structure returning the solution of the local eikonal problem
 *
 * @tparam PHDIM The physical dimension
 */
template <int N, std::size_t PHDIM> struct EikonalSolution// N is the number of unknowns
{
  using Vector=typename apsc::LineSearch_traits<N>::Vector;
  double  value; //! The value at the new point
  Vector  lambda; //! The value(s) of lambda (foot of the characteristics)
  int     status; //! 0= converged 1=no descent direction 2=no convergence
};

/*!
 * @brief Driver for the local solver
 *
 * @tparam PHDIM The physical dimension
 */
template <int N, int PHDIM> class solveEikonalLocalProblem//N is the number of vertices of the simplex
{
public:
  using Vector = typename apsc::LineSearch_traits<N - 2>::Vector;
  using Matrix = typename apsc::LineSearch_traits<N - 2>::Matrix;
  //! I pass a simplex structure and the values in the constructor
  //! @todo To save memory and time I have to store references in Phi
  template <typename SIMPLEX, typename VALUES>
  solveEikonalLocalProblem(SIMPLEX &&simplex, VALUES &&values)
    : my_phi{std::forward<SIMPLEX>(simplex), std::forward<VALUES>(values)}
  {}
  /*!
   * Solves the local problem
   */
  EikonalSolution<N - 2, PHDIM>          //restituisce una struct
  operator()() const
  {
    typename apsc::OptimizationData<N - 2> optimizationData;
    optimizationData.NumberOfVariables = N  - 2;
    optimizationData.costFunction = [this](const Vector &x) {
      return this->my_phi(x);
    };
    optimizationData.gradient = [this](const Vector &x) {
      return this->my_phi.gradient(x);
    };

    optimizationData.hessian = [this](const Vector &x) {
      return this->my_phi.hessian(x);
    };
    Vector initialPoint;
    if constexpr(N == 4)
      {
        setBounds(optimizationData, {0., 0.}, {1.0, 1.0});
      }
    else
      {
        setBounds(optimizationData, {0.0}, {1.0});
      }
    initialPoint.fill(0.333);           //put lambdas = 1/3 all elements in the vector, it is the initial value
    apsc::LinearSearchSolver<N - 2> solver(optimizationData,
                                    std::make_unique<apsc::NewtonDirection<N - 2>>(),
                                    optimizationOptions, lineSearchOptions);
    solver.setInitialPoint(initialPoint);
    auto [finalValues, numIter, status] = solver.solve();
#ifdef VERBOSE
    if(status == 0)
      std::cout << "Solver converged" << std::endl;
    else
      std::cout << "Solver DID NOT converge" << std::endl;

    std::cout << "Point found=" << finalValues.currentPoint.transpose()
              << "\nCost function value=" << finalValues.currentCostValue
              << "\nGradient   =" << finalValues.currentGradient.transpose()
              << "\nGradient norm=" << finalValues.currentGradient.norm()
              << "\nNumber of iterations=" << numIter << "\nStatus=" << status
              << std::endl;
#endif
    return {finalValues.currentCostValue, finalValues.currentPoint, status};
  }
  static void setLineSearchOptions(apsc::LineSearchOptions<N - 2> const & lso)
		{
	  lineSearchOptions=lso;
		}
  static void setOptimizationOptions(apsc::OptimizationOptions<N - 2> const & oop)
 		{
 	  optimizationOptions=oop;
 		}
private:
  Eikonal::Phi<N - 2, PHDIM>                     my_phi;
  inline static apsc::LineSearchOptions<N - 2>   lineSearchOptions;
  inline static apsc::OptimizationOptions<N - 2> optimizationOptions;
  inline static std::unique_ptr<apsc::DescentDirectionBase<N - 2>>
    descentDirectionFunction = std::make_unique<apsc::NewtonDirection<N - 2>>();
};

} // namespace Eikonal
#endif

