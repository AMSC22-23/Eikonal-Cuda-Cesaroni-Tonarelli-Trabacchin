/*
 * DescentDirectionFactory.cpp
 *
 *  Created on: Mar 24, 2022
 *      Author: forma
 */
#include "DescentDirectionFactory.hpp"
namespace apsc
{
    template<int N>
DescentDirectionFactory<N> &
loadDirections()
{
  // get the factory
  DescentDirectionFactory<N> &theFactory = DescentDirectionFactory<N>::Instance();
  //
  theFactory.add("GradientDirection",
                 []() { return std::make_unique<GradientDirection<N>>(); });
  theFactory.add("BFGSDirection",
                 []() { return std::make_unique<BFGSDirection<N>>(); });
  theFactory.add("BFGSIDirection",
                 []() { return std::make_unique<BFGSIDirection<N>>(); });
  theFactory.add("BBDirection",
                 []() { return std::make_unique<BBDirection<N>>(); });
  theFactory.add("CGDirection",
                 []() { return std::make_unique<CGDirection<N>>(); });
  theFactory.add("NewtonDirection",
                 []() { return std::make_unique<NewtonDirection<N>>(); });
  return theFactory;
}
} // namespace apsc
