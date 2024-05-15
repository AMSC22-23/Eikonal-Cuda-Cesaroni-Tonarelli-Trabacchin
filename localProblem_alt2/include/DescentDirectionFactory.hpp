/*
 * DecenntDirectionFactory.hpp
 *
 *  Created on: Mar 24, 2022
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONFACTORY_HPP_
#define EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONFACTORY_HPP_
#include "Factory.hpp"
#include "DescentDirections.hpp"
namespace apsc
{
  /*!
   * The Factory of descent directions. an instance of my generic Factory
   */
   template<int N>
  using DescentDirectionFactory=GenericFactory::Factory<DescentDirectionBase<N>,std::string>;
  /*!
   * @brief Load descent directions in the factory.
   *
   * Look at the source code for details. The definition is in the source file.
   * Here, for simplicity, I do not use the constructor attribute to have the automatic registration.
   * The function returns the reference to the only factory in the code.
   */
   template<int N>
  DescentDirectionFactory<N> & loadDirections();
  /*!
   * Anothe possibility is to have the factory as global variable
   */
  //extern DescentDirectionFactory & descentDirectionFactory;


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


}



#endif /* EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONFACTORY_HPP_ */
