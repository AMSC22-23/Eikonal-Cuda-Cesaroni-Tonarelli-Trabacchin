/*
 * Phi.hpp
 *
 *  Created on: Jun 16, 2022
 *      Author: forma
 */

#ifndef EIKONAL_PHI_HPP_
#define EIKONAL_PHI_HPP_
#include "LineSearch_traits.hpp"
#include "SimplexData.hpp"
#include <cmath>
namespace Eikonal
{
template<int N, std::size_t PHDIM>//N is the number of unknown in Phi
struct Phi
{
	// The vector of the unknowns lambdas
	using Vector=typename apsc::LineSearch_traits<N>::Vector;
	using Scalar=typename apsc::LineSearch_traits<N>::Scalar;
	using Matrix=typename apsc::LineSearch_traits<N>::Matrix;
	using VectorExt=typename Eikonal_traits<PHDIM, N>::VectorExt;
	// The dimension of the unknwons (phisical dim -1)
	static constexpr std::size_t DIM=N;
	//@todo implementare move semantic per velocizzare
	Phi(SimplexData<PHDIM, DIM  + 2> const & simplex, VectorExt const & values):simplexData{simplex},values{values}
	{
		du(0)=values(0)-values(DIM);// u31 (u21)
		du(DIM)=values(DIM); // u3 (u2)
		if constexpr (DIM==2)
		{
			du(1)=values(1)-values(DIM);
		}
		lambdaExt(DIM)=1.0;
	};
	// a more efficient implementation chaches some quantities
	// here I just use the expressions straight away to avoid
	// errors
	  Scalar operator()(Vector const & v)const
	  {
		  lambdaExt.template topRows<DIM>()=v;
		  return lambdaExt.dot(du) + normL();
	  }

	  decltype(auto) gradient(Vector const & v)const
	  {
		  lambdaExt.template topRows<DIM>()=v;
		  return du.template topRows<DIM>()+
				  simplexData.MM_Matrix.template block<DIM,DIM+1>(0,0)*lambdaExt/normL();
	  }

	  Matrix hessian(Vector const & v)const
	  {
		  lambdaExt.template topRows<DIM>()=v;
		  auto n = 1./normL();
		  auto n3= n*n*n;
		  //if constexpr (PHDIM==3u)
		//{
		  Vector part{simplexData.MM_Matrix.template block<DIM,DIM+1>(0,0)*lambdaExt};
		  Matrix parta;
		  parta=simplexData.MM_Matrix.template block<DIM,DIM>(0,0);
		  Matrix partb;
		  partb=part*part.transpose();
		  return n*parta-n3*partb;
		//}
		  //else
		  //{
			  //Matrix const & M=simplexData.MM_Matrix;
			  //return (M(0,0)*M(1,1)-M(0,1)*M(0,1))/
					  //std::pow(lambdaExt(0)*lambdaExt(0)*M(1,1)+2.*lambdaExt(0)*M(0,1)+M(1,1), 3./2.);
		  //}
	  }

	  Scalar normL() const
	  {
		  return std::sqrt(
				  lambdaExt.transpose()*
				  simplexData.MM_Matrix*lambdaExt
				  );
	  }

	  SimplexData<PHDIM, DIM + 2> simplexData;
	  VectorExt values;
	  VectorExt du;

	  mutable VectorExt lambdaExt;	// can change value from a const function because declared as mutable
};
}




#endif /* EIKONAL_PHI_HPP_ */
