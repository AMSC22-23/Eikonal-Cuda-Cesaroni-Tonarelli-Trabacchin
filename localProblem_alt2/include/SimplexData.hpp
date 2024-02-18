/*
 * SimplexData.hpp
 *
 *  Created on: Jun 17, 2022
 *      Author: forma
 */

#ifndef EIKONAL_SIMPLEXDATA_HPP_
#define EIKONAL_SIMPLEXDATA_HPP_
#include "Eikonal_traits.hpp"
namespace Eikonal
{
template<std::size_t PHDIM, int N>
  struct SimplexData
  {
	using AnisotropyM=typename Eikonal_traits<PHDIM, N-2>::AnisotropyM;
	using MMatrix=typename Eikonal_traits<PHDIM, N-2>::MMatrix;
    using EMatrix = typename Eikonal_traits<PHDIM, N-2>::EMatrix;
	using Point=typename Eikonal_traits<PHDIM, N - 2>::Point;

	//! This constructor just takes of vector that describes the simplex

	  SimplexData(std::array<std::array<double,PHDIM>,N> const & p,
			  AnisotropyM const &M=AnisotropyM::Identity())
	  {
		  for (auto i=0;i<N;++i)
			  points[i]=Eigen::Map<Point>(const_cast<double*>(p[i].data()));
		  setup(M);
	  }

	  SimplexData(std::array<Point,N> const & p,
			  AnisotropyM const &M=AnisotropyM::Identity()):
		  points{p}
	  {
		  setup(M);
	  };
	  std::array<Point,N> points;
	  MMatrix MM_Matrix;
	  EMatrix E;
  private:
	  void setup(AnisotropyM const &M)
	  {
		  E.col(0)=points[N-2]-points[0];//e13 or e11
		  if constexpr(N == 4)
    		{
			  E.col(1)=points[2]-points[1]; //e23
    		}
		  E.col(N-2)=points[N-1]-points[N-2]; //e34 or e23
		  MM_Matrix=E.transpose()*M*E;
	  }
  };
}




#endif /* EIKONAL_SIMPLEXDATA_HPP_ */
