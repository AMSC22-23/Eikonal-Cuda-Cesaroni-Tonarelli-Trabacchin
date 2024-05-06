#ifndef KERNELS
#define KERNELS
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <set>
#include <array>
//#include "../localProblem_alt2/include/Phi.hpp"
//#include "../localProblem_alt2/include/solveEikonalLocalProblem.hpp"
#include "Mesh.cuh"
#include "LocalSolver.cuh"
#include "CudaEikonalTraits.cuh"
constexpr int D = 3;
/*
using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
using Matrix = typename Eikonal::Eikonal_traits<D,2>::AnisotropyM;
using VectorV = typename Eigen::Matrix<double,4,1>;
*/




template <typename Float>
__global__ void setSolutionsToInfinity(Float* solutions_dev, Float infinity_value, size_t size_sol){
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sol){
        solutions_dev[threadId] = infinity_value;
    }
}

template <typename Float>
__global__ void setSolutionsSourcesAndDomains(Float* solutions_dev, int* source_nodes_dev, int* active_domains_dev, int* partitions_vertices_dev, int partitions_number, size_t size_sources){
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sources) {

        solutions_dev[source_nodes_dev[threadId]] = 0.0;
        bool found = false;
        for(int i = 0; i < partitions_number && !found; i++) {
            if (source_nodes_dev[threadId] <= partitions_vertices_dev[i]) {
                active_domains_dev[i] = 1; //there is no need to use atomic operations since if multiple threads try to access the memory location they all write 1
                found = true;
            }
        }
    }
}

template <typename Float>
__global__ void domainSweep(int domain_id, const int*  __restrict__ partitions_vertices_dev, int* __restrict__ partitions_tetra_dev, Float* __restrict__ geo_dev, const int* __restrict__ tetra_dev,
                            const TetraConfig* __restrict__ shapes_dev, const int* __restrict__ ngh, const Float* __restrict__ M_dev, Float* __restrict__ solutions_dev, int* __restrict__ active_domains_dev,
                            int num_partitions, int num_vertices, int num_tetra, int shapes_size, Float infinity_value, Float tol){
                                
    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;
                                
    unsigned int nodeIdDomain = threadIdx.x + blockIdx.x * blockDim.x;
    /*if(nodeIdDomain == 0) {
        printf("searching domain %d\n", domain_id);
    }*/
    unsigned int nodeIdGlobal = ((domain_id != 0) ? (partitions_vertices_dev[domain_id - 1]) : 0) + nodeIdDomain;
    
    //std::array<VectorExt, 4> coordinates;
    VectorExt coordinates[4];
    VectorV values;
    const Float* M;
    Float minimum_sol = infinity_value;
    // each thread takes a node and compute the solution looping over all its associated tetrahedra
    if (nodeIdGlobal < num_vertices){
        for(int i = ngh[nodeIdGlobal]; i < ((nodeIdGlobal != num_vertices - 1) ? ngh[nodeIdGlobal+1]: shapes_size); i++){
            // call local solver on tetra[shapes_dev[i].tetra_index] using configuration shapes_dev[i].tetra_config
            for(int j = 0; j < D + 1; j++){
                for(int k = 0; k < D; k++) {
                    coordinates[j][k] =  geo_dev[D * tetra_dev[(D+1) * shapes_dev[i].tetra_index + j] + k];
                }
                values[j] = solutions_dev[tetra_dev[(D+1) * shapes_dev[i].tetra_index + j]];
            }
            M = M_dev + shapes_dev[i].tetra_index * 6;
            auto [sol, lambda1, lambda2] = LocalSolver<D, Float>::solve(coordinates, values, M, D+1-shapes_dev[i].tetra_config);
            //printf("sol = %f\n", sol);
            if(sol < minimum_sol) {
                minimum_sol = sol;
            }
        }

        if(-minimum_sol + solutions_dev[nodeIdGlobal] > tol) {
            //printf("min sol: %f %f\n",minimum_sol,  solutions_dev[nodeIdGlobal]);
            atomicExch(&solutions_dev[nodeIdGlobal], minimum_sol);
            //solutions_dev[nodeIdGlobal] = minimum_sol;
            
            
            bool found = false;
            for (int i = ngh[nodeIdGlobal]; i < ((nodeIdGlobal != num_vertices - 1) ? ngh[nodeIdGlobal+1]: shapes_size); i++) { //for each tetra associated with the node
                for (int j = 0; j < D + 1; j++) { //for each vertex in the tetra
                    found = false;
                    for (int k = 0; k < num_partitions && !found; k++) { // activate the domains associated with the vertices
                        if (tetra_dev[shapes_dev[i].tetra_index * (D + 1) + j] <= partitions_vertices_dev[k]) {
                            int old_active = atomicExch(&active_domains_dev[k], 1);
                            //active_domains_dev[k] = 1;
                            found = true;
                        }
                    }
                }
            }
        }
    }

}

#endif