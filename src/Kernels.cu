#ifndef KERNELS
#define KERNELS
#include <vector>
#include <cuda.h>
#include <random>
#include <set>
#include "../localProblem_alt2/include/Phi.hpp"
#include "../localProblem_alt2/include/solveEikonalLocalProblem.hpp"
#include "Mesh.cuh"
#include "LocalSolver.cuh"

constexpr int D = 3;
using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
using Matrix = typename Eikonal::Eikonal_traits<D,2>::AnisotropyM;
using VectorV = typename Eigen::Matrix<double,4,1>;


template <typename Float>
__global__ void setSolutionsToInfinity(Float* solutions_dev, Float infinity_value, size_t size_sol){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sol){
        solutions_dev[threadId] = infinity_value;
    }
}

template <typename Float>
__global__ void setSolutionsSourcesAndDomains(Float* solutions_dev, int* source_nodes_dev, int* active_domains_dev, int* partitions_vertices_dev, int partitions_number, size_t size_sources){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sources) {
        solutions_dev[source_nodes_dev[threadId]] = 0;

        for(int i = 0; i < partitions_number; i++){
            if(source_nodes_dev[threadId] < partitions_vertices_dev[i])
                active_domains_dev[i] = 1; //there is no need to use atomic operations since if multiple threads try to access the memory location they all write 1
        }
    }
}

template <typename Float>
__global__ void domainSweep(int domain_id, int* partitions_vertices_dev, int* partitions_tetra_dev, Float* geo_dev, int* tetra_dev,
                            TetraConfig* shapes_dev, int* ngh, Float* M_dev, Float* solutions_dev, int* active_domains_dev,
                            int num_partitions, int num_vertices, int num_tetra, int shapes_size, Float infinity_value){
    int nodeIdDomain = threadIdx.x + blockIdx.x * blockDim.x;
    int nodeIdGlobal = partitions_vertices_dev[domain_id] + nodeIdDomain;
    std::array<VectorExt, 4> coordinates;
    VectorV values;
    Float* M;
    Float minimum_sol = infinity_value;
    Float lambda1, lambda2;
    // each thread takes a node and compute the solution looping over all its associated tetrahedra
    if (nodeIdGlobal < num_vertices){
        for(int i = ngh[nodeIdGlobal]; i < (nodeIdGlobal != num_vertices - 1) ? ngh[nodeIdGlobal+1]: shapes_size; i++){
            // call local solver on tetra[shapes_dev[i].tetra_index] using configuration shapes_dev[i].tetra_config
            for(int j = 0; j < D + 1; j++){
                for(int k = 0; k < D; k++) {
                    coordinates[j][k] = geo_dev[tetra_dev[shapes_dev[i].tetra_index + j] + k];
                }
                values[j] = solutions_dev[tetra_dev[shapes_dev[i].tetra_index + j]];
            }
            M = M_dev + shapes_dev[i].tetra_index * 6;
            auto [sol, lambda1, lambda2] = LocalSolver<D, Float>::solve(coordinates, values, M, D + 1 - shapes_dev[i].tetra_config);
            if(sol < minimum_sol)
                minimum_sol = sol;
        }
        // TODO modifiy solutions_dev[nodeIdGlobal] to minimum_sol using an atomic
        // TODO modify active domain
    }
}

// ad ogni iterazione esterna, facciamo un elenco dei domini attivi (quelli che hanno bisogno di essere sweepati)
// data questa lista, assegniamo un blocco per ogni sottodominio attivo e facciamo lo sweep, teniamo
// tutti i dati relativi al sottodominio nella shared memory (vertici, tetraedri, matrici M, soluzioni parziali)
// quando modifichiamo la soluzione di un vertice, dobbiamo verificare a quelle sottodominio appartiene quel vertice
// e se quel vertice appartiene al sottodominio N (diverso da quello attuale) allora rendiamo attivo N per
// la prossima iterazione. i dati del partition vector possono essere memorizzati nella read-only cache
// per poter verificare il dominio di appartenenza dei vertici

#endif