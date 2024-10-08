#ifndef KERNELS
#define KERNELS
#include <vector>
#include <cuda.h>
#include <random>
#include <set>
#include <array>
#include "Mesh.cuh"
#include "LocalSolver.cuh"
#include "CudaEikonalTraits.cuh"
#include <cmath>
constexpr int D = 3;

// function, callable only from device code, designed to replace the value
// at a given memory address if the new value is smaller than the current value,
// ensuring atomicity and correctness in concurrent GPU computations
template <typename Float>
__device__ Float atomicSwapIfLess(Float* address, Float value) {
    Float swap, old;
    swap = *address;
    do {
        if(swap <= value) {
            break;
        }
        old = swap;
    } while((swap = atomicCAS(address, swap, value)) != old);
    return swap;
}

template <>
__device__ float atomicSwapIfLess(float* address, float value) {
    float swap, old;
    swap = *address;
    do {
        if(swap <= value) {
            break;
        }
        old = swap;

    } while((swap = __uint_as_float(atomicCAS((unsigned int*)address, __float_as_uint(swap), __float_as_uint(value)))) != old);
    return swap;
}

template <>
__device__ double atomicSwapIfLess(double* address, double value) {
    double swap, old;
    swap = *address;
    do {
        if(swap <= value) {
            break;
        }
        old = swap;
    } while((swap = __longlong_as_double(atomicCAS((unsigned long long int*)address, __double_as_longlong(swap), __double_as_longlong(value)))) != old);
    return swap;
}

// function to set solution_dev to infinity value
template <typename Float>
__global__ void setSolutionsToInfinity(Float* solutions_dev, Float infinity_value, size_t size_sol){
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sol){
        solutions_dev[threadId] = infinity_value;
    }
}

// method to initialize solutions in source nodes to 0.
// input parameters:
// - vector storing solutions on device
// - vector storing source nodes on device
// - size of vector storing source nodes
template <typename Float>
__global__ void setSolutionsSources(Float* solutions_dev, int* source_nodes_dev, size_t size_sources){
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < size_sources) {
        solutions_dev[source_nodes_dev[threadId]] = 0.0;
    }
}



//requires size < total number of vertices in mesh
//requires tid + domain_begin < total number of vertices in mesh
//length(map) == length(output)
//requires map being the exclusive scan of a boolean array predicate
//requires max(map) < lenght(output) : ok, map is non-decreasing and the max value is in the last position. At most its value is equals to length(map) == length(output)
//requires doesn't exist i,j such that map[i] == map[j] && predicate[i] == predicate[j] == 1 : ok, suppose (without loss of generality) that  i < j, then map[i] = a and map[j] = b + predicate[i] + a
//requires map containing all and only numbers from 0 to max(map), satisfied as map is non-decreasing and the max difference between map[i+1] and map[i] is 1.
//ensures that 1) if predicate[i] == 1 than exists j such that output[j] == i + domain_begin 2)if map[i] is written to, then for each 0 <= j < i are also written to. (satisfied)
//ok, as if predicate[i] == 1 then output[map[j]] is assigned to i + domain_begin. Moreover, if predicate[k] == 1 than map[k] != map[j]


// input parameters:
// - vector storing the addresses
// - predicate vector
// - domain size
// - vector in which result will be stored
// - index corresponding to where the domain starts
__global__ void compact(int* map, int* predicate, size_t size, int* output, int domain_begin) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size) {
        if(predicate[tid] == 1) {
            output[map[tid]] = tid + domain_begin;
        }
    }
}

// count number of neighbours for each node
//requires length(result) < active_nodes
//requires length(ngh) >= shapes_size
//requires cList being output of compact kernel
//active nodes < num_vertices, ok
//requires cList[i] < active_nodes, ok (see requires compact kernel)
//requires : given shapes an array so that if ngh[i] = a and ngh[i+1] = b then from shapes[a](inclusive) to shapes[b](exclusive) are tetrahedra near vertex i. If i+1 >= len(ngh) then ngh[i+1] is assumed to be shapes_size
//ensures result[tid] contains the number of tetrahedra near to cList[tid]. ok, result[tid] = ngh[cList[tid] + 1] - ngh[cList[tid]] which is the number of tetrahedra near to cList[tid] (see requiere)
__global__ void count_Nbhs(int* cList, int* ngh, int* result, size_t active_nodes, size_t num_vertices, size_t shapes_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < active_nodes) {
        int temp = cList[tid];
        result[tid] = ((temp != num_vertices - 1) ? ngh[temp+1]: shapes_size) - ngh[temp];
    }
}

//requires cList being an array of vertices indexes, with length up to active_nodes
//requires nbhNr being the array st nbhNr[i] == number of near tetrahedra to cList[i], with length up to len(cList)
//requires sAd to satisfy the same properties as map in kernel compact, plus len(sAd) < active_nodes
//requires len(shapes) at least max(ngh[cList[tid]] + nbhNr[tid]), ok

//ensures elemListSize == max(sAd[tid] + nbhNr[tid] - 1) + 1, with tid < active_nodes. satisfied as the maximum is achived for tid == active_nodes - 1
//ensures if sAd[tid] = k then elemList from sAd[tid](inclusive) to sAd[tid] + nbhNr[tid] (exclusive) is filled with only and all cList[tid] neighbours
__global__ void gather_elements(int* sAd, int* cList, int* nbhNr, TetraConfig* elemList, size_t active_nodes, size_t* elemListSize, int* ngh_dev, TetraConfig* shapes_dev) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0) {
        *elemListSize = 
        sAd[active_nodes - 1] + 
        nbhNr[active_nodes - 1];
    }
    if(tid < active_nodes) {
        int begin = ngh_dev[cList[tid]];
        int end = begin + nbhNr[tid]; 
        for(int i = begin; i < end; i++) {
            elemList[sAd[tid] + i - begin] = shapes_dev[i];
        }
    }
}



//number of blocks is assumed to be precise
// elemList contains neighbouring tetrahedra of each node in active list (cList)
template <int D, typename Float>
__global__ void constructPredicate(TetraConfig* elemList, size_t* elemListSize, int active_nodes, int* sAddr, int* tetra_dev, Float* geo_dev, Float* solutions_dev, int* predicate, Float* M_dev, Float tol, int* active_list_dev, int domain_begin, int domain_size) {
    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

    int sAddr_index = blockIdx.x;
    int elemList_index = sAddr[sAddr_index] + threadIdx.x;
    int upper_bound = ((sAddr_index != active_nodes - 1) ? sAddr[sAddr_index + 1] : (int)*elemListSize);
    __shared__ Float shared_sol;
    Float old_sol;
    VectorExt coordinates[4];
    VectorV values;
    TetraConfig tetra;
    if(elemList_index < upper_bound) {
        // compute tetra index
        tetra = elemList[elemList_index];
        // old_sol = solusions_dev[node], where node is index of the node associated with tetra.config
        old_sol = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1]];
        if(threadIdx.x == 0) {
            shared_sol = old_sol;
        }
    }
    __syncthreads();
    if(elemList_index < upper_bound) {
        // we retrieve coordinates of all nodes in tetra
        for(int j = 0; j < D + 1; j++){
            for(int k = 0; k < D; k++) {
                coordinates[j][k] =  geo_dev[D * tetra_dev[(D+1) * tetra.tetra_index + j] + k];
            }
            // retrieve solutions of all nodes in tetra
            values[j] = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]];
        }
        // retrieve velocity matrix associated to tetra
        const Float* M;
        M = M_dev + tetra.tetra_index * 6;
        // we call the solve method of the LocalSolver
        Float sol = LocalSolver<D, Float>::solve(coordinates, values, M, D+1-tetra.tetra_config);
        // shared_sol <- min(shared_sol, sol)
        atomicSwapIfLess<Float>(&shared_sol, sol);        
    }
    __syncthreads();

    if(elemList_index < upper_bound) {
        int pred = std::abs(old_sol - shared_sol) < tol * (1 + 0.5 * (shared_sol + old_sol));
        if(pred == 1) {
            for(int j = 0; j < D + 1; j++) {
                // for each node in tetra except for the node
                if(j != tetra.tetra_config - 1) {
                    if(solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]] - shared_sol > tol) {
                        predicate[tetra_dev[(D+1) * tetra.tetra_index + j]] = 1;
                    }
                }
            }
            // we set to 2 converged nodes: active_list[node]=2
            if(threadIdx.x == 0) {
                active_list_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1] - domain_begin] = 2;
            }
        }

        // solution [node] = min(solution[node],shared sol)
        if(threadIdx.x == 0) {
            atomicSwapIfLess<Float>(&solutions_dev[tetra_dev[tetra.tetra_index * (D+1) + tetra.tetra_config - 1]], shared_sol);
        }
    }
}


//requires length(active_list) == predicate_size
//requires length(predicate) == predicate_size
//ensures if predicate[tid] ==1 then acive_list[tid] == 0, with tid < predicate_size
__global__ void removeConvergedNodes(int* active_list, int size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size) {
        if(active_list[tid] == 2) {
            active_list[tid] = 0;
        }
    }
}

// process all nodes set for further processing
template <int D, typename Float>
__global__ void processNodes(TetraConfig* elemList, size_t* elemListSize, int active_nodes, int* sAddr, int* tetra_dev, Float* geo_dev, Float* solutions_dev, int* active_list, Float* M_dev, Float tol, int domain_begin) {
    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

    int sAddr_index = blockIdx.x;
    int elemList_index = sAddr[sAddr_index] + threadIdx.x;
    int upper_bound = ((sAddr_index != active_nodes - 1) ? sAddr[sAddr_index + 1] : (int)*elemListSize);
    __shared__ Float shared_sol;
    Float old_sol;
    VectorExt coordinates[4];
    VectorV values;
    TetraConfig tetra;
    int tetra_dev_val;
    if(elemList_index < upper_bound) {
        tetra = elemList[elemList_index];
        tetra_dev_val = tetra_dev[tetra.tetra_index * (D+1) + tetra.tetra_config - 1];
        if(active_list[tetra_dev_val - domain_begin] != 0) {
            return;
        }
        // old_sol = solution[node], node is index of the node associated with tetra.config
        old_sol = solutions_dev[tetra_dev_val];
        if(threadIdx.x == 0) {
            shared_sol = old_sol;
        }
    }
    __syncthreads();
    // we retrieve coordinates of all nodes in tetra
    if(elemList_index < upper_bound) {
        for(int j = 0; j < D + 1; j++){
            for(int k = 0; k < D; k++) {
                coordinates[j][k] =  geo_dev[D * tetra_dev[(D+1) * tetra.tetra_index + j] + k];
            }
            // retrieve solutions of all nodes in tetra
            values[j] = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]];
        }
        // retrieve velocity matrix associated to tetra
        const Float* M;
        M = M_dev + tetra.tetra_index * 6;
        // solve local problem
        Float sol = LocalSolver<D, Float>::solve(coordinates, values, M, D+1-tetra.tetra_config);
        // shared_sol = min(shared_sol, sol)
        atomicSwapIfLess<Float>(&shared_sol, sol);        
    }
    __syncthreads();

    if(elemList_index < upper_bound) {
        if(threadIdx.x == 0) {
            if(shared_sol < old_sol) {
                // solution[node] <- shared_sol
                atomicSwapIfLess<Float>(&solutions_dev[tetra_dev_val], shared_sol);
                //solutions_dev[tetra_dev_val] = shared_sol;
                // active_lis[node] <- 1
                active_list[tetra_dev_val - domain_begin] = 1;
            }
        }
    }
}

// method to propagate activation of nodes across domains
__global__ void propagatePredicate(int* active_list_dev, int domain_size, int begin_domain, int* predicate) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < domain_size) {
            if(predicate[begin_domain + tid] == 1) {
                active_list_dev[tid] = 1;
            }
        predicate[begin_domain + tid] = 0;
    }
}

#endif