#ifndef KERNELS
#define KERNELS
#include <vector>
#include <cuda.h>
#include <random>
#include <set>
#include <array>
//#include "../localProblem_alt2/include/Phi.hpp"
//#include "../localProblem_alt2/include/solveEikonalLocalProblem.hpp"
#include "Mesh.cuh"
#include "LocalSolver.cuh"
#include "CudaEikonalTraits.cuh"
#include <cmath>
constexpr int D = 3;
/*
using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
using Matrix = typename Eikonal::Eikonal_traits<D,2>::AnisotropyM;
using VectorV = typename Eigen::Matrix<double,4,1>;
*/

template <typename Float>
__device__ Float atomicSwapIfLess(Float* address, Float value) {
    Float swap, old;
    swap = *address;


    do {
        if(swap <= value) {
            break;
        }
        old = swap;

    }while((swap = atomicCAS(address, swap, value)) != old);
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

    }while((swap = __uint_as_float(atomicCAS((unsigned int*)address, __float_as_uint(swap), __float_as_uint(value)))) != old);
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

    }while((swap = __longlong_as_double(atomicCAS((unsigned long long int*)address, __double_as_longlong(swap), __double_as_longlong(value)))) != old);
    return swap;

}


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



//requires size < total number of vertices in mesh
//requires tid + domain_begin < total number of vertices in mesh
//length(map) == length(output)
//requires map being the exclusive scan of a boolean array predicate
//reqquires max(map) < lenght(output) : ok, map is non-decreasing and the max value is in the last position. At most its value is equals to length(map) == length(output)
//requires doesn't exist i,j such that map[i] == map[j] && predicate[i] == predicate[j] == 1 : ok, suppose (without loss of generality) that  i < j, then map[i] = a and map[j] = b + predicate[i] + a
//requires map containing all and only numbers from 0 to max(map), satisfied as map is non-decreasing and the max difference between map[i+1] and map[i] is 1.
//ensures that 1) if predicate[i] == 1 than exists j such that output[j] == i + domain_begin 2)if map[i] is written to, then for each 0 <= j < i are also written to. (satisfied)
//ok, as if predicate[i] == 1 then output[map[j]] is assigned to i + domain_begin. Moreover, if predicate[k] == 1 than map[k] != map[j]

__global__ void compact(int* map, int* predicate, size_t size, int* output, int domain_begin) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < size) {
        if(predicate[tid] == 1) {
            output[map[tid]] = tid + domain_begin;
        }
    }

}


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

/*__global__ void count_Nbhs_vertices(int* cList, int* ngh, int* tetra, int* result, size_t active_nodes, size_t num_vertices, size_t shapes_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < active_nodes) {
        int temp = cList[tid];

    }
}*/

//requires cList being an array of vertices indexes, with length up to active_nodes
//requires nbhNr being the array st nbhNr[i] == number of near tetrahedra to cList[i], with length up to len(cList)
//requires sAd to satisfy the same properties as map in kernel compact, plus len(sAd) < active_nodes
//requires len(shapes) at least max(ngh[cList[tid]] + nbhNr[tid]), ok

//ensures elemListSize == max(sAd[tid] + nbhNr[tid] - 1) + 1, with tid < active_nodes. satisfied as the maximum is achived for tid == active_nodes - 1
//ensures if sAd[tid] = k then elemList from sAd[tid](inclusive) to sAd[tid] + nbhNr[tid] (exclusive) is filled with only and all cList[tid] neighbours
__global__ void gather_elements(int* sAd, int* cList, int* nbhNr, TetraConfig* elemList, size_t active_nodes, size_t* elemListSize, int* ngh_dev, TetraConfig* shapes_dev) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0) {
        //printf("active nodes %d %p %p %p %p %p\n", active_nodes , &sAd[0], &nbhNr[active_nodes - 1], &cList[tid], &ngh_dev[cList[tid]], &nbhNr[tid]);

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
template <int D, typename Float>
__global__ void construct_predicate_shared_memory(TetraConfig* elemList, size_t* elemListSize, int active_nodes, int* sAddr, int* tetra_dev, Float* geo_dev, Float* solutions_dev, int* predicate, Float* M_dev, Float tol, int* active_list_dev, int domain_begin, int domain_size) {

    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

    int sAddr_index = blockIdx.x;
    int elemList_index = sAddr[sAddr_index] + threadIdx.x;
    int upper_bound = ((sAddr_index != active_nodes - 1) ? sAddr[sAddr_index + 1] : (int)*elemListSize);
    __shared__ Float shared_sol;
    Float old_sol;
    //int old_active_status;
    VectorExt coordinates[4];
    VectorV values;
    TetraConfig tetra;
    if(elemList_index < upper_bound) {

        tetra = elemList[elemList_index];
        old_sol = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1]];
        //old_active_status = active_list_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1] - domain_begin];
        if(threadIdx.x == 0) {
            shared_sol = old_sol;
        }
    }
    __syncthreads();
    if(elemList_index < upper_bound) {

        for(int j = 0; j < D + 1; j++){
            for(int k = 0; k < D; k++) {
                coordinates[j][k] =  geo_dev[D * tetra_dev[(D+1) * tetra.tetra_index + j] + k];
            }
            values[j] = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]];
        }

        const Float* M;
        M = M_dev + tetra.tetra_index * 6;
        Float sol = LocalSolver<D, Float>::solve(coordinates, values, M, D+1-tetra.tetra_config);
        /*if(tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1] == 39) {
            printf("solution found for 39 : %f tetra_index : %d, %d %d \n", shared_sol, tetra.tetra_index, elemList_index, upper_bound);
        }*/
        atomicSwapIfLess<Float>(&shared_sol, sol);        
    }
    __syncthreads();

    if(elemList_index < upper_bound) {

        int pred = std::abs(old_sol - shared_sol) < tol * (1 + 0.5 * (shared_sol + old_sol));
        if(pred == 1) {
            for(int j = 0; j < D + 1; j++) {
                if(j != tetra.tetra_config - 1) {
                    if(solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]] - shared_sol > 0) {
                        predicate[tetra_dev[(D+1) * tetra.tetra_index + j]] = 1;
                        //printf("vertex %d added to predicate\n", tetra_dev[(D+1) * tetra.tetra_index + j]);
                    }
                }
            }
        
            if(threadIdx.x == 0) {
                active_list_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1] - domain_begin] = 2;
            }
        
            
        }


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


template <int D, typename Float>
__global__ void domain_Sweep_shared_memory(TetraConfig* elemList, size_t* elemListSize, int active_nodes, int* sAddr, int* tetra_dev, Float* geo_dev, Float* solutions_dev, int* active_list, Float* M_dev, Float tol, int domain_begin) {

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
        tetra = elemList[elemList_index];
        old_sol = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + tetra.tetra_config - 1]];
        if(threadIdx.x == 0) {
            shared_sol = old_sol;
        }
    }
    __syncthreads();
    if(elemList_index < upper_bound && active_list[tetra_dev[tetra.tetra_index * (D+1) + tetra.tetra_config - 1] - domain_begin] == 0) {
        for(int j = 0; j < D + 1; j++){
            for(int k = 0; k < D; k++) {
                coordinates[j][k] =  geo_dev[D * tetra_dev[(D+1) * tetra.tetra_index + j] + k];
            }
            values[j] = solutions_dev[tetra_dev[(D+1) * tetra.tetra_index + j]];
        }

        const Float* M;
        M = M_dev + tetra.tetra_index * 6;
        Float sol = LocalSolver<D, Float>::solve(coordinates, values, M, D+1-tetra.tetra_config);
        atomicSwapIfLess<Float>(&shared_sol, sol);        
    }
    __syncthreads();

    if(elemList_index < upper_bound) {
        
        if(threadIdx.x == 0) {
            if(shared_sol < old_sol) {
                atomicSwapIfLess<Float>(&solutions_dev[tetra_dev[tetra.tetra_index * (D+1) + tetra.tetra_config - 1]], shared_sol);
                active_list[tetra_dev[tetra.tetra_index * (D+1) + tetra.tetra_config - 1] - domain_begin] = 1;
            }
        }
    }
}

__global__ void propagatePredicate(int* active_list_dev, int domain_size, int begin_domain, int* predicate) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < domain_size) {
        //if(active_list_dev[tid] != 2) {
            if(predicate[begin_domain + tid] == 1) {
                active_list_dev[tid] = 1;
                //printf("added node %d \n", tid + begin_domain);
            }
            //active_list_dev[tid] |= predicate[begin_domain + tid];
        //}
        predicate[begin_domain + tid] = 0;
    }
}

template <typename Float>
__global__ void activeListPrint(int* active_list, Float* solutions, int begin_domain, int size) {
    for(int i = 0; i < size; i++) {
        printf("%d (%f) ", active_list[i], solutions[active_list[i]]);
    }
    printf("\n");
}

__global__ void elemListPrint(TetraConfig* elem_list, int* tetra_dev, size_t* size) {
    for(int i = 0; i < *size; i++) {
        printf("(%d %d %d %d)\n", tetra_dev[4*elem_list[i].tetra_index + 0], tetra_dev[4*elem_list[i].tetra_index + 1], tetra_dev[4*elem_list[i].tetra_index + 2], tetra_dev[4*elem_list[i].tetra_index + 3]);
    }
    printf("\n");
}


__global__ void checkZero(int* begin, int* end) {
    while(begin!=end) {
        if(*begin != 0) {
            printf("error %d\n", *begin);
        }
        begin++;
    }
}

__global__ void active_print(int* list, int size) {
    for(int i = 0; i < size; i++) {
        if(list[i] == 1) {
            //printf("active lists contains %d\n", i);
        }
    }
}


#endif