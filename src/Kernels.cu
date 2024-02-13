#ifndef KERNELS
#define KERNELS
#include <vector>
#include <cuda.h>
#include <random>
#include <set>

__global__ void partition_mesh_kernel(int* visitedNodes, int dim_d, int* neighbors, int dim_ne, int* indices, int* frontier,
                                      int dim_f, int* new_frontier, int* dim_nf){
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        printf("neighbors size gpu : %d\n", dim_ne);
    }
    if (threadIdx.x + blockDim.x * blockIdx.x < dim_f) {
        int vertex = frontier[threadIdx.x + blockDim.x * blockIdx.x];
        int end = (vertex == dim_d - 1) ? dim_ne : indices[vertex + 1];
        for (int i = indices[vertex]; i < end; i++) {
            if(i>=dim_ne) {
                printf("error1 i=%d vertex=%d\n",i,vertex);
            }
            int v = neighbors[i];
            if(v>=dim_d) {
                printf("error2\n");
            }
            int old = atomicCAS(&visitedNodes[v], 0, visitedNodes[vertex]);
            if (old == 0) {
                int index = atomicAdd(dim_nf, 1);
                new_frontier[index] = v;
            }
        }
    }
}

int* partition_mesh_host(const std::vector<int>* neighbors, const std::vector<int>* indices, int n_sub){
    int *visitedNodes_cpu, *visitedNodes_gpu, *frontier_cpu, *frontier_gpu,
                 *new_frontier_gpu, *neighbors_gpu, *indices_gpu;
    int dim_d, dim_ne, dim_f, *dim_nf;
    dim_d = indices->size();
    dim_ne = neighbors->size();
    std::cout << "neighbors size cpu: " << dim_ne << std::endl;
    dim_f = n_sub;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,dim_d - 1);
    std::vector<std::set<int>> v(n_sub);
    frontier_cpu = (int*)malloc(n_sub*sizeof(int));
    visitedNodes_cpu = (int*)malloc(dim_d*sizeof(int));
    for(int i = 0; i < n_sub; i++) {
        frontier_cpu[i] = (int)dist(rng);
    }
    for(int i = 0; i < dim_d; i++){
        visitedNodes_cpu[i] = 0;
    }

    for(int i = 0; i < n_sub; i++) {
        visitedNodes_cpu[frontier_cpu[i]] = i + 1;
    }

    //allocate and initialize visitedNodes
    cudaMalloc(&visitedNodes_gpu, indices->size() * sizeof(int));
    cudaMemcpy(visitedNodes_gpu, visitedNodes_cpu, dim_d * sizeof(int), cudaMemcpyHostToDevice);
    //allocate frontier
    cudaMalloc(&frontier_gpu, dim_d * sizeof(int));
    //allocate and initialize indices
    cudaMalloc(&indices_gpu, dim_d * sizeof(int));
    cudaMemcpy(indices_gpu, indices->data(), dim_d * sizeof(int), cudaMemcpyHostToDevice);
    //allocate and initialize neighbors
    cudaMalloc(&neighbors_gpu, dim_ne * sizeof(int));
    cudaMemcpy(neighbors_gpu, neighbors->data(), dim_ne * sizeof(int), cudaMemcpyHostToDevice);
    //allocate and initialize new_frontier (which will be assigned to frontier in the while loop)
    cudaMalloc(&new_frontier_gpu, dim_d * sizeof(int));
    //allocate and initialize dim_new_frontier (which will be assigned to dim_frontier in the while loop)
    cudaMalloc(&dim_nf, sizeof(int));
    cudaMemset(dim_nf, 0, sizeof(int));
    cudaMemcpy(frontier_gpu, frontier_cpu, dim_d * sizeof(int), cudaMemcpyHostToDevice);
    while(dim_f != 0){
        //copy new_frontier into frontier

        //copy dim_nf into dim_f
        //cudaMemcpy(&dim_f, dim_nf, sizeof(int), cudaMemcpyDeviceToHost);
        //set dim_nf to zero
        partition_mesh_kernel<<<5,4>>>(visitedNodes_gpu, dim_d, neighbors_gpu, dim_ne, indices_gpu, frontier_gpu,
                dim_f, new_frontier_gpu, dim_nf);
        cudaMemcpy(&dim_f, dim_nf, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(frontier_gpu, new_frontier_gpu, dim_d * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(dim_nf, 0, sizeof(int));

    }
    cudaMemcpy(visitedNodes_cpu, visitedNodes_gpu, dim_d*sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << err << std::endl;
        std::cout << "execution failed" << std::endl;
        exit(1);
    }
    return visitedNodes_cpu;

}
#endif
