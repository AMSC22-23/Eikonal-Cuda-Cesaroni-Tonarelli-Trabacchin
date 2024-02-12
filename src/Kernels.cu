#include <vector>
#include <cuda.h>
#include <random>
#include <set>

__global__ void partition_mesh_kernel(int* visitedNodes, int dim_d, int* neighbors, int dim_ne, int* indices, int* frontier,
                                      int dim_f, int* new_frontier, int* dim_nf){
    if (threadIdx.x + blockDim.x * blockIdx.x < dim_f) {
        int vertex = frontier[threadIdx.x + blockDim.x * blockIdx.x];
        int end = vertex == dim_d - 1 ? dim_ne : indices[vertex + 1];
        for (int i = indices[vertex]; i < end; i++) {
            int v = neighbors[i];
            int old = atomicCAS(&visitedNodes[v], 0, visitedNodes[vertex]);
            if (old == 0) {
                int index = atomicAdd(dim_nf, 1);
                new_frontier[index] = v;
            }
        }
    }
}

void partition_mesh(std::vector<int>* neighbors, std::vector<int>* indices, int n_sub){
    int *visitedNodes_cpu, *visitedNodes_gpu, *frontier_cpu, *frontier_gpu, *new_frontier_cpu, *new_frontier_gpu, *neighbors_gpu;
    int dim_d, dim_ne, dim_f, *dim_nf;
    dim_d = indices->size();
    dim_ne = neighbors->size();
    dim_f = n_sub;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,dim_d - 1); // distribution in range [1, 6]
    std::vector<std::set<int>> v(n_sub);
    for(int i = 0; i < n_sub; i++) {
        new_frontier_cpu[i] = (int)dist(rng);
    }
    for(int i = 0; i < dim_d; i++){
        visitedNodes_cpu[i] = 0;
    }

    cudaMalloc(&visitedNodes_gpu, indices->size() * sizeof(int));
    cudaMemcpy(visitedNodes_gpu, visitedNodes_cpu, dim_d * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&frontier_gpu, indices->size() * sizeof(int));

    cudaMalloc(&new_frontier_gpu, dim_d * sizeof(int));
    cudaMemcpy(new_frontier_gpu, new_frontier_cpu, dim_d * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dim_nf, sizeof(int));
    cudaMemset(&dim_nf, n_sub, sizeof(int));

    while(true){
        cudaMemcpy(frontier_gpu, new_frontier_gpu, dim_d * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&dim_f, dim_nf, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(&dim_nf, 0, sizeof(int));
        partition_mesh_kernel<<<5,4>>>(visitedNodes_gpu, dim_d, neighbors->data() ,dim_ne, indices, frontier_gpu,
                dim_f, new_frontier_gpu, dim_nf);

    }

}