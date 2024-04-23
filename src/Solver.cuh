#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH

#include "Mesh.cuh"
#include "Kernels.cu"
#include <cuda.h>

template <int D,typename Float>
class Solver {
public:
    Solver(Mesh<D, Float>* mesh) : mesh(mesh){
        gpuDataTransfer();
    }

    void gpuDataTransfer(){
        // Allocate memory in device
        cudaMalloc(&active_domains_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&partitions_vertices_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&partitions_tetra_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&geo_dev, sizeof(Float) * mesh->getNumberVertices() * D);
        cudaMalloc(&tetra_dev, sizeof(int) * mesh->get_tetra().size());
        cudaMalloc(&shapes_dev, sizeof(TetraConfig) * mesh->getShapes().size());
        cudaMalloc(&ngh_dev, sizeof(int) * mesh->get_ngh().size());
        cudaMalloc(&M_dev, sizeof(Float) * mesh->get_M().size());
        cudaMalloc(&solutions_dev, sizeof(Float) * mesh->getNumberVertices());
        // Copy data from host to device
        cudaMemcpy(partitions_vertices_dev, mesh->getPartitionVertices().data(), sizeof(int) * mesh->getPartitionsNumber(), cudaMemcpyHostToDevice);
        cudaMemcpy(partitions_tetra_dev, mesh->getPartitionTetra().data(), sizeof(int) * mesh->getPartitionsNumber(), cudaMemcpyHostToDevice);
        cudaMemcpy(geo_dev, mesh->getGeo().data(), sizeof(Float) * mesh->getNumberVertices() * D, cudaMemcpyHostToDevice);
        cudaMemcpy(tetra_dev, mesh->get_tetra().data(), sizeof(int) * mesh->get_tetra().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(shapes_dev, mesh->getShapes().data(), sizeof(TetraConfig) * mesh->getShapes().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ngh_dev, mesh->get_ngh().data(), sizeof(int) * mesh->get_ngh().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(M_dev, mesh->get_M().data(), sizeof(Float) * mesh->get_M().size(), cudaMemcpyHostToDevice);
    }

    void solve(std::vector<int> source_nodes, Float infinity_value, const std::string& output_file_name){
        std::vector<cudaStream_t*> streams;
        int* active_domains;
        bool check = false;

        // allocate host memory
        active_domains = (int*)malloc(sizeof(int) * mesh->getPartitionsNumber());

        // prepare the data
        setSolutionsAndActiveDomains(source_nodes, infinity_value);
        // create streams, one for each domain
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamCreate(streams[i]);
        }

        // loop used to perform domain sweeps
        while(!check){
            cudaMemcpy(active_domains, active_domains_dev, sizeof(int) * mesh->getPartitionsNumber(), cudaMempcyDeviceToHost);
            check = false;
            // check if all domains are not active; in that case, the computation is done and there is no need to do other domain sweeps
            for(int i = 0; i < mesh->getPartitionsNumber() && !check; i++){
                if(active_domains[i] == 1) check = true;
            }
            cudaMemcset(active_domains_dev, 0, sizeof(int) * mesh->getPartitionsNumber());
            // perform sweep over active domains
            for(int i = 0; i < mesh->getPartitionsNumber() && !check; i++){
                if(active_domains[i] == 1){
                    // TODO compute right number of blocks and threads per block
                    domainSweep<<<1024, 2, 0, streams[i]>>>(i, partitions_vertices_dev, partitions_tetra_dev, geo_dev, tetra_dev,
                                      shapes_dev, ngh, M_dev, solutions_dev, active_domains_dev, mesh->getPartitionsNumber(),
                                      mesh->getNumberVertices(), mesh->getNumberTetra(), mesh->getShapes().size(), infinity_value);
                }
            }
            cudaDeviceSynchronize();
        }

        // copy solutions from device to host and print them into a vtk file to allow visualization
        Float* solutions;
        cudaMemcpy(solutions, solutions_dev.data(), sizeof(Float) * mesh->getNumberVertices().size(), cudaMemcpyDeviceToHost);
        mesh.getSolutionsVTK(output_file_name, solutions);

        // destroy streams
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamDestroy(streams[i]);
        }
        // free host memory
        delete active_domains;
        delete solutions;
    }



private:
    Mesh<D>* mesh;
    int* partitions_vertices_dev;
    int* partitions_tetra_dev;
    Float* geo_dev;
    int* tetra_dev;
    TetraConfig* shapes_dev;
    int* ngh;
    Float* M_dev;
    Float* solutions_dev;
    int* active_domains_dev;

    void setSolutionsAndActiveDomains(std::vector<int>& source_nodes, Float infinity_value){
        int* source_nodes_dev;
        cudaMalloc(&source_nodes_dev, sizeof(int) * source_nodes.size());
        cudaMemcpy(source_nodes_dev, source_nodes.data(), sizeof(int) * source_nodes.size(), cudaMempcyHostToDevice);
        int numBlocksInfinity = mesh->getNumberVertices() / 1024 + 1;
        setSolutionsToInfinity<<<1024, numBlocksInfinity>>>(solutions_dev, infinity_value, mesh->getNumberVertices());
        int numBlocksSources = source_nodes.size() / 32 + 1;
        cudaMemset(active_domains_dev, 0, sizeof(int) * mesh->getPartitionsNumber());
        setSolutionsSourcesAndDomains<<<32, numBlocksSources>>>(solutions_dev, source_nodes_dev, source_nodes.size());
        cudaFree(&source_nodes_dev);
    }


};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
