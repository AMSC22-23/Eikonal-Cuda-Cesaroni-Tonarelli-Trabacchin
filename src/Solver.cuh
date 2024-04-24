#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH

#include "Mesh.cuh"
#include "Kernels.cu"
#include <cuda.h>

constexpr int NUM_THREADS = 1024;
constexpr int SIZE_WARP = 32;

template <int D,typename Float>
class Solver {
public:
    Solver(Mesh<D, Float>* mesh) : mesh(mesh){
        gpuDataTransfer();
    }

    void gpuDataTransfer(){
        printf("GPU data transfer started\n");
        // Allocate memory in device
        cudaMalloc(&active_domains_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMemset(active_domains_dev,0,sizeof(int) * mesh->getPartitionsNumber());
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
        printf("GPU data transfer completed\n");
    }

    void solve(std::vector<int> source_nodes, Float tol, Float infinity_value, const std::string& output_file_name){
        std::cout << "Start solve..." << std::endl;
        std::vector<cudaStream_t> streams;
        int* active_domains;
        bool check = true;

        // allocate host memory
        active_domains = (int*)malloc(sizeof(int) * mesh->getPartitionsNumber());

        // prepare the data
        setSolutionsAndActiveDomains(source_nodes, infinity_value);
        // create streams, one for each domain
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamCreate(&streams[i]);
        }

        // loop used to perform domain sweeps
        while(check){
            std::cout<<"while" << std::endl;
            cudaMemcpy(active_domains, active_domains_dev, sizeof(int) * mesh->getPartitionsNumber(), cudaMemcpyDeviceToHost);
            check = false;
            std::cout<<"ok1" << std::endl;
            // check if all domains are not active; in that case, the computation is done and there is no need to do other domain sweeps
            for(int i = 0; i < mesh->getPartitionsNumber() && !check; i++){
                if(active_domains[i] == 1) check = true;
            }
            std::cout<<"ok2" << std::endl;
            cudaMemset(active_domains_dev, 0, sizeof(int) * mesh->getPartitionsNumber());
            std::cout<<"ok3 part = " << mesh->getPartitionsNumber() << std::endl;
            // perform sweep over active domains
            /**
            for(int i = 0; i < mesh->getPartitionsNumber() && check; i++){
                std::cout << "loop i = " << i << std::endl;
                if(active_domains[i] == 1){
                    std::cout<<"ok4" << std::endl;
                    int numBlocks = (partitions_vertices_dev[i] -  ((i == 0) ? -1 : partitions_vertices_dev[i-1])) / NUM_THREADS + 1;
                    std::cout<<"before kernel" << std::endl;
                    domainSweep<<<NUM_THREADS, numBlocks, 0, streams[i]>>>(i, partitions_vertices_dev, partitions_tetra_dev, geo_dev, tetra_dev,
                                      shapes_dev, ngh_dev, M_dev, solutions_dev, active_domains_dev, mesh->getPartitionsNumber(),
                                      mesh->getNumberVertices(), mesh->getNumberTetra(), mesh->getShapes().size(), infinity_value, tol);

                }
            }*/
            cudaDeviceSynchronize();
        }

        // copy solutions from device to host and print them into a vtk file to allow visualization
        Float* solutions;
        cudaMemcpy(solutions, solutions_dev, sizeof(Float) * mesh->getNumberVertices(), cudaMemcpyDeviceToHost);
        mesh->getSolutionsVTK(output_file_name, solutions);

        // destroy streams
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamDestroy(streams[i]);
        }
        // free host memory
        delete active_domains;
        delete solutions;
    }



private:
    Mesh<D, Float>* mesh;
    int* partitions_vertices_dev;
    int* partitions_tetra_dev;
    Float* geo_dev;
    int* tetra_dev;
    TetraConfig* shapes_dev;
    int* ngh_dev;
    Float* M_dev;
    Float* solutions_dev;
    int* active_domains_dev;

    void setSolutionsAndActiveDomains(std::vector<int>& source_nodes, Float infinity_value){
        int* source_nodes_dev;
        cudaMalloc(&source_nodes_dev, sizeof(int) * source_nodes.size());
        cudaMemcpy(source_nodes_dev, source_nodes.data(), sizeof(int) * source_nodes.size(), cudaMemcpyHostToDevice);
        int numBlocksInfinity = mesh->getNumberVertices() / NUM_THREADS + 1;
        setSolutionsToInfinity<<<NUM_THREADS, numBlocksInfinity>>>(solutions_dev, infinity_value, mesh->getNumberVertices());
        int numBlocksSources = source_nodes.size() / SIZE_WARP + 1;
        cudaMemset(active_domains_dev, 0, sizeof(int) * mesh->getPartitionsNumber());
        setSolutionsSourcesAndDomains<<<SIZE_WARP, numBlocksSources>>>(solutions_dev, source_nodes_dev, active_domains_dev, partitions_vertices_dev, mesh->getPartitionsNumber(), source_nodes.size());
        cudaFree(&source_nodes_dev);
    }

};

#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH