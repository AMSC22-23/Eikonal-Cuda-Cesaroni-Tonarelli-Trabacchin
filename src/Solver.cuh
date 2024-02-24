#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH

#include "Mesh.cuh"
#include <cuda.h>

template <int D,typename Float>
class Solver {
public:
    Solver( Mesh<D>* mesh, int tetrahedra_per_block):mesh(mesh), tetrahedra_per_block(tetrahedra_per_block) {}

    void gpu_data_transfer(){

        //allocate and initialize partitions_vertices
        cudaMalloc(& partitions_dev, mesh->get_partitions().size()*sizeof(int));
        cudaMemcpy(partitions_dev, mesh->get_partitions().ptr(), mesh->get_partitions().size()*sizeof(int), cudaMemcpyHostToDevice);
    }





private:
    Mesh<D>* mesh;
    int* partitions_dev;
    int tetrahedra_per_block;
    int* global_to_local_mapping;


};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
