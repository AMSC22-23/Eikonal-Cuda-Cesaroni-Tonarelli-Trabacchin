
#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MULTI_DOMAIN_MESH_H
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MULTI_DOMAIN_MESH_H
#include "Mesh.cu"
#include "Kernels.cu"

template<int D>
class MultiDomainMesh {
public:
    MultiDomainMesh(const Mesh<D>* mesh, int n) {

    }
private:

    void partition_mesh(const Mesh<D>* mesh, int n) {
        partition_mesh_host(mesh->getVectorNeighbors(), mesh->getVectorNeighborsIndices(), n);
    }
};
#endif