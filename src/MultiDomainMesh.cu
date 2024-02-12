#include "Mesh.cu"
#include "Kernels.cu"
#include <random>

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