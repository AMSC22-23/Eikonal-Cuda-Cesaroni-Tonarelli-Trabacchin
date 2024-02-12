#include "Mesh.cu"
#include <random>
template<int D>
class MultiDomainMesh {
public:
    MultiDomainMesh(const Mesh<D>* mesh, int n) {

    }
private:


    void partition_mesh(const Mesh<D>* mesh, int n) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0,mesh->getNumberVertices() - 1); // distribution in range [1, 6]
        std::vector<std::set<int>> v(n);
        for(int i = 0; i < n; i++) {
            v[i].insert((int)dist(rng));
        }


    }
};