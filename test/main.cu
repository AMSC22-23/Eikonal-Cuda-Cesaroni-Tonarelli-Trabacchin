#include "Mesh.cu"

int main() {
    Mesh<3> mesh("../test/meshes/input_meshes/cube-20.vtk", 4);
    std::cout << mesh.getNumberVertices() << std::endl;
    int* sol;
    sol = new int[mesh.getNumberVertices()];
    memset(sol, 0, mesh.getNumberVertices()* sizeof(int));
    for(int i = 0; i < mesh.partitions.size(); i++){
        int begin = mesh.partitions[i];
        int end = (i == mesh.partitions.size() - 1) ? mesh.getNumberVertices() : mesh.partitions[i+1];
        for (int j = begin; j < end; j++){
            sol[j] = i+1;
        }
    }

    mesh.getSolutionsVTK("../test/meshes/output_meshes/cube-20-new.vtk", sol);

    return 0;
}