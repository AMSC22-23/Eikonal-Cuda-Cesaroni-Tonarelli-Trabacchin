#include "Mesh.cuh"

int main() {
    Mesh<3, double> mesh("../test/meshes/input_meshes/cube-20.vtk",4, " ");


    return 0;
}