//
// Created by Melanie Tonarelli on 12/02/24.
//
#include "MultiDomainMesh.cu"
#include "TetrahedricalMesh.cu"
int main() {
    TetrahedricalMesh<3> mesh("../test/meshes/input_meshes/cube-20.vtk");

    MultiDomainMesh<3> mdmesh;
    int* sol = mdmesh.partition_mesh(&mesh, 5);
    mdmesh.getSolutionsVTK(mesh, "../test/meshes/output_meshes/cube-20.vtk", sol);
}