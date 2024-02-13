
#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MULTI_DOMAIN_MESH_H
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MULTI_DOMAIN_MESH_H
#include "Mesh.cu"
#include "Kernels.cu"

template<int D>
class MultiDomainMesh {
public:
    /*MultiDomainMesh(const Mesh<D>* mesh, int n) {

    }*/

    int* partition_mesh(const Mesh<D>* mesh, int n) {
        return partition_mesh_host(&(mesh->getVectorNeighbors()), &(mesh->getVectorNeighborsIndices()), n);
    }

    void getSolutionsVTK(Mesh<D>& mesh, const std::string& output_file_name, int* solutions){
        std::ofstream output_file(output_file_name);

        std::string input = mesh.getFilenameInputMesh();
        std::ifstream input_file(input);

        std::string line;
        if (input_file && output_file) {
            while (std::getline(input_file, line)) {
                output_file << line << "\n";
            }
        }
        else {
            printf("Cannot read File");
        }
        input_file.close();

        output_file << "POINT_DATA " << mesh.getOriginalNumberOfVertices() << std::endl;
        output_file << "SCALARS solution double 1" << std::endl;
        output_file << "LOOKUP_TABLE default" << std::endl;
        for(int i = 0; i < mesh.getOriginalNumberOfVertices(); i++){
            double solution = solutions[mesh.getMapVertex(i)];
            output_file << solution << " ";
        }
        output_file << std::endl;
        output_file.flush();

        output_file.close();
    }
};
#endif