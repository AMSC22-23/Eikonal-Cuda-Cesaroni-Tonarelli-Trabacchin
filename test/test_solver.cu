#include "../src/Solver.cuh"
#include "../src/Mesh.cuh"
#include "../src/CudaEikonalTraits.cuh"

#include <string>
#include <iostream>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[]){
    constexpr int D = 3;
    const double tol = 1e-3;
    using VectorExt = typename CudaEikonalTraits<float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<float, D>::Matrix;

    const double infinity_value = 2000;
    if(argc == 4)
    {
        std::cout << "Start main..." << std::endl;
        // Retrieve parameters
        std::string input_fileName = argv[1];
        int num_parts = std::atoi(argv[2]);
        std::string output_fileName = argv[3];
        std::string fileName = "../test/meshes/output_meshes/" + output_fileName + ".vtk";

        // Setting velocity matrix
        //Matrix M {1, 0, 0, 0, 1, 0, 0, 0, 1};
        Matrix M;
        M << 1,0,0,
            0,1,0,
            0,0,1;
            


        // Instantiating mesh
        std::cout << "creating mesh" << std::endl;
        Mesh<D,float> mesh(input_fileName, num_parts, M);
        std::cout << "Created mesh..." << std::endl;
        // Setting boundary
        std::vector<int> boundary;
        boundary.push_back(mesh.getNearestVertex(std::array<float, D>({0, 0, 0})));
        // boundary.push_back(mesh.getNearestVertex(std::array<double, D>({1, 1, 1})));
        std::cout << "Boundary set..." << std::endl;

        // Instantiating Eikonal Solver
        Solver<D,float> solver(&mesh);
        std::cout << "Initialised solver..." << std::endl;

        // Solve
        auto start1 = std::chrono::high_resolution_clock::now();
        solver.solve(boundary, tol, infinity_value, fileName);
        auto stop1 = std::chrono::high_resolution_clock::now();
        std::cout << "solved..." << std::endl;


        // Performance Result Table
        std::cout << "===============================================" << std::endl;
        std::cout << "               EIKONAL SOLVER" << std::endl;
        std::cout << "Input: " << input_fileName << std::endl;
        std::cout << "Number of partitions = " << num_parts << std::endl;
        std::cout << std::endl;
        std::cout << "Execution time = " <<
                  std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count() << std::endl;
        std::cout << "Output can be found in ../test/output_meshes" << std::endl <<
                  "with name: " << output_fileName << ".vtk" << std::endl;
        std::cout << "===============================================" << std::endl;
    }
    else
    {
        std::cout << "Wrong argument passed to the program\n";
    }

    return 0;
}