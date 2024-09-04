#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MESH_H
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_MESH_H
#include <array>
#include <vector>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cmath>
#include <tuple>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include "../src/CudaEikonalTraits.cuh"
#define IDXTYPEWIDTH 32
#define REALTYPEWIDTH 64
#include "../lib/METIS/include/metis.h"

// struct storing the index associated to the tetrahedron and
// its configuration (we have 4 possible configurations)
struct TetraConfig {
    int tetra_index;
    int tetra_config;
};

// The class encapsulates all the geometric information about the domain and has 2 constructors:
// - The first constructor reads the matrices associated with each tetrahedron from a file.
//   It takes as parameters the mesh file path, the number of partitions into which the subdomain is divided,
//   and the matrix file path.
// - The second constructor initializes all the matrices to the same value, with the matrix provided as a constructor parameter

template <int D, typename Float>
class Mesh {
public:

    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

    Mesh(const std::string& mesh_file_path, int nparts, const std::string& matrix_file_path) : partitions_number(nparts){
        // we read the mesh from file and store the tetrahedra in sets, each set element represents a tetrahedron
        std::set<std::set<int>> sets = Mesh<D, Float>::init_mesh(mesh_file_path, 4);
        
    
        tetra.resize(sets.size() * (D+1));
        int i = 0;
        for(auto &t : sets) {
            if(t.size() != D+1) continue;
            int j = 0;
            for(auto &v : t) {
                tetra[i+j] = v;
                j++;
            }
            i += D+1;
        }

        // we read the matrices from file and store them in a vector of matrices
        std::vector<Matrix> tempM = readMatrices(matrix_file_path);
        // perform the partition with Metis providing the vector of matrices
        execute_metis_api(tempM);

        std::vector<std::vector<int>> g;
        g.resize(getNumberVertices());
        for(i = 0; i < tetra.size(); i += D+1) {
            for(int j = 0; j < D + 1; j++) {
                g[tetra[j+i]].push_back(i/(D+1));
            }
        }


        ngh.resize(getNumberVertices());
        ngh[0] = 0;
        unsigned int cont = 0;
        for(i = 1; i < ngh.size(); i++) {
            ngh[i] = ngh[i-1] + g[i - 1].size();
            cont += g[i-1].size();
        }
        cont += g[ngh.size()-1].size();
        shapes.resize(cont);
        cont = 0;

        for(i = 0; i < g.size(); i++) {
            for(int j = 0; j < g[i].size(); j++) {
                shapes[cont].tetra_index = g[i][j];
                int tetra_to_search = g[i][j];
                for(int k = 0; k < 4; k++) {
                    if(tetra[4 * tetra_to_search + k] == i) {
                        shapes[cont].tetra_config = k + 1;
                    }
                }
                cont++;
            }
        }
    }

    Mesh(const std::string& mesh_file_path, int nparts, const Matrix& velocity) : partitions_number(nparts){
        // sets contains the tetrahedra. Each set element represents a tetrahedron
        std::set<std::set<int>> sets = Mesh<D, Float>::init_mesh(mesh_file_path, 4);
        // tetra is filled with the indices of the vertices that form the tetrahedron
        tetra.resize(sets.size() * (D+1));
        unsigned int i = 0;
        for(auto &t : sets) {
            int j = 0;
            for(auto &v : t) {
                tetra[i+j] = v;
                j++;
            }
            i += D+1;
        }

        std::vector<Matrix> tempM;
        // we have a matrix associated to each tetrahedron
        tempM.resize(tetra.size()/(D+1));



        for(i = 0; i < tetra.size() / (D + 1); i++){
            // We initialize each matrix using the one provided as a constructor parameter.
            tempM[i] = velocity;
        }
        // we perform the partitioning providing the vector of matrices to the method
        execute_metis_api(tempM);


        std::vector<std::vector<int>> g;
        g.resize(getNumberVertices());
        for(i = 0; i < tetra.size(); i += D+1) {
            for(int j = 0; j < D + 1; j++) {
                g[tetra[j+i]].push_back(i/(D+1));
            }
        }

        ngh.resize(getNumberVertices());
        ngh[0] = 0;
        unsigned int cont = 0;
        for(i = 1; i < ngh.size(); i++) {
            ngh[i] = ngh[i-1] + g[i - 1].size();
            cont += g[i-1].size();
        }
        cont += g[ngh.size()-1].size();
        shapes.resize(cont);

        cont = 0;


        for(i = 0; i < g.size(); i++) {
            for(int j = 0; j < g[i].size(); j++) {
                shapes[cont].tetra_index = g[i][j];
                int tetra_to_search = g[i][j];
                for(int k = 0; k < 4; k++) {
                    if(tetra[4 * tetra_to_search + k] == i) {
                        shapes[cont].tetra_config = k + 1;
                    }
                }
                cont++;
            }
        }

    }


    void execute_metis_api(const std::vector<Matrix>& tempM) {
        if(partitions_number > 1) {
            idx_t tetra_number = tetra.size()/(D+1);
            idx_t vertices_number = geo.size()/D;
            idx_t objval;
            std::vector<idx_t> elements_subdivision;
            elements_subdivision.resize(tetra_number + 1);
            for(int i = 0; i < tetra_number+1; i++) {
                elements_subdivision[i] = 4*i;
            }
            std::vector<idx_t> parts_vertices;
            parts_vertices.resize(vertices_number);
            std::vector<idx_t> parts_tetra;
            parts_tetra.resize(tetra_number);
            idx_t ncomm = 3;
            idx_t metis_result = METIS_PartMeshDual(&tetra_number, &vertices_number, elements_subdivision.data() , tetra.data(), 0, 0, &ncomm, &partitions_number, 0, 0, &objval, parts_tetra.data(), parts_vertices.data());
            if(metis_result != METIS_OK) {
                std::cout << metis_result << std::endl;
                exit(-1);
            }

            reorderPartitions(parts_vertices);
            reorderTetra(parts_tetra, tempM);
        } else {
            std::vector<int> parts(getNumberVertices(), 0);
            reorderPartitions(parts);
            parts = std::vector<int>(getNumberTetra(), 0);
            reorderTetra(parts, tempM);
        }
    }


    void execute_metis(const std::vector<Matrix>& tempM) {
        if(partitions_number > 1) {
            // we generate the input file for metis
            print_file_metis();
            // we execute the METIS command-line tool
            int ret_code = system(("../lib/METIS/build/programs/mpmetis metis_input.txt  -contig  -ncommon=3  " + std::to_string(partitions_number) + " > /dev/null").c_str());
            // we check the return code
            if(ret_code!=0) {
                exit(ret_code);
            }
            // we read the partitioning results for vertices from METIS
            std::vector<int> parts = read_metis_vertices_output();
            // we reorder the vertices according to the partitions they belong to
            // (geo will be reordered)
            reorderPartitions(parts);
            // we read the partitioning results for tetrahedra from METIS
            parts = read_metis_tetra_output();
            // we reorder the tetrahedra according to the partition they belong to
            // (tetra will be reordered)
            reorderTetra(parts, tempM);
        } else {
            // in this case the domain is not partitioned into subdomains
            std::vector<int> parts(getNumberVertices(), 0);
            reorderPartitions(parts);
            parts = std::vector<int>(getNumberTetra(), 0);
            reorderTetra(parts, tempM);
        }
    }

    std::string toString() {
        size_t cont = 0;
        int index = 0;
        std::string res;
        while(true) {
            res+= "vertex " + std::to_string(cont) + ": " ;
            for(size_t i = index; i < (cont < ngh.size()-1 ? ngh[cont+1] : shapes.size()); i += vertices_per_shape - 1){
                res += std::to_string(shapes[i].tetra_index) + " " + std::to_string(shapes[i+1].tetra_index) + ", ";
                index = i + vertices_per_shape - 1;
            }
            cont++;
            if(cont == ngh.size()){
                break;
            }
            res += "\n";
        }
        return res;
    }

    // method returns the total number of vertices
    int getNumberVertices() const {
        return geo.size() / D;
    };


    /*int getVerticesPerShape() const {
        return vertices_per_shape;
    }*/

    // method returns the total number of tetrahedra
    int getNumberTetra() const{
        return tetra.size() / (D+1);
    }

    /*std::vector<int> getNeighbors(size_t vertex) const {
        std::set<int> n;
        for(size_t i = ngh[vertex]; i < (vertex != ngh.size() -1 ? ngh[vertex + 1] : shapes.size()); i++){
            n.insert(shapes[i].tetra_index);
        }
        std::vector<int> res(n.begin(), n.end());
        return res;
    }*/

    /*std::vector<int> getShapes(size_t vertex) const {
        std::vector<int> shapes_v;
        for(size_t i = ngh[vertex]; i < (vertex != ngh.size() -1 ? ngh[vertex + 1] : shapes.size()); i++){
            shapes_v.emplace_back(shapes[i]);
        }
        return shapes_v;
    }*/

    // method that provided a vertex (index) returns its coordinates
    template<typename V>
    V getCoordinates(int vertex) const{
        V coord;
        for(int i = D * vertex; i < D * vertex + D; i++){
            coord[i - D * vertex] = geo[i];
        }
        return coord;
    }

    VectorExt getCoordinates_(int vertex) const{
        VectorExt coord;
        for(int i = D * vertex; i < D * vertex + D; i++){
            coord[i - D * vertex] = geo[i];
        }
        return coord;
    }


    std::array<Float, D> getCoordinates(int vertex) const {
        std::array<Float,D> coord;
        for(int i = D * vertex; i < D * vertex + D; i++){
            coord[i - D * vertex] = geo[i];
        }
        return coord;
    }

    int getPartitionsNumber() const {
        return partitions_number;
    }

    const std::vector<int>& getPartitionVertices() const{
        return partitions_vertices;
    }

    const std::vector<int>& getPartitionTetra() const{
        return partitions_tetrahedra;
    }

    const std::vector<Float>& getGeo() const {
        return geo;
    }

    const std::vector<TetraConfig>& getShapes() const {
        return shapes;
    }

    int getNearestVertex(std::array<Float, D> coordinates) const {
        Float min_distance = std::numeric_limits<Float>::max();
        int min_vertex = 0;
        for(int i = 0; i < getNumberVertices(); i++){
            Float distance = getDistance(coordinates, getCoordinates(i));
            if(distance < min_distance){
                min_distance = distance;
                min_vertex = i;
            }
        }
        return min_vertex;
    }



    void print_file_metis(){
        std::ofstream output_file("metis_input.txt");
        // number of tetrahedra
        output_file << tetra.size()/(D+1) << std::endl;

        for(int i = 0; i < tetra.size(); i += D+1) {
            for(int j = 0; j < D+1; j++) {
                // METIS uses 1-based indexing for enumerating
                output_file << tetra[i+j] + 1 << " ";
            }
            output_file << std::endl;
        }
        
        output_file.close();
    }

    std::vector<int> read_metis_vertices_output() {
        std::ifstream mesh_file (("metis_input.txt.npart." + std::to_string(partitions_number)).c_str() );
        std::vector<int> parts(getNumberVertices());
        for(int & part : parts) {
            mesh_file >> part;
        }
        mesh_file.close();
        return parts;
    }

    std::vector<int> read_metis_tetra_output() {
        std::ifstream mesh_file (("metis_input.txt.epart." + std::to_string(partitions_number)).c_str() );
        std::vector<int> parts(tetra.size()/4);
        for(int & part : parts) {
            mesh_file >> part;
        }
        mesh_file.close();
        return parts;
    }

    void getSolutionsVTK(const std::string& output_file_name, Float* solutions){
        std::ofstream output_file(output_file_name);
        if(!output_file.is_open()) {
            std::cout << "output file not opened " << output_file_name << std::endl;
        }
        //header
        output_file << "# vtk DataFile Version 3.0\n" <<
                    "#This file was generated by the deal.II library on 2023/12/1 at 11:15:24\n";
        output_file << "ASCII\n" << "DATASET UNSTRUCTURED_GRID\n\n";

        // points
        output_file << "POINTS " << getNumberVertices() << " Float\n";
        for(int i = 0; i < getNumberVertices(); i++){
            for(int j = 0; j < D; j++){
                output_file << std::setprecision(15) << geo[j + i * D] << " ";
            }
            output_file << "\n";
        }
        output_file << "\n";

        //  cells
    
        int num_shapes = tetra.size()/(D+1);
        output_file << "CELLS        " << num_shapes << " " << num_shapes * 5 << std::endl;

        for(int i = 0; i < tetra.size(); i += D+1) {
            output_file << D+1 << " ";
            for(int j = 0; j < D+1; j++) {
                output_file << "  " << tetra[i+j];
            }
            output_file << std::endl;
        }

        // cell_types
        int n = 10;
        std::cout << std::endl;
        output_file << "CELL_TYPES " << num_shapes << std::endl;
        for(int i = 0; i < num_shapes; i++){
            output_file << n << " ";
        }

        // look_up table
        output_file << std::endl;
        output_file << "POINT_DATA " << getNumberVertices() << std::endl;
        output_file << "SCALARS solution Float 1" << std::endl;
        output_file << "LOOKUP_TABLE default" << std::endl;
        for(int i = 0; i < getNumberVertices(); i++){
            output_file << solutions[i] << " ";
        }
        output_file << std::endl;

        output_file.flush();
        output_file.close();
    }

    const std::vector<int>& get_tetra() const{
        return tetra;
    }

    const std::vector<Float>& get_M() const{
        return M;
    }

    const std::vector<int>& get_ngh() const {
        return ngh;
    }

    const std::vector<int>& get_partitions() const {
        return partitions_vertices;
    }


protected:

// method that, provided the coordinates of 2 nodes, returns the distance
    Float getDistance(std::array<Float, D> c1, std::array<Float, D> c2) const {
        Float res = 0;
        for(int i = 0; i < D; i++){
            res += (c1[i] - c2[i]) * (c1[i] - c2[i]);
        }
        res = std::sqrt(res);
        return res;
    }

    // we reorder the tetrahedra according to the partition they belong to
    void reorderTetra(std::vector<int> partitions_vector, std::vector<Matrix> tempM){

        //partitions_tetrahedra.resize(partitions_number);
        std::vector<int> pos;
        pos.resize(partitions_vector.size());
        std::iota(pos.begin(), pos.end(),0);
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return partitions_vector[i] < partitions_vector[j];});
        std::vector<int> reordered_tetra;
        reordered_tetra.resize(tetra.size());
        //partitions_tetrahedra.push_back(0);
        // The matrix is symmetric so we can store only 6 Floats for each tetrahedron
        M.resize(6*pos.size());
        for(int i = 0; i < pos.size(); i++){
            /*if(i!=0 && partitions_vector[pos[i]]!= partitions_vector[pos[i-1]]){
                partitions_tetrahedra.push_back(4*i);
            }*/
            for(int j=0; j<D+1; j++){
                reordered_tetra[4*i+j] = tetra[4*pos[i]+j];
            }
            VectorExt x1 = getCoordinates<VectorExt>(tetra[4*pos[i]]);
            VectorExt x2 = getCoordinates<VectorExt>(tetra[4*pos[i]+1]);
            VectorExt x3 = getCoordinates<VectorExt>(tetra[4*pos[i]+2]);
            VectorExt x4 = getCoordinates<VectorExt>(tetra[4*pos[i]+3]);
            VectorExt e12 = x2+(-x1);
            VectorExt e13 = x3+(-x1);
            VectorExt e23 = x3+(-x2);
            VectorExt e14 = x4+(-x1);
            VectorExt e24 = x4+(-x2);
            VectorExt e34 = x4+(-x3);
            M[i * 6]     = e12.transpose() * (tempM[pos[i]] * e12);
            M[i * 6 + 1] = e13.transpose() * (tempM[pos[i]] * e13);
            M[i * 6 + 2] = e23.transpose() * (tempM[pos[i]] * e23);
            M[i * 6 + 3] = e14.transpose() * (tempM[pos[i]] * e14);
            M[i * 6 + 4] = e24.transpose() * (tempM[pos[i]] * e24);
            M[i * 6 + 5] = e34.transpose() * (tempM[pos[i]] * e34);
        }
        tetra = reordered_tetra;
    }

// we provide to the method a vector of integers where each entry represents the partition assignment of a vertex.
// For example, if partitions_vector[i] = 2, then the i-th vertex belongs to partition 2.
    void reorderPartitions(std::vector<int> partitions_vector) {
        partitions_vertices.resize(partitions_number);
        std::vector<int> map_vertices;
        std::vector<int> pos;
        pos.resize(partitions_vector.size());
        std::iota(pos.begin(), pos.end(),0);
        // we sort the vertices based on the partition assignments,
        // grouping vertices in their partition assignment
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return partitions_vector[i] < partitions_vector[j];});
        size_t current_index = 0;
        size_t prec;
        std::vector<int> same;
        std::vector<Float> reordered_geo;
        reordered_geo.resize(0);
        map_vertices.resize(geo.size() / D);
        int cont_partitions = 0;
        // we map each original vertex index to its new index after sorting
        for(int i = 0; i < pos.size(); i++) {
            map_vertices[pos[i]] = i;
        }
        while(current_index < pos.size()){
            prec = current_index;
            current_index++;
            // we store in same vertices belonging to the same partition
            same.push_back(pos[prec]);
            while(true){
                if( current_index < pos.size() && partitions_vector[pos[prec]] == partitions_vector[pos[current_index]]){
                    same.push_back(pos[current_index]);
                    current_index++;
                } else{
                    partitions_vertices[cont_partitions] = current_index;
                    cont_partitions++;
                    for(int j : same){
                        // the reordered coordinates are stored in reordered_geo
                        for(int i = 0; i < D; i++) {
                            reordered_geo.push_back(geo[j*D+i]);
                        }
                    }

                    break;
                }
            }
            same.clear();
        }

        for(auto &v : tetra) {
            v = map_vertices[v];
        }

        geo = reordered_geo;

    }

    // method to remove duplicated vertices from geo, it returns a vector storing the mapping
    std::vector<int> removeDuplicateVertices(){
        // Vector that will store the mapping of old vertex indices to new indices in the reduced set.
        std::vector<int> map_vertices;
        std::vector<int> pos;
        // size is equal to the number of vertices
        pos.resize(geo.size()/D);
        std::iota(pos.begin(), pos.end(),0);
        // The indices are sorted based on the verticesCompare function
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return verticesCompare(i,j) == 1; });

        size_t current_index = 0;
        size_t prec;
        // same will store the indices of duplicated vertices
        std::vector<int> same;
        // reduced_geo stores the indices of all the vertices without duplicates
        std::vector<Float> reduced_geo;
        reduced_geo.resize(0);
        // map_vertices has the size of the original number of vertices
        map_vertices.resize(geo.size() / D);

        while(current_index < pos.size()){
            prec = current_index;
            current_index++;
            same.push_back(pos[prec]);
            while(true){
                // verticesCompare returns 0 if two vertices are equal
                if( current_index < pos.size() && verticesCompare(pos[prec], pos[current_index]) == 0){
                    same.push_back(pos[current_index]);
                    current_index++;
                } else{
                    for(int j : same){
                        map_vertices[j] = (int)reduced_geo.size() / D;
                    }
                    for(int i=0; i<D; i++){
                        reduced_geo.push_back(geo[same[0]*D+i]);
                    }
                    break;
                }
            }
            same.clear();
        }
        geo = reduced_geo;
        return map_vertices;
    }

    // This function is used to determine the order of vertices
    int verticesCompare(int i, int j) const {
        for(int k = 0; k < D; k++){
            if(geo[D * i + k] < geo[D * j + k]){
                return 1;
            } else if (geo[D * i + k] > geo[D * j + k]){
                return -1;
            }
        }
        return 0;
    }


    // This method is responsible for reading a set of matrices from a file
    // and storing them in a vector of Matrix objects
    std::vector<Matrix> readMatrices(const std::string& matrix_file_path){
        std::ifstream matrix_file (matrix_file_path);
        std::vector<Matrix> matrices;
        if(matrix_file.is_open()){
            // a matrix for each tetrahedron
            matrices.resize(tetra.size()/4);
            std::string buffer;
            std::array<Float,6> n;
            for(int i = 0; i <tetra.size()/4; i++){
                matrix_file>>buffer;
                sscanf(buffer.c_str(),"%f %f %f %f %f %f", &n[0], &n[1], &n[2], &n[3], &n[4], &n[5]);
                matrices[i] << n[0], n[1], n[2],
                        n[1], n[3], n[4],
                        n[2], n[4], n[5];
            }
            matrix_file.close();

        }
        return matrices;
    }

    // method to read the mesh from file
    std::set<std::set<int>> init_mesh(const std::string& mesh_file_path, int vertices_per_shape_) {
        std::set<std::set<int>> sets;
        vertices_per_shape = vertices_per_shape_;
        std::ifstream mesh_file (mesh_file_path);
        if(mesh_file.is_open()) {
            // we ignore the information that isn't relevant
            std::string buffer;

            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);

            mesh_file>>buffer;
            int vertices_number;
            mesh_file>>vertices_number;
            mesh_file>>buffer;
            // we populate geo with the coordinates of each vertex
            geo.resize(vertices_number*D);
            int ignore;
            for(int i = 0; i < vertices_number; i++){
                for(int j = 0; j < 3; j++){
                    if(j < D)
                        mesh_file >> geo[D * i + j];
                    else mesh_file >> ignore;
                }

            }
            // We remove vertices having the same coordinates and store
            // the mapping from original vertex indices to their corresponding indices in the reduced set of vertices
            std::vector<int> map_vertices = removeDuplicateVertices();

            // now geo contains coordinates of vertices without duplicates
            vertices_number = geo.size()/D;
            mesh_file>>buffer;
            int triangle_number;
            mesh_file>>triangle_number;
            mesh_file>>buffer;
            ngh.resize(vertices_number);
            int num;
            for(int i=0; i<triangle_number; i++){
                int element_type;
                mesh_file>>element_type;
                std::set<int> tmp;
                for(int j=0; j<element_type; j++){
                    mesh_file>>num;
                    tmp.insert(map_vertices[num]);
                }
                sets.insert(tmp);

            }
            mesh_file.close();

        } else {
            std::cout << "Couldn't open mesh file." << std::endl;
        }
        // sets will contain all the tetrahedra
        return sets;
    }


    // vector storing the coordinates of the vertices. It stores the x,y and z values for the first vertex,
    // followed by the x,y and z values and so on.
    std::vector<Float> geo;

    // Vector that, for each vertex, stores the shapes associated to it
    std::vector<TetraConfig> shapes;

    // Vector storing tetrahedra : indices of the vertices in groups of 4 for each tetrahedron
    std::vector<int> tetra;

    // vector defining the boundaries in shapes
    std::vector<int> ngh;

    // number of subdomains
    int partitions_number;

    // vector of matrices, each matrix is associated to a tetrahedron
    std::vector<Float> M;


    /*std::vector<int> neighbors;
    std::vector<int> indices; */

    int vertices_per_shape = 4;

    // vector defining the boundaries of each partition in geo (i.e. defines the set of vertices in each partition)
    std::vector<int> partitions_vertices;

    // vector defining the boundaries of each partition in tetra (i.e. defines the set of tetrahedra in each partition)
    std::vector<int> partitions_tetrahedra;


};
#endif
