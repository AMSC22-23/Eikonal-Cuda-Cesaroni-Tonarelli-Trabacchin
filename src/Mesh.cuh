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

struct TetraConfig {
    int tetra_index;
    int tetra_config;
};


template <int D, typename Float>
class Mesh {
public:

    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

    Mesh(const std::string& mesh_file_path, int nparts, const std::string& matrix_file_path) : partitions_number(nparts){
        std::set<std::set<int>> sets = Mesh<D, Float>::init_mesh(mesh_file_path, 4);
        std::cout << "init_mesh completed" << std::endl;
        tetra.resize(sets.size() * (D+1));
        int i = 0;
        for(auto &t : sets) {
            int j = 0;
            for(auto &v : t) {
                tetra[i+j] = v;
                j++;
            }
            i += D+1;
        }

        std::vector<Matrix> tempM = readMatrices(matrix_file_path);
        execute_metis(tempM);

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
        std::cout << "actual shapes size " << cont << std::endl;
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

        std::cout << "after shapes size " << cont << std::endl;


        /*
        for(auto &x : g) {
            for(auto &y : x) {
                shapes[cont] = y;
                cont++;
            }
        }
         */

        /*indices.resize(getNumberVertices());
        for(int i = 0; i < getNumberVertices(); i++){
            indices[i] = neighbors.size();
            std::vector<int> n = getNeighbors(i);
            neighbors.insert(neighbors.end(), n.begin(), n.end());
        }*/
    }

    Mesh(const std::string& mesh_file_path, int nparts, Matrix velocity) : partitions_number(nparts){
        std::set<std::set<int>> sets = Mesh<D, Float>::init_mesh(mesh_file_path, 4);
        std::cout << "init_mesh completed" << std::endl;
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
        tempM.resize(tetra.size()/(D+1));
        for(i = 0; i < tetra.size() / (D + 1); i++){
            tempM[i] = velocity;
        }
        std::cout<<"tetra" << std::endl;
        execute_metis(tempM);

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
        std::cout << "actual shapes size " << cont << std::endl;

        cont = 0;

        for(i = 0; i < g.size(); i++) {
            for(int j = 0; j < g[i].size(); j++) {
                shapes[cont].tetra_index = g[i][j];
                int tetra_to_search = g[i][j];
                bool sanity_check = false;
                for(int k = 0; k < 4; k++) {
                    if(tetra[4 * tetra_to_search + k] == i) {
                        shapes[cont].tetra_config = k + 1;
                        sanity_check = true;
                    }
                }
                if(!sanity_check) {
                    std::cout << "sanity error" << std::endl;
                }
                cont++;
            }
        }

        std::cout << "after shapes size " << cont << std::endl;


        for(int i = 0; i < shapes.size(); i++) {
            if(shapes[i].tetra_config<1||shapes[i].tetra_config>4) {
                std::cout << "error with shapes" << std::endl;
            }
        }



        /*
        for(auto &x : g) {
            for(auto &y : x) {
                shapes[cont] = y;
                cont++;
            }
        }
         */

        /*indices.resize(getNumberVertices());
        for(int i = 0; i < getNumberVertices(); i++){
            indices[i] = neighbors.size();
            std::vector<int> n = getNeighbors(i);
            neighbors.insert(neighbors.end(), n.begin(), n.end());
        }*/
    }


    void execute_metis(std::vector<Matrix> tempM) {
        print_file_metis();
        int ret_code = system(("../lib/METIS/build/programs/mpmetis metis_input.txt  -contig  -ncommon=3  " + std::to_string(partitions_number)).c_str());
        if(ret_code!=0) {
            exit(ret_code);
        }
        std::vector<int> parts = read_metis_vertices_output();
        reorderPartitions(parts);
        parts = read_metis_tetra_output();
        reorderTetra(parts, tempM);

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

    int getNumberVertices() const {
        return geo.size() / D;
    };

    int getVerticesPerShape() const {
        return vertices_per_shape;
    }

    int getNumberTetra() const{
        return tetra.size() / 4;
    }

    std::vector<int> getNeighbors(size_t vertex) const {
        std::set<int> n;
        for(size_t i = ngh[vertex]; i < (vertex != ngh.size() -1 ? ngh[vertex + 1] : shapes.size()); i++){
            n.insert(shapes[i].tetra_index);
        }
        std::vector<int> res(n.begin(), n.end());
        return res;
    }

    std::vector<int> getShapes(size_t vertex) const {
        std::vector<int> shapes_v;
        for(size_t i = ngh[vertex]; i < (vertex != ngh.size() -1 ? ngh[vertex + 1] : shapes.size()); i++){
            shapes_v.emplace_back(shapes[i]);
        }
        return shapes_v;
    }

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

    std::vector<int>& getPartitionVertices() {
        return partitions_vertices;
    }

    std::vector<int>& getPartitionTetra() {
        return partitions_tetrahedra;
    }

    std::vector<Float>& getGeo() {
        return geo;
    }

    std::vector<TetraConfig>& getShapes() {
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

    /*const std::vector<int>& getVectorNeighbors() const{
        return neighbors;
    }

    const std::vector<int>& getVectorNeighborsIndices() const{
        return indices;
    }*/

    void print_file_metis(){
        std::ofstream output_file("metis_input.txt");

        output_file << tetra.size()/(D+1) << std::endl;

        for(int i = 0; i < tetra.size(); i += D+1) {
            for(int j = 0; j < D+1; j++) {
                output_file << tetra[i+j] + 1 << " ";
            }
            output_file << std::endl;
        }
        /*
        for(auto &s : element_set) {
            for(auto &x : s) {
                output_file << x+1 << " ";
            }
            output_file << std::endl;
        }
         */
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
        } else {
            std::cout << "output file path: " << output_file_name << std::endl;
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
        /*auto [num_shapes, str] = getStringMeshShapes();
        output_file << "CELLS        " << num_shapes << " " << num_shapes * 5 << std::endl;
        output_file << str << std::endl;*/
        int num_shapes = tetra.size()/(D+1);
        output_file << "CELLS        " << num_shapes << " " << num_shapes * 5 << std::endl;

        for(int i = 0; i < tetra.size(); i += D+1) {
            output_file << D+1 << " ";
            for(int j = 0; j < D+1; j++) {
                output_file << "  " << tetra[i+j];
            }
            output_file << std::endl;
        }



        /*for(auto x: s){
            output_file << D+1 << "  ";
            for(auto y: x){
                output_file << "  " << y;
            }
            output_file << std::endl;
        }
         */

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
        std::cout << "writing finished" << std::endl;
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


    Float getDistance(std::array<Float, D> c1, std::array<Float, D> c2) const {
        Float res = 0;
        for(int i = 0; i < D; i++){
            res += (c1[i] - c2[i]) * (c1[i] - c2[i]);
        }
        res = std::sqrt(res);
        return res;
    }

    void reorderTetra(std::vector<int> partitions_vector, std::vector<Matrix> tempM){

        partitions_tetrahedra.resize(partitions_number);
        std::vector<int> pos;
        pos.resize(partitions_vector.size());
        std::iota(pos.begin(), pos.end(),0);
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return partitions_vector[i] < partitions_vector[j];});
        std::vector<Float> reordered_tetra;
        reordered_tetra.resize(tetra.size());
        partitions_tetrahedra.push_back(0);
        M.resize(6*pos.size());
        for(int i = 0; i < pos.size(); i++){
            if(i!=0 && partitions_vector[pos[i]]!= partitions_vector[pos[i-1]]){
                partitions_tetrahedra.push_back(4*i);
            }
            for(int j=0; j<D+1; j++){
                reordered_tetra[4*i+j] = tetra[4*pos[i]+j];
            }
            VectorExt x1 = getCoordinates<VectorExt>(tetra[4*i]);
            VectorExt x2 = getCoordinates<VectorExt>(tetra[4*i+1]);
            VectorExt x3 = getCoordinates<VectorExt>(tetra[4*i+2]);
            VectorExt x4 = getCoordinates<VectorExt>(tetra[4*i+3]);
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
    }

    void reorderPartitions(std::vector<int> partitions_vector) {
        partitions_vertices.resize(partitions_number);
        std::vector<int> map_vertices;
        std::vector<int> pos;
        pos.resize(partitions_vector.size());
        std::iota(pos.begin(), pos.end(),0);
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return partitions_vector[i] < partitions_vector[j];});
        size_t current_index = 0;
        size_t prec;
        std::vector<int> same;
        std::vector<Float> reordered_geo;
        reordered_geo.resize(0);
        map_vertices.resize(geo.size() / D);
        //std::vector<int> reordered_shapes(shapes.size());
        //std::vector<int> reordered_ngh(ngh.size());
        int cont_partitions = 0;
        //reordered_ngh[0] = 0;
        for(int i = 0; i < pos.size(); i++) {
            map_vertices[pos[i]] = i;
        }
        while(current_index < pos.size()){
            prec = current_index;
            current_index++;
            same.push_back(pos[prec]);
            while(true){
                if( current_index < pos.size() && partitions_vector[pos[prec]] == partitions_vector[pos[current_index]]){
                    same.push_back(pos[current_index]);
                    current_index++;
                } else{
                    partitions_vertices[cont_partitions] = current_index;
                    cont_partitions++;
                    for(int j : same){
                        //map_vertices[j] = (int)reordered_geo.size() / D;
                        for(int i = 0; i < D; i++) {
                            reordered_geo.push_back(geo[j*D+i]);
                        }
                        /*
                        int begin = ngh[j];
                        int end = (j != ngh.size() - 1) ? ngh[j + 1] : shapes.size();
                        reordered_ngh[cont_ngh] = cont_shapes;
                        for(int i = begin; i < end; i++) {
                            reordered_shapes[cont_shapes] = map_vertices[shapes[i]];
                            cont_shapes++;
                        }

                        cont_ngh++;
                        */
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
        //shapes = reordered_shapes;
        //ngh = reordered_ngh;


    }

    std::vector<int> removeDuplicateVertices(){
        std::vector<int> map_vertices;
        std::vector<int> pos;
        pos.resize(geo.size()/D);
        std::iota(pos.begin(), pos.end(),0);
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return verticesCompare(i,j) == 1; });

        size_t current_index = 0;
        size_t prec;
        std::vector<int> same;
        std::vector<Float> reduced_geo;
        reduced_geo.resize(0);
        map_vertices.resize(geo.size() / D);

        while(current_index < pos.size()){
            prec = current_index;
            current_index++;
            same.push_back(pos[prec]);
            while(true){
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

    std::vector<Matrix> readMatrices(const std::string& matrix_file_path){
        std::ifstream matrix_file (matrix_file_path);
        std::vector<Matrix> matrices;
        if(matrix_file.is_open()){
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

    std::set<std::set<int>> init_mesh(const std::string& mesh_file_path, int vertices_per_shape_) {
        std::set<std::set<int>> sets;
        vertices_per_shape = vertices_per_shape_;
        std::ifstream mesh_file (mesh_file_path);
        if(mesh_file.is_open()) {
            std::string buffer;

            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);
            std::getline(mesh_file, buffer);

            mesh_file>>buffer;
            int vertices_number;
            mesh_file>>vertices_number;
            mesh_file>>buffer;
            geo.resize(vertices_number*D);
            int ignore;
            for(int i = 0; i < vertices_number; i++){
                for(int j = 0; j < 3; j++){
                    if(j < D)
                        mesh_file >> geo[D * i + j];
                    else mesh_file >> ignore;
                }

            }
            std::vector<int> map_vertices = removeDuplicateVertices();


            vertices_number = geo.size()/D;
            mesh_file>>buffer;
            int triangle_number;
            mesh_file>>triangle_number;
            mesh_file>>buffer;
            ngh.resize(vertices_number);
            int num;
            for(int i=0; i<triangle_number; i++){
                int shape_type;
                mesh_file>>shape_type;
                if(shape_type != 4) {
                    continue;
                }
                std::set<int> tmp;
                for(int j=0; j<vertices_per_shape; j++){
                    mesh_file>>num;
                    tmp.insert(map_vertices[num]);
                }
                sets.insert(tmp);

            }
            mesh_file.close();

        } else {
            std::cout << "Couldn't open mesh file." << std::endl;
        }
        return sets;
    }


    std::vector<Float> geo; // Coordinates of the vertices
    std::vector<TetraConfig> shapes; // For each vertex, the shapes associated to it (contains only the other three vertices in the shape)
    std::vector<int> tetra; // Tetrahedra
    std::vector<int> ngh; // Defines the boundaries in shapes
    int partitions_number; // Number of partitions
    std::vector<Float> M; // vector of matrices, each matrix associated with a tetrahedron


    /*std::vector<int> neighbors;
    std::vector<int> indices; */
    int vertices_per_shape = 4;

    std::vector<int> partitions_vertices; // Defines the boundaries of each partition in geo (i.e. defines the set of vertices in each partition)
    std::vector<int> partitions_tetrahedra; // Defines the boundaries of each partition in tetra (i.e. defines the set of tetrahedra in each partition)


};
#endif
