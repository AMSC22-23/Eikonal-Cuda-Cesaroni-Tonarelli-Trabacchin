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

template <int D>
class Mesh {
public:

    Mesh(const std::string& mesh_file_path, int nparts) : partitions_number(nparts){
        std::set<std::set<int>> sets = Mesh<D>::init_mesh(mesh_file_path, 4);
        std::vector<std::vector<int>> g;
        g.resize(getNumberVertices());
        for(auto &s : sets) {
            for(auto &j : s) {
                for(auto &i : s) {
                    if(i != j) {
                        g[j].push_back(i);

                    }
                }
            }
        }

        ngh.resize(getNumberVertices());
        ngh[0] = 0;
        int cont = 0;
        for(int i = 1; i < ngh.size(); i++) {
            ngh[i] = ngh[i-1] + g[i - 1].size();
            cont += g[i-1].size();
        }
        cont += g[ngh.size()-1].size();
        shapes.resize(cont);
        cont = 0;
        for(auto &x : g) {
            for(auto &y : x) {
                shapes[cont] = y;
                cont++;
            }
        }



        execute_metis();
        indices.resize(getNumberVertices());
        for(int i = 0; i < getNumberVertices(); i++){
            indices[i] = neighbors.size();
            std::vector<int> n = getNeighbors(i);
            neighbors.insert(neighbors.end(), n.begin(), n.end());
        }
    }

    void execute_metis() {
        print_file_metis();
        system(("../lib/METIS/build/programs/mpmetis metis_input.txt  -contig  -ncommon=3  " + std::to_string(partitions_number)).c_str());
        std::vector<int> parts = read_metis_output();
        reorderPartitions(parts);
    }

    std::string toString() {
        size_t cont = 0;
        int index = 0;
        std::string res = "";
        while(true) {
            res+= "vertex " + std::to_string(cont) + ": " ;
            for(size_t i = index; i < (cont < ngh.size()-1 ? ngh[cont+1] : shapes.size()); i += vertices_per_shape - 1){
                res += std::to_string(shapes[i]) + " " + std::to_string(shapes[i+1]) + ", ";
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

    std::vector<int> getNeighbors(size_t vertex) const {
        std::set<int> n;
        for(size_t i = ngh[vertex]; i < (vertex != ngh.size() -1 ? ngh[vertex + 1] : shapes.size()); i++){
            n.insert(shapes[i]);
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

    std::array<double, D> getCoordinates(int vertex) const{
        std::array<double,D> coord;
        for(int i = D * vertex; i < D * vertex + D; i++){
            coord[i - D * vertex] = geo[i];
        }
        return coord;
    }



    int getNearestVertex(std::array<double, D> coordinates) const {
        double min_distance = std::numeric_limits<double>::max();
        int min_vertex = 0;
        for(int i = 0; i < getNumberVertices(); i++){
            double distance = getDistance(coordinates, getCoordinates(i));
            if(distance < min_distance){
                min_distance = distance;
                min_vertex = i;
            }
        }
        return min_vertex;
    }

    const std::vector<int>& getVectorNeighbors() const{
        return neighbors;
    }

    const std::vector<int>& getVectorNeighborsIndices() const{
        return indices;
    }

    void print_file_metis(){
        std::ofstream output_file("metis_input.txt");
        //output_file << this->shapes.size()/D << std::endl;
        std::set<std::set<int>> element_set = this->retrieveShapes();
        output_file << element_set.size() << std::endl;
        for(auto &s : element_set) {
            for(auto &x : s) {
                output_file << x+1 << " ";
            }
            output_file << std::endl;
        }
        output_file.close();
        return;
        for(int i = 0; i < this->getNumberVertices(); i++) {
            int begin = ngh[i];
            int end = (i == this->getNumberVertices() - 1)?shapes.size():ngh[i+1];
            for(int j = begin; j < end; j++) {
                output_file << i << " ";
                for (int k = 0; k < D; k++) {
                    output_file << shapes[j + k] << " ";
                }
                output_file << std::endl;
            }

        }

    }

    std::vector<int> read_metis_output() {
        std::ifstream mesh_file (("metis_input.txt.npart." + std::to_string(partitions_number)).c_str() );
        std::vector<int> parts(getNumberVertices());
        for(int & part : parts) {
            mesh_file >> part;
        }
        mesh_file.close();
        return parts;
    }



    void getSolutionsVTK(const std::string& output_file_name, int* solutions){
        std::ofstream output_file(output_file_name);

        //header
        output_file << "# vtk DataFile Version 3.0\n" <<
                       "#This file was generated by the deal.II library on 2023/12/1 at 11:15:24\n";
        output_file << "ASCII\n" << "DATASET UNSTRUCTURED_GRID\n\n";

        // points
        output_file << "POINTS " << getNumberVertices() << " double\n";
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
        std::set<std::set<int>> s = retrieveShapes();
        int num_shapes = s.size();
        output_file << "CELLS        " << num_shapes << " " << num_shapes * 5 << std::endl;
        for(auto x: s){
            output_file << D+1 << "  ";
            for(auto y: x){
                output_file << "  " << y;
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
        output_file << "SCALARS solution double 1" << std::endl;
        output_file << "LOOKUP_TABLE default" << std::endl;
        for(int i = 0; i < getNumberVertices(); i++){
            output_file << solutions[i] << " ";
        }
        output_file << std::endl;

        output_file.flush();
        output_file.close();
    }

protected:

    double getDistance(std::array<double, D> c1, std::array<double, D> c2) const {
        double res = 0;
        for(int i = 0; i < D; i++){
            res += (c1[i] - c2[i]) * (c1[i] - c2[i]);
        }
        res = std::sqrt(res);
        return res;
    }


    void reorderPartitions(std::vector<int> partitions_vector) {
        partitions.resize(partitions_number);
        std::vector<int> map_vertices;
        std::vector<int> pos;
        pos.resize(partitions_vector.size());
        std::iota(pos.begin(), pos.end(),0);
        std::sort(pos.begin(), pos.end(), [&](std::size_t i, std::size_t j) { return partitions_vector[i] < partitions_vector[j];});
        size_t current_index = 0;
        size_t prec;
        std::vector<int> same;
        std::vector<double> reordered_geo;
        reordered_geo.resize(0);
        map_vertices.resize(geo.size() / D);
        std::vector<int> reordered_shapes(shapes.size());
        std::vector<int> reordered_ngh(ngh.size());
        int cont_shapes = 0;
        int cont_ngh = 0;
        int cont_partitions = 0;
        reordered_ngh[0] = 0;
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
                    partitions[cont_partitions] = current_index;
                    cont_partitions++;
                    for(int j : same){
                        //map_vertices[j] = (int)reordered_geo.size() / D;
                        for(int i = 0; i < D; i++) {
                            reordered_geo.push_back(geo[j*D+i]);
                        }
                        int begin = ngh[j];
                        int end = (j != ngh.size() - 1) ? ngh[j + 1] : shapes.size();
                        reordered_ngh[cont_ngh] = cont_shapes;
                        for(int i = begin; i < end; i++) {
                            reordered_shapes[cont_shapes] = map_vertices[shapes[i]];
                            cont_shapes++;
                        }

                        cont_ngh++;
                    }

                    break;
                }
            }
            same.clear();
        }

        geo = reordered_geo;
        shapes = reordered_shapes;
        ngh = reordered_ngh;


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
        std::vector<double> reduced_geo;
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
                mesh_file>>buffer;
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


    std::vector<double> geo;
    std::vector<int> shapes;
    std::vector<int> ngh;
    //std::vector<int> partitions;
    int partitions_number;


    std::vector<int> neighbors;
    std::vector<int> indices;
    int vertices_per_shape = 0;
public:
    std::vector<int> partitions;

private:
    std::set<std::set<int>> retrieveShapes(){

        std::set<std::set<int>> s;
        for(int i = 0; i < this->getNumberVertices(); i++) {
            int begin = ngh[i];
            int end = (i == this->getNumberVertices() - 1) ? shapes.size() : ngh[i + 1];
            for(int j = begin; j < end; j+=D){
                std::set<int> tmp;
                tmp.insert(i);
                for (int k = 0; k < D; k++) {
                    tmp.insert(shapes[j + k]);
                }
                s.insert(tmp);
            }
        }
        return s;
    }



};
#endif
