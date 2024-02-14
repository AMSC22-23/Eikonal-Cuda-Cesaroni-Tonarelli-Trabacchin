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

template <int D>
class Mesh {
public:

    Mesh(const std::string& mesh_file_path) {
        std::vector<std::set<int>> sets = Mesh<D>::init_mesh(mesh_file_path, 4);
        int cont = 0;
        for(int i=0; i < getNumberVertices(); i++) {
            std::set<std::set<int>> ngh_set;
            Mesh<D>::ngh[i] = cont;
            for(const auto& x: sets[i]){
                std::vector<int> tmp (std::min(sets[i].size(), sets[x].size() ), 0);
                std::vector<int>::iterator end;
                end = std::set_intersection(sets[i].begin(), sets[i].end(),
                                            sets[x].begin(), sets[x].end(),
                                            tmp.begin());
                std::vector<int>::iterator it;
                std::set<int> current_set;

                for(it=tmp.begin(); it!=end; it++){
                    current_set.insert(*it);
                }

                for(it = tmp.begin(); it != end; it++){
                    std::vector<int>::iterator end2;
                    std::vector<int> tmp2(std::min(current_set.size(), sets[*it].size()), 0);
                    std::vector<int>::iterator it2;
                    end2 = std::set_intersection(current_set.begin(), current_set.end(),
                                                 sets[*it].begin(), sets[*it].end(),
                                                 tmp2.begin());
                    for(it2 = tmp2.begin(); it2 != end2; it2++){
                        //shapes.push_back(x);
                        //shapes.push_back(*it);
                        //shapes.push_back(*it2);

                        //cont += vertices_per_shape - 1;
                        std::set<int> group;
                        group.insert(x);
                        group.insert(*it);
                        group.insert(*it2);
                        auto[iterator, inserted] = ngh_set.insert(group);
                        if(inserted) {
                            cont += vertices_per_shape - 1;
                            shapes.push_back(x);
                            shapes.push_back(*it);
                            shapes.push_back(*it2);
                        }
                    }

                }
            }
        }
        indices.resize(getNumberVertices());
        for(int i = 0; i < getNumberVertices(); i++){
            indices[i] = neighbors.size();
            std::vector<int> n = getNeighbors(i);
            neighbors.insert(neighbors.end(), n.begin(), n.end());

        }
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

    int getMapVertex(int vertex) const {
        return map_vertices[vertex];
    }

    int getOriginalNumberOfVertices() const{
        return map_vertices.size();
    }

    std::string getFilenameInputMesh() const {
        return filename_input_mesh;
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

    void print_file_metis() const{
        std::ofstream output_file("metis_input.txt");
        output_file << this->shapes.size()/D << std::endl;
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

    void getSolutionsVTK(const std::string& output_file_name, int* solutions){
        std::ofstream output_file(output_file_name);

        std::string input = getFilenameInputMesh();
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

        output_file << "POINT_DATA " << getOriginalNumberOfVertices() << std::endl;
        output_file << "SCALARS solution double 1" << std::endl;
        output_file << "LOOKUP_TABLE default" << std::endl;
        for(int i = 0; i < getOriginalNumberOfVertices(); i++){
            double solution = solutions[getMapVertex(i)];
            output_file << solution << " ";
        }
        output_file << std::endl;
        output_file.flush();

        output_file.close();
    }

    void getSolutionsVTK2(const std::string& output_file_name, int* solutions){
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
        std::set<std::set<int>> s = nonloso();
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

    void removeDuplicateVertices(){
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

    std::vector<std::set<int>> init_mesh(const std::string& mesh_file_path, int vertices_per_shape_) {
        filename_input_mesh = mesh_file_path;
        vertices_per_shape = vertices_per_shape_;
        std::ifstream mesh_file (mesh_file_path);
        std::vector<std::set<int>> sets;
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
            removeDuplicateVertices();


            vertices_number = geo.size()/D;
            mesh_file>>buffer;
            int triangle_number;
            mesh_file>>triangle_number;
            mesh_file>>buffer;
            sets.resize(vertices_number);
            ngh.resize(vertices_number);
            for(int i=0; i<triangle_number; i++){
                mesh_file>>buffer;
                std::vector<int> tmp;
                tmp.resize(vertices_per_shape);
                for(int j=0; j<vertices_per_shape; j++){
                    mesh_file>>tmp[j];
                    tmp[j] = map_vertices[tmp[j]];
                }
                for(int j=0; j < vertices_per_shape; j++){
                    for(int k=0; k < vertices_per_shape; k++){
                        if(j!=k){
                            sets[tmp[j]].insert(tmp[k]);
                        }
                    }
                }

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
    std::vector<int> neighbors;
    std::vector<int> indices;
    int vertices_per_shape = 0;
    std::vector<int> map_vertices;
    std::string filename_input_mesh;

private:
    std::set<std::set<int>> nonloso(){
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


    std::tuple<int, std::string> getStringMeshShapes(){
        bool flag = false;
        std::set<std::set<int>> s;
        int num_shapes = 0;
        std::string res;

        for(int i = 0; i < this->getNumberVertices(); i++) {
            int begin = ngh[i];
            int end = (i == this->getNumberVertices() - 1) ? shapes.size(): ngh[i+1];

            for(int j = begin; j < end; j+=D) {
                flag = false;
                std::set<int> tmp;
                tmp.insert(i);
                for (int k = 0; k < D; k++) {
                    if(shapes[j + k] < i) {
                        flag = true;
                    }
                    tmp.insert(shapes[j + k]);
                }

                if(flag){
                    std::cout << "flag true" << std::endl;
                    if(s.find(tmp) == s.end()){
                        std::cout << "NON CONTENUTO!!!!\n";
                        flag = false;
                    }
                }
                else
                {
                    if(s.find(tmp) != s.end()) {
                        std::cout << "GIA' CONTENUTO!\n";
                        flag = true;
                    }
                }


                if (!flag) {
                    num_shapes++;
                    res.append(std::to_string(D + 1) + "    " + std::to_string(i));
                    for (int k = 0; k < D; k++) {
                        res.append("  " + std::to_string(shapes[j + k]));
                    }
                    s.insert(tmp);
                    res.append("\n");
                }
            }
        }
        return {num_shapes, res};
    }
};
#endif
