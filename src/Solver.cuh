#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH

#include "Mesh.cuh"
#include "Kernels.cu"
#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <functional>
#include <thrust/async/scan.h>




constexpr int NUM_THREADS = 1024;
constexpr int SIZE_WARP = 32;


template <int N>
struct map_N_to_0 : std::unary_function<int, int> {
    __host__ __device__ int operator()(int i) {
        return (i==N) ? 0 : i;
    }
};


template <int D,typename Float>
class Solver {
public:
    Solver(Mesh<D, Float>* mesh) : mesh(mesh){
        gpuDataTransfer();
    }


    static void cudaCheck(std::string mess) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cout << "cuda err = " << err << std::endl;
            std::cout << "mess = "<< mess << std::endl;

            exit(1);
        }
    }

    void gpuDataTransfer(){
        //printf("GPU data transfer started\n");
        // Allocate memory in device
        cudaMalloc(&active_domains_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMemset(active_domains_dev,0,sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&partitions_vertices_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&partitions_tetra_dev, sizeof(int) * mesh->getPartitionsNumber());
        cudaMalloc(&geo_dev, sizeof(Float) * mesh->getNumberVertices() * D);
        cudaMalloc(&tetra_dev, sizeof(int) * mesh->get_tetra().size());
        cudaMalloc(&shapes_dev, sizeof(TetraConfig) * mesh->getShapes().size());
        cudaMalloc(&ngh_dev, sizeof(int) * mesh->get_ngh().size());
        cudaMalloc(&M_dev, sizeof(Float) * mesh->get_M().size());
        cudaMalloc(&solutions_dev, sizeof(Float) * mesh->getNumberVertices());
        // Copy data from host to device
        cudaMemcpy(partitions_vertices_dev, mesh->getPartitionVertices().data(), sizeof(int) * mesh->getPartitionsNumber(), cudaMemcpyHostToDevice);
        cudaMemcpy(partitions_tetra_dev, mesh->getPartitionTetra().data(), sizeof(int) * mesh->getPartitionsNumber(), cudaMemcpyHostToDevice);
        cudaMemcpy(geo_dev, mesh->getGeo().data(), sizeof(Float) * mesh->getNumberVertices() * D, cudaMemcpyHostToDevice);
        cudaMemcpy(tetra_dev, mesh->get_tetra().data(), sizeof(int) * mesh->get_tetra().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(shapes_dev, mesh->getShapes().data(), sizeof(TetraConfig) * mesh->getShapes().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ngh_dev, mesh->get_ngh().data(), sizeof(int) * mesh->get_ngh().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(M_dev, mesh->get_M().data(), sizeof(Float) * mesh->get_M().size(), cudaMemcpyHostToDevice);
        //printf("GPU data transfer completed\n");
    }

    //note: PartitionsVertices is an array such that (assuming vertices are clustered according to their subdomain, i.e. )
    //1)length(PartitionsVertices) == number of subdomains in the mesh
    //2)PartitionsVertices[i] == k <==> vertices from PartitionsVertices[i-1](inclusive) to PartitionsVertices[i](exclusive) belong to subdomains i. If i==0, then PartitionsVertices[-1] is assumed to be equals to 0, Moreover PartitionsVertices[i], with 0 <= i < mesh->getPartitionsNumber() - 1, is the index of the first node belonging to sudomain i+1
    void solve(std::vector<int> source_nodes, Float tol, Float infinity_value, const std::string& output_file_name){

        using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
        using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
        using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

        //std::cout << "domain 1 begin = " << mesh->getPartitionVertices()[0] << " end =" << mesh->getPartitionVertices()[1] << std::endl;


        std::vector<cudaStream_t> streams;
        streams.resize(mesh->getPartitionsNumber());
        int* active_domains;
        active_domains = (int*)malloc(sizeof(int) * mesh->getPartitionsNumber());
        std::vector<thrust::host_vector<int>> active_lists(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> sAddrs(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> cLists(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> active_lists_dev(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> nbhNrs(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<TetraConfig>> elemLists(mesh->getPartitionsNumber());
        //std::vector<thrust::device_vector<int>> s(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> predicate(mesh->getPartitionsNumber());
        thrust::device_vector<size_t> elemListSizes(mesh->getPartitionsNumber());


        for(int i = 0; i < mesh->getPartitionsNumber(); i++) {
            size_t vec_size = getDomainSize(i);
            sAddrs[i] = thrust::device_vector<int>(vec_size);//shorter
            cLists[i] = thrust::device_vector<int>(vec_size);//shorter
            nbhNrs[i] = thrust::device_vector<int>(vec_size);//shorter
            active_lists_dev[i] = thrust::device_vector<int>(vec_size);//shorter
            predicate[i] = thrust::device_vector<int>(mesh->getNumberVertices(), 0);//long
            elemLists[i] = thrust::device_vector<TetraConfig>(getTetraNumberinDomain(i));//shorter
            //s[i] = thrust::device_vector<int>(mesh->getNumberVertices());
            active_lists[i] = thrust::host_vector<int>(vec_size, 0);
        }

        setup(source_nodes, infinity_value, active_lists, active_lists_dev);

        // create streams, one for each domain
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamCreate(&streams[i]);
        }

        thrust::plus<int> binary_op;
        map_N_to_0<2> unary_op;

        int cont = 0;
        int inner_cont = 0;
        bool not_converged = true;
        Float* solutions = (Float*)malloc(sizeof(Float) * mesh->getNumberVertices());

        while(not_converged) {
            cont++;
            not_converged = false;
            #pragma omp parallel for num_threads(mesh->getPartitionsNumber()) nowait
            //std::cout << "-----------------------------" << std::endl;
            for(int domain = 0; domain < mesh->getPartitionsNumber(); domain++) {
                size_t domain_size = getDomainSize(domain);
                size_t begin_domain = getBeginDomain(domain);
                size_t end_domain = begin_domain + domain_size;
                int numBlocks = domain_size / NUM_THREADS + 1;
                int active_nodes = thrust::count(thrust::cuda::par.on(streams[domain]), active_lists_dev[domain].begin(), active_lists_dev[domain].end(), 1);
                
                if(active_nodes != 0) {
                    not_converged = true;
                }
                while(active_nodes > 0) {
                    inner_cont++;
                    thrust::transform_exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), active_lists_dev[domain].begin(), active_lists_dev[domain].end(), sAddrs[domain].begin(), unary_op, 0, binary_op);  //sAddrs may be shorter
                    compact<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(active_lists_dev[domain].data()), domain_size ,thrust::raw_pointer_cast(cLists[domain].data()), begin_domain);
                    count_Nbhs<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(cLists[domain].data()), ngh_dev, thrust::raw_pointer_cast(nbhNrs[domain].data()), active_nodes, mesh->getNumberVertices() ,mesh->getShapes().size());
                    thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), nbhNrs[domain].begin(), nbhNrs[domain].begin() + active_nodes, sAddrs[domain].begin());
                    //activeListPrint<Float><<<1,1>>>(thrust::raw_pointer_cast(cLists[domain].data()), solutions_dev, begin_domain, active_nodes);
                    gather_elements<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(cLists[domain].data()), thrust::raw_pointer_cast(nbhNrs[domain].data()),
                                                                                    thrust::raw_pointer_cast(elemLists[domain].data()), active_nodes, thrust::raw_pointer_cast(&elemListSizes[domain]), ngh_dev, shapes_dev);
                    construct_predicate_shared_memory<D, Float><<<active_nodes, 128, 0, streams[domain]>>>(thrust::raw_pointer_cast(elemLists[domain].data()), thrust::raw_pointer_cast(&elemListSizes[domain]), active_nodes, thrust::raw_pointer_cast(sAddrs[domain].data()), tetra_dev, geo_dev, solutions_dev, thrust::raw_pointer_cast(predicate[domain].data()), M_dev, tol, thrust::raw_pointer_cast(active_lists_dev[domain].data()), begin_domain, domain_size);
                    thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin() + begin_domain, predicate[domain].begin() + end_domain, sAddrs[domain].begin());
                    compact<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(predicate[domain].data()) + begin_domain, domain_size,thrust::raw_pointer_cast(cLists[domain].data()), begin_domain);
                    int active_neighbors_node = thrust::count(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin() + begin_domain, predicate[domain].begin() + end_domain, 1);//count of the current domain activated nodes, other will be processed by other domains in a successive iteration
                    if(active_neighbors_node != 0) {
                        count_Nbhs<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(cLists[domain].data()), ngh_dev, thrust::raw_pointer_cast(nbhNrs[domain].data()), active_neighbors_node, mesh->getNumberVertices() ,mesh->getShapes().size());
                        thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), nbhNrs[domain].begin(), nbhNrs[domain].begin() + active_neighbors_node, sAddrs[domain].begin());
                        cudaStreamSynchronize(streams[domain]);
                        gather_elements<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(cLists[domain].data()), thrust::raw_pointer_cast(nbhNrs[domain].data()),
                                                                                    thrust::raw_pointer_cast(elemLists[domain].data()), active_neighbors_node, thrust::raw_pointer_cast(&elemListSizes[domain]), ngh_dev, shapes_dev);
                        domain_Sweep_shared_memory<D, Float><<<active_neighbors_node, 128, 0, streams[domain]>>>(thrust::raw_pointer_cast(elemLists[domain].data()),thrust::raw_pointer_cast(&elemListSizes[domain]),active_neighbors_node, thrust::raw_pointer_cast(sAddrs[domain].data()),tetra_dev, geo_dev, solutions_dev, thrust::raw_pointer_cast(active_lists_dev[domain].data()), M_dev, tol, begin_domain);
                    }

                    removeConvergedNodes<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(active_lists_dev[domain].data()), domain_size);
                    active_nodes = thrust::count(thrust::cuda::par_nosync.on(streams[domain]), active_lists_dev[domain].begin(), active_lists_dev[domain].end(), 1);
                    thrust::fill(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin() + begin_domain, predicate[domain].begin() + end_domain, 0);

                    //cudaStreamSynchronize(streams[domain]);
                    //cudaStreamSynchronize(streams[domain]);
                    //cudaMemcpy(solutions, solutions_dev, sizeof(Float) * mesh->getNumberVertices(), cudaMemcpyDeviceToHost);
                    //mesh->getSolutionsVTK(output_file_name+std::to_string(inner_cont), solutions);
                }

            }

            cudaDeviceSynchronize();

            for(int domain = 0; domain < mesh->getPartitionsNumber() && not_converged; domain++) {
                size_t domain_size = getDomainSize(domain);
                size_t begin_domain = getBeginDomain(domain);
                for(int other_domain = 0; other_domain < mesh->getPartitionsNumber(); other_domain++) {
                    int numBlocks = domain_size / NUM_THREADS + 1;
                    if(domain != other_domain) {
                        propagatePredicate<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(active_lists_dev[domain].data()), domain_size, begin_domain, thrust::raw_pointer_cast(&predicate[other_domain][0]));
                    }
                }
            }

            cudaDeviceSynchronize();
        }

        
        cudaMemcpy(solutions, solutions_dev, sizeof(Float) * mesh->getNumberVertices(), cudaMemcpyDeviceToHost);
        /*
            begin debug
        
        Matrix M;
        M << 0.01,0,0,
            0,0.01,0,
            0,0,0.01;
        
            double error = 0;

        for(int i = 0; i < mesh->getNumberVertices(); i++) {

            VectorExt v;
            for(int j = 0; j < D; j++) {
                v[j] = mesh->getGeo()[D*i+j];
            }
            error += std::abs(std::sqrt(v.transpose()*M*v) - solutions[i]);
            std::cout << i <<": " << solutions[i] << " actual: " << std::sqrt(v.transpose()*M*v) <<  std::endl;
        }
        std::cout << error/mesh->getNumberVertices() << std::endl;


        
            end debug
        */
        mesh->getSolutionsVTK(output_file_name, solutions);

        // destroy streams
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamDestroy(streams[i]);
        }
        // free host memory
        delete active_domains;
        delete solutions;







    }



private:
    Mesh<D, Float>* mesh;
    int* partitions_vertices_dev;
    int* partitions_tetra_dev;
    Float* geo_dev;
    int* tetra_dev;
    TetraConfig* shapes_dev;
    int* ngh_dev;
    Float* M_dev;
    Float* solutions_dev;
    int* active_domains_dev;

    void setup(std::vector<int>& source_nodes, Float infinity_value, std::vector<thrust::host_vector<int>>& active_lists, std::vector<thrust::device_vector<int>>& active_lists_dev){
        int* source_nodes_dev;
        cudaMalloc(&source_nodes_dev, sizeof(int) * source_nodes.size());
        cudaMemcpy(source_nodes_dev, source_nodes.data(), sizeof(int) * source_nodes.size(), cudaMemcpyHostToDevice);
        int numBlocksInfinity = mesh->getNumberVertices() / NUM_THREADS + 1;
        setSolutionsToInfinity<<<NUM_THREADS, numBlocksInfinity>>>(solutions_dev, infinity_value, mesh->getNumberVertices());
        cudaDeviceSynchronize();
        int numBlocksSources = source_nodes.size() / SIZE_WARP + 1;
        cudaMemset(active_domains_dev, 0, sizeof(int) * mesh->getPartitionsNumber());
        setSolutionsSourcesAndDomains<<<SIZE_WARP, numBlocksSources>>>(solutions_dev, source_nodes_dev, active_domains_dev, partitions_vertices_dev, mesh->getPartitionsNumber(), source_nodes.size());
        cudaCheck("domain cuda set");
        cudaDeviceSynchronize();
        cudaFree(source_nodes_dev);
        //std::cout << "source nodes: ";
        for(auto source : source_nodes) {
            for(int i = mesh->get_ngh()[source]; i < ((source != mesh->getNumberVertices() - 1) ? mesh->get_ngh()[source+1] : mesh->getShapes().size()); i++) {
                for(int j = 0; j < D +1; j++) {
                    int v = mesh->get_tetra()[(D+1)*mesh->getShapes()[i].tetra_index + j];
                    if(v != source) {
                        int domain = getDomain(v);
                        active_lists[domain][v - getBeginDomain(domain)] = 1;
                        //std::cout << "setting source at " << v << std::endl;
                    }

                }
            }
        }


        for(int i = 0; i < active_lists.size(); i++) {
            active_lists_dev[i] = active_lists[i];
        }

    }

    int getDomain(int vertex_index) {
        for(int i = 0; i < mesh->getPartitionsNumber(); i++) {
            if(vertex_index < mesh->getPartitionVertices()[i]) {
                return i;
            }
        }
        return -1;
    }

    //requires domain lies in range [0, mesh->getPartitionsNumber())
    //requires ngh is an array such that ngh[i] is the index in msh->shapes of the first tetrahedra near node i
    //ensures that return value is number of tetrahedra id domain 'domain', satisfied as
    //1)end + 1 is the index in mesh->shapes of the first tetrahedra near node PartitionsVertices[domain] -> is also the index in shapes of the first tetrahedra belonging to domain 'domain + 1'. Therefore end is the index of the last tetrahedra (in shapes) belonging to domain 'domain'(inclusive)
    //2) start, similarly to point 1) is the last tetrahedra belonging to domain 'domain-1'(inclusive)
    //therefore end - start is actually what is meant to be
    int getTetraNumberinDomain(int domain) {
        int end = (domain != mesh->getPartitionsNumber() - 1) ? (mesh->get_ngh()[getBeginDomain(domain + 1)] -1) : mesh->getShapes().size() - 1;
        int start = mesh->get_ngh()[getBeginDomain(domain)] - 1;
        return end - start;
    }


    //requires domain lies in range
    int getBeginDomain(int domain) {
        return (domain != 0) ? mesh->getPartitionVertices()[domain - 1] : 0;
    }


    int getDomainSize(int domain) {
        return getBeginDomain(domain+1) - getBeginDomain(domain);
    }

};

#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH