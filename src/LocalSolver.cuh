#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#include <cmath>
#include <limits>
#include <cuda.h>
#include "../src/CudaEikonalTraits.cuh"

// class responsible for the implementation of the local solver
template <size_t D, typename Float>
class LocalSolver {
    /*using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
    using Matrix = typename Eikonal::Eikonal_traits<D,2>::AnisotropyM;
    using VectorV = typename Eigen::Matrix<double,4,1>;*/
    using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
    using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
    using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

public:
    //M is supposed to point at the beginning of the relevant fragment of the M array (M is a 6-element array)
    __host__ __device__ static Float solve(VectorExt* coordinates, VectorV values, const Float* M,  int shift) {

        int lookup_vertices[9];
        lookup_vertices[1] = 0; // 0 -> 0 0 0 1 -> 2^0=1
        lookup_vertices[2] = 1; // 1 -> 0 0 1 0 -> 2^1=2
        lookup_vertices[4] = 2; // 2 -> 0 1 0 0 -> 2^2=4
        lookup_vertices[8] = 3; // 3 -> 1 0 0 0 -> 2^3=8

        int lookup_edges_1[13];

        lookup_edges_1[3] = 0;  // edge_01 -> 0 0 1 1 -> 3
        lookup_edges_1[5] = 0;  // edge_02 -> 0 1 0 1 -> 5
        lookup_edges_1[6] = 1;  // edge_12 -> 0 1 1 0 -> 6
        lookup_edges_1[9] = 0;  // edge_03 -> 1 0 0 1 -> 9
        lookup_edges_1[10] = 1; // edge_13 -> 1 0 1 0 -> 10
        lookup_edges_1[12] = 2; // edge_23 -> 1 1 0 0 -> 12

        int lookup_edges_2[13];
        lookup_edges_2[3] = 1;
        lookup_edges_2[5] = 2;
        lookup_edges_2[6] = 2;
        lookup_edges_2[9] = 3;
        lookup_edges_2[10] = 3;
        lookup_edges_2[12] = 3;


        Float lambda21;
        Float lambda22;
        Float lambda11;
        Float lambda12;
        Float lambda1;
        Float lambda2;

        // (e_02)^T*M*(e_02)
        Float alpha1 = computeScalarProductDiagonal(0,2,0,2, M, shift, lookup_edges_1, lookup_edges_2);
        // (e_12)^T*M*(e_02)
        Float alpha2 = computeScalarProduct(1,2,0,2, M, shift, lookup_edges_1, lookup_edges_2);
        // (e_23)^T*M*(e_02)
        Float alpha3 = computeScalarProduct(2,3,0,2, M, shift, lookup_edges_1, lookup_edges_2);

        // (e_02)^T*M*(e_12)
        Float beta1 = alpha2;
        // (e_12)^T*M*(e_12)
        Float beta2 = computeScalarProductDiagonal(1,2,1,2,M, shift, lookup_edges_1, lookup_edges_2);
        // (e_23)^T*M*(e_12)
        Float beta3 = computeScalarProduct(2,3,1,2,M, shift, lookup_edges_1, lookup_edges_2);

        // (e_02)^T*M*(e_23)
        Float gamma1 = alpha3;
        // (e_12)^T*M*(e_23)
        Float gamma2 = beta3;
        // (e_23)^T*M*(e_23)
        Float gamma3 = computeScalarProductDiagonal(2,3,2,3,M, shift, lookup_edges_1, lookup_edges_2);

        // M_prime = [alpha, beta, gamma]
        Matrix M_prime;
        M_prime << alpha1, beta1, gamma1,
                alpha2, beta2, gamma2,
                alpha3, beta3, gamma3;




        int phi31_gray_code = getGrayCode(0, 2);
        int phi32_gray_code = getGrayCode(1, 2);
        int phi31_gray_code_rotated, phi31_gray_code_rotated_sign, phi32_gray_code_rotated, phi32_gray_code_rotated_sign;
        rotate(phi31_gray_code, shift, &phi31_gray_code_rotated, &phi31_gray_code_rotated_sign);
        rotate(phi32_gray_code, shift, &phi32_gray_code_rotated, &phi32_gray_code_rotated_sign);

        int phi31_actual_index1 = lookup_edges_1[phi31_gray_code_rotated];
        int phi31_actual_index2 = lookup_edges_2[phi31_gray_code_rotated];
        int phi32_actual_index1 = lookup_edges_1[phi32_gray_code_rotated];
        int phi32_actual_index2 = lookup_edges_2[phi32_gray_code_rotated];

        solve3D(phi31_gray_code_rotated_sign*(values[phi31_actual_index2] - values[phi31_actual_index1]), phi32_gray_code_rotated_sign*(values[phi32_actual_index2] - values[phi32_actual_index1]), alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3, &lambda11, &lambda21, &lambda12, &lambda22);

        bool acceptable11 = !std::isnan(lambda11) && lambda11 > 0 && lambda11 < 1;
        bool acceptable12 = !std::isnan(lambda12) && lambda12 > 0 && lambda12 < 1;
        bool acceptable21 = !std::isnan(lambda21) && lambda21 > 0 && lambda21 < 1;
        bool acceptable22 = !std::isnan(lambda22) && lambda22 > 0 && lambda22 < 1;

        //xy (lambda11, lambda21), (lambda12, lambda22)
        Float phi4_1 = computeSolution3D(lambda11, lambda21, values, coordinates, M_prime,  shift, lookup_vertices, lookup_edges_1, lookup_edges_2);
        Float phi4_2 = computeSolution3D(lambda12, lambda22, values, coordinates, M_prime,  shift, lookup_vertices, lookup_edges_1, lookup_edges_2);
        if(acceptable21 && acceptable11 && acceptable12 && acceptable22) {
            if(phi4_1 < phi4_2 && phi4_1 >= 0) {
                return phi4_1;
            } else {
                return phi4_2;
            }
        } else if((!acceptable12 || !acceptable22) && acceptable21 && acceptable11 && phi4_1 >=0) {
            lambda1 = lambda11;
            lambda2 = lambda21;
            //Float phi4 = phi4_1; //computeSolution3D(lambda1, lambda2, values, coordinates, M,  shift);
            return phi4_1;
        } else if(acceptable12 && acceptable22 && (!acceptable21 || !acceptable11) && phi4_2 >=0) {
            lambda1 = lambda21;
            lambda2 = lambda22;
            //Float phi4 = computeSolution3D(lambda1, lambda2, values, coordinates, M,  shift);
            return phi4_2;
        } else {
            Float last_resort1 = computeSolution3D(1, 0, values, coordinates, M_prime,  shift, lookup_vertices, lookup_edges_1, lookup_edges_2);
            Float last_resort2 = computeSolution3D(0, 1, values, coordinates, M_prime, shift, lookup_vertices, lookup_edges_1, lookup_edges_2);
            Float last_resort3 = computeSolution3D(0, 0, values, coordinates, M_prime, shift, lookup_vertices, lookup_edges_1, lookup_edges_2);

            if(last_resort3 < last_resort1 && last_resort3 < last_resort2 && last_resort3 >=0) {
                 return last_resort3;
            }
            else if(last_resort1 < last_resort2 && last_resort1 >=0) {
                return last_resort1;
            } else {
                return last_resort2;
            }
        }
    }


    __host__ __device__ static Float computeSolution3D(Float lambda1, Float lambda2, VectorV& values, VectorExt* coordinates, const Matrix& M, int shift, int* lookup_vertices, int* lookup_edges_1, int* lookup_edges_2) {
        int rotated_0;
        int sign_0_ignore;
        int rotated_1;
        int sign_1_ignore;
        int rotated_2;
        int sign_2_ignore;
        /*auto[rotated_0, sign_0_ignore] = rotate(getGrayCode(0), shift);
        auto[rotated_1, sign_1_ignore] = rotate(getGrayCode(1), shift);
        auto[rotated_2, sign_2_ignore] = rotate(getGrayCode(2), shift);*/
        rotate(getGrayCode(0), shift, &rotated_0, &sign_0_ignore);
        rotate(getGrayCode(1), shift, &rotated_1, &sign_1_ignore);
        rotate(getGrayCode(2), shift, &rotated_2, &sign_2_ignore);
        rotated_0 = lookup_vertices[rotated_0];
        rotated_1 = lookup_vertices[rotated_1];
        rotated_2 = lookup_vertices[rotated_2];
        return lambda1*values[rotated_0] + lambda2*values[rotated_1] + (1 - lambda1 - lambda2)*values[rotated_2]
               + computeP(coordinates, M, lambda1, lambda2, shift, lookup_edges_1, lookup_edges_2);
    }


    __host__ __device__ static Float computeP(VectorExt* coordinates, const Matrix& M, Float lambda1, Float lambda2, int shift, int* lookup_edges_1, int* lookup_edges_2) {
        /*Float M_prime[3][3];
        //TODO consider improving the M_prime management
        M_prime[0][0] = computeScalarProductDiagonal(0,2,0,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[1][0] = computeScalarProduct(1,2,0,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[2][0] = computeScalarProduct(2,3,0,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[0][1] = M_prime[1][0];//computeScalarProduct(0,2,1,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[1][1] = computeScalarProductDiagonal(1,2,1,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[2][1] = computeScalarProduct(2,3,1,2,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[0][2] = M_prime[2][0];//computeScalarProduct(0,2,2,3,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[1][2] = M_prime[2][1];//computeScalarProduct(1,2,2,3,M,shift, lookup_edges_1, lookup_edges_2);
        M_prime[2][2] = computeScalarProductDiagonal(2,3,2,3,M,shift, lookup_edges_1, lookup_edges_2);
        Matrix M_prime_ ;
        M_prime_ << M_prime[0][0], M_prime[0][1], M_prime[0][2],
        M_prime[1][0], M_prime[1][1], M_prime[1][2],
        M_prime[2][0], M_prime[2][1], M_prime[2][2] ;*/


        VectorExt lambda ;//{lambda1, lambda2, 1};
        lambda << lambda1, lambda2, 1;
        Float computedP = std::sqrt(lambda.transpose() * ( M * lambda ));
        return computedP;
    }


    __host__ __device__ static void solve3D(Float phi1, Float phi2, Float alpha1, Float alpha2, Float alpha3, Float beta1,
                                            Float beta2, Float beta3, Float gamma1, Float gamma2, Float gamma3,
                                            Float* lambda11, Float* lambda21, Float* lambda12, Float* lambda22) {
        Float a, b, c, d, e, f, g, h, k, a_hat, b_hat, c_hat, delta;

        Float phi1_phi1 = phi1 * phi1;
        Float alpha3_alpha3 = alpha3 * alpha3;

        a = phi2 * (alpha1 - alpha3) - phi1 * (beta1 - beta3);
        b = phi2 * (alpha2 - alpha3) - phi1 * (beta2 - beta3);
        c = phi2 * alpha3 - phi1 * beta3;
        d = phi1_phi1 * (alpha1 - alpha3 - gamma1 + gamma3) - (alpha1 * alpha1 + alpha3_alpha3 - 2 * alpha1 * alpha3);
        e = phi1_phi1 * (beta2 - beta3 - gamma2 + gamma3) - (alpha2 * alpha2 + alpha3_alpha3 - 2 * alpha2 * alpha3);
        f = phi1_phi1 * (alpha2 - alpha3 + beta1 - beta3 - gamma1 - gamma2 + 2 * gamma3) - 2 * (alpha1 - alpha3) * (alpha2 - alpha3);
        g = phi1_phi1 * (alpha3 + gamma1 - 2 * gamma3) - (-2 * alpha3_alpha3 + 2 * alpha1 * alpha3);
        h = phi1_phi1 * (beta3 + gamma2 - 2 * gamma3) - (2 * alpha2 * alpha3 - 2 * alpha3_alpha3);
        k = phi1_phi1 * gamma3 - alpha3_alpha3;

        a_hat = d * b * b / (a * a) + e - f * b / a;
        b_hat = 2 * b * c * d / (a * a) - f * c / a - b * g / a + h;
        c_hat = k + d * c * c / (a * a) - c * g / a;

        delta = std::sqrt(b_hat * b_hat - 4 * a_hat * c_hat);

        if(b >= 0) {
            *lambda21 = (-b_hat - delta) / (2 * a_hat);
            *lambda22 = 2 * c_hat / (-b_hat - delta);
        } else {
            *lambda21 = (-b_hat + delta) / (2 * a_hat);
            *lambda22 = 2 * c_hat / (-b_hat + delta);
        }

        *lambda11 = (- b * (*lambda21) - c) / a;
        *lambda12 = (- b * (*lambda22) - c) / a;
    }


    //for face x,y phi = phi(y) - phi(x), alpha = e(x,y)'Me(x,y), beta = e(x,4)'Me(x,y), gamma = e(x,4)'Me(e,4)
    __host__ __device__ static void solve2D(Float phi, Float alpha, Float beta, Float gamma, Float* lambda1, Float* lambda2){
        Float a = (alpha - phi * phi) * alpha;
        Float b = 2 * beta * (phi * phi - alpha);
        Float c = beta * beta - phi * phi * gamma;

        Float delta = std::sqrt(b * b - 4 * a * c);
        /*std::cout << "discriminant: " << delta << " a = " << a << " b = "
                  << b << " c = " << c <<  std::endl;
        */
        if(b >= 0) {
            *lambda1 = (-b - delta) / (2 * a);
            *lambda2 = 2 * c / (-b - delta);
        } else {
            *lambda1 = (-b + delta) / (2 * a);
            *lambda2 = 2 * c / (-b + delta);
        }
    }

    __host__ __device__ static bool check_gray(int gray, int & l1_new, int & l2_new) {
        if(gray == 3 || gray == 5 || gray == 6 || gray == 9 || gray == 10 || gray==12) {
            //printf("ok!!!\n");
            return true;
        }
        else {
            auto [l1, l2] = getOriginalNumbers(gray);
            l1_new = l1;
            l2_new = l2;
            printf("wrong gray code %d %d\n", l1_new, l2_new);
            return false;
        }
    }

    __host__ __device__ static Float computeScalarProduct(int k1, int k2, int l1, int l2, const Float* M, int shift, int* lookup_edges_1, int* lookup_edges_2) {
        int k_gray = getGrayCode(k1, k2);
        int l_gray = getGrayCode(l1, l2);
        int k_gray_rotated;
        int sign1;
        int l_gray_rotated;
        int sign2;
        //auto [k_gray_rotated, sign1] = rotate(k_gray, shift);
        //auto [l_gray_rotated, sign2] = rotate(l_gray, shift);
        rotate(k_gray, shift, &k_gray_rotated, &sign1);
        rotate(l_gray, shift, &l_gray_rotated, &sign2);
        k_gray = k_gray_rotated;
        l_gray = l_gray_rotated;
        //auto [k1_new, k2_new] = getOriginalNumbers(k_gray);
        k1 = lookup_edges_1[k_gray];
        k2 = lookup_edges_2[k_gray];
        //auto [l1_new, l2_new] = getOriginalNumbers(l_gray);
        l1 = lookup_edges_1[l_gray];
        l2 = lookup_edges_2[l_gray];
        //check_gray(l_gray, l1_new, l2_new);


        /*k1 = k1_new;
        k2 = k2_new;
        l1 = l1_new;
        l2 = l2_new;*/

        //if(k_gray != l_gray) {
        int s_gray = k_gray ^ l_gray;
        //auto [s1, s2] = getOriginalNumbers(s_gray);
        int s1 = lookup_edges_1[s_gray];
        int s2 = lookup_edges_2[s_gray];
        int sign = (2 * (s_gray > k_gray) - 1) * (2 * (s_gray > l_gray) - 1) * sign1 * sign2;
        return sign * 0.5 * (M[getMIndex(k1, k2)] + M[getMIndex(l1, l2)] - M[getMIndex(s1, s2)]);
        //} else {
        //return M[getMIndex(k1,k2)];
        //}
    }

    __host__ __device__ static Float computeScalarProductDiagonal(int k1, int k2, int l1, int l2, const Float* M, int shift, int* lookup_edges_1, int* lookup_edges_2) {
        int k_gray = getGrayCode(k1, k2);
        int k_gray_rotated;
        int sign1;
        rotate(k_gray, shift, &k_gray_rotated, &sign1);
        k_gray = k_gray_rotated;
        k1 = lookup_edges_1[k_gray];
        k2 = lookup_edges_2[k_gray];

        return  M[getMIndex(k1,k2)];

    }

    // method to retrieve edge
    __host__ __device__ static int getGrayCode(int k, int l) {
        return 1<<k | 1<<l;
    }

    // method to retrieve vertex
    __host__ __device__ static int getGrayCode(int k) {
        return 1<<k;
    }

    // invert getGrayCode (edge)
    __host__ __device__ static auto getOriginalNumbers(int gray) {
        if(gray == 3) {
            return std::make_tuple(0,1); // edge_01
        }
        else if(gray == 5) {
            return std::make_tuple(0,2); // edge_02
        }
        else if(gray == 6) {
            return std::make_tuple(1,2); // edge_12
        }
        else if(gray == 9) {
            return std::make_tuple(0,3); // edge_03
        }
        else if(gray == 10) {
            return std::make_tuple(1,3); // edge_13
        }
        else if(gray == 12) {
            return std::make_tuple(2,3); // edge_23
        }
        else {
            printf("wrong gray code double %d\n", gray);
            return std::make_tuple(0,0);
        }
    }

    // invert getGrayCode (vertex)
    __host__ __device__ static auto getOriginalNumber(int gray) {
        if(gray == 1) {
            return 0;
        }
        else if(gray == 2) {
            return 1;
        }
        else if(gray == 4) {
            return 2;
        }
        else if(gray == 8) {
            return 3;
        }
        else {
            printf("wrong gray code single %d\n", gray);
            return 0;
        }
    }


    //ok
    __host__ __device__ static int getMIndex(int s1, int s2) {
        return (1<<(s2-1)) - 1 + s1;
    }

    //return 0 if sign is +, -1 otherwise
    __host__ __device__ static void rotate(int k, int shift, int * result, int * sign) {
        int carry = (k << (4 - shift)) & 0x0000000F;
        k = (k >> shift) | carry;
        carry = carry - ((carry >> 1) & 0x55555555);
        carry = (carry & 0x33333333) + ((carry >> 2) & 0x33333333);
        int count = ((carry + (carry >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
        int mod = count & 1;
        *result = k;
        *sign = (mod == 0 ? 1 : -1);
        //return std::make_tuple(k, mod == 0 ? 1 : -1);
    }

};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH

