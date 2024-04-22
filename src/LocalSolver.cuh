#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#include <cmath>
#include <limits>
#include <tuple>

template <int D, typename Float>
class LocalSolver {
    using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
    using Matrix = typename Eikonal::Eikonal_traits<D,2>::AnisotropyM;

public:
    //M is supposed to point at the beginning of the relevant fragment of the M array (M is a 6-element array)
    static auto solve(std::array<VectorExt, 4> coordinates, VectorExt values, Float* M, Matrix velocity) {


        Float lambda21;
        Float lambda22;
        Float lambda11;
        Float lambda12;
        Float lambda1;
        Float lambda2;


        Float alpha1 = computeScalarProduct(0,2,0,2, M);
        //(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha2 = computeScalarProduct(1,2,0,2, M);
        //(coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha3 = computeScalarProduct(2,3,0,2, M);
        //(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[0]);

        Float beta1 = alpha2;//(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        Float beta2 = computeScalarProduct(1,2,1,2,M);
        //(coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        Float beta3 = computeScalarProduct(2,3,1,2,M);
        //(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[1]);

        Float gamma1 = alpha3;
        //(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        Float gamma2 = beta3;//(coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        Float gamma3 = computeScalarProduct(2,3,2,3,M);
        //(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[3] - coordinates[2]);


        solve3D(values[2] - values[0], values[2] - values[1], alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3, &lambda11, &lambda21, &lambda12, &lambda22);
        std::cout << "phi1 = " << values[2] - values[0] << std::endl;
        std::cout << "phi2 = " << values[2] - values[1] << std::endl;
        std::cout << "alpha1 = " << alpha1 << std::endl;
        std::cout << "alpha2 = " << alpha2 << std::endl;
        std::cout << "alpha3 = " << alpha3 << std::endl;
        std::cout << "beta1 = " << beta1 << std::endl;
        std::cout << "beata2 = " << beta2 << std::endl;
        std::cout << "beta3 = " << beta3 << std::endl;
        std::cout << "gamma1 = " << gamma1 << std::endl;
        std::cout << "gamma2 = " << gamma2 << std::endl;
        std::cout << "gamma3 = " << gamma3 << std::endl;

        std::cout << "lambda11: " << lambda11 << std::endl;
        std::cout << "lambda12: " << lambda12 << std::endl;
        std::cout << "lambda21: " << lambda21 << std::endl;
        std::cout << "lambda22: " << lambda22 << std::endl;



        Float acceptable12 = !std::isnan(lambda11) && lambda11 > 0 && lambda11 < 1;
        Float acceptable11 = !std::isnan(lambda12) && lambda12 > 0 && lambda12 < 1;
        Float acceptable21 = !std::isnan(lambda21) && lambda21 > 0 && lambda21 < 1;
        Float acceptable22 = !std::isnan(lambda22) && lambda22 > 0 && lambda22 < 1;

//xy (lambda11, lambda21), (lambda12, lambda22)
        if(acceptable21 && acceptable11 && acceptable12 && acceptable22) {
            Float phi4_1 = computeSolution3D(lambda11, lambda21, values, coordinates, M, velocity);
            Float phi4_2 = computeSolution3D(lambda12, lambda22, values, coordinates, M, velocity);
            if(phi4_1 < phi4_2) {
                return std::make_tuple(phi4_1, lambda11, lambda21);
            } else {
                return std::make_tuple(phi4_2, lambda12, lambda22);
            }
        } else if(!acceptable12 && !acceptable22 && acceptable21 && acceptable11) {
            lambda1 = lambda11;
            lambda2 = lambda21;
            Float phi4 = computeSolution3D(lambda1, lambda2, values, coordinates, M, velocity);
            return std::make_tuple(phi4, lambda1, lambda2);

        } else if(acceptable12 && acceptable22 && !acceptable12 && !acceptable11) {
            lambda1 = lambda21;
            lambda2 = lambda22;
            Float phi4 = computeSolution3D(lambda1, lambda2, values, coordinates, M, velocity);
            return std::make_tuple(phi4, lambda1, lambda2);

        } else {
            /*
            Float alpha, beta, gamma, phi;
            Float phi4[6];
            std::array<Float, 2> lambdas[6];
            bool acceptable[6] = {false};
            //face 1,3
            alpha = computeScalarProduct(0, 2, 0, 2, M);
            beta  = computeScalarProduct(0, 3, 0, 2, M);
            gamma = computeScalarProduct(0, 3, 0, 3, M);
            phi = values[2] - values[0];
            solve2D(phi, alpha, beta, gamma, &lambda1, &lambda2);

            if(!std::isnan(lambda1)) {
                if(lambda1 >= 0 && lambda1 <= 1) {
                    phi4[0] = computeSolution2D(lambda1, values, coordinates, M, 0, 2, velocity);
                    lambdas[0] = {lambda1,0};
                    acceptable[0] = true;
                }
                if(lambda2 >= 0 && lambda2 <= 1) {
                    phi4[1] = computeSolution2D(lambda2, values, coordinates, M, 0, 2, velocity);
                    lambdas[1] = {0, lambda2};
                    acceptable[1] = true;
                }
            }

            //face 1,2
            alpha = computeScalarProduct(0, 1, 0, 1, M);
            beta  = computeScalarProduct(0, 3, 0, 1, M);
            gamma = computeScalarProduct(0, 3, 0, 3, M);
            phi = values[1] - values[0];
            solve2D(phi, alpha, beta, gamma, &lambda1, &lambda2);
            if(!std::isnan(lambda1)) {
                if(lambda1 >= 0 && lambda1 <= 1) {
                    phi4[2] = computeSolution2D(lambda1, values, coordinates, M, 0, 1, velocity);
                    lambdas[2] = {lambda1,0};
                    acceptable[2] = true;
                }
                if(lambda2 >= 0 && lambda2 <= 1) {
                    phi4[3] = computeSolution2D(lambda2, values, coordinates, M, 0, 1, velocity);
                    lambdas[3] = {0,lambda2};
                    acceptable[3] = true;
                }
            }

            //face 2,3
            alpha = computeScalarProduct(1, 2, 1, 2, M);
            beta  = computeScalarProduct(1, 3, 1, 2, M);
            gamma = computeScalarProduct(1, 3, 1, 3, M);
            phi = values[2] - values[1];
            solve2D(phi, alpha, beta, gamma, &lambda1, &lambda2);
            if(!std::isnan(lambda1)) {
                if(lambda1 >= 0 && lambda1 <= 1) {
                    phi4[4] = computeSolution2D(lambda1, values, coordinates, M, 1, 2, velocity);
                    lambdas[4] = {lambda1,0};
                    acceptable[4] = true;
                }
                if(lambda2 >= 0 && lambda2 <= 1) {
                    lambdas[5] = {0,lambda2};
                    phi4[5] = computeSolution2D(lambda2, values, coordinates, M, 1, 2, velocity);
                    acceptable[5] = true;
                }
            }

            //find smallest acceptable solution
            Float minimum_value = std::numeric_limits<Float>::max();
            int minimum_index;
            bool minimum_found = false;
            for(int i = 0; i < 6; i++) {
                if(acceptable[i] && phi4[i] < minimum_value) {
                    minimum_value = phi4[i];
                    minimum_index = i;
                    minimum_found = true;
                }
            }

            if(minimum_found) {
                return std::make_tuple(minimum_value, lambdas[minimum_index][0], lambdas[minimum_index][1]);
            }
             */
            //if 3d solutions are not acceptable we resort to these solutions
            //TODO check whether this approach leads to convergence, as this is an arbitrary assumption
            Float last_resort1 = computeSolution3D(1, 0, values, coordinates, M, velocity);
            Float last_resort2 = computeSolution3D(0, 1, values, coordinates, M, velocity);
            std::cout << "last resort" << std::endl;
            if(last_resort1 < last_resort2) {
                return std::make_tuple(last_resort1, 1.0, 0.0);
            } else {
                return std::make_tuple(last_resort2, 0.0, 1.0);
            }
            // return (last_resort1 < last_resort2) ? last_resort1 : last_resort2;


        }


    }

    static Float computeSolution3D(Float lambda1, Float lambda2, VectorExt values, std::array<VectorExt, 4> &coordinates, Float* M, Matrix velocity) {
        return lambda1*values[0] + lambda2*values[1] + (1 - lambda1 - lambda2)*values[2] + computeP(coordinates, M, lambda1, lambda2, velocity);
    }

    static Float computeSolution2D(Float lambda, VectorExt values, std::array<VectorExt, 4> &coordinates, Float* M, int x, int y, Matrix velocity) {
        return lambda * (values[y] - values[x])  + values[x] + computeP2D(lambda, coordinates, M, x, y, velocity);
    }

    static Float computeP2D(Float lambda, std::array<VectorExt, 4> &coordinates, Float* M, int x, int y, Matrix velocity) {
        std::cout << "P2d" << std::endl;
        VectorExt x5 = lambda*coordinates[y] + (1 - lambda) * coordinates[x];
        VectorExt e54 = coordinates[3] - x5;


        Float p[D];
        p[0] = (coordinates[1] - coordinates[0]).transpose() * e54;
        p[1] = (coordinates[2] - coordinates[0]).transpose() * e54;
        p[2] = (coordinates[3] - coordinates[0]).transpose() * e54;
        Float P = 0;
        for(int i = 0; i < D; i++) {
            for(int j = 0; j < D; j++) {
                P += p[i]*p[j]* computeScalarProduct(0, i + 1, 0, j + 1, M);
            }
        }
        P = sqrt(P);
        return sqrt(e54.transpose() * velocity * e54);
    }


    static Float computeP(std::array<VectorExt, 4> &coordinates, Float* M, Float lambda1, Float lambda2, Matrix velocity) {

        std::cout << "P3d" << std::endl;

        Float M_prime[3][3];
//TODO consider improving the M_prime management
        M_prime[0][0] = computeScalarProduct(0,2,0,2,M);
        M_prime[1][0] = computeScalarProduct(1,2,0,2,M);
        M_prime[2][0] = computeScalarProduct(2,3,0,2,M);
        M_prime[0][1] = computeScalarProduct(0,2,1,2,M);
        M_prime[1][1] = computeScalarProduct(1,2,1,2,M);
        M_prime[2][1] = computeScalarProduct(2,3,1,2,M);
        M_prime[0][2] = computeScalarProduct(0,2,2,3,M);
        M_prime[1][2] = computeScalarProduct(1,2,2,3,M);
        M_prime[2][2] = computeScalarProduct(2,3,2,3,M);
        Matrix M_prime_;
        M_prime_ << M_prime[0][0], M_prime[0][1], M_prime[0][2],
                M_prime[1][0], M_prime[1][1], M_prime[1][2],
                M_prime[2][0], M_prime[2][1], M_prime[2][2];
        VectorExt lambda;
        lambda << lambda1, lambda2, 1 ;
        Float computedP = std::sqrt(lambda.transpose() * M_prime_ * lambda);
        VectorExt x5 = lambda1*coordinates[0] + lambda2*coordinates[1] + (1 - lambda1 - lambda2)*coordinates[2];
        VectorExt e54 = coordinates[3] - x5;
        Float correctP = std::sqrt(e54.transpose() * velocity * e54);
        if(computedP != correctP) {
            std::cout << "wrong P " << computedP << " " << correctP << std::endl;
        }
        return correctP;
    }

    static Float computeP1(std::array<VectorExt, 4> &coordinates, Float* M, Float lambda1, Float lambda2, Matrix velocity) {
        VectorExt x5 = lambda1*coordinates[0] + lambda2*coordinates[1] + (1 - lambda1 - lambda2)*coordinates[2];
        VectorExt e54 = coordinates[3] - x5;
        //return std::sqrt(e54.transpose() * velocity * e54);
        Float p[D];
        p[0] = (coordinates[1] - coordinates[0]).transpose() * e54;
        p[1] = (coordinates[2] - coordinates[0]).transpose() * e54;
        p[2] = (coordinates[3] - coordinates[0]).transpose() * e54;
        Float P = 0;
        for(int i = 0; i < D; i++) {
            for(int j = 0; j < D; j++) {
                P += p[i]*p[j] * computeScalarProduct(0, i + 1, 0, j + 1, M);
            }
        }
        P = sqrt(P);
        if (P != std::sqrt(e54.transpose() * velocity * e54)) {
            std::cout << "wrong scalar product: " << lambda1 << " " << lambda2 << std::endl;
        } else {
            std::cout << "correct scalar product: " << lambda1 << " " << lambda2 << std::endl;
        }
        return P;
    }





    static void solve3D(Float phi1, Float phi2, Float alpha1, Float alpha2, Float alpha3, Float beta1,
                        Float beta2, Float beta3, Float gamma1, Float gamma2, Float gamma3,
                        Float* lambda11, Float* lambda21, Float* lambda12, Float* lambda22) {
        Float a, b, c, d, e, f, g, h, k, a_hat, b_hat, c_hat, delta;

        a = phi2 * (alpha1 - alpha3) - phi1 * (beta1 - beta3);
        b = phi2 * (alpha2 - alpha3) - phi1 * (beta2 - beta3);
        c = phi2 * alpha3 - phi1 * beta3;
        d = phi1 * phi1 * (alpha1 - alpha3 - gamma1 + gamma3) - (alpha1 * alpha1 + alpha3 * alpha3 - 2 * alpha1 * alpha3);
        e = phi1 * phi1 * (beta2 - beta3 - gamma2 + gamma3) - (alpha2 * alpha2 + alpha3 * alpha3 - 2 * alpha2 * alpha3);
        f = phi1* phi1 * (alpha2 - alpha3 + beta1 - beta3 - gamma1 - gamma2 + 2 * gamma3) - 2 * (alpha1 - alpha3) * (alpha2 - alpha3);
        g = phi1 * phi1 * (alpha3 + gamma1 - 2 * gamma3) - (-2 * alpha3 * alpha3 + 2 * alpha1 * alpha3);
        h = phi1 * phi1 * (beta3 + gamma2 - 2 * gamma3) - (2 * alpha2 * alpha3 - 2 * alpha3 * alpha3);
        k = phi1* phi1 * gamma3 - alpha3 * alpha3;

        a_hat = d * b * b / (a * a) + e - f * b / a;
        b_hat = 2 * b * c * d / (a * a) - f * c / a - b * g / a + h;
        c_hat = k + d * c * c / (a * a) - c * g / a;

        /*if (b_hat * b_hat - 4 * a_hat * c_hat < 0){
               printf("Discriminant is negative\n");
           }
           */
        delta = std::sqrt(b_hat * b_hat - 4 * a_hat * c_hat);
        /* std::cout << "discriminant: " << delta << " a = " << a_hat << " b = "
                   << b_hat << " c = " << c_hat <<  std::endl;
         */
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
    static void solve2D(Float phi, Float alpha, Float beta, Float gamma, Float* lambda1, Float* lambda2){
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

    static Float computeScalarProduct(int k1, int k2, int l1, int l2, Float* M) {
        int k_gray = getGrayCode(k1, k2);
        int l_gray = getGrayCode(l1, l2);
        if(k_gray != l_gray) {
            int s_gray = k_gray ^ l_gray;
            auto [s1, s2] = getOriginalNumbers(s_gray);
            int sign = (2 * (s_gray > k_gray) - 1) * (2 * (s_gray > l_gray) - 1);
            return sign * 0.5 * (M[getMIndex(k1, k2)] + M[getMIndex(l1, l2)] - M[getMIndex(s1, s2)]);
        } else {
            return ( (k1 < k2) ? (M[getMIndex(k1,k2)]) : M[getMIndex(k2,k1)]);
        }
    }

    static int getGrayCode(int k, int l) {
        return 1<<k | 1<<l;
    }

    //invert getGrayCode
    static auto getOriginalNumbers(int gray) {

        if(gray == 3) {
            return std::make_tuple(0,1);
        }
        else if(gray == 5) {
            return std::make_tuple(0,2);
        }
        else if(gray == 6) {
            return std::make_tuple(1,2);
        }
        else if(gray == 9) {
            return std::make_tuple(0,3);
        }
        else if(gray == 10) {
            return std::make_tuple(1,3);
        }
        else if(gray == 12) {
            return std::make_tuple(2,3);
        }
        else {
            printf("wrong gray code\n");
            return std::make_tuple(0,0);
        }
    }

    //ok
    static int getMIndex(int s1, int s2) {
        return (1<<(s2-1)) - 1 + s1;
    }


};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH

