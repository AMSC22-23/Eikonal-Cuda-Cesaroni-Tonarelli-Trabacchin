#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#include <cmath>

template <int D, typename Float>
class LocalSolver {
    using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
public:
    static Float solve(std::array<VectorExt, 2> coordinates, VectorExt values, Float phi1, Float phi2, Float* M
    ) {


        Float lambda21;
        Float lambda22;
        Float lambda11;
        Float lambda12;
        Float lambda1;
        Float lambda2;


        Float alpha1 = computeScalarProduct(0,2,0,2, M);//(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha2 = computeScalarProduct(1,2,0,2, M); //(coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha3 = computeScalarProduct(2,3,0,2, M);//(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[0]);

        Float beta1 = alpha2;//(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        Float beta2 = computeScalarProduct(1,2,1,2,M);//coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        Float beta3 = computeScalarProduct(2,3,1,2,M);//(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[1]);

        Float gamma1 = alpha3;//(coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        Float gamma2 = beta3;//(coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        Float gamma3 = computeScalarProduct(2,3,2,3,M);//(coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[3] - coordinates[2]);


        solve3D(values[2] - values[0], values[2] - values[1], alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3, &lambda11, &lambda21, &lambda12, &lambda22);
        Float nan12 = isnan(lambda12);
        Float nan11 = isnan(lambda11);
        Float nan21 = isnan(lambda21);
        Float nan22 = isnan(lambda22);


        VectorExt x5 = lambda11*coordinates[0] + lambda12*coordinates[1] + (1 - lambda11 - lambda12)*coordinates[2];
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


        if(!nan12 && !nan11 && !nan21 && !nan22) {
            Float value1, value2;

            Float phi4_1 = lambda11*values[0] + lambda12*values[1] + (1 - lambda11 - lambda12)*values[2] + P;
            Float phi4_2 = lambda21*values[0] + lambda22*values[1] + (1 - lambda21 - lambda22)*values[2] + P;
            if(phi4_1 < phi4_2) {
                return phi4_1;
            } else {
                return phi4_2;
            }

            //both couple
        } else if(nan21 && nan22 && !nan12 && !nan11) {
            *lambda1 = lambda11;
            *lambda2 = lambda12;
            Float phi4 = lambda1*values[0] + lambda2*values[1] + (1 - lambda1 - lambda2)*values[2] + P;
            return phi4;

        } else if(!nan21 && !nan22 && nan12 && nan11) {
            *lambda1 = lambda21;
            *lambda2 = lambda22;
            Float phi4 = lambda1*values[0] + lambda2*values[1] + (1 - lambda1 - lambda2)*values[2] + P;
            return phi4;

        } else {
            //2d problems
        }
    }
private:
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
        std::cout << "discriminant: " << delta << " a = " << a_hat << " b = "
                  << b_hat << " c = " << c_hat <<  std::endl;

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

    static void solve2D(Float phi, Float alpha, Float beta, Float gamma, Float* lambda1, Float* lambda2){
        Float a = (alpha - phi * phi) * alpha;
        Float b = 2 * beta * (phi * phi - alpha);
        Float c = beta * beta - phi * phi * gamma;

        Float delta = std::sqrt(b * b - 4 * a * c);
        std::cout << "discriminant: " << delta << " a = " << a << " b = "
                  << b << " c = " << c <<  std::endl;

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
            return sign * 0.5 * (M[getMIndex(k1, k2) + M[getMIndex(l1, l2) - M[getMIndex(s1, s2)]]]);
        }
    }

    static int getGrayCode(int k, int l) {
        return 1<<k | 1<<l;
    }

    //invert getGrayCode
    static auto getOriginalNumbers(int gray) {
        if(gray == 0) {
            return std::make_tuple(0,0);
        }
        else if(gray == 3) {
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

     static int getMIndex(int s1, int s2) {
        return (1<<(s2-1)) - 1 + s1;
    }



};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH

