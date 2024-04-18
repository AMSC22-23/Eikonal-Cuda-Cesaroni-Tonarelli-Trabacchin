#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#include <cmath>

template <typename Float>
class LocalSolver {
    using VectorExt = typename Eikonal::Eikonal_traits<3, 2>::VectorExt;
public:
    static Float solve(std::array<VectorExt, 2> coordinates, VectorExt values, Float phi1, Float phi2, Float alpha1, Float alpha2, Float alpha3, Float beta1,
                      Float beta2, Float beta3, Float gamma1, Float gamma2, Float gamma3
                      ) {


        Float lambda21;
        Float lambda22;
        Float lambda11;
        Float lambda12;
        Float lambda1;
        Float lambda2;


        Float alpha1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[0]);
        Float alpha3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[0]);

        Float beta1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        Float beta2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[2] - coordinates[1]);
        double beta3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[2] - coordinates[1]);

        double gamma1 = (coordinates[2] - coordinates[0]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        double gamma2 = (coordinates[2] - coordinates[1]).transpose() * velocity * (coordinates[3] - coordinates[2]);
        double gamma3 = (coordinates[3] - coordinates[2]).transpose() * velocity * (coordinates[3] - coordinates[2]);


        solve3D(values[2] - values[0], values[2] - values[1], alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3, &lambda11, &lambda21, &lambda12, &lambda22);
        bool nan12 = isnan(lambda12);
        bool nan11 = isnan(lambda11);
        bool nan21 = isnan(lambda21);
        bool nan22 = isnan(lambda22);
        if(!nan12 && !nan11 && !nan21 && !nan22) {
            double value1, value2;
            VectorExt x5 = lambda11*coordinates[0] + lambda12*coordinates[1] + (1 - lambda11 - lambda12)*coordinates[2];

            Float phi4 = lambda11*values[0] + lambda12*values[1] + (1 - lambda11 - lambda12)*values[2] + std::sqrt((coordinates[3] - x5).transpose() * );
            //both couple
        } else if(nan21 && nan22 && !nan12 && !nan11) {
            *lambda1 = lambda11;
            *lambda2 = lambda12;

        } else if(!nan21 && !nan22 && nan12 && nan11) {
            *lambda1 = lambda21;
            *lambda2 = lambda22;
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



};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH

