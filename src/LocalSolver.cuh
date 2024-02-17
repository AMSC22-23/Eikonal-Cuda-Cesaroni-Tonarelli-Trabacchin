##ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
#include <cmath>

template <typename Float>
class LocalSolver {
public:
    static void solve(Float phi1, Float phi2, Float alpha1, Float alpha2, Float alpha3, Float beta1,
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

#ifdef DEBUG
        if (b_hat * b_hat - 4 * a_hat * c_hat < 0){
               printf("Discrimant is negative\n");
               return 0;
           }
#endif

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

};


#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_LOCALSOLVER_CUH
