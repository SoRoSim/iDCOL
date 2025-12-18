// examples/main.cpp
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include "core/idcol_newton.hpp"
#include "core/radial_bounds.hpp"
#include "core/shape_core.hpp"

//small helper
static inline double deg2rad(double deg) {
    constexpr double PI = 3.14159265358979323846;
    return deg * PI / 180.0;
}

// Row-major packer to match your polytope layout: params = [m; beta; A(row-major); b]
static Eigen::VectorXd pack_polytope_params_rowmajor(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& A,
    const Eigen::VectorXd& b,
    double beta)
{
    const int m = static_cast<int>(A.rows());
    if (A.cols() != 3) throw std::runtime_error("A must be m x 3");
    if (b.size() != m) throw std::runtime_error("b must be length m");

    Eigen::VectorXd params(3 + 3*m + m);
    params(0) = static_cast<double>(m);
    params(1) = beta;
    params(2) = 1; //A length scale. Replace with max(rout1, rout2)

    // A in column-major (MATLAB reshape(A,[],1) order):
    // [A(0,0) A(1,0) ... A(m-1,0)  A(0,1) ... A(m-1,1)  A(0,2) ... A(m-1,2)]
    double* outA = params.data() + 3;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < m; ++i) {
            outA[j*m + i] = A(i,j);
        }
    }

    // b
    params.segment(3 + 3*m, m) = b;
    return params;
}

int main() {
    using namespace idcol;

    // ----------------- Polytope 1 in world frame -----------------
    Eigen::Matrix<double, 6, 3, Eigen::RowMajor> A1;
    A1 <<  1, 0, 0,
          -1, 0, 0,
           0, 1, 0,
           0,-1, 0,
           0, 0, 1,
           0, 0,-1;

    Eigen::VectorXd b1(6);
    b1 << 1, 1, 1, 1, 1, 1;

    // ----------------- Polytope 2 in local frame -----------------
    Eigen::Matrix<double, 6, 3, Eigen::RowMajor> A2 = A1;

    Eigen::VectorXd b2(6);
    b2 << 1, 1, 2, 2, 1.5, 1.5;

    // ----------------- Poses g1, g2 ------------------------------
    Eigen::Matrix4d g1 = Eigen::Matrix4d::Identity(); //no lose of generality

    // (A) Different every run (recommended)
    std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        )
    );

    // (B) Reproducible (uncomment if needed)
    // std::mt19937 rng(12345);

    // uniform distributions
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_real_distribution<double> unifDeg(0.0, 360.0);

    // ---------- Rotation (Z–X–Y, MATLAB-style) ----------
    auto deg2rad = [](double deg) {
        constexpr double PI = 3.14159265358979323846;
        return deg * PI / 180.0;
    };

    double angZ = deg2rad(unifDeg(rng));
    double angX = deg2rad(unifDeg(rng));
    double angY = deg2rad(unifDeg(rng));

    //angZ = 1; angX = 2; angY = -1;

    Eigen::Matrix3d R =
        Eigen::AngleAxisd(angZ, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(angX, Eigen::Vector3d::UnitX()).toRotationMatrix() *
        Eigen::AngleAxisd(angY, Eigen::Vector3d::UnitY()).toRotationMatrix();

    // ---------- Translation ----------
    Eigen::Vector3d r;
    r << -3.0 + 6.0 * unif01(rng),
        -3.0 + 6.0 * unif01(rng),
        -3.0 + 6.0 * unif01(rng);

    //r << 1.1, -2.21 , 1.51;

    // ---------- Pose g2 ----------
    Eigen::Matrix4d g2 = Eigen::Matrix4d::Identity();
    g2.topLeftCorner<3,3>() = R;
    g2.topRightCorner<3,1>() = r;

    // ----------------- Pack params like MATLAB -------------------
    const double beta = 20.0;
    Eigen::VectorXd params1 = pack_polytope_params_rowmajor(A1, b1, beta);
    Eigen::VectorXd params2 = pack_polytope_params_rowmajor(A2, b2, beta);

    ProblemData P;
    P.g1 = g1;
    P.g2 = g2;
    P.shape_id1 = 2; //polytope
    P.shape_id2 = 2;
    P.params1 = params1;
    P.params2 = params2;

    RadialBoundsOptions optr;
    optr.num_starts = 1000;
    RadialBounds bounds1 = compute_radial_bounds_local(2, params1, optr);
    RadialBounds bounds2 = compute_radial_bounds_local(2, params2, optr);

    double d = r.norm();
    double alpha_min = d / (bounds1.Rout + bounds2.Rout);
    double alpha_max = d / (bounds1.Rin + bounds2.Rin);
    Eigen::Vector3d u = r / d;

    std::cout << "alpha_min  = " << alpha_min  << "\n";
    std::cout << "alpha_max = " << alpha_max << "\n";

    NewtonOptions opt;
    opt.L = 1; //scale factor for x. change with std::max(bounds1.Rout,bounds2.Rout)?
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    // Solving a surrogate problem !
    std::cout << "||r|| = " << r.norm() << "\n";
    r /= alpha_min;
    g2.topRightCorner<3,1>() = r;
    P.g2 = g2;

    double alpha_max_scaled = alpha_max/alpha_min;
    // alpha_max_scaled = 1

    double s_min = 0;
    double s_max = std::log(alpha_max_scaled);

    //std::cout << "alpha_min_scaled  = " << alpha_min  << "\n";
    std::cout << "alpha_max_scaled = " << alpha_max_scaled << "\n";

    // ----------------- Initial guess -----------------------------
    Eigen::Vector3d x0 = 0.5 * ( (bounds1.Rout) * u + (r - bounds2.Rout * u) ); //(change)
    double alpha0 = std::sqrt(alpha_max_scaled);

    double phi0;
    Eigen::Vector4d grad0;
    shape_eval_global_ax_phi_grad(g1, x0, alpha0, 2, params1, phi0, grad0);
    double lambda10 = (alpha0) / (r.transpose() * grad0.head<3>()); // from stationarity equations
    shape_eval_global_ax_phi_grad(g2, x0, alpha0, 2, params2, phi0, grad0);
    double lambda20 = -(alpha0) / (r.transpose() * grad0.head<3>()); // from stationarity equations

    //std::cout << "x0_scaled = " << x0.transpose() << "\n";
    //std::cout << "alpha0_scaled  = " << alpha0  << "\n";
    //std::cout << "lambda10_scaled  = " << lambda10  << "\n";
    //std::cout << "lambda20_scaled  = " << lambda20  << "\n";

    opt.s_min = s_min;
    opt.s_max = s_max;

    // start timer
    int N = 10000; //put it to 1 if no need to check time
    auto t0 = std::chrono::high_resolution_clock::now();

    idcol::NewtonResult res;
    for (int i=1;i<N;i++) {
        res = idcol::solve_idcol_newton(P, x0, alpha0, lambda10, lambda20, opt);
    }

    // stop timer
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "converged: " << (res.converged ? "true" : "false") << "\n";
    std::cout << "x: " << (res.x * alpha_min).transpose() << "\n"; // back to real pblm
    std::cout << "alpha: " << res.alpha * alpha_min << "\n"; // back to real pblm
    std::cout << "F_norm: " << res.final_F_norm << "\n";
    std::cout << "attempts_used: " << res.attempts_used << "\n";
    std::cout << "iters_used: " << res.iters_used << "\n";
    std::cout << "msg: " << res.message << "\n";
    // duration
    std::chrono::duration<double, std::micro> dt = (t1 - t0) / N;

    std::cout << "Time: " << dt.count() << " us\n";

    return res.converged ? 0 : 1;
}
