// examples/main.cpp
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include "core/idcol_newton.hpp"

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

    Eigen::VectorXd params(2 + 3*m + m);
    params(0) = static_cast<double>(m);
    params(1) = beta;

    // A in row-major: [a11 a12 a13 a21 a22 a23 ...]
    double* outA = params.data() + 2;
    for (int i = 0; i < m; ++i) {
        outA[3*i + 0] = A(i,0);
        outA[3*i + 1] = A(i,1);
        outA[3*i + 2] = A(i,2);
    }

    // b
    params.segment(2 + 3*m, m) = b;
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
    Eigen::Matrix4d g1 = Eigen::Matrix4d::Identity();

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

    Eigen::Matrix3d R =
        Eigen::AngleAxisd(angZ, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(angX, Eigen::Vector3d::UnitX()).toRotationMatrix() *
        Eigen::AngleAxisd(angY, Eigen::Vector3d::UnitY()).toRotationMatrix();

    // ---------- Translation ----------
    Eigen::Vector3d r;
    r << 1.0 + 2.0 * unif01(rng),
        1.0 + 2.0 * unif01(rng),
        1.0 + 2.0 * unif01(rng);

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
    P.shape_id1 = 2;  // your convex polytope smooth-max
    P.shape_id2 = 2;
    P.params1 = params1;
    P.params2 = params2;

    // ----------------- Initial guess -----------------------------
    Eigen::Vector3d x0 = r / 2.0;
    double alpha0 = 1.0;
    double lambda10 = 1.0;
    double lambda20 = 1.0;

    NewtonOptions opt;
    opt.L = 1.0;         // later: use bounding-sphere radius
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

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
    std::cout << "x: " << res.x.transpose() << "\n";
    std::cout << "alpha: " << res.alpha << "\n";
    std::cout << "F_norm: " << res.final_F_norm << "\n";
    std::cout << "attempts_used: " << res.attempts_used << "\n";
    std::cout << "iters_used: " << res.iters_used << "\n";
    std::cout << "msg: " << res.message << "\n";
    // duration
    std::chrono::duration<double, std::micro> dt = (t1 - t0) / N;

    std::cout << "Time: " << dt.count() << " us\n";

    return res.converged ? 0 : 1;
}
