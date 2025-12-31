// examples/main.cpp
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <cmath>
#include <chrono>
#include "core/idcol_newton.hpp"
#include "core/radial_bounds.hpp"
#include "core/shape_core.hpp"
#include "core/idcol_solve.hpp"


// Fix for M_PI undefined
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// ----- so(3) hat operator -----
static inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<     0.0, -v.z(),  v.y(),
          v.z(),   0.0,  -v.x(),
         -v.y(),  v.x(),   0.0;
    return S;
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
    
    params(0) = beta;
    params(1) = static_cast<double>(m);
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

    // ----------------- Polytope--------------
    Eigen::MatrixXd A1(8,3);
    A1 <<  1,  1,  1,
        1, -1, -1,
        -1,  1, -1,
        -1, -1,  1,
        -1, -1, -1,
        -1,  1,  1,
        1, -1,  1,
        1,  1, -1;
    A1 *= (1.0 / std::sqrt(3.0));

    Eigen::VectorXd b1(8);
    b1 << 1.0, 1.0, 1.0, 1.0,
        5.0/3.0, 5.0/3.0, 5.0/3.0, 5.0/3.0;

    // ----------------- Superellipsoid--------------
    const double a = 0.5;
    const double b = 1.0;
    const double c = 1.5;

    // ----------------- Superelliptic Cylinder--------------
    const double r = 1.0;
    const double h = 2.0; //half-height

    // ----------------- Truncated Cone--------------
    const double rb = 1.0;
    const double rt = 1.5;
    const double ac = 1.5;
    const double bc = 1.5;

        
    const double beta = 20.0;
    const int n = 8;
    
    Eigen::VectorXd params_poly = pack_polytope_params_rowmajor(A1, b1, beta);
    Eigen::Vector4d params_se;
    params_se << n, a, b, c;
    Eigen::Vector3d params_sec;
    params_sec << n, r, h;
    Eigen::Matrix<double, 5, 1> params_tc;
    params_tc << beta, rb, rt, ac, bc;

    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    RadialBounds bounds_poly = compute_radial_bounds_local(2, params_poly, optr);
    RadialBounds bounds_se = compute_radial_bounds_local(3, params_se, optr);
    RadialBounds bounds_sec = compute_radial_bounds_local(4, params_sec, optr);
    RadialBounds bounds_tc = compute_radial_bounds_local(5, params_tc, optr);

    //std::cout << "r_1,in  = " << bounds_tc.Rin  << "\n";
    //std::cout << "r_1,out  = " << bounds_tc.Rout  << "\n";
    //std::exit(0);

    // ----------------- Poses g1, g2 ------------------------------
    Eigen::Matrix4d g1 = Eigen::Matrix4d::Identity(); //no lose of generality
    Eigen::Matrix4d g2 = Eigen::Matrix4d::Identity(); //no lose of generality

    ProblemData P;
    P.g1 = g1;
    P.shape_id1 = 2; 
    P.shape_id2 = 3; 
    P.params1 = params_poly;
    P.params2 = params_se;

    // Face-face case
    /*Eigen::Vector3d u(2.0, -1.0, 2.0);
    u.normalize();
    const double theta = M_PI / 3;

    // g_here(1:3,1:3) = expm(skew(u*theta))  == exp(theta * skew(u))
    g2.topLeftCorner<3,3>() = (theta * skew(u)).exp();

    // g_here(1:3,4) = [2;2;2]
    g2.topRightCorner<3,1>() << 1.0, 0.0, 2.0;
    P.g2 = g2;
    */

    P.g2 <<
        0.821168,   0.557509,  -0.121927,  -1.82107,
        0.123649,  0.0347633,   0.991717,  -2.76868,
        0.557129,  -0.829443,  -0.0403888, -0.302319,
        0, 0, 0, 1;

    SolveData S;
    S.P = P;
    S.bounds1 = bounds_poly;
    S.bounds2 = bounds_se;
    
    NewtonOptions opt;
    opt.L = 1; //scale factor for x. change with: bounds_poly.Rout + bounds_poly.Rout?
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    Eigen::Vector3d x;
    double alpha;
    double lambda1;
    double lambda2;

    //idcol::Guess guess;



    // Build surrogate schedule (default is {1,3}, but you can set explicitly)    
    idcol::SurrogateOptions sopt;
    sopt.fS_values = {1};   // or {1,2,4}, etc.

    
    idcol::SolveResult out;
    //Call with no initial guess
    auto t0 = std::chrono::high_resolution_clock::now();
    out = idcol::idcol_solve(S, opt, std::nullopt, sopt); 
    auto t1 = std::chrono::high_resolution_clock::now();

    // Extract solution (original space)
    x       = out.newton.x;
    alpha   = out.newton.alpha;
    lambda1 = out.newton.lambda1;
    lambda2 = out.newton.lambda2;

    double phi_star;
    Eigen::Vector4d grad_star;

    shape_eval_global_xa_phi_grad(P.g1, x, alpha, P.shape_id1, P.params1, phi_star, grad_star);

    double t_total_us =
    std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();

    if (out.newton.converged) {
        std::cout << "[iDCOL] converged = 1\n"
                << "        time_us = " << t_total_us << "\n"
                << "        fS_used = " << out.fS_used << "\n"
                << "        fS_attempts = " << out.fS_attempts_used << "\n"
                << "        iters = " << out.newton.iters_used << "\n"
                << "        ||F|| = " << out.newton.final_F_norm << "\n"
                << "        alpha = " << alpha << "\n"
                << "        x = " << x.transpose() << "\n"
                << "        lambda1 = " << lambda1 << "\n"
                << "        lambda2 = " << lambda2 << "\n"
                << "        normal = " << grad_star.head<3>().transpose() << "\n";
    } else {
        std::cout << "[iDCOL] converged = 0\n"
                << "        time_us = " << t_total_us << "\n"
                << "        fS_used = " << out.fS_used << "\n"
                << "        fS_attempts = " << out.fS_attempts_used << "\n"
                << "        iters = " << out.newton.iters_used << "\n"
                << "        ||F|| = " << out.newton.final_F_norm << "\n"
                << "        msg = " << out.newton.message << "\n";
    }

}
