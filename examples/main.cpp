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


static Eigen::VectorXd pack_polytope_params(
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

struct ShapeSpec {
    int shape_id;
    Eigen::VectorXd params;
    RadialBounds bounds;
    std::string name;
};

static ShapeSpec make_poly(const Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>& A,
                           const Eigen::VectorXd& b, double beta, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 2;
    s.name = "poly";
    s.params = pack_polytope_params(A, b, beta);
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_tc(double beta, double rb, double rt, double ac, double bc, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 3;
    s.name = "tc";
    Eigen::Matrix<double,5,1> p; p << beta, rb, rt, ac, bc;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_se(double n, double a, double b, double c, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 4;
    s.name = "se";
    Eigen::Vector4d p; p << n, a, b, c;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_sec(double n, double r, double h, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 5;
    s.name = "sec";
    Eigen::Vector3d p; p << n, r, h;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
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

        // ----------------- Smooth truncated Cone--------------
    const double rb = 1.0;
    const double rt = 1.5;
    const double ac = 1.5;
    const double bc = 1.5;

    // ----------------- Superellipsoid--------------
    const double a = 0.5;
    const double b = 1.0;
    const double c = 1.5;

    // ----------------- Superelliptic Cylinder--------------
    const double r = 1.0;
    const double h = 2.0; //half-height
        
    const double beta = 20.0;
    const int n = 8;


    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    auto poly = make_poly(A1, b1, beta, optr);
    auto se   = make_se(n, a, b, c, optr);
    auto sec  = make_sec(n, r, h, optr);
    auto tc   = make_tc(beta, rb, rt, ac, bc, optr);


    // ----------------- Relative pose g ------------------------------

    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();

    ProblemData P;

    P.shape_id1 = poly.shape_id; 
    P.shape_id2 = poly.shape_id;  
    P.params1 = poly.params;
    P.params2 = poly.params;

    P.g <<
        0.821168,   0.557509,  -0.121927,  -1.82107,
        0.123649,  0.0347633,   0.991717,  -2.76868,
        0.557129,  -0.829443,  -0.0403888, -0.302319,
        0, 0, 0, 1;

    //P.g.topRightCorner<3,1>() *= 10;

    SolveData S;
    S.P = P;
    S.bounds1 = poly.bounds;
    S.bounds2 = poly.bounds;
    
    NewtonOptions opt;
    opt.L = 1; //scale factor
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = true;

    Eigen::Vector3d x;
    double alpha;
    double lambda1;
    double lambda2;

    // Build surrogate schedule (default is {1,3}, but you can set explicitly)    
    idcol::SurrogateOptions sopt;
    sopt.fS_values = {1};   // or {1,2,4}, etc.
    sopt.enable_scaling = false;

    
    idcol::SolveResult out;
    //Call with no initial guess
    auto t0 = std::chrono::high_resolution_clock::now();
    out = idcol::idcol_solve(S);
    //out = idcol::idcol_solve(S, std::nullopt, opt, sopt); 
    auto t1 = std::chrono::high_resolution_clock::now();

    // Extract solution (original space)
    x       = out.newton.x;
    alpha   = out.newton.alpha;
    lambda1 = out.newton.lambda1;
    lambda2 = out.newton.lambda2;

    double phi_star;
    Eigen::Vector4d grad_star;

    shape_eval_global_xa_phi_grad(Eigen::Matrix4d::Identity(), x, alpha, P.shape_id1, P.params1, phi_star, grad_star);

    double t_total_us =
    std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();

    if (out.newton.converged) {
        std::cout << "[iDCOL] converged = 1\n"
              << "        time_us = " << t_total_us << "\n"
              << "        fS_used = " << out.fS_used << "\n"
              << "        fS_attempts = " << out.fS_attempts_used << "\n"
              << "        F = \n" << out.newton.F << "\n"
              << "        J = \n" << out.newton.J << "\n"
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
