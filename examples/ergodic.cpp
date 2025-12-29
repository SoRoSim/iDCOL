// examples/main.cpp
#include <Eigen/Dense>
#include <Eigen/Geometry>
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

// Helper to construct the pose g(t)
Eigen::Matrix4d getSystematicPose(double t, double r_min, double r_max) {
    
    // 1. Frequencies (Irrational ratios for ergodicity)
    const double f1 = std::sqrt(2.0);
    const double f2 = std::sqrt(3.0);
    const double f3 = std::sqrt(5.0);
    const double f4 = std::sqrt(7.0);
    const double f5 = std::sqrt(11.0);
    const double f6 = std::sqrt(13.0);
    const double f7 = std::sqrt(17.0);

    const double TWO_PI = 2.0 * M_PI;

    // ---------- Translation (Spherical Sweep) ----------
    // Radial distance oscillating between r_min and r_max
    //double r_val = r_min + 0.5 * (r_max - r_min) * (1.0 + std::sin(TWO_PI * f1 * t));
    // smooth parameter in [0,1]
    double u = 0.5 * (1.0 + std::sin(TWO_PI * f1 * t));

    // approximate uniform-in-volume radial mapping
    double r_val = std::cbrt(
        r_min*r_min*r_min +
        (r_max*r_max*r_max - r_min*r_min*r_min) * u
    );

    
    // Latitude (theta) and Longitude (phi)
    double theta = (M_PI / 2.0) * std::sin(TWO_PI * f2 * t);
    double phi   = TWO_PI * f3 * t;

    Eigen::Vector3d pos;
    pos << r_val * std::cos(theta) * std::cos(phi),
           r_val * std::cos(theta) * std::sin(phi),
           r_val * std::sin(theta);

    // ---------- Orientation (Uniform Quaternion Sweep) ----------
    // Use Sine/Cosine mix to ensure the vector never hits [0,0,0,0]
    double v1 = std::sin(TWO_PI * f4 * t);
    double v2 = std::cos(TWO_PI * f5 * t);
    double v3 = std::sin(TWO_PI * f6 * t);
    double v4 = std::cos(TWO_PI * f7 * t);

    Eigen::Quaterniond q(v1, v2, v3, v4); // (w, x, y, z)
    q.normalize(); // Ensures it sits on the unit hypersphere

    // ---------- Assemble Pose g2 ----------
    Eigen::Matrix4d g2 = Eigen::Matrix4d::Identity();
    g2.topLeftCorner<3,3>() = q.toRotationMatrix();
    g2.topRightCorner<3,1>() = pos;

    return g2;
}

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
    s.params = pack_polytope_params_rowmajor(A, b, beta);
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_se(double n, double a, double b, double c, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 3;
    s.name = "se";
    Eigen::Vector4d p; p << n, a, b, c;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_sec(double n, double r, double h, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 4;
    s.name = "sec";
    Eigen::Vector3d p; p << n, r, h;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_tc(double beta, double rb, double rt, double ac, double bc, const RadialBoundsOptions& optr)
{
    ShapeSpec s;
    s.shape_id = 5;
    s.name = "tc";
    Eigen::Matrix<double,5,1> p; p << beta, rb, rt, ac, bc;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static void run_case(const ShapeSpec& s1, const ShapeSpec& s2)
{
    using namespace idcol;

    ProblemData P;
    P.g1 = Eigen::Matrix4d::Identity();
    P.shape_id1 = s1.shape_id;
    P.shape_id2 = s2.shape_id;
    P.params1 = s1.params;
    P.params2 = s2.params;

    NewtonOptions opt;
    opt.L = 1; 
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    SurrogateOptions sopt;
    sopt.fS_values = {1, 3, 5, 7};

    double r_min = 0.1 * std::min(s1.bounds.Rin,  s2.bounds.Rin);
    double r_max = 2.0 * std::max(s1.bounds.Rout, s2.bounds.Rout);

    const double t_max = 100.0;
    const double dt = 1e-4;
    const int N = static_cast<int>(std::round(t_max / dt)) + 1;

    int failed = 0;
    Eigen::Vector3d x0; double alpha0=1.0, lambda10=1.0, lambda20=1.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        double t = i * dt;

        P.g2 = getSystematicPose(t, r_min, r_max);

        // No warm start (keep your current behavior)
        SolveResult out = idcol_solve(P, s1.bounds, s2.bounds, opt, std::nullopt, sopt);

        // If you want warm-start: pass Guess after i>0
        x0 = out.newton.x; alpha0 = out.newton.alpha; lambda10 = out.newton.lambda1; lambda20 = out.newton.lambda2;

        if (!out.newton.converged){
            /*
            std::cout << "\nFAIL case " << s1.name << "-" << s2.name
                    << " i=" << i << " t=" << t << "\n";

            std::cout << "g2=\n" << P.g2 << "\n";
            std::cout << "x=" << out.newton.x.transpose() << "\n";
            std::cout << "alpha=" << out.newton.alpha
                    << " lambda1=" << out.newton.lambda1
                    << " lambda2=" << out.newton.lambda2 << "\n";
            std::cout << "iters=" << out.newton.iters_used
                    << " ||F||=" << out.newton.final_F_norm
                    << " msg=" << out.newton.message << "\n";
            std::exit(0);
            */
            failed++;
            

        } 
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count() / double(N);

    double success = 100.0 * double(N - failed) / double(N);

    std::cout << "CASE " << s1.name << "-" << s2.name
              << " | avg_us=" << avg_us
              << " | success=" << success << "%\n";
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
    
    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    auto poly = make_poly(A1, b1, beta, optr);
    auto se   = make_se(n, a, b, c, optr);
    auto sec  = make_sec(n, r, h, optr);
    auto tc   = make_tc(beta, rb, rt, ac, bc, optr);

    std::vector<ShapeSpec> shapes = {poly, se, sec, tc};

    for (const auto& s1 : shapes)
        for (const auto& s2 : shapes)
            run_case(s1, s2);
    //run_case(sec, sec);

    /*
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
    P.shape_id1 = 2; //polytope
    P.shape_id2 = 2; //polytope
    P.params1 = params_poly;
    P.params2 = params_poly;
    
    NewtonOptions opt;
    opt.L = 1; //scale factor for x. change with: bounds_poly.Rout + bounds_poly.Rout?
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    Eigen::Vector3d x0;
    double alpha0;
    double lambda10;
    double lambda20;

    double t_max = 100.0; 
    double dt = 0.0001;
    int N = static_cast<int>(std::round(t_max / dt)) + 1;
    int failed_count = 0; // Initialize counter
    idcol::Guess guess;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {

        double t = i * dt;

        // ---------- ergodic pose g2 ----------
        g2 = getSystematicPose(t, 0.1 * bounds_poly.Rin, 2 * bounds_poly.Rout);
        
        // Build surrogate schedule (default is {1,3}, but you can set explicitly)
        P.g2 = g2;
        idcol::SurrogateOptions sopt;
        sopt.fS_values = {1, 3};   // or {1,2,4}, etc.

        
        idcol::SolveResult out;
        //Call with no initial guess
        //out = idcol::idcol_solve(P, bounds_poly, bounds_poly, opt, std::nullopt, sopt); 

        // Call with initial guess
        if (i==0){
            out = idcol::idcol_solve(P, bounds_poly, bounds_poly, opt, std::nullopt, sopt); // replace std::nullopt with guess
        }
        else{
            guess.x       = x0;
            guess.alpha   = alpha0;
            guess.lambda1 = lambda10;
            guess.lambda2 = lambda20;
            out = idcol::idcol_solve(P, bounds_poly, bounds_poly, opt, guess, sopt);
        }

        // Extract solution (original space)
        x0       = out.newton.x;
        alpha0   = out.newton.alpha;
        lambda10 = out.newton.lambda1;
        lambda20 = out.newton.lambda2;

        // (Optional) diagnostics
        // bool ok = out.newton.converged;
        // int fS_used = out.fS_used;
        // int fS_tries = out.fS_attempts_used;

        if (!out.newton.converged) {
            failed_count++;
            std::cout << "FAILED case #" << failed_count << "\n";
            std::cout << "FAILED case index: " << i << "\n";
            std::cout << "[iDCOL] converged = " << out.newton.converged << "\n"
                        << "        fS_used = " << out.fS_used << "\n"
                        << "        fS_attempts = " << out.fS_attempts_used << "\n"
                        << "        iters = " << out.newton.iters_used << "\n"
                        << "        ||F|| = " << out.newton.final_F_norm << "\n"
                        << "        msg = " << out.newton.message << "\n";    
        }
    }

    // stop timer
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t_total = std::chrono::duration<double, std::micro>(t1 - t0);
    double t_avg = t_total.count() / static_cast<double>(N);

    std::cout << "========================================\n";
    std::cout << "BENCHMARK SUMMARY (Deterministic Sweep)\n";
    std::cout << "========================================\n";
    std::cout << "Avg time: " << t_avg << " us\n";

    double success_rate = (static_cast<double>(N - failed_count) / N) * 100.0;
    std::cout << "Success Rate: " << success_rate << " %\n";
    */
}
