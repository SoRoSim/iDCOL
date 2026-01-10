#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

#include "core/idcol_solve.hpp"

using namespace idcol;
using Clock = std::chrono::steady_clock;

// Local copy of ShapeSpec and helpers from examples/ergodic.cpp
struct ShapeSpec {
    int shape_id;
    Eigen::VectorXd params;
    RadialBounds bounds;
    std::string name;
};

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

    double* outA = params.data() + 3;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < m; ++i) {
            outA[j*m + i] = A(i,j);
        }
    }

    params.segment(3 + 3*m, m) = b;
    return params;
}

static ShapeSpec make_poly_local() {
    Eigen::Matrix<double,8,3> A1;
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

    RadialBoundsOptions optr; optr.num_starts = 100;
    ShapeSpec s;
    s.shape_id = 2;
    s.name = "poly";
    s.params = pack_polytope_params_rowmajor(A1, b1, 20.0);
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

static ShapeSpec make_tc_local() {
    const double beta = 20.0;
    const double rb = 1.0;
    const double rt = 1.5;
    const double ac = 1.5;
    const double bc = 1.5;
    RadialBoundsOptions optr; optr.num_starts = 100;
    ShapeSpec s;
    s.shape_id = 5;
    s.name = "tc";
    Eigen::Matrix<double,5,1> p; p << beta, rb, rt, ac, bc;
    s.params = p;
    s.bounds = compute_radial_bounds_local(s.shape_id, s.params, optr);
    return s;
}

void run_solve_benchmark(const ShapeSpec &s1, const ShapeSpec &s2, int runs) {
    SolveResult res;
    SolveData S;
    ProblemData P;
    P.g1 = Eigen::Matrix4d::Identity();
    P.g2 = Eigen::Matrix4d::Identity(); P.g2.topRightCorner<3,1>() = Eigen::Vector3d(1.0, 0.0, 0.0);
    P.shape_id1 = s1.shape_id;
    P.shape_id2 = s2.shape_id;
    P.params1 = s1.params;
    P.params2 = s2.params;
    S.P = P;
    S.bounds1 = s1.bounds;
    S.bounds2 = s2.bounds;

    SurrogateOptions sopt;
    sopt.enable_scaling = false;

    NewtonOptions opt;
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    // warmup
    for (int i = 0; i < 2; ++i) idcol_solve(S, std::nullopt, opt, sopt);

    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < runs; ++i) {
        res = idcol_solve(S, std::nullopt, opt, sopt);
    }
    Clock::time_point t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    std::cout << "Case " << s1.name << "-" << s2.name << ": runs=" << runs << " total_ms=" << ms << " avg_ms=" << (ms / runs) << "\n";
    std::cout << "  converged=" << res.newton.converged << " iters=" << res.newton.iters_used << " final_F_norm=" << res.newton.final_F_norm << "\n";
}

void run_shape_eval_benchmark(const ShapeSpec &s, int runs) {
    // measure shape_eval_global_xa cost at a sample x,alpha
    Eigen::Vector3d x; x << 0.1, 0.2, 0.3;
    double alpha = 1.0;
    double phi; Eigen::Vector4d grad; Eigen::Matrix4d H;

    // g transform: use identity for benchmarking
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();

    // warmup
    for (int i = 0; i < 2; ++i) shape_eval_global_xa(g, x, alpha, s.shape_id, s.params, phi, grad, H);

    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < runs; ++i) {
        shape_eval_global_xa(g, x, alpha, s.shape_id, s.params, phi, grad, H);
    }
    Clock::time_point t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    std::cout << "Shape " << s.name << ": eval runs=" << runs << " total_ms=" << ms << " avg_micro=" << (ms * 1000.0 / runs) << " us\n";
}

int main() {
    auto poly = make_poly_local();
    auto tc   = make_tc_local();

    std::cout << "Benchmark: idcol_solve (poly-poly)\n";
    run_solve_benchmark(poly, poly, 50);

    std::cout << "Benchmark: idcol_solve (tc-tc)\n";
    run_solve_benchmark(tc, tc, 50);

    std::cout << "Benchmark: shape eval (poly)\n";
    run_shape_eval_benchmark(poly, 10000);

    std::cout << "Benchmark: shape eval (tc)\n";
    run_shape_eval_benchmark(tc, 10000);

    return 0;
}
