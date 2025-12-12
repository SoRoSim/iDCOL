#include "mex.h"
#include <Eigen/Dense>
#include <cmath>
#include "core/shape_core.hpp"
#include <chrono>

int n_repeats = 1000; // or pass as an optional argument

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// -----------------------------------------------------------------------------
// Evaluate KKT residual F and Jacobian J for variables
// z = [ x (3); s; lambda1; lambda2 ]
// -----------------------------------------------------------------------------
static void eval_F_J(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const Matrix4d& g1,
    const Matrix4d& g2,
    int shape_id1,
    int shape_id2,
    const VectorXd& params1,
    const VectorXd& params2,
    Vector6d& F,
    Matrix6d& J)
{
    // alpha = e^s > 0
    double alpha = std::exp(s);

    // Evaluate phi1, grad1, H1
    double  phi1;
    Vector4d grad1;
    Matrix4d H1;
    shape_eval_global_ax(g1, x, alpha, shape_id1, params1, phi1, grad1, H1);

    // Evaluate phi2, grad2, H2
    double  phi2;
    Vector4d grad2;
    Matrix4d H2;
    shape_eval_global_ax(g2, x, alpha, shape_id2, params2, phi2, grad2, H2);

    // Split gradients
    Vector3d g1x = grad1.head<3>();
    double   g1a = grad1(3);

    Vector3d g2x = grad2.head<3>();
    double   g2a = grad2(3);

    // Split Hessians via block views (no copies)
    const auto H1_xx = H1.block<3,3>(0,0);
    const auto H1_xa = H1.block<3,1>(0,3);       // d²phi1 / dx dα
    double     H1_aa = H1(3,3);

    const auto H2_xx = H2.block<3,3>(0,0);
    const auto H2_xa = H2.block<3,1>(0,3);       // d²phi2 / dx dα
    double     H2_aa = H2(3,3);

    // -----------------------
    // Build residual F(z)
    // -----------------------
    F(0) = phi1;                                 // F1 = φ1
    F(1) = phi2;                                 // F2 = φ2
    F.segment<3>(2) = lambda1 * g1x + lambda2 * g2x; // F3..5
    F(5) = 1.0 + lambda1 * g1a + lambda2 * g2a; // F6

    // -----------------------
    // Build Jacobian J
    // z = [x(3), s, lambda1, lambda2]
    // -----------------------
    J.setZero();

    // Row 0: F1 = φ1
    J.block<1,3>(0,0) = g1x.transpose();       // dF1/dx
    J(0,3) = g1a * alpha;                      // dF1/ds

    // Row 1: F2 = φ2
    J.block<1,3>(1,0) = g2x.transpose();       // dF2/dx
    J(1,3) = g2a * alpha;                      // dF2/ds

    // Rows 2..4: Fx = λ1 ∇x φ1 + λ2 ∇x φ2
    for (int j = 0; j < 3; ++j) {
        int row = 2 + j;

        // dF_j/dx = λ1 H1_xx(j,:) + λ2 H2_xx(j,:)
        J.block<1,3>(row, 0) =
            lambda1 * H1_xx.row(j) + lambda2 * H2_xx.row(j);

        // dF_j/ds = (λ1 H1_xa(j) + λ2 H2_xa(j)) * α
        double dFjs = (lambda1 * H1_xa(j) + lambda2 * H2_xa(j)) * alpha;
        J(row, 3) = dFjs;

        // dF_j/dλ1 = (∇x φ1)_j
        J(row, 4) = g1x(j);
        // dF_j/dλ2 = (∇x φ2)_j
        J(row, 5) = g2x(j);
    }

    // Row 5: F6 = 1 + λ1 g1a + λ2 g2a
    // dF6/dx = λ1 (d(dφ1/dα)/dx) + λ2 (d(dφ2/dα)/dx)
    // H_ax = (H_xa)^T since H is symmetric
    Eigen::RowVector3d dF6_dx =
        lambda1 * H1_xa.transpose() + lambda2 * H2_xa.transpose();
    J.block<1,3>(5,0) = dF6_dx;

    // dF6/ds = λ1 (dg1a/ds) + λ2 (dg2a/ds)
    // dg1a/dα = H1_aa, so dg1a/ds = H1_aa * α
    double dF6_ds = (lambda1 * H1_aa + lambda2 * H2_aa) * alpha;
    J(5,3) = dF6_ds;

    // dF6/dλ1 = g1a, dF6/dλ2 = g2a
    J(5,4) = g1a;
    J(5,5) = g2a;
}


static void eval_F( // no J
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const Matrix4d& g1,
    const Matrix4d& g2,
    int shape_id1,
    int shape_id2,
    const VectorXd& params1,
    const VectorXd& params2,
    Vector6d& F)
{
    // alpha = e^s > 0
    double alpha = std::exp(s);

    // φ1, ∂φ1/∂x, ∂φ1/∂α
    double  phi1;
    Vector4d grad1;
    shape_eval_global_ax_phi_grad(
        g1, x, alpha, shape_id1, params1,
        phi1, grad1
    );

    // φ2, ∂φ2/∂x, ∂φ2/∂α
    double  phi2;
    Vector4d grad2;
    shape_eval_global_ax_phi_grad(
        g2, x, alpha, shape_id2, params2,
        phi2, grad2
    );

    // split gradients; avoid extra Vector3d copies for no reason
    const Vector3d g1x = grad1.head<3>();
    const double   g1a = grad1(3);

    const Vector3d g2x = grad2.head<3>();
    const double   g2a = grad2(3);

    // F1 = φ1
    F(0) = phi1;

    // F2 = φ2
    F(1) = phi2;

    // F3..5 = λ1 ∇x φ1 + λ2 ∇x φ2
    F.segment<3>(2) = lambda1 * g1x + lambda2 * g2x;

    // F6 = 1 + λ1 ∂φ1/∂α + λ2 ∂φ2/∂α
    F(5) = 1.0 + lambda1 * g1a + lambda2 * g2a;
}


// -----------------------------------------------------------------------------
// MEX interface:
// [z_opt, F_opt, J_opt] = idcol_newton_mex( ...
//     g1, g2, shape_id1, params1, shape_id2, params2, x0, alpha0, ...
//     [lambda10, lambda20, max_iters, tol] )
//
// z_opt = [x; alpha; lambda1; lambda2]  (6x1)
// F_opt = KKT residual at solution      (6x1)
// J_opt = Jacobian dF/d[x; s; lambda1; lambda2] at solution (6x6)
// -----------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 8) {
        mexErrMsgTxt("Usage: [z_opt, F_opt, J_opt] = idcol_newton_mex(g1, g2, shape_id1, params1, shape_id2, params2, x0, alpha0, [lambda10, lambda20, max_iters, tol])");
    }
    if (nlhs != 3) {
        mexErrMsgTxt("Need 3 outputs: z_opt (6x1), F_opt (6x1), J_opt (6x6).");
    }

    // --- g1 ---
    if (mxGetM(prhs[0]) != 4 || mxGetN(prhs[0]) != 4) {
        mexErrMsgTxt("g1 must be 4x4.");
    }
    double* g1_ptr = mxGetPr(prhs[0]);
    Matrix4d g1;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            g1(row,col) = g1_ptr[row + 4*col];

    // --- g2 ---
    if (mxGetM(prhs[1]) != 4 || mxGetN(prhs[1]) != 4) {
        mexErrMsgTxt("g2 must be 4x4.");
    }
    double* g2_ptr = mxGetPr(prhs[1]);
    Matrix4d g2;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            g2(row,col) = g2_ptr[row + 4*col];

    // --- shape_id1 ---
    int shape_id1 = static_cast<int>(mxGetScalar(prhs[2]));

    // --- params1 ---
    if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3])) {
        mexErrMsgTxt("params1 must be real double.");
    }
    double* p1 = mxGetPr(prhs[3]);
    mwSize nP1 = mxGetNumberOfElements(prhs[3]);
    VectorXd params1(nP1);
    for (mwSize i = 0; i < nP1; ++i) params1(i) = p1[i];

    // --- shape_id2 ---
    int shape_id2 = static_cast<int>(mxGetScalar(prhs[4]));

    // --- params2 ---
    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5])) {
        mexErrMsgTxt("params2 must be real double.");
    }
    double* p2 = mxGetPr(prhs[5]);
    mwSize nP2 = mxGetNumberOfElements(prhs[5]);
    VectorXd params2(nP2);
    for (mwSize i = 0; i < nP2; ++i) params2(i) = p2[i];

    // --- x0 ---
    if (mxGetM(prhs[6]) != 3 || mxGetN(prhs[6]) != 1) {
        mexErrMsgTxt("x0 must be 3x1.");
    }
    double* x0_ptr = mxGetPr(prhs[6]);
    Vector3d x0;
    x0 << x0_ptr[0], x0_ptr[1], x0_ptr[2];

    // --- alpha0 ---
    double alpha0 = mxGetScalar(prhs[7]);
    if (alpha0 <= 0.0) {
        mexErrMsgTxt("alpha0 must be > 0.");
    }
    double s0 = std::log(alpha0);

    // Optional: lambda10, lambda20, max_iters, tol
    double lambda10 = 1.0;
    double lambda20 = 1.0;
    int    max_iters = 20;
    double tol = 1e-10;

    if (nrhs >= 9)  lambda10 = mxGetScalar(prhs[8]);
    if (nrhs >= 10) lambda20 = mxGetScalar(prhs[9]);
    if (nrhs >= 11) max_iters = static_cast<int>(mxGetScalar(prhs[10]));
    if (nrhs >= 12) tol = mxGetScalar(prhs[11]);

    // Working variables (will be reset each repeat)
    Vector3d x;
    double s;
    double lambda1, lambda2;

    // Newton loop on F(z) = 0
    Vector6d F;
    Matrix6d J;

    // bounds on s = log(alpha)
    const double s_min = -4.605170185988092;  // log(1e-2)
    const double s_max =  4.605170185988092;  // log(1e2)

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n_repeats; ++k) {

        // RESET INIT GUESS EVERY REPEAT (this is the key)
        x = x0;
        s = s0;
        lambda1 = lambda10;
        lambda2 = lambda20;

        for (int iter = 0; iter < max_iters; ++iter) {
            // clamp s inside bounds before evaluating
            if (s < s_min) s = s_min;
            if (s > s_max) s = s_max;

            eval_F_J(x, s, lambda1, lambda2,
                    g1, g2, shape_id1, shape_id2,
                    params1, params2,
                    F, J);

            double Fn = F.norm();
            if (Fn < tol) {
                break;
            }

            // Regularize Jacobian slightly if near singular
            /* Matrix6d J_reg = J;
            Eigen::FullPivLU<Matrix6d> lu(J_reg);
            if (!lu.isInvertible()) {
                J_reg += 1e-9 * Matrix6d::Identity();
                lu.compute(J_reg);
                mexWarnMsgTxt("Newton: Jacobian is near-singular, regularizing.\n");
            }
            Vector6d dz = lu.solve(-F); */

            // Cheaper LU: assume J is usually well-conditioned in narrow phase
            Eigen::PartialPivLU<Matrix6d> lu(J);
            Vector6d dz = lu.solve(-F);



            // Simple trust region in s to avoid huge alpha jumps
            /*double max_step_s = 1.0; // limit change in s per iteration
            if (std::abs(dz(3)) > max_step_s) {
                double scale = max_step_s / std::abs(dz(3));
                dz *= scale;
            }*/

            // Backtracking line search on merit function 0.5*||F||^2
            double Fnorm2 = F.squaredNorm();
            double t = 1.0;
            const double beta = 0.5;   // step shrink factor
            const double c    = 1e-4;  // sufficient decrease parameter

            Vector3d x_trial;
            double  s_trial, lambda1_trial, lambda2_trial;
            Vector6d F_trial;

            bool accepted = false;
            for (int ls = 0; ls < 10; ++ls) {

                x_trial       = x + t * dz.segment<3>(0);
                s_trial       = s + t * dz(3);

                lambda1_trial = lambda1 + t * dz(4);
                lambda2_trial = lambda2 + t * dz(5);

                // clamp s_trial to keep alpha in [alpha_min, alpha_max]
                if (s_trial < s_min) s_trial = s_min;
                if (s_trial > s_max) s_trial = s_max;

                eval_F(x_trial, s_trial, lambda1_trial, lambda2_trial,
                g1, g2, shape_id1, shape_id2,
                params1, params2,
                F_trial);

                double Fnorm2_trial = F_trial.squaredNorm();
                if (Fnorm2_trial <= Fnorm2 * (1.0 - c * t)) {
                    accepted = true;
                    break;
                }

                t *= beta;
            }

            if (!accepted) {
                mexWarnMsgTxt("Newton: line search failed to reduce residual, stopping.\n");
                break;
            }

            // Accept trial step
            x       = x_trial;
            s       = s_trial;
            lambda1 = lambda1_trial;
            lambda2 = lambda2_trial;

            // Optional: clamp lambdas to avoid blow-up
            /*const double lambda_max = 1e3;
            if (lambda1 > lambda_max)  lambda1 = lambda_max;
            if (lambda1 < -lambda_max) lambda1 = -lambda_max;
            if (lambda2 > lambda_max)  lambda2 = lambda_max;
            if (lambda2 < -lambda_max) lambda2 = -lambda_max;*/

            // Stopping criterion on step size
            if ((t * dz).norm() < 1e-14) {
                break;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count(); // seconds
    double per_call = 1e6 * elapsed / n_repeats; // microseconds per solve

    mexPrintf("avg per solve = %.3f us\n", per_call);
    
    // Final evaluation for outputs
    eval_F_J(x, s, lambda1, lambda2,
             g1, g2, shape_id1, shape_id2,
             params1, params2,
             F, J);

    double alpha = std::exp(s);

    // --- Output z_opt = [x; alpha; lambda1; lambda2] ---
    plhs[0] = mxCreateDoubleMatrix(6, 1, mxREAL);
    double* z_out = mxGetPr(plhs[0]);
    z_out[0] = x(0);
    z_out[1] = x(1);
    z_out[2] = x(2);
    z_out[3] = alpha;
    z_out[4] = lambda1;
    z_out[5] = lambda2;

    // --- Output F_opt ---
    plhs[1] = mxCreateDoubleMatrix(6, 1, mxREAL);
    double* F_out = mxGetPr(plhs[1]);
    for (int i = 0; i < 6; ++i) {
        F_out[i] = F(i);
    }

    // --- Output J_opt ---
    plhs[2] = mxCreateDoubleMatrix(6, 6, mxREAL);
    double* J_out = mxGetPr(plhs[2]);
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            J_out[row + 6*col] = J(row,col);
        }
    }
}
