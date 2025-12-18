#include "mex.h"
#include <Eigen/Dense>
#include <cmath>
#include "core/idcol_kkt.hpp"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;


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
        mexErrMsgTxt("Usage: [z_opt, F_opt, J_opt] = idcol_newton_mex(g1, g2, shape_id1, params1, shape_id2, params2, x0, alpha0, [lambda10, lambda20, L, max_iters, tol])");
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

    idcol::ProblemData P;
    P.g1 = g1;
    P.g2 = g2;
    P.shape_id1 = shape_id1;
    P.shape_id2 = shape_id2;
    P.params1 = params1;
    P.params2 = params2;

    // --- x0 ---
    if (mxGetM(prhs[6]) != 3 || mxGetN(prhs[6]) != 1) mexErrMsgTxt("x0 must be 3x1.");
    double* x0_ptr = mxGetPr(prhs[6]);
    Vector3d x0;
    x0 << x0_ptr[0], x0_ptr[1], x0_ptr[2];

    // --- alpha0 ---
    double alpha0 = mxGetScalar(prhs[7]);
    if (alpha0 <= 0.0) mexErrMsgTxt("alpha0 must be > 0.");
    double s0 = std::log(alpha0);

    // Optional: lambda10, lambda20, L, max_iters, tol
    double lambda10 = 1.0;
    double lambda20 = 1.0;
    double L = 1.0; //scaling
    int    max_iters = 30;
    double tol = 1e-10;

    if (nrhs >= 9)  lambda10  = mxGetScalar(prhs[8]);
    if (nrhs >= 10) lambda20  = mxGetScalar(prhs[9]);
    if (nrhs >= 11) L         = mxGetScalar(prhs[10]);
    if (nrhs >= 12) max_iters = static_cast<int>(mxGetScalar(prhs[11]));
    if (nrhs >= 13) tol       = mxGetScalar(prhs[12]);

    if (!(L > 0.0) || !std::isfinite(L)) L = 1.0; //to avoid user nonsense

    // Newton parameters
    const int    max_ls   = 8;
    const double beta_ls  = 0.5;
    const double c_armijo = 1e-4;

    // scaling
    Eigen::Matrix<double,6,1> D;
    D << L, L, L, 1.0, 1.0, 1.0;

    // trust-ish limits (soft, no projection)
    const double max_step_s = 1.0;     // limit |Δs| per accepted step
    const double max_step   = 1.0 * L; // limit overall step norm (units mostly meters)

    auto run_newton = [&](Vector3d& x, double& s, double& lambda1, double& lambda2,
                          Vector6d& F, Matrix6d& J) -> bool
    {
        bool solved = false;

        for (int iter = 0; iter < max_iters; ++iter) {

            idcol::eval_F_J(x, s, lambda1, lambda2, P, F, J);

            const double Fn = F.norm();
            if (Fn < tol) { solved = true; break; }

            // Merit m(z) = 0.5 ||F||^2 and gradient g = J^T F
            const double m0 = 0.5 * F.squaredNorm();
            Vector6d g_m;
            g_m.noalias() = J.transpose() * F;


            // Build JD = J * diag(D)
            Matrix6d JD = J;
            for (int k = 0; k < 6; ++k) JD.col(k) *= D(k);

            // Compute step in scaled coords: JD * dz_hat = -F
            Vector6d dz_hat;
            bool have_step = false;

            /*Eigen::FullPivLU<Matrix6d> lu(JD); //Partial is enuf! otherwise replace
            if (lu.isInvertible()) {
                dz_hat = lu.solve(-F);
                have_step = true;
            }*/
            
            Eigen::PartialPivLU<Matrix6d> lu(JD);
            dz_hat = lu.solve(-F);

            // cheap sanity check
            if (dz_hat.allFinite() && dz_hat.norm() <= 1e6) { have_step = true; }

            // LM fallback if Newton solve looks singular
            if (!have_step) {
                Vector6d rhs; rhs.noalias() = -JD.transpose() * F;
                double mu = 1e-8;
                for (int k = 0; k < 10; ++k) {
                    Matrix6d A = JD.transpose() * JD + mu * Matrix6d::Identity();
                    Eigen::LLT<Matrix6d> llt(A);
                    if (llt.info() == Eigen::Success) {
                        dz_hat = llt.solve(rhs);
                        have_step = true;
                        break;
                    }
                    mu *= 10.0;
                }
                if (!have_step) {
                    mexWarnMsgTxt("Newton: LM fallback failed (LLT). Stopping.\n");
                    break;
                }
            }

            // Unscale: dz = diag(D) * dz_hat
            Vector6d dz = D.array() * dz_hat.array();

            // Soft step limits (no projection)
            if (std::isfinite(dz(3)) && std::abs(dz(3)) > max_step_s) {
                dz *= (max_step_s / std::abs(dz(3)));
            }
            double dz_norm = dz.norm();
            if (std::isfinite(dz_norm) && dz_norm > max_step) {
                dz *= (max_step / dz_norm);
            }

            // Ensure descent for merit
            double slope = g_m.dot(dz);
            if (!(slope < 0.0) || !std::isfinite(slope)) {
                // fall back to steepest descent
                Vector6d dz_sd = -g_m;
                double slope_sd = g_m.dot(dz_sd); // = -||g_m||^2
                if (!(slope_sd < 0.0) || !std::isfinite(slope_sd)) {
                    mexWarnMsgTxt("Newton: Non-descent direction and invalid gradient. Stopping.\n");
                    break;
                }
                dz = dz_sd;
                slope = slope_sd;

                // limit steepest-descent too
                if (std::isfinite(dz(3)) && std::abs(dz(3)) > max_step_s) {
                    dz *= (max_step_s / std::abs(dz(3)));
                }
                dz_norm = dz.norm();
                if (std::isfinite(dz_norm) && dz_norm > max_step) {
                    dz *= (max_step / dz_norm);
                }
            }

            // Backtracking line search on m(z)=0.5||F||^2
            double t = 1.0;
            bool accepted = false;

            Vector3d x_trial;
            double s_trial, lambda1_trial, lambda2_trial;
            Vector6d F_trial;

            for (int ls = 0; ls < max_ls; ++ls) {

                x_trial       = x + t * dz.segment<3>(0);
                s_trial       = s + t * dz(3);
                lambda1_trial = lambda1 + t * dz(4);
                lambda2_trial = lambda2 + t * dz(5);

                idcol::eval_F(x_trial, s_trial, lambda1_trial, lambda2_trial, P, F_trial);

                const double m_trial = 0.5 * F_trial.squaredNorm();

                // Armijo: m(z+t dz) <= m0 + c t slope
                if (std::isfinite(m_trial) && (m_trial <= m0 + c_armijo * t * slope)) {
                    accepted = true;
                    break;
                }

                t *= beta_ls;
            }

            if (!accepted) {
                // ---------- Rescue: LM damping sweep (only when Armijo fails) ----------
                const double m0 = 0.5 * F.squaredNorm();
                Vector6d g;   g.noalias()   = J.transpose() * F;
                Matrix6d JTJ;   JTJ.noalias() = J.transpose() * J;

                bool rescued = false;

                double mu = 1e-3;  // start damped
                for (int tr = 0; tr < 10; ++tr) {

                    Matrix6d A = JTJ + mu * Matrix6d::Identity();
                    Eigen::LLT<Matrix6d> llt(A);
                    if (llt.info() != Eigen::Success) {
                        mu *= 10.0;
                        continue;
                    }

                    Vector6d dz_r = llt.solve(-g);

                    // soft step limits (use your existing max_step_s/max_step)
                    if (std::isfinite(dz_r(3)) && std::abs(dz_r(3)) > max_step_s)
                        dz_r *= (max_step_s / std::abs(dz_r(3)));

                    double dzr_norm = dz_r.norm();
                    if (std::isfinite(dzr_norm) && dzr_norm > max_step)
                        dz_r *= (max_step / dzr_norm);

                    // trial
                    Vector3d x_r = x + dz_r.segment<3>(0);
                    double   s_r = s + dz_r(3);
                    double   l1_r = lambda1 + dz_r(4);
                    double   l2_r = lambda2 + dz_r(5);

                    Vector6d F_r;
                    idcol::eval_F(x_r, s_r, l1_r, l2_r, P, F_r);

                    const double m_r = 0.5 * F_r.squaredNorm();

                    // ACCEPT: any real decrease is good in rescue mode
                    if (std::isfinite(m_r) && (m_r < m0)) {
                        x = x_r;
                        s = s_r;
                        lambda1 = l1_r;
                        lambda2 = l2_r;
                        rescued = true;
                        break;
                    }

                    mu *= 10.0;
                }

                if (!rescued) {
                    mexWarnMsgTxt("Newton: Armijo failed; rescue failed.\n");
                    break;
                }

                // rescued: continue main iterations
                continue;
            }


            // accept
            x       = x_trial;
            s       = s_trial;
            lambda1 = lambda1_trial;
            lambda2 = lambda2_trial;

            if ((t * dz).norm() < 1e-14) break;
        }
    return solved;
    };

    // Initial guess
    Vector3d x = x0;
    double s = s0;
    double lambda1 = lambda10, lambda2 = lambda20;
    Vector6d F;
    Matrix6d J;

    bool converged = run_newton(x, s, lambda1, lambda2, F, J);

    // Only do restart logic if first try failed (fast path stays fast)
    if (!converged) {

        Vector3d best_x = x;
        double   best_s = s;
        double   best_l1 = lambda1, best_l2 = lambda2;
        double   best_Fn = F.norm();

        auto attempt = [&](double s_init) -> bool {
            Vector3d x_try = x0;
            double   s_try = s_init;
            double   l1_try = lambda10, l2_try = lambda20;
            Vector6d F_try;
            Matrix6d J_try;

            bool ok = run_newton(x_try, s_try, l1_try, l2_try, F_try, J_try);

            const double Fn = F_try.norm();
            if (std::isfinite(Fn) && Fn < best_Fn) {
                best_Fn = Fn;
                best_x = x_try; best_s = s_try; best_l1 = l1_try; best_l2 = l2_try;
            }

            if (ok) {
                x = x_try; s = s_try; lambda1 = l1_try; lambda2 = l2_try;
                F = F_try; J = J_try;
            }

            return ok;
        };

        // Try #2 and #3
        converged = attempt(s0 + 0.1);
        if (!converged) converged = attempt(s0 - 0.1);

        // Try #4: go further in the better direction (based on best residual so far)
        if (!converged) {
            const double dir = (best_s > s0) ? 1.0 : ((best_s < s0) ? -1.0 : 0.0);
            if (dir != 0.0) converged = attempt(s0 + 0.2 * dir);
        }

        // Try #5: go further in the better direction (based on best residual so far)
        if (!converged) {
            const double dir = (best_s > s0) ? 1.0 : ((best_s < s0) ? -1.0 : 0.0);
            converged = attempt(s0 + 0.5 * dir);
        }

        // If still not converged: return best solution found + warn
        if (!converged) {
            x = best_x; s = best_s; lambda1 = best_l1; lambda2 = best_l2;
            mexWarnMsgTxt("Newton: not converged after s-restarts. Returning best; need better initial guess.\n");
        }
    }

    // Final evaluation for outputs
    idcol::eval_F_J(x, s, lambda1, lambda2, P, F, J);

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
