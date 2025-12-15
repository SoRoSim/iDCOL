#include "idcol_newton.hpp"
#include <iostream>

namespace idcol {

NewtonResult solve_idcol_newton(
    const ProblemData& P,
    const Vector3d& x0,
    double alpha0,
    double lambda10,
    double lambda20,
    const NewtonOptions& opt_in)
{
    NewtonResult out;
    NewtonOptions opt = opt_in;

    // Basic input sanity
    if (!(alpha0 > 0.0) || !std::isfinite(alpha0)) {
        out.converged = false;
        out.message = "alpha0 must be finite and > 0.";
        return out;
    }
    if (!(opt.L > 0.0) || !std::isfinite(opt.L)) opt.L = 1.0;

    const double s0 = std::log(alpha0);

    // scaling vector D for [x(3), s, lambda1, lambda2]
    Eigen::Matrix<double,6,1> D;
    D << opt.L, opt.L, opt.L, 1.0, 1.0, 1.0;

    const double max_step = opt.max_step_mult * opt.L;

    auto run_newton = [&](Vector3d& x, double& s, double& lambda1, double& lambda2,
                          Vector6d& F, Matrix6d& J,
                          int& iters_used) -> bool
    {
        bool solved = false;
        iters_used = 0;

        for (int iter = 0; iter < opt.max_iters; ++iter) {
            iters_used = iter + 1;

            eval_F_J(x, s, lambda1, lambda2, P, F, J);

            const double Fn = F.norm();
            if (Fn < opt.tol) { solved = true; break; }

            const double m0 = idcol::merit_from_F(F);

            Vector6d g_m;
            g_m.noalias() = J.transpose() * F;

            // Build JD = J * diag(D)
            Matrix6d JD = J;
            for (int k = 0; k < 6; ++k) JD.col(k) *= D(k);

            // Solve JD * dz_hat = -F
            Vector6d dz_hat;
            bool have_step = false;

            Eigen::PartialPivLU<Matrix6d> lu(JD);
            dz_hat = lu.solve(-F);

            if (dz_hat.allFinite() && dz_hat.norm() <= opt.dz_hat_norm_max) {
                have_step = true;
            }

            // LM fallback if LU step is bad
            if (!have_step) {
                Vector6d rhs;
                rhs.noalias() = -JD.transpose() * F;

                double mu = 1e-8;
                for (int k = 0; k < 10; ++k) {
                    Matrix6d A;
                    A.noalias() = JD.transpose() * JD;
                    A.diagonal().array() += mu;

                    Eigen::LLT<Matrix6d> llt(A);
                    if (llt.info() == Eigen::Success) {
                        dz_hat = llt.solve(rhs);
                        have_step = dz_hat.allFinite();
                        if (have_step) break;
                    }
                    mu *= 10.0;
                }
                if (!have_step) {
                    if (opt.verbose) std::cerr << "[idcol] LM fallback failed (LLT)\n";
                    break;
                }
            }

            // Unscale: dz = diag(D) * dz_hat
            Vector6d dz = D.array() * dz_hat.array();

            // Soft step limits
            if (std::isfinite(dz(3)) && std::abs(dz(3)) > opt.max_step_s)
                dz *= (opt.max_step_s / std::abs(dz(3)));

            double dz_norm = dz.norm();
            if (std::isfinite(dz_norm) && dz_norm > max_step)
                dz *= (max_step / dz_norm);

            // Ensure descent direction for merit
            double slope = g_m.dot(dz);
            if (!(slope < 0.0) || !std::isfinite(slope)) {
                Vector6d dz_sd = -g_m;
                double slope_sd = g_m.dot(dz_sd);

                if (!(slope_sd < 0.0) || !std::isfinite(slope_sd)) {
                    if (opt.verbose) std::cerr << "[idcol] Non-descent and invalid gradient\n";
                    break;
                }

                dz = dz_sd;
                slope = slope_sd;

                if (std::isfinite(dz(3)) && std::abs(dz(3)) > opt.max_step_s)
                    dz *= (opt.max_step_s / std::abs(dz(3)));

                dz_norm = dz.norm();
                if (std::isfinite(dz_norm) && dz_norm > max_step)
                    dz *= (max_step / dz_norm);
            }

            // Backtracking line search on m(z) = 0.5 ||F||^2
            double t = 1.0;
            bool accepted = false;

            Vector3d x_trial;
            double s_trial, l1_trial, l2_trial;
            Vector6d F_trial;

            for (int ls = 0; ls < opt.max_ls; ++ls) {
                x_trial = x + t * dz.segment<3>(0);
                s_trial = s + t * dz(3);
                l1_trial = lambda1 + t * dz(4);
                l2_trial = lambda2 + t * dz(5);

                eval_F(x_trial, s_trial, l1_trial, l2_trial, P, F_trial);

                const double m_trial = idcol::merit_from_F(F_trial);
                if (std::isfinite(m_trial) && (m_trial <= m0 + opt.c_armijo * t * slope)) {
                    accepted = true;
                    break;
                }
                t *= opt.beta_ls;
            }

            if (!accepted) {
                // Rescue: LM sweep on unscaled J 
                const double m0r = idcol::merit_from_F(F);

                Vector6d g;
                g.noalias() = J.transpose() * F;

                Matrix6d JTJ;
                JTJ.noalias() = J.transpose() * J;

                bool rescued = false;
                double mu = 1e-3;

                for (int tr = 0; tr < 10; ++tr) {
                    Matrix6d A = JTJ;
                    A.diagonal().array() += mu;

                    Eigen::LLT<Matrix6d> llt(A);
                    if (llt.info() != Eigen::Success) {
                        mu *= 10.0;
                        continue;
                    }

                    Vector6d dz_r = llt.solve(-g);

                    if (std::isfinite(dz_r(3)) && std::abs(dz_r(3)) > opt.max_step_s)
                        dz_r *= (opt.max_step_s / std::abs(dz_r(3)));

                    double dzr_norm = dz_r.norm();
                    if (std::isfinite(dzr_norm) && dzr_norm > max_step)
                        dz_r *= (max_step / dzr_norm);

                    Vector3d x_r = x + dz_r.segment<3>(0);
                    double   s_r = s + dz_r(3);
                    double   l1_r = lambda1 + dz_r(4);
                    double   l2_r = lambda2 + dz_r(5);

                    Vector6d F_r;
                    eval_F(x_r, s_r, l1_r, l2_r, P, F_r);

                    const double m_r = idcol::merit_from_F(F_r);
                    if (std::isfinite(m_r) && (m_r < m0r)) {
                        x = x_r; s = s_r; lambda1 = l1_r; lambda2 = l2_r;
                        rescued = true;
                        break;
                    }

                    mu *= 10.0;
                }

                if (!rescued) {
                    if (opt.verbose) std::cerr << "[idcol] Armijo failed; rescue failed\n";
                    break;
                }

                continue; // rescued step accepted
            }

            // Accept line search step
            x = x_trial;
            s = s_trial;
            lambda1 = l1_trial;
            lambda2 = l2_trial;

            if ((t * dz).norm() < 1e-14) break;
        }

        return solved;
    };

    // Initial guess
    Vector3d x = x0;
    double s = s0;
    double lambda1 = lambda10;
    double lambda2 = lambda20;

    Vector6d F;
    Matrix6d J;
    int iters_used = 0;

    bool converged = run_newton(x, s, lambda1, lambda2, F, J, iters_used);

    int attempts_used = 1;

    // Restart logic (optional)
    if (!converged && opt.enable_restarts) {
        Vector3d best_x = x;
        double   best_s = s;
        double   best_l1 = lambda1, best_l2 = lambda2;
        double   best_Fn = F.norm();

        auto attempt = [&](double s_init) -> bool {
            ++attempts_used;

            Vector3d x_try = x0;
            double s_try = s_init;
            double l1_try = lambda10;
            double l2_try = lambda20;

            Vector6d F_try;
            Matrix6d J_try;
            int it_try = 0;

            bool ok = run_newton(x_try, s_try, l1_try, l2_try, F_try, J_try, it_try);

            const double Fn = F_try.norm();
            if (std::isfinite(Fn) && Fn < best_Fn) {
                best_Fn = Fn;
                best_x = x_try; best_s = s_try; best_l1 = l1_try; best_l2 = l2_try;
            }

            if (ok) {
                x = x_try; s = s_try; lambda1 = l1_try; lambda2 = l2_try;
                F = F_try; J = J_try;
                iters_used = it_try;
            }
            return ok;
        };

        converged = attempt(s0 + opt.s_restart_delta);
        if (!converged) converged = attempt(s0 - opt.s_restart_delta);

        if (!converged) {
            const double dir = (best_s > s0) ? 1.0 : ((best_s < s0) ? -1.0 : 0.0);
            if (dir != 0.0) converged = attempt(s0 + opt.s_restart_extra * dir);
        }

        if (!converged) {
            x = best_x; s = best_s; lambda1 = best_l1; lambda2 = best_l2;
        }
    }

    // Final eval for returned outputs
    eval_F_J(x, s, lambda1, lambda2, P, out.F, out.J);

    out.x = x;
    out.s = s;
    out.alpha = std::exp(s);
    out.lambda1 = lambda1;
    out.lambda2 = lambda2;

    out.converged = converged;
    out.iters_used = iters_used;
    out.attempts_used = attempts_used;
    out.final_F_norm = out.F.norm();

    if (converged) {
        out.message = "converged";
    } else {
        out.message = opt.enable_restarts
            ? "not converged (returned best from restarts)"
            : "not converged (no restarts)";
    }

    return out;
}

} // namespace idcol
