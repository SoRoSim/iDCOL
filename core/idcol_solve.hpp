// idcol_solve.hpp
#pragma once

#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <limits>
#include <cmath>
#include <string>

#include "shape_core.hpp"
#include "idcol_newton.hpp"
#include "radial_bounds.hpp"


namespace idcol {

struct SolveData {
    ProblemData P;
    RadialBounds bounds1;
    RadialBounds bounds2;
};    

inline bool shape_uses_n(int shape_id) {
    return (shape_id == 3) || (shape_id == 4); // SE or SEC
}

inline int get_integer_n_or_default(const Eigen::VectorXd& params, int default_n = 2) {
    if (params.size() == 0) return default_n;
    const double n_raw = params[0];
    const int n = static_cast<int>(std::round(n_raw));
    if (n <= 0 || std::fabs(n_raw - n) > 1e-9) return default_n;
    return n;
}

inline void set_n_in_params(Eigen::VectorXd& params, int n) {
    if (params.size() > 0) params[0] = double(n);
}


struct Guess {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();   // in body-1 frame
    double alpha   = 1.0;
    double lambda1 = 0.0;
    double lambda2 = 0.0;
};

struct SurrogateOptions {
    // Default schedule: try fS=1, then fS=3, 5, 7 progressively if not converged
    std::vector<int> fS_values = {1, 3, 5, 7};
};

struct SolveResult {
    NewtonResult newton;          // NewtonResult semantics untouched (attempts_used remains Newton-internal)
    int  fS_used = 1;             // which fS produced the returned result
    int  fS_attempts_used = 0;    // how many fS values were tried by this wrapper
    bool used_surrogate = false;  // true if we ran via the surrogate transform
};

// Helper: map a guess from original space -> surrogate space with scale_factor = fS/alpha_min
inline Guess map_guess_to_surrogate(const Guess& g, double scale_factor) {
    Guess gs = g;
    gs.x       = g.x * scale_factor;
    gs.alpha   = g.alpha * scale_factor;
    gs.lambda1 = g.lambda1 * scale_factor;
    gs.lambda2 = g.lambda2 * scale_factor;
    return gs;
}

// Helper: map solution from surrogate -> original
inline void map_solution_to_original(NewtonResult& res, double scale_factor) {
    res.x       /= scale_factor;
    res.alpha   /= scale_factor;
    res.lambda1 /= scale_factor;
    res.lambda2 /= scale_factor;
    res.J *= scale_factor; // surrogate problem has same F.
}

// A thin wrapper around solve_idcol_newton:
// - optional user guess (original space)
// - otherwise auto-guess in surrogate space
// - try fS schedule (default {1,3}) until converged or exhausted
inline SolveResult idcol_solve(const SolveData& S,
                              const NewtonOptions& opt_in,
                              const std::optional<Guess>& user_guess = std::nullopt,
                              const SurrogateOptions& sopt = SurrogateOptions{})
{
    const ProblemData& P_in = S.P;
    const RadialBounds& bounds1 = S.bounds1;
    const RadialBounds& bounds2 = S.bounds2;

    static thread_local int solve_depth = 0; // to avoid recursion

    struct SolveDepthGuard {
        int& d;
        SolveDepthGuard(int& d_) : d(d_) { ++d; }
        ~SolveDepthGuard() { --d; }
    };

    SolveDepthGuard _depth_guard(solve_depth);
    
    SolveResult out;

    // Work on a local copy because we overwrite P.g2 for the surrogate problem
    ProblemData P = P_in;

    // Original relative translation (in g2)
    const Eigen::Matrix4d g2 = P.g2;
    const Eigen::Vector3d r0 = g2.topRightCorner<3,1>();
    const double d0 = r0.norm();

    // Degenerate: coincident origins -> cannot define direction u = r/d
    if (!std::isfinite(d0) || d0 <= 1e-15) {
        out.newton.converged    = false;
        out.newton.iters_used   = 0;
        out.newton.final_F_norm = std::numeric_limits<double>::infinity();
        out.newton.message =
            "idcol_solve: degenerate relative translation (||r|| ~ 0). "
            "Cannot build surrogate (u = r/||r||).";

        out.used_surrogate   = false;
        out.fS_used          = (sopt.fS_values.empty() ? 1 : sopt.fS_values.front());
        out.fS_attempts_used = 0;
        return out;
    }

    const Eigen::Vector3d u = r0 / d0;

    // Compute alpha bounds (original)
    const double alpha_min = d0 / (bounds1.Rout + bounds2.Rout);
    const double alpha_max = d0 / (bounds1.Rin  + bounds2.Rin);

    auto attempt = [&](int fS)->NewtonResult {
        NewtonOptions opt = opt_in;

        // Surrogate scaling: rS = r0 / (alpha_min / fS) = r0 * (fS/alpha_min)
        const double scale_factor = static_cast<double>(fS) / alpha_min;

        // Build surrogate g2S with controlled separation
        Eigen::Matrix4d g2S = g2;
        Eigen::Vector3d rS  = u * (bounds1.Rout + bounds2.Rout) * static_cast<double>(fS);
        g2S.topRightCorner<3,1>() = rS;
        P.g2 = g2S;

        // Scaled alpha bounds
        const double alpha_min_scaled = static_cast<double>(fS);
        const double alpha_max_scaled = (alpha_max / alpha_min) * static_cast<double>(fS);

        opt.s_min = std::log(alpha_min_scaled);
        opt.s_max = std::log(alpha_max_scaled);

        // ----------------- Initial guess -----------------
        Guess g0;

        if (user_guess.has_value()) {
            // User guess is in original space; map to surrogate space
            g0 = map_guess_to_surrogate(*user_guess, scale_factor);
        } else {
            // Auto guess directly in surrogate space (matches your snippet)
            g0.x = bounds1.Rout * u;
            g0.alpha = std::sqrt(alpha_min_scaled * alpha_max_scaled);

            double phi_tmp;
            Eigen::Vector4d grad_tmp;

            // lambda1 from stationarity on body 1
            shape_eval_global_xa_phi_grad(P.g1, g0.x, g0.alpha,
                                          P.shape_id1, P.params1, phi_tmp, grad_tmp);
            const double denom1 = rS.dot(grad_tmp.head<3>());
            if (std::abs(denom1) < 1e-14 || !std::isfinite(denom1)) {
                g0.lambda1 = 1.0; // fallback
            } else {
                g0.lambda1 = (g0.alpha) / denom1;
            }

            // lambda2 from stationarity on body 2 (surrogate)
            shape_eval_global_xa_phi_grad(P.g2, g0.x, g0.alpha,
                                          P.shape_id2, P.params2, phi_tmp, grad_tmp);
            const double denom2 = rS.dot(grad_tmp.head<3>());
            if (std::abs(denom2) < 1e-14 || !std::isfinite(denom2)) {
                g0.lambda2 = 1.0; // fallback
            } else {
                g0.lambda2 = -(g0.alpha) / denom2;
            }
        }

        // Solve surrogate
        NewtonResult resS = solve_idcol_newton(P, g0.x, g0.alpha, g0.lambda1, g0.lambda2, opt);

        // Map back to original space before returning
        map_solution_to_original(resS, scale_factor);
        return resS;
    };

    // fS schedule (default {1,3})
    out.used_surrogate   = true;
    out.fS_attempts_used = 0;

    // If schedule is empty: behave like fS=1
    if (sopt.fS_values.empty()) {
        out.newton = attempt(1);
        out.fS_used = 1;
        out.fS_attempts_used = 1;
        return out;
    }

    for (std::size_t k = 0; k < sopt.fS_values.size(); ++k) {
        const int fS = sopt.fS_values[k];

        out.newton = attempt(fS);
        out.fS_used = fS;
        out.fS_attempts_used = static_cast<int>(k) + 1;

        if (out.newton.converged) {
            break;
        }
    }

    if (solve_depth == 1 && !out.newton.converged) { //continuation strategy for SE shapes

        const bool body1_has_n = shape_uses_n(P_in.shape_id1);
        const bool body2_has_n = shape_uses_n(P_in.shape_id2);

        const int n1_target = body1_has_n ? get_integer_n_or_default(P_in.params1, 2) : 0;
        const int n2_target = body2_has_n ? get_integer_n_or_default(P_in.params2, 2) : 0;
        const int n_target  = std::max(n1_target, n2_target);

        // trigger only if there is something to continue
        if (n_target > 2 && (body1_has_n || body2_has_n)) {

            std::vector<int> n_schedule;

            // Always start from the easy case
            n_schedule.push_back(2);

            // Intermediate steps (even n)
            for (int nk = 4; nk < n_target; nk += 2) {
                n_schedule.push_back(nk);
            }

            // Ensure we end exactly at n_target (odd or even)
            if (n_schedule.back() != n_target) {
                n_schedule.push_back(n_target);
            }

            // start from the best we already have
            idcol::SolveResult best_out = out;
            double best_norm = out.newton.final_F_norm;

            std::optional<idcol::Guess> guess_k = std::nullopt;

            for (int nk : n_schedule) {
                if (nk > n_target) break;

                // build modified solve data (copy P + bounds)
                idcol::SolveData Sk = S;

                if (body1_has_n) {
                    const int n1k = std::min(nk, n1_target);
                    set_n_in_params(Sk.P.params1, n1k);
                }
                if (body2_has_n) {
                    const int n2k = std::min(nk, n2_target);
                    set_n_in_params(Sk.P.params2, n2k);
                }

                // solve at this nk, warm-starting if we have a guess
                idcol::SolveResult out_k = idcol::idcol_solve(Sk, opt_in, guess_k, sopt);

                // update best
                if (std::isfinite(out_k.newton.final_F_norm) && out_k.newton.final_F_norm < best_norm) {
                    best_out = out_k;
                    best_norm = out_k.newton.final_F_norm;
                }

                if (out_k.newton.converged) {
                    // warm start next stage
                    idcol::Guess g;
                    g.x = out_k.newton.x;
                    g.alpha = out_k.newton.alpha;
                    g.lambda1 = out_k.newton.lambda1;
                    g.lambda2 = out_k.newton.lambda2;
                    guess_k = g;

                    // if we reached target, return this
                    if (nk == n_target) {
                        out = out_k;
                        out.newton.message += " | converged via n-continuation";
                        return out;
                    }
                } else {
                    // if stage fails, still warm start from best found so far
                    idcol::Guess g;
                    g.x = best_out.newton.x;
                    g.alpha = best_out.newton.alpha;
                    g.lambda1 = best_out.newton.lambda1;
                    g.lambda2 = best_out.newton.lambda2;
                    guess_k = g;
                }
            }

            // if continuation didn't fully converge, return best we found
            out = best_out;
            out.newton.message += " | attempted n-continuation (fallback)";
        }
    }

    return out;
}

} // namespace idcol
