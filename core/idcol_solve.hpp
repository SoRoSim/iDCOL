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

// NOTE: make sure the declaration of shape_eval_global_ax_phi_grad(...) is visible
// by including the header where it is declared, e.g.
//   #include "shape_core.hpp"
// or whatever file provides its prototype.

namespace idcol {

struct Guess {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();   // in body-1 frame
    double alpha   = 1.0;
    double lambda1 = 0.0;
    double lambda2 = 0.0;
};

struct SurrogateOptions {
    // Default schedule: try fS=1, then fS=3 if not converged
    std::vector<int> fS_values = {1, 3};
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
}

// A thin wrapper around solve_idcol_newton:
// - optional user guess (original space)
// - otherwise auto-guess in surrogate space
// - try fS schedule (default {1,3}) until converged or exhausted
inline SolveResult idcol_solve(const ProblemData& P_in,
                              const RadialBounds& bounds1,
                              const RadialBounds& bounds2,
                              const NewtonOptions& opt_in,
                              const std::optional<Guess>& user_guess = std::nullopt,
                              const SurrogateOptions& sopt = SurrogateOptions{})
{
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
            shape_eval_global_ax_phi_grad(P.g1, g0.x, g0.alpha,
                                          P.shape_id1, P.params1, phi_tmp, grad_tmp);
            const double denom1 = rS.dot(grad_tmp.head<3>());
            if (std::abs(denom1) < 1e-14 || !std::isfinite(denom1)) {
                g0.lambda1 = 1.0; // fallback
            } else {
                g0.lambda1 = (g0.alpha) / denom1;
            }

            // lambda2 from stationarity on body 2 (surrogate)
            shape_eval_global_ax_phi_grad(P.g2, g0.x, g0.alpha,
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

    return out;
}

} // namespace idcol
