#pragma once

#include <Eigen/Dense>
#include <optional>

#include "idcol_solve.hpp"
#include "idcol_implicitfamily.hpp"   // ShapeSpec

namespace idcol {

class ContactPair {
public:
    ContactPair(const ShapeSpec& s1, const ShapeSpec& s2,
                NewtonOptions opt = NewtonOptions{},
                SurrogateOptions sopt = SurrogateOptions{})
        : opt_(opt), sopt_(sopt)
    {
        S_.P.shape_id1 = s1.shape_id;
        S_.P.shape_id2 = s2.shape_id;
        S_.P.params1   = s1.params;
        S_.P.params2   = s2.params;
        S_.bounds1     = s1.bounds;
        S_.bounds2     = s2.bounds;
    }

    // Solve with automatic warm start (reuses last converged solution)
    SolveResult solve(const Eigen::Matrix4d& g)
    {
        S_.P.g = g;

        const std::optional<Guess> user_guess =
            have_guess_ ? std::optional<Guess>(last_guess_) : std::nullopt;

        SolveResult out = idcol_solve(S_, user_guess, opt_, sopt_);

        if (out.newton.converged) {
            last_guess_.x       = out.newton.x;
            last_guess_.alpha   = out.newton.alpha;
            last_guess_.lambda1 = out.newton.lambda1;
            last_guess_.lambda2 = out.newton.lambda2;
            have_guess_ = true;
        }
        return out;
    }

    void reset_guess()
    {
        have_guess_ = false;
        last_guess_ = Guess{};
    }

    // Optional: expose options for advanced users
    NewtonOptions& newton_options() { return opt_; }
    SurrogateOptions& surrogate_options() { return sopt_; }

    const SolveData& solve_data() const { return S_; }

private:
    SolveData S_;
    NewtonOptions opt_;
    SurrogateOptions sopt_;
    Guess last_guess_;
    bool have_guess_ = false;
};

} // namespace idcol
