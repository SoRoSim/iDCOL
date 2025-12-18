#include "radial_bounds.hpp"
#include "shape_core.hpp"

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <random>
#include <iostream>

static inline Eigen::Vector3d random_unit_vector(std::mt19937& rng)
{
    // Marsaglia method
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    double x1, x2, s;
    do {
        x1 = unif(rng);
        x2 = unif(rng);
        s = x1*x1 + x2*x2;
    } while (s <= 1e-16 || s >= 1.0);

    double f = 2.0 * std::sqrt(1.0 - s);
    return Eigen::Vector3d(x1*f, x2*f, 1.0 - 2.0*s).normalized();
}

// Find y0 = t*u on the surface phi(y0)=0 via bracketing + bisection.
// Assumes origin is inside: phi(0) < 0, and outside corresponds to phi > 0.
static inline bool ray_to_surface(
    const Eigen::Vector3d& u_unit,
    int shape_id,
    const Eigen::VectorXd& params,
    const RadialBoundsOptions& opt,
    Eigen::Vector3d& y0_out)
{
    // bracket t in [0, t_hi] such that phi(0) < 0 and phi(t_hi*u) >= 0
    double t_hi = opt.t0;

    double phi_hi = 0.0;
    Eigen::Vector3d gtmp;

    bool bracketed = false;
    for (int i = 0; i < opt.max_grow_steps; ++i) {
        shape_eval_local_phi_grad(t_hi * u_unit, shape_id, params, phi_hi, gtmp);
        if (phi_hi >= 0.0) { bracketed = true; break; }
        t_hi *= opt.grow;
    }
    if (!bracketed) return false;

    // bisection
    double a = 0.0, b = t_hi;
    for (int it = 0; it < opt.bisect_iters; ++it) {
        double m = 0.5 * (a + b);

        double phi_m;
        shape_eval_local_phi_grad(m * u_unit, shape_id, params, phi_m, gtmp);

        if (std::abs(phi_m) <= opt.init_phi_tol) { a = b = m; break; }
        if (phi_m <= 0.0) a = m;
        else b = m;
    }

    double t = 0.5 * (a + b);
    y0_out = t * u_unit;

    // final check
    double phi_end;
    shape_eval_local_phi_grad(y0_out, shape_id, params, phi_end, gtmp);
    return std::abs(phi_end) <= 10.0 * opt.init_phi_tol;
}

// Plain Newton on KKT system:
// F(y,k) = [ y + k*grad_phi(y) ; phi(y) ] = 0
static inline bool newton_kkt_plain(
    int shape_id,
    const Eigen::VectorXd& params,
    const RadialBoundsOptions& opt,
    Eigen::Vector3d& y,
    double& k)
{
    for (int it = 0; it < opt.newton_iters; ++it) {
        double phi;
        Eigen::Vector3d grad;
        Eigen::Matrix3d H;
        shape_eval_local(y, shape_id, params, phi, grad, H);
        
        Eigen::Vector3d r1 = y + k * grad;

        Eigen::Vector4d F;
        F.head<3>() = r1;
        F(3) = phi;

        if (F.norm() <= opt.F_tol) return true;

        Eigen::Matrix4d J = Eigen::Matrix4d::Zero();
        J.block<3,3>(0,0) = Eigen::Matrix3d::Identity() + k * H;
        J.block<3,1>(0,3) = grad;
        J.block<1,3>(3,0) = grad.transpose();

        Eigen::Vector4d delta = J.fullPivLu().solve(-F);
        if (!delta.allFinite()) return false;

        y += delta.head<3>();
        k += delta(3);
    }
    return false;
}

RadialBounds compute_radial_bounds_local(
    int shape_id,
    const Eigen::VectorXd& params,
    const RadialBoundsOptions& opt)
{
    // must contain origin (for ray bracketing)
    {
        double phi0;
        Eigen::Vector3d g0;
        shape_eval_local_phi_grad(Eigen::Vector3d::Zero(), shape_id, params, phi0, g0);
        if (!(phi0 < 0.0)) {
            throw std::runtime_error("compute_radial_bounds_local: requires phi(0) < 0 (origin inside).");
        }
    }

    double best_min2 = std::numeric_limits<double>::infinity();
    double best_max2 = 0.0;
    Eigen::Vector3d best_xin = Eigen::Vector3d::Zero();
    Eigen::Vector3d best_xout = Eigen::Vector3d::Zero();

    int n_converged = 0;

    std::mt19937 rng(opt.rng_seed);

    for (int s = 0; s < opt.num_starts; ++s) {
        // 1) random direction
        Eigen::Vector3d u = random_unit_vector(rng);

        // 2) project to surface
        Eigen::Vector3d y0;
        if (!ray_to_surface(u, shape_id, params, opt, y0)) continue;
        //std::cout << "y0 = " << y0.transpose() << "\n";

        // 3) seed k0 from y + k*grad = 0 => k = -(y·g)/(g·g)
        double phi_init;
        Eigen::Vector3d g_init;
        shape_eval_local_phi_grad(y0, shape_id, params, phi_init, g_init);
        double gg = g_init.squaredNorm();
        if (!(gg > 0.0) || !std::isfinite(gg)) continue;
        double k0 = -(y0.dot(g_init)) / gg;

        //std::cout << "k0 = " << k0 << "\n";

        // 4) Newton
        Eigen::Vector3d y = y0;
        double k = k0;

        if (!newton_kkt_plain(shape_id, params, opt, y, k)) continue;

        //std::cout << "y = " << y.transpose() << "\n";

        // 5) accept only if ||F|| small (re-check with fresh grad/H if you want)
        double phi_chk;
        Eigen::Vector3d g_chk;
        shape_eval_local_phi_grad(y, shape_id, params, phi_chk, g_chk);
        Eigen::Vector3d r1_chk = y + k * g_chk;
        Eigen::Vector4d F_chk;
        F_chk.head<3>() = r1_chk;
        F_chk(3) = phi_chk;

        if (F_chk.norm() > opt.F_tol) continue;

        // accept
        n_converged++;

        double r2 = y.squaredNorm();
        if (!std::isfinite(r2)) continue;

        if (r2 < best_min2) { best_min2 = r2; best_xin = y; }
        if (r2 > best_max2) { best_max2 = r2; best_xout = y; }
    }

    if (n_converged == 0 || !std::isfinite(best_min2) || !(best_max2 > 0.0)) {
        throw std::runtime_error("compute_radial_bounds_local: no converged KKT solutions (increase num_starts or relax F_tol).");
    }

    RadialBounds out;
    out.Rin2 = best_min2;
    out.Rout2 = best_max2;
    out.Rin  = std::sqrt(best_min2);
    out.Rout = std::sqrt(best_max2);
    out.xin  = best_xin;
    out.xout = best_xout;
    return out;
}
