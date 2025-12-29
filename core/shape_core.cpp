#include "shape_core.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iostream>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;



inline double pow_even_int(double u, int n) {
    // u^(2n)
    double u2  = u * u;
    double res = 1.0;
    for (int k = 0; k < n; ++k) res *= u2;
    return res;
}

inline double pow_2n_minus_1(double u, int n) {
    // u^(2n-1)
    if (n == 1) return u;
    double u2  = u * u;
    double res = 1.0;
    for (int k = 0; k < n - 1; ++k) res *= u2;
    return u * res;
}

inline double pow_2n_minus_2(double u, int n) {
    // u^(2n-2)
    if (n == 1) return 1.0;
    double u2  = u * u;
    double res = 1.0;
    for (int k = 0; k < n - 1; ++k) res *= u2;
    return res;
}


void shape_eval_local_phi_grad(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector3d& grad_phi)
{
    const double* p = params.data();
    const std::size_t nParams = static_cast<std::size_t>(params.size());

    phi = 0.0;
    grad_phi.setZero();
    const double eps = 1e-9;

    auto fail = [&](const char* msg){ throw std::runtime_error(msg); };

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1)
    {
        if (nParams < 1) fail("Sphere needs params(1) = radius.");

        double R = p[0];
        double R2inv = 1 / (R * R);
        double r2 = y.squaredNorm();

        phi = r2 * R2inv - 1;
        grad_phi = 2.0 * y * R2inv;
        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Convex polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2)
    {
        if (nParams < 3) fail("Polytope params too short.");

        double beta = p[0];
        int m = (int)p[1];
        double Lscale = p[2];

        if (m <= 0)      fail("Polytope: m must be positive.");
        if (beta <= 0.0) fail("Polytope: beta must be > 0.");
        if (Lscale <= 0.0) fail("Polytope: Lscale must be > 0.");

        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (nParams != expected)
            fail("Polytope params size must be 3 + 4*m.");

        const double* A_data = p + 3;
        const double* b_data = p + 3 + 3*m;

        Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,3>> A(A_data, m, 3);
        Eigen::Map<const Eigen::VectorXd> b(b_data, m);

        // 1) find zmax = max_i (a_i' y - b_i)
        double zmax = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            double zi = A.row(i).dot(y) - b(i);
            if (zi > zmax) zmax = zi;
        }

        // 2) accumulate sum_ez and grad in one pass
        double sum_ez = 0.0;
        grad_phi.setZero();
        for (int i = 0; i < m; ++i) {
            double zi = A.row(i).dot(y) - b(i);
            double wi_unnorm = std::exp(beta * (zi - zmax));
            sum_ez += wi_unnorm;
            grad_phi.noalias() += wi_unnorm * A.row(i).transpose();
        }

        // 3) finalize phi and normalize grad
        phi = zmax + (1.0 / beta) * std::log(sum_ez);

        if (sum_ez > 0.0) {
            grad_phi /= sum_ez;
        } else {
            // extremely pathological case; shouldn't really happen
            grad_phi.setZero();
        }

        phi /= Lscale;
        grad_phi /= Lscale;

        return;
    }

    // ---------------------------------
    // shape_id = 3 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 3) {
        if (nParams < 4) {
            fail("Superellipsoid needs params = [n; a; b; c].");
        }

        double n_raw = p[0];
        double a     = p[1];
        double b     = p[2];
        double c     = p[3];

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            fail("Superellipsoid: n must be a positive integer.");
        }

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        const int e = 2 * n;  // exponent m = 2n

        // precompute inverses
        const double inv_a = 1.0 / a;
        const double inv_b = 1.0 / b;
        const double inv_c = 1.0 / c;

        // normalized coords
        const double u1 = x1 * inv_a;
        const double u2 = x2 * inv_b;
        const double u3 = x3 * inv_c;

        // S = sum u^e
        const double t1 = pow_even_int(u1, n);  // u1^(2n)
        const double t2 = pow_even_int(u2, n);
        const double t3 = pow_even_int(u3, n);

        double S = t1 + t2 + t3;

        // ---- NEW: phi = S^(1/e) - 1 (L_{2n} norm - 1) ----
        // Protect fractional powers near S=0 (only relevant deep inside)
        const double S_eps = 1e-16;
        const double S_safe = (S > S_eps) ? S : S_eps;

        const double inv_e = 1.0 / double(e);
        const double q = std::pow(S_safe, inv_e);   // q = S^(1/e)

        phi = q - 1.0;

        // ---- Derivatives ----
        // q'(S)  = (1/e) S^(1/e - 1)
        const double qprime  = inv_e * std::pow(S_safe, inv_e - 1.0);

        // dS/dx_i = e * u_i^(e-1) * (1/s_i) = e * u_i^(2n-1) * inv_s
        const double dSdx1 = double(e) * pow_2n_minus_1(u1,n) * inv_a;
        const double dSdx2 = double(e) * pow_2n_minus_1(u2,n) * inv_b;
        const double dSdx3 = double(e) * pow_2n_minus_1(u3,n) * inv_c;

        // grad phi = q'(S) * grad S
        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        return;
    }


    // ---------------------------------
    // shape_id = 4 : Superelliptic cylinder
    // ---------------------------------
    else if (shape_id == 4) {
        if (nParams < 3) {
            fail("Superelliptic cylinder needs params = [n; R; h].");
        }

        double n_raw = p[0];
        double R     = p[1];
        double h     = p[2];

        if (R <= 0.0 || h <= 0.0) {
            fail("Superelliptic cylinder: R, h must be > 0.");
        }

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            fail("Superelliptic cylinder: n must be a positive integer.");
        }

        const int e = 2 * n;                 // exponent m = 2n
        const double inv_e = 1.0 / double(e);

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        const double r2 = x2 * x2 + x3 * x3;

        // inverses
        const double inv_R  = 1.0 / R;
        const double inv_R2 = inv_R * inv_R;
        const double inv_h  = 1.0 / h;

        // ---- Build S = axial + radial ----
        // axial: (x1/h)^(2n)
        const double ua = x1 * inv_h;
        const double Sa = pow_even_int(ua,n);

        // radial: (r/R)^(2n) = (r2/R^2)^n
        double Sr = 0.0;
        if (r2 > eps) {
            const double q = r2 * inv_R2;      // (r/R)^2
            double qn = 1.0;
            for (int k = 0; k < n; ++k) qn *= q; // q^n = (r/R)^(2n)
            Sr = qn;
        } else {
            Sr = 0.0;
        }

        double S = Sa + Sr;

        // phi = S^(1/e) - 1 
        const double S_eps = 1e-16;
        const double S_safe = (S > S_eps) ? S : S_eps;

        const double qS = std::pow(S_safe, inv_e); // S^(1/e)
        phi = qS - 1.0;

        const double qprime  = inv_e * std::pow(S_safe, inv_e - 1.0);
        const double qsecond = inv_e * (inv_e - 1.0) * std::pow(S_safe, inv_e - 2.0);

        // ---- Derivatives of S ----
        // Axial:
        // dSa/dx1  = e * (x1/h)^(e-1) * (1/h)
        double dSdx1  = double(e) * pow_2n_minus_1(ua,n) * inv_h;

        // Radial:
        // Sr = (r/R)^e
        // dSr/dx2 = e * Sr * x2 / r2
        // dSr/dx3 = e * Sr * x3 / r2
        double dSdx2 = 0.0, dSdx3 = 0.0;

        if (r2 > eps && Sr > 0.0) {
            const double r4 = r2 * r2;
            const double coeff = double(e) * Sr / r2;

            dSdx2 = coeff * x2;
            dSdx3 = coeff * x3;
        }

        // ---- Gradient of phi ----
        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        return;
    }

    // ---------------------------------
    // shape_id = 5 : Truncated cone
    // ---------------------------------
   else if (shape_id == 5)
    {
        if (nParams < 5)
            fail("Truncated cone needs params = [beta; Rb; Rt; a; b].");
        
        double beta = p[0];
        double Rb   = p[1];
        double Rt   = p[2];
        double a    = p[3];
        double b    = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0)
            fail("Truncated cone params must all be > 0.");

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;
        double h  = a + b;

        // linear interpolation of radius along x1
        double t  = (x1 + a) / h;        // in [0,1] ideally
        double Rx = Rb + (Rt - Rb) * t;
        double Rx2 = Rx * Rx;
        double Rx3 = Rx2 * Rx;

        double phi_side;
        Vector3d grad_side = Vector3d::Zero();

        if (Rx > 0.0 && r2 > 0.0) {
            // side surface implicit: r^2 / Rx^2 - 1 = 0
            phi_side = r2 / Rx2 - 1.0;

            double inv_Rx2 = 1.0 / Rx2;
            grad_side(1) = 2.0 * x2 * inv_Rx2;
            grad_side(2) = 2.0 * x3 * inv_Rx2;

            double dRx_dx1 = (Rt - Rb) / h;
            grad_side(0) = -2.0 * r2 / Rx3 * dRx_dx1;
        } else {
            // safely "inside" or degenerate case: treat side as inactive
            phi_side = -1.0;
        }

        double phi_bot = -x1 / a - 1;  // bottom plane
        double phi_top =  x1 / b - 1;  // top plane

        Vector3d grad_bot(-1.0 / a, 0.0, 0.0);
        Vector3d grad_top( 1.0 / b, 0.0, 0.0);

        // smooth-max of [phi_side, phi_bot, phi_top]
        double mphi = std::max(phi_side, std::max(phi_bot, phi_top));

        double e_side = std::exp(beta * (phi_side - mphi));
        double e_bot  = std::exp(beta * (phi_bot  - mphi));
        double e_top  = std::exp(beta * (phi_top  - mphi));

        double sum_e = e_side + e_bot + e_top;

        phi = mphi + (1.0 / beta) * std::log(sum_e);

        double inv_sum_e = 1.0 / sum_e;
        double w_side = e_side * inv_sum_e;
        double w_bot  = e_bot  * inv_sum_e;
        double w_top  = e_top  * inv_sum_e;

        grad_phi = w_side * grad_side + w_bot * grad_bot + w_top * grad_top;
        return;
    }

    else
        fail("Unknown shape_id (valid: 1–5).");
}

void shape_eval_global_ax_phi_grad(
    const Matrix4d& g,
    const Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Vector4d& grad)
{
    if (alpha <= 0.0) throw std::runtime_error("shape_eval_global_ax_phi_grad: alpha must be > 0.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // y = R' * (x - r) / alpha
    Vector3d y = R.transpose() * (x - r) / alpha;

    // local shape evaluation
    double phi_local = 0.0;
    Vector3d grad_y  = Vector3d::Zero();
    shape_eval_local_phi_grad(y, shape_id, params, phi_local, grad_y);

    phi = phi_local;

    // ---------------------------------------------------
    // gradient wrt x
    // dphi/dx = (1/alpha) * R * grad_y
    // ---------------------------------------------------
    Vector3d grad_x = (1.0/alpha) * (R * grad_y);

    // ---------------------------------------------------
    // gradient wrt alpha
    // dphi/dalpha = - (1/alpha) * y' * grad_y
    // ---------------------------------------------------
    double grad_alpha = -(1.0/alpha) * y.dot(grad_y);

    // pack
    grad.segment<3>(0) = grad_x;
    grad(3) = grad_alpha;
}

void shape_eval_local(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector3d& grad_phi,
    Matrix3d& hess_phi)
{
    const double* p = params.data();
    std::size_t nParams = params.size();

    phi = 0.0;
    grad_phi.setZero();
    hess_phi.setZero();
    const double eps = 1e-9;

    auto fail = [&](const char* msg){ throw std::runtime_error(msg); };

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1) {
        if (nParams < 1) {
            fail("Sphere needs params(1) = radius.");
        }
        double R = p[0];
        double R2inv = 1 / (R * R);
        double r2 = y.squaredNorm();
        phi = r2 * R2inv - 1;        // phi(y) = ||y||^2 / R^2 - 1
        grad_phi = 2.0 * y * R2inv;      // ∂phi/∂y = 2y / R^2
        hess_phi = 2.0 * Matrix3d::Identity() * R2inv; // ∂²phi/∂y² = 2I₃ / R^2

        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Convex polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2) {
        if (nParams < 3) {
            fail("Polytope params too short.");
        }

        double beta = p[0];
        int m = static_cast<int>(p[1]);
        double Lscale = p[2];

        if (m <= 0)      fail("Polytope: m must be positive.");
        if (beta <= 0.0) fail("Polytope: beta must be > 0.");
        if (Lscale <= 0.0) fail("Polytope: Lscale must be > 0.");

        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (nParams != expected)
            fail("Polytope params size must be 3 + 4*m.");

        const double* A_data = p + 3;
        const double* b_data = p + 3 + 3*m;

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3>> A(A_data, m, 3);
        Eigen::Map<const Eigen::VectorXd> b(b_data, m);

        // -----------------------------
        // 1) Find zmax = max_i (a_i' y - b_i)
        // -----------------------------
        double zmax;
        {
            // use the first face to initialize
            Eigen::Vector3d a0 = A.row(0).transpose();
            double z0 = a0.dot(y) - b(0);
            zmax = z0;

            for (int i = 1; i < m; ++i) {
                Eigen::Vector3d ai = A.row(i).transpose();
                double zi = ai.dot(y) - b(i);
                if (zi > zmax) {
                    zmax = zi;  
                }
            }
        }

        // -----------------------------
        // 2) Accumulate sum_u, grad, and M
        //    u_i = exp(beta * (z_i - zmax))
        //    G   = sum u_i * a_i
        //    M   = sum u_i * a_i a_i'
        // -----------------------------
        double sum_u = 0.0;
        grad_phi.setZero();
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

        for (int i = 0; i < m; ++i) {
            Eigen::Vector3d ai = A.row(i).transpose();
            double zi = ai.dot(y) - b(i);
            double ui = std::exp(beta * (zi - zmax));  // unnormalized weight

            sum_u += ui;
            grad_phi.noalias() += ui * ai;
            M.noalias()        += ui * (ai * ai.transpose());
        }

        // -----------------------------
        // 3) Finalize phi, grad, hess
        // -----------------------------
        if (sum_u <= 0.0) {
            // extremely pathological; treat as "far inside"
            phi = zmax;      // arbitrary; shouldn't really happen
            grad_phi.setZero();
            hess_phi.setZero();
            return;
        }
        
        phi = zmax + (1.0 / beta) * std::log(sum_u);
        
        // grad = (1/sum_u) * G
        grad_phi /= sum_u;

        // A^T diag(w) A = (1/sum_u) * M
        // A^T (w w^T) A = grad * grad^T
        // H = beta * (A^T diag(w) A - A^T w w^T A)
        hess_phi = (beta / sum_u) * M - beta * (grad_phi * grad_phi.transpose());

        phi      /= Lscale;
        grad_phi /= Lscale;
        hess_phi /= Lscale;

        return;
    }

    // ---------------------------------
    // shape_id = 3 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 3) {
        if (nParams < 4) {
            fail("Superellipsoid needs params = [n; a; b; c].");
        }

        double n_raw = p[0];
        double a     = p[1];
        double b     = p[2];
        double c     = p[3];

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            fail("Superellipsoid: n must be a positive integer.");
        }

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        const int e = 2 * n;  // exponent m = 2n

        // precompute inverses
        const double inv_a = 1.0 / a;
        const double inv_b = 1.0 / b;
        const double inv_c = 1.0 / c;

        // normalized coords
        const double u1 = x1 * inv_a;
        const double u2 = x2 * inv_b;
        const double u3 = x3 * inv_c;

        // S = sum u^e
        const double t1 = pow_even_int(u1,n);  // u1^(2n)
        const double t2 = pow_even_int(u2,n);
        const double t3 = pow_even_int(u3,n);

        double S = t1 + t2 + t3;

        // ---- NEW: phi = S^(1/e) - 1 (L_{2n} norm - 1) ----
        // Protect fractional powers near S=0 (only relevant deep inside)
        const double S_eps = 1e-16;
        const double S_safe = (S > S_eps) ? S : S_eps;

        const double inv_e = 1.0 / double(e);
        const double q = std::pow(S_safe, inv_e);   // q = S^(1/e)

        phi = q - 1.0;

        // ---- Derivatives ----
        // q'(S)  = (1/e) S^(1/e - 1)
        // q''(S) = (1/e)(1/e - 1) S^(1/e - 2)
        const double qprime  = inv_e * std::pow(S_safe, inv_e - 1.0);
        const double qsecond = inv_e * (inv_e - 1.0) * std::pow(S_safe, inv_e - 2.0);

        // dS/dx_i = e * u_i^(e-1) * (1/s_i) = e * u_i^(2n-1) * inv_s
        const double dSdx1 = double(e) * pow_2n_minus_1(u1,n) * inv_a;
        const double dSdx2 = double(e) * pow_2n_minus_1(u2,n) * inv_b;
        const double dSdx3 = double(e) * pow_2n_minus_1(u3,n) * inv_c;

        // grad phi = q'(S) * grad S
        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        // d2S/dx_i^2 = e(e-1) u_i^(e-2) * (1/s_i^2) = e(e-1) u_i^(2n-2) * inv_s^2
        const double d2Sdx1 = double(e) * double(e - 1) * pow_2n_minus_2(u1,n) * (inv_a * inv_a);
        const double d2Sdx2 = double(e) * double(e - 1) * pow_2n_minus_2(u2,n) * (inv_b * inv_b);
        const double d2Sdx3 = double(e) * double(e - 1) * pow_2n_minus_2(u3,n) * (inv_c * inv_c);

        // Hess phi = q'(S) * Hess S + q''(S) * (grad S)(grad S)^T
        hess_phi.setZero();

        // q'(S) * Hess S (diagonal)
        hess_phi(0,0) += qprime * d2Sdx1;
        hess_phi(1,1) += qprime * d2Sdx2;
        hess_phi(2,2) += qprime * d2Sdx3;

        // q''(S) * gradS * gradS^T (dense rank-1 update)
        Eigen::Vector3d gS(dSdx1, dSdx2, dSdx3);
        hess_phi.noalias() += qsecond * (gS * gS.transpose());

        return;
    }

    // ---------------------------------
    // shape_id = 4 : Superelliptic cylinder
    // ---------------------------------
   else if (shape_id == 4) {
        if (nParams < 3) {
            fail("Superelliptic cylinder needs params = [n; R; h].");
        }

        double n_raw = p[0];
        double R     = p[1];
        double h     = p[2];

        if (R <= 0.0 || h <= 0.0) {
            fail("Superelliptic cylinder: R, h must be > 0.");
        }

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            fail("Superelliptic cylinder: n must be a positive integer.");
        }

        const int e = 2 * n;                 // exponent m = 2n
        const double inv_e = 1.0 / double(e);

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        const double r2 = x2 * x2 + x3 * x3;

        // inverses
        const double inv_R  = 1.0 / R;
        const double inv_R2 = inv_R * inv_R;
        const double inv_h  = 1.0 / h;

        // ---- Build S = axial + radial ----
        // axial: (x1/h)^(2n)
        const double ua = x1 * inv_h;
        const double Sa = pow_even_int(ua,n);

        // radial: (r/R)^(2n) = (r2/R^2)^n
        double Sr = 0.0;
        if (r2 > eps) {
            const double q = r2 * inv_R2;      // (r/R)^2
            double qn = 1.0;
            for (int k = 0; k < n; ++k) qn *= q; // q^n = (r/R)^(2n)
            Sr = qn;
        } else {
            Sr = 0.0;
        }

        double S = Sa + Sr;

        // phi = S^(1/e) - 1 
        const double S_eps = 1e-16;
        const double S_safe = (S > S_eps) ? S : S_eps;

        const double qS = std::pow(S_safe, inv_e); // S^(1/e)
        phi = qS - 1.0;

        const double qprime  = inv_e * std::pow(S_safe, inv_e - 1.0);
        const double qsecond = inv_e * (inv_e - 1.0) * std::pow(S_safe, inv_e - 2.0);

        // ---- Derivatives of S ----
        // Axial:
        // dSa/dx1  = e * (x1/h)^(e-1) * (1/h)
        // d2Sa/dx1 = e(e-1) (x1/h)^(e-2) * (1/h^2)
        double dSdx1  = double(e) * pow_2n_minus_1(ua,n) * inv_h;
        double d2Sdx1 = double(e) * double(e - 1) * pow_2n_minus_2(ua,n) * (inv_h * inv_h);

        // Radial:
        // Sr = (r/R)^e
        // dSr/dx2 = e * Sr * x2 / r2
        // dSr/dx3 = e * Sr * x3 / r2
        // Hessian matches your existing closed form
        double dSdx2 = 0.0, dSdx3 = 0.0;
        double d2Sdx2 = 0.0, d2Sdx3 = 0.0, d2Sdx2dx3 = 0.0;

        if (r2 > eps && Sr > 0.0) {
            const double r4 = r2 * r2;
            const double coeff = double(e) * Sr / r2;

            dSdx2 = coeff * x2;
            dSdx3 = coeff * x3;

            d2Sdx2    = double(e) * Sr / r4 * ((double(e) - 1.0) * x2 * x2 + x3 * x3);
            d2Sdx3    = double(e) * Sr / r4 * ((double(e) - 1.0) * x3 * x3 + x2 * x2);
            d2Sdx2dx3 = double(e) * (double(e) - 2.0) * Sr * x2 * x3 / r4;
        }

        // ---- Gradient of phi ----
        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        // ---- Hessian of phi ----
        // H = q'(S) * Hess(S) + q''(S) * gradS * gradS^T
        hess_phi.setZero();

        // q'(S) * Hess(S) (axial + radial)
        hess_phi(0,0) += qprime * d2Sdx1;
        hess_phi(1,1) += qprime * d2Sdx2;
        hess_phi(2,2) += qprime * d2Sdx3;
        hess_phi(1,2) += qprime * d2Sdx2dx3;
        hess_phi(2,1) += qprime * d2Sdx2dx3;

        // rank-1 update from q''(S) * gradS*gradS^T
        Eigen::Vector3d gS(dSdx1, dSdx2, dSdx3);
        hess_phi.noalias() += qsecond * (gS * gS.transpose());

        return;
    }

    // ---------------------------------
    // shape_id = 5 : Truncated cone
    // ---------------------------------
   else if (shape_id == 5) {
        if (nParams < 5) {
            fail("Truncated cone needs params = [beta; Rb; Rt; a; b].");
        }
        
        double beta = p[0];
        double Rb   = p[1];
        double Rt   = p[2];
        double a    = p[3];
        double b    = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0) {
            fail("Truncated cone: Rb, Rt, a, b, beta all must be > 0.");
        }

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;
        double h  = a + b;

        double t   = (x1 + a) / h;
        double Rx  = Rb + (Rt - Rb) * t;
        double Rx2 = Rx * Rx;
        double Rx3 = Rx2 * Rx;

        double phi_side = 0.0;
        Vector3d grad_side = Vector3d::Zero();
        Matrix3d hess_side = Matrix3d::Zero();

        if (Rx > 0.0 && r2 > 0.0) {
            double term_side = r2 / Rx2;   // r^2 / Rx^2
            phi_side = term_side - 1.0;

            double inv_Rx2 = 1.0 / Rx2;

            // grad wrt x2, x3
            grad_side(1) = 2.0 * x2 * inv_Rx2;
            grad_side(2) = 2.0 * x3 * inv_Rx2;

            // Rx depends linearly on x1
            double dRx_dx1 = (Rt - Rb) / h;

            // grad wrt x1
            grad_side(0) = -2.0 * r2 / Rx3 * dRx_dx1;

            // Hessian
            hess_side.setZero();
            double dRx_dx1_sq = dRx_dx1 * dRx_dx1;
            double inv_Rx4 = 1.0 / (Rx2 * Rx2);

            hess_side(0,0) = 6.0 * r2 * inv_Rx4 * dRx_dx1_sq;
            hess_side(1,1) = 2.0 * inv_Rx2;
            hess_side(2,2) = 2.0 * inv_Rx2;

            double H12 = -4.0 * x2 * dRx_dx1 / Rx3;
            double H13 = -4.0 * x3 * dRx_dx1 / Rx3;

            hess_side(0,1) = H12;
            hess_side(1,0) = H12;
            hess_side(0,2) = H13;
            hess_side(2,0) = H13;
        } else {
            phi_side = -1.0;
            grad_side.setZero();
            hess_side.setZero();
        }

        double phi_bot = -x1 / a - 1;
        double phi_top =  x1 / b - 1;

        Vector3d grad_bot(-1.0 / a, 0.0, 0.0);
        Vector3d grad_top( 1.0 / b, 0.0, 0.0);

        // smooth-max over [phi_side, phi_bot, phi_top]
        double mphi = std::max(phi_side, std::max(phi_top, phi_bot));

        double e_side = std::exp(beta * (phi_side - mphi));
        double e_bot  = std::exp(beta * (phi_bot  - mphi));
        double e_top  = std::exp(beta * (phi_top  - mphi));

        double sum_e = e_side + e_bot + e_top;

        phi = mphi + (1.0 / beta) * std::log(sum_e);

        double inv_sum_e = 1.0 / sum_e;
        double w_side = e_side * inv_sum_e;
        double w_bot  = e_bot  * inv_sum_e;
        double w_top  = e_top  * inv_sum_e;

        // gradient of smooth-max
        grad_phi = w_side * grad_side + w_bot * grad_bot + w_top * grad_top;

        // Hessian of smooth-max:
        // H = sum w_i H_i + beta * ( sum w_i g_i g_i^T - (sum w_i g_i)(sum w_i g_i)^T )
        Matrix3d H_sum = Matrix3d::Zero();
        H_sum += w_side * hess_side;  // only side has nonzero Hessian

        Matrix3d G2 = Matrix3d::Zero();
        G2 += w_side * (grad_side * grad_side.transpose());
        G2 += w_bot  * (grad_bot  * grad_bot.transpose());
        G2 += w_top  * (grad_top  * grad_top.transpose());

        hess_phi = H_sum + beta * (G2 - grad_phi * grad_phi.transpose());

        return;
    }

    else {
        fail("Unknown shape_id. Implemented: 1..5.");
    }
}

void shape_eval_global_ax(
    const Matrix4d& g,
    const Vector3d& x,
    double alpha,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector4d& grad,
    Matrix4d& H)
{
    if (alpha <= 0.0) throw std::runtime_error("shape_eval_global_ax: alpha must be > 0.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // y = R' * (x - r) / alpha
    Vector3d y = R.transpose() * (x - r) / alpha;

    // local eval
    double phi_local;
    Vector3d grad_y;
    Matrix3d H_y;
    shape_eval_local(y, shape_id, params, phi_local, grad_y, H_y);

    phi = phi_local;

    // chain rule
    // grad_x = (1/alpha) * R * grad_y
    Vector3d grad_x = (1.0/alpha) * (R * grad_y);

    // grad_alpha = -(1/alpha) * y' * grad_y
    double grad_alpha = -(1.0/alpha) * y.dot(grad_y);

    // H_xx = (1/alpha^2) * R * H_y * R'
    Matrix3d H_xx = (1.0/(alpha*alpha)) * (R * H_y * R.transpose());

    // H_xa = -(1/alpha^2) * R * (H_y * y + grad_y)
    Vector3d H_xa = -(1.0/(alpha*alpha)) * (R * (H_y * y + grad_y));

    // H_aa = (1/alpha^2) * ( y' H_y y + 2 y' grad_y )
    double H_aa = (1.0/(alpha*alpha)) *
                  ( y.transpose() * H_y * y + 2.0 * y.dot(grad_y) );

    // pack gradient [dphi/dx; dphi/dalpha]
    grad.segment<3>(0) = grad_x;
    grad(3) = grad_alpha;

    // pack Hessian [ H_xx  H_xa
    //                H_xa' H_aa ]
    H.setZero();
    H.block<3,3>(0,0) = H_xx;
    H.block<3,1>(0,3) = H_xa;
    H.block<1,3>(3,0) = H_xa.transpose();
    H(3,3) = H_aa;
}

