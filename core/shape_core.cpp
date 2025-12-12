#include "mex.h"              // for mexErrMsgTxt (keep for now)
#include "shape_core.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;

void shape_eval_local_phi_grad(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector3d& grad_phi)
{
    const double* p = params.data();
    const int nParams = params.size();

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
        double r2 = y.squaredNorm();

        phi = r2 - R * R;
        grad_phi = 2.0 * y;
        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Convex polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2)
    {
        if (nParams < 3) fail("Polytope params too short.");

        int m = (int)p[0];
        double beta = p[1];

        if (m <= 0)      fail("Polytope: m must be positive.");
        if (beta <= 0.0) fail("Polytope: beta must be > 0.");

        const int expected = 2 + 4 * m;
        if (nParams != expected)
            fail("Polytope params size must be 2 + 4*m.");

        const double* A_data = p + 2;
        const double* b_data = p + 2 + 3*m;

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

        return;
    }

    // ---------------------------------
    // shape_id = 3 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 3)
    {
        if (nParams < 4)
            fail("Superellipsoid needs params = [a; b; c; n].");

        double a     = p[0];
        double b     = p[1];
        double c     = p[2];
        double n_raw = p[3];

        int n = (int)std::round(n_raw);
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9)
            fail("Superellipsoid: n must be a positive integer.");

        // exponent e = 2n
        int e = 2 * n;

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        // precompute inverse scales to avoid repeated divisions
        double inv_a = 1.0 / a;
        double inv_b = 1.0 / b;
        double inv_c = 1.0 / c;

        // helper: (x/s)^(2n) = ( (x/s)^2 )^n
        auto pow_even_int = [&](double x, double inv_s) {
            double base = x * inv_s;     // x/s
            double base2 = base * base;  // (x/s)^2
            double result = 1.0;
            for (int k = 0; k < n; ++k) {
                result *= base2;         // (x/s)^(2n)
            }
            return result;
        };

        double tx1 = pow_even_int(x1, inv_a);
        double tx2 = pow_even_int(x2, inv_b);
        double tx3 = pow_even_int(x3, inv_c);

        double S = tx1 + tx2 + tx3;
        phi = S - 1.0;

        double d1 = 0.0;
        double d2 = 0.0;
        double d3 = 0.0;

        if (std::fabs(x1) > eps) {
            d1 = e * tx1 / x1;
        }
        if (std::fabs(x2) > eps) {
            d2 = e * tx2 / x2;
        }
        if (std::fabs(x3) > eps) {
            d3 = e * tx3 / x3;
        }

        grad_phi << d1, d2, d3;
        return;
    }

    // ---------------------------------
    // shape_id = 4 : Superelliptic cylinder
    // ---------------------------------
    else if (shape_id == 4)
    {
        if (nParams < 3)
            fail("Superelliptic cylinder needs params = [R; h; n].");

        double R     = p[0];
        double h     = p[1];
        double n_raw = p[2];

        if (R <= 0.0 || h <= 0.0)
            fail("Superelliptic cylinder: R, h must be > 0.");

        int n = (int)std::round(n_raw);
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9)
            fail("Superelliptic cylinder: n must be a positive integer.");

        int e = 2 * n;

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;

        // precompute inverses
        double inv_R  = 1.0 / R;
        double inv_R2 = inv_R * inv_R;
        double inv_h  = 1.0 / h;

        // axial term: term_a = (x1/h)^(2n) = ((x1/h)^2)^n
        double base_a  = x1 * inv_h;
        double base_a2 = base_a * base_a;
        double term_a  = 1.0;
        for (int k = 0; k < n; ++k) {
            term_a *= base_a2;
        }

        // radial term: term_r = (r/R)^(2n) = (r2/R^2)^n
        double term_r = 0.0;
        if (r2 > eps) {
            double q = r2 * inv_R2;  // (r^2 / R^2)
            double qn = 1.0;
            for (int k = 0; k < n; ++k) {
                qn *= q;
            }
            term_r = qn;
        }

        double S = term_r + term_a;
        phi = S - 1.0;

        double d1 = 0.0;
        double d2 = 0.0;
        double d3 = 0.0;

        // dS/dx1 from axial term
        if (std::fabs(x1) > eps) {
            double inv_x1 = 1.0 / x1;
            d1 = e * term_a * inv_x1;
        }

        // dS/dx2, dS/dx3 from radial term
        if (r2 > eps && term_r > 0.0) {
            // from derivation: dS/dx2 = e * term_r * x2 / r2, same for x3
            double coeff = e * term_r / r2;
            d2 = coeff * x2;
            d3 = coeff * x3;
        }

        grad_phi << d1, d2, d3;
        return;
    }

    // ---------------------------------
    // shape_id = 5 : Truncated cone
    // ---------------------------------
   else if (shape_id == 5)
    {
        if (nParams < 5)
            fail("Truncated cone needs params = [Rb; Rt; a; b; beta].");

        double Rb   = p[0];
        double Rt   = p[1];
        double a    = p[2];
        double b    = p[3];
        double beta = p[4];

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

        double phi_bot = -x1 - a;  // bottom plane
        double phi_top =  x1 - b;  // top plane

        Vector3d grad_bot(-1.0, 0.0, 0.0);
        Vector3d grad_top( 1.0, 0.0, 0.0);

        // smooth-max of [phi_side, phi_bot, phi_top]
        double mphi = std::max(phi_side, std::max(phi_bot, phi_top));

        double e_side = std::exp(beta * (phi_side - mphi));
        double e_bot  = std::exp(beta * (phi_bot  - mphi));
        double e_top  = std::exp(beta * (phi_top  - mphi));

        double sum_e = e_side + e_bot + e_top;

        phi = (1.0 / beta) * (mphi + std::log(sum_e));

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
    if (alpha == 0.0) {
        throw std::runtime_error("shape_eval_global_ax_phi_grad: alpha must be nonzero.");
    }

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
    mwSize nParams = static_cast<mwSize>(params.size());

    phi = 0.0;
    grad_phi.setZero();
    hess_phi.setZero();
    const double eps = 1e-9;

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1) {
        if (nParams < 1) {
            mexErrMsgTxt("Sphere needs params(1) = radius.");
        }
        double R = p[0];

        double r2 = y.squaredNorm();
        phi = r2 - R * R;        // phi(y) = ||y||^2 - R^2
        grad_phi = 2.0 * y;      // ∂phi/∂y = 2y
        hess_phi = 2.0 * Matrix3d::Identity(); // ∂²phi/∂y² = 2I₃
    }

    // -----------------------------------------
    // shape_id = 2 : Convex polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2) {
        if (nParams < 3) {
            mexErrMsgTxt("Polytope params too short.");
        }

        int m = static_cast<int>(p[0]);
        double beta = p[1];
        if (m <= 0) {
            mexErrMsgTxt("Polytope: m must be positive.");
        }
        if (beta <= 0.0) {
            mexErrMsgTxt("Polytope: beta must be > 0.");
        }

        mwSize expected = 2 + 4 * static_cast<mwSize>(m);
        if (nParams != expected) {
            mexErrMsgTxt("Polytope: params size must be 2 + 4*m (m, beta, A(:), b).");
        }

        double* A_data = const_cast<double*>(p + 2);
        double* b_data = const_cast<double*>(p + 2 + 3 * m);

        Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, 3> > A(A_data, m, 3);
        Eigen::Map< Eigen::VectorXd > b(b_data, m);

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
    }


    // ---------------------------------
    // shape_id = 3 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 3) {
        if (nParams < 4) {
            mexErrMsgTxt("Superellipsoid needs params = [a; b; c; n].");
        }
        double a     = p[0];
        double b     = p[1];
        double c     = p[2];
        double n_raw = p[3];

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            mexErrMsgTxt("Superellipsoid: n must be a positive integer.");
        }

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        int e = 2 * n;  // exponent

        // precompute inverses to avoid repeated divisions
        double inv_a = 1.0 / a;
        double inv_b = 1.0 / b;
        double inv_c = 1.0 / c;

        // helper: (x/s)^(2n) = ((x/s)^2)^n
        auto pow_even_int = [&](double x, double inv_s) {
            double base  = x * inv_s;      // x/s
            double base2 = base * base;    // (x/s)^2
            double res   = 1.0;
            for (int k = 0; k < n; ++k) {
                res *= base2;              // ((x/s)^2)^n = (x/s)^(2n)
            }
            return res;
        };

        double tx1 = pow_even_int(x1, inv_a);
        double tx2 = pow_even_int(x2, inv_b);
        double tx3 = pow_even_int(x3, inv_c);

        double S = tx1 + tx2 + tx3;
        phi = S - 1.0;

        double dSdx1  = 0.0;
        double dSdx2  = 0.0;
        double dSdx3  = 0.0;
        double d2Sdx1 = 0.0;
        double d2Sdx2 = 0.0;
        double d2Sdx3 = 0.0;

        if (std::fabs(x1) > eps) {
            double inv_x1  = 1.0 / x1;
            double inv_x12 = inv_x1 * inv_x1;
            dSdx1  = e * tx1 * inv_x1;
            d2Sdx1 = e * (e - 1.0) * tx1 * inv_x12;
        }
        if (std::fabs(x2) > eps) {
            double inv_x2  = 1.0 / x2;
            double inv_x22 = inv_x2 * inv_x2;
            dSdx2  = e * tx2 * inv_x2;
            d2Sdx2 = e * (e - 1.0) * tx2 * inv_x22;
        }
        if (std::fabs(x3) > eps) {
            double inv_x3  = 1.0 / x3;
            double inv_x32 = inv_x3 * inv_x3;
            dSdx3  = e * tx3 * inv_x3;
            d2Sdx3 = e * (e - 1.0) * tx3 * inv_x32;
        }

        grad_phi(0) = dSdx1;
        grad_phi(1) = dSdx2;
        grad_phi(2) = dSdx3;

        hess_phi.setZero();
        hess_phi(0,0) = d2Sdx1;
        hess_phi(1,1) = d2Sdx2;
        hess_phi(2,2) = d2Sdx3;
    }


    // ---------------------------------
    // shape_id = 4 : Superelliptic cylinder
    // ---------------------------------
   else if (shape_id == 4) {
        if (nParams < 3) {
            mexErrMsgTxt("Superelliptic cylinder needs params = [R; h; n].");
        }
        double R     = p[0];
        double h     = p[1];
        double n_raw = p[2];

        if (R <= 0.0 || h <= 0.0) {
            mexErrMsgTxt("Superelliptic cylinder: R, h must be > 0.");
        }

        int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            mexErrMsgTxt("Superelliptic cylinder: n must be a positive integer.");
        }
        int e = 2 * n;

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;

        // precompute inverses
        double inv_R  = 1.0 / R;
        double inv_R2 = inv_R * inv_R;
        double inv_h  = 1.0 / h;

        double term_radial = 0.0;
        double term_axial  = 0.0;

        // axial term: (x1/h)^(2n) = ((x1/h)^2)^n
        double base_a  = x1 * inv_h;
        double base_a2 = base_a * base_a;
        term_axial     = 1.0;
        for (int k = 0; k < n; ++k) {
            term_axial *= base_a2;
        }

        // radial term: (r/R)^(2n) = (r2/R^2)^n
        if (r2 > eps) {
            double q  = r2 * inv_R2;  // r^2 / R^2
            double qn = 1.0;
            for (int k = 0; k < n; ++k) {
                qn *= q;
            }
            term_radial = qn;
        } else {
            term_radial = 0.0;
        }

        double S = term_radial + term_axial;
        phi = S - 1.0;

        double dSdx1      = 0.0;
        double dSdx2      = 0.0;
        double dSdx3      = 0.0;
        double d2Sdx1     = 0.0;
        double d2Sdx2     = 0.0;
        double d2Sdx3     = 0.0;
        double d2Sdx2dx3  = 0.0;

        // axial derivatives
        if (std::fabs(x1) > eps && h > 0.0) {
            double inv_x1  = 1.0 / x1;
            double inv_x12 = inv_x1 * inv_x1;
            dSdx1  = e * term_axial * inv_x1;
            d2Sdx1 = e * (e - 1.0) * term_axial * inv_x12;
        }

        // radial derivatives
        if (r2 > eps && R > 0.0 && term_radial > 0.0) {
            double r4    = r2 * r2;
            double coeff = e * term_radial / r2;

            dSdx2 = coeff * x2;
            dSdx3 = coeff * x3;

            d2Sdx2     = e * term_radial / r4 * ((e - 1.0) * x2 * x2 + x3 * x3);
            d2Sdx3     = e * term_radial / r4 * ((e - 1.0) * x3 * x3 + x2 * x2);
            d2Sdx2dx3  = e * (e - 2.0) * term_radial * x2 * x3 / r4;
        }

        grad_phi(0) = dSdx1;
        grad_phi(1) = dSdx2;
        grad_phi(2) = dSdx3;

        hess_phi.setZero();
        hess_phi(0,0) = d2Sdx1;
        hess_phi(1,1) = d2Sdx2;
        hess_phi(2,2) = d2Sdx3;
        hess_phi(1,2) = d2Sdx2dx3;
        hess_phi(2,1) = d2Sdx2dx3;
    }

    // ---------------------------------
    // shape_id = 5 : Truncated cone
    // ---------------------------------
   else if (shape_id == 5) {
        if (nParams < 5) {
            mexErrMsgTxt("Truncated cone needs params = [Rb; Rt; a; b; beta].");
        }
        double Rb   = p[0];
        double Rt   = p[1];
        double a    = p[2];
        double b    = p[3];
        double beta = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0) {
            mexErrMsgTxt("Truncated cone: Rb, Rt, a, b, beta all must be > 0.");
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

        double phi_bot = -x1 - a;
        double phi_top =  x1 - b;

        Vector3d grad_bot(-1.0, 0.0, 0.0);
        Vector3d grad_top( 1.0, 0.0, 0.0);

        // smooth-max over [phi_side, phi_bot, phi_top]
        double max_phi = std::max(phi_side, std::max(phi_top, phi_bot));

        double e_side = std::exp(beta * (phi_side - max_phi));
        double e_bot  = std::exp(beta * (phi_bot  - max_phi));
        double e_top  = std::exp(beta * (phi_top  - max_phi));

        double sum_e = e_side + e_bot + e_top;

        phi = (1.0 / beta) * (max_phi + std::log(sum_e));

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
    }

    else {
        mexErrMsgTxt("Unknown shape_id. Implemented: 1..5.");
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
    if (alpha == 0.0) {
        mexErrMsgTxt("shape_eval_global_ax: alpha must be nonzero.");
    }

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

