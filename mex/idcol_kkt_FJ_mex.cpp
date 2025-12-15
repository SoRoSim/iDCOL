// idcol_kkt_FJ_mex.cpp
//
// [F,J] = idcol_kkt_FJ_mex(z, g1, g2, shape_id1, shape_id2, params1, params2)
//
// z = [ x(3); alpha; lambda1; lambda2 ]  (6x1)
// F = 6x1, J = 6x6
//
// Requires: core/shape_core.hpp to provide shape_eval_global_ax (or equivalent).
// You said: #include "core/shape_core.hpp" has shape_eval, so we just call it.

#include "mex.h"
#include <Eigen/Dense>
#include <cmath>

#include "core/shape_core.hpp" 

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix4d;

using Vector6d = Eigen::Matrix<double,6,1>;
using Matrix6d = Eigen::Matrix<double,6,6>;

// If your header uses a different function name/signature, adapt ONLY this call-site.
// Expected outputs:
//   phi: scalar
//   grad: 4x1 with [dphi/dx; dphi/dalpha]
//   H: 4x4 Hessian wrt [x; alpha]
//
// Example signature assumed (same as we used before):
// void shape_eval_global_ax(const Matrix4d& g, const Vector3d& x, double alpha,
//                           int shape_id, const VectorXd& params,
//                           double& phi, Vector4d& grad, Matrix4d& H);

static inline int getIntScalar(const mxArray* a)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("idcol:Type","shape_id must be a real double scalar.");
    }
    return static_cast<int>(mxGetScalar(a));
}

static inline void requireRealDouble(const mxArray* a, const char* name)
{
    if (!mxIsDouble(a) || mxIsComplex(a)) {
        mexErrMsgIdAndTxt("idcol:Type","%s must be real double.", name);
    }
}

static void eval_F_J_alpha(
    const Vector3d& x,
    double alpha,
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
    // Evaluate shape 1
    double  phi1;
    Vector4d grad1;
    Matrix4d H1;
    shape_eval_global_ax(g1, x, alpha, shape_id1, params1, phi1, grad1, H1);

    // Evaluate shape 2
    double  phi2;
    Vector4d grad2;
    Matrix4d H2;
    shape_eval_global_ax(g2, x, alpha, shape_id2, params2, phi2, grad2, H2);

    // Split gradients
    const Vector3d g1x = grad1.head<3>();
    const double   g1a = grad1(3);

    const Vector3d g2x = grad2.head<3>();
    const double   g2a = grad2(3);

    // Split Hessians
    const Eigen::Matrix3d H1_xx = H1.block<3,3>(0,0);
    const Eigen::Vector3d H1_xa = H1.block<3,1>(0,3);   // d²phi1/dx dα
    const double          H1_aa = H1(3,3);

    const Eigen::Matrix3d H2_xx = H2.block<3,3>(0,0);
    const Eigen::Vector3d H2_xa = H2.block<3,1>(0,3);
    const double          H2_aa = H2(3,3);

    // -----------------------
    // Residual F
    // z = [x(3); alpha; lambda1; lambda2]
    // -----------------------
    F.setZero();
    F(0) = phi1;                         // φ1 = 0
    F(1) = phi2;                         // φ2 = 0
    F.segment<3>(2) = lambda1*g1x + lambda2*g2x; // stationarity in x
    F(5) = 1.0 + lambda1*g1a + lambda2*g2a;      // stationarity in alpha (your form)

    // -----------------------
    // Jacobian J = dF/dz
    // columns: [dx(3), dα, dλ1, dλ2]
    // -----------------------
    J.setZero();

    // Row 0: F1 = φ1
    J.block<1,3>(0,0) = g1x.transpose();  // dφ1/dx
    J(0,3) = g1a;                         // dφ1/dα

    // Row 1: F2 = φ2
    J.block<1,3>(1,0) = g2x.transpose();  // dφ2/dx
    J(1,3) = g2a;                         // dφ2/dα

    // Rows 2..4: Fx = λ1 ∇xφ1 + λ2 ∇xφ2
    for (int j = 0; j < 3; ++j) {
        const int row = 2 + j;

        // d/dx: λ1 H1_xx(j,:) + λ2 H2_xx(j,:)
        J.block<1,3>(row,0) =
            lambda1 * H1_xx.row(j) + lambda2 * H2_xx.row(j);

        // d/dα: λ1 H1_xa(j) + λ2 H2_xa(j)
        J(row,3) = lambda1 * H1_xa(j) + lambda2 * H2_xa(j);

        // d/dλ1, d/dλ2:
        J(row,4) = g1x(j);
        J(row,5) = g2x(j);
    }

    // Row 5: F6 = 1 + λ1 g1a + λ2 g2a
    // d/dx: λ1 * d(g1a)/dx + λ2 * d(g2a)/dx = λ1 * H1_ax + λ2 * H2_ax
    // H_ax = (H_xa)^T because Hessian is symmetric
    J.block<1,3>(5,0) = lambda1 * H1_xa.transpose() + lambda2 * H2_xa.transpose();

    // d/dα: λ1 * d(g1a)/dα + λ2 * d(g2a)/dα = λ1*H1_aa + λ2*H2_aa
    J(5,3) = lambda1 * H1_aa + lambda2 * H2_aa;

    // d/dλ1, d/dλ2:
    J(5,4) = g1a;
    J(5,5) = g2a;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("idcol:Args",
            "Usage: [F,J] = idcol_kkt_FJ_mex(z,g1,g2,shape_id1,shape_id2,params1,params2)");
    }
    if (nlhs < 1 || nlhs > 2) {
        mexErrMsgIdAndTxt("idcol:Args","Output must be [F] or [F,J].");
    }

    // z
    requireRealDouble(prhs[0], "z");
    if (mxGetNumberOfElements(prhs[0]) != 6) {
        mexErrMsgIdAndTxt("idcol:Dim","z must be 6x1: [x(3); alpha; lambda1; lambda2].");
    }
    const double* zptr = mxGetPr(prhs[0]);
    const Vector3d x(zptr[0], zptr[1], zptr[2]);
    const double alpha   = zptr[3];
    const double lambda1 = zptr[4];
    const double lambda2 = zptr[5];

    // g1, g2 (4x4)
    requireRealDouble(prhs[1], "g1");
    requireRealDouble(prhs[2], "g2");
    if (mxGetM(prhs[1]) != 4 || mxGetN(prhs[1]) != 4 ||
        mxGetM(prhs[2]) != 4 || mxGetN(prhs[2]) != 4) {
        mexErrMsgIdAndTxt("idcol:Dim","g1 and g2 must be 4x4.");
    }
    const double* g1ptr = mxGetPr(prhs[1]);
    const double* g2ptr = mxGetPr(prhs[2]);

    // MATLAB column-major matches Eigen default column-major => direct map
    const Eigen::Map<const Matrix4d> g1(g1ptr);
    const Eigen::Map<const Matrix4d> g2(g2ptr);

    // shape IDs
    const int shape_id1 = getIntScalar(prhs[3]);
    const int shape_id2 = getIntScalar(prhs[4]);

    // params
    requireRealDouble(prhs[5], "params1");
    requireRealDouble(prhs[6], "params2");
    const mwSize n1 = mxGetNumberOfElements(prhs[5]);
    const mwSize n2 = mxGetNumberOfElements(prhs[6]);

    const Eigen::Map<const VectorXd> params1(mxGetPr(prhs[5]), (int)n1);
    const Eigen::Map<const VectorXd> params2(mxGetPr(prhs[6]), (int)n2);

    Vector6d F;
    Matrix6d J;

    eval_F_J_alpha(x, alpha, lambda1, lambda2, g1, g2, shape_id1, shape_id2, params1, params2, F, J);

    // Output F
    plhs[0] = mxCreateDoubleMatrix(6,1,mxREAL);
    Eigen::Map<Vector6d>(mxGetPr(plhs[0])) = F;

    // Output J if requested
    if (nlhs == 2) {
        plhs[1] = mxCreateDoubleMatrix(6,6,mxREAL);
        Eigen::Map<Matrix6d>(mxGetPr(plhs[1])) = J;
    }
}
