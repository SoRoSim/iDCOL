#include "mex.h"
#include <Eigen/Dense>
#include "core/shape_core.hpp"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Inputs:
//   g       : 4x4 homogeneous transform
//   x       : 3x1 (same frame as g)
//   alpha   : scalar
//   shape_id: scalar
//   params  : vector
//
// Outputs:
//   phi     : scalar
//   grad    : 4x1   [dphi/dx; dphi/dalpha]
//   H       : 4x4   Hessian wrt [x; alpha]

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs != 5) {
        mexErrMsgTxt("Usage: [phi, grad, H] = shape_global_ax_mex(g, x, alpha, shape_id, params)");
    }
    if (nlhs != 3) {
        mexErrMsgTxt("Need 3 outputs: phi, grad, H.");
    }

    // --- g (4x4) ---
    if (mxGetM(prhs[0]) != 4 || mxGetN(prhs[0]) != 4) {
        mexErrMsgTxt("g must be 4x4.");
    }
    double* g_ptr = mxGetPr(prhs[0]);
    Matrix4d g;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            g(row,col) = g_ptr[row + 4*col];

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // --- x (3x1) ---
    if (!(mxGetM(prhs[1]) == 3 && mxGetN(prhs[1]) == 1)) {
        mexErrMsgTxt("x must be 3x1.");
    }
    double* x_ptr = mxGetPr(prhs[1]);
    Vector3d x(x_ptr[0], x_ptr[1], x_ptr[2]);

    // --- alpha ---
    double alpha = mxGetScalar(prhs[2]);
    if (alpha == 0.0) {
        mexErrMsgTxt("alpha must be nonzero.");
    }

    // --- shape_id ---
    int shape_id = static_cast<int>(mxGetScalar(prhs[3]));

    // --- params ---
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4])) {
        mexErrMsgTxt("params must be real double.");
    }
    double* p = mxGetPr(prhs[4]);
    mwSize nParams = mxGetNumberOfElements(prhs[4]);
    VectorXd params(nParams);
    for (mwSize i = 0; i < nParams; ++i) params(i) = p[i];

    // ---- local y ----
    Vector3d y = R.transpose() * (x - r) / alpha;

    // ---- call core on y ----
    double phi;
    Vector3d grad_y;
    Matrix3d H_y;
    shape_eval_local(y, shape_id, params, phi, grad_y, H_y);

    // ---- chain rule to (x, alpha) ----

    // grad_x = (1/alpha) * R * grad_y
    Vector3d grad_x = (1.0/alpha) * (R * grad_y);

    // grad_alpha = -(1/alpha) * y' * grad_y
    double grad_alpha = -(1.0/alpha) * y.dot(grad_y);

    // H_xx = (1/alpha^2) * R * H_y * R'
    Matrix3d H_xx = (1.0/(alpha*alpha)) * (R * H_y * R.transpose());

    // H_xa = -(1/alpha^2) * R * (H_y * y + grad_y)
    Vector3d H_xa = -(1.0/(alpha*alpha)) * (R * (H_y * y + grad_y));

    // H_aa = (1/alpha^2) * ( y' H_y y + 2 y' grad_y )
    double H_aa = (1.0/(alpha*alpha)) * ( y.transpose() * H_y * y
                                          + 2.0 * y.dot(grad_y) );

    // ---- pack outputs ----

    // phi
    plhs[0] = mxCreateDoubleScalar(phi);

    // grad (4x1)
    plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
    double* g_out = mxGetPr(plhs[1]);
    g_out[0] = grad_x(0);
    g_out[1] = grad_x(1);
    g_out[2] = grad_x(2);
    g_out[3] = grad_alpha;

    // H (4x4), column-major
    plhs[2] = mxCreateDoubleMatrix(4, 4, mxREAL);
    double* H_out = mxGetPr(plhs[2]);

    // top-left 3x3 = H_xx
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            H_out[row + 4*col] = H_xx(row,col);

    // last column (0..2) = H_xa
    for (int row = 0; row < 3; ++row)
        H_out[row + 4*3] = H_xa(row);

    // last row (0..2) = H_xa'
    for (int col = 0; col < 3; ++col)
        H_out[3 + 4*col] = H_xa(col);

    // bottom-right
    H_out[3 + 4*3] = H_aa;
}
