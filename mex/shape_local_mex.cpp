#include "mex.h"
#include <Eigen/Dense>
#include "core/shape_core.hpp"

// Inputs:
//   prhs[0] : y (3x1 double)
//   prhs[1] : shape_id (double scalar)
//   prhs[2] : params (double array)
//
// Outputs:
//   plhs[0] : phi      (scalar)
//   plhs[1] : grad_phi (3x1 double)   wrt y
//   plhs[2] : hess_phi (3x3 double)   wrt y

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs != 3) {
        mexErrMsgTxt("Expected 3 inputs: y(3x1), shape_id, params.");
    }
    if (nlhs != 3) {
        mexErrMsgTxt("Expected 3 outputs: phi, grad_phi, hess_phi.");
    }

    // --- y ---
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
        mxGetNumberOfElements(prhs[0]) != 3) {
        mexErrMsgTxt("y must be a real 3-element double vector.");
    }
    double* y_ptr = mxGetPr(prhs[0]);
    Eigen::Vector3d y(y_ptr[0], y_ptr[1], y_ptr[2]);

    // --- shape_id ---
    int shape_id = static_cast<int>(mxGetScalar(prhs[1]));

    // --- params ---
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgTxt("params must be a real double array.");
    }
    double* p = mxGetPr(prhs[2]);
    mwSize nParams = mxGetNumberOfElements(prhs[2]);
    Eigen::VectorXd params(nParams);
    for (mwSize i = 0; i < nParams; ++i) {
        params(i) = p[i];
    }

    // --- call core C++ geometry ---
    double phi;
    Eigen::Vector3d grad_phi;
    Eigen::Matrix3d hess_phi;
    shape_eval_local(y, shape_id, params, phi, grad_phi, hess_phi);

    // --------- set outputs ---------
    // phi
    plhs[0] = mxCreateDoubleScalar(phi);

    // grad_phi (3x1)
    plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
    double* g_out = mxGetPr(plhs[1]);
    g_out[0] = grad_phi(0);
    g_out[1] = grad_phi(1);
    g_out[2] = grad_phi(2);

    // hess_phi (3x3, column-major)
    plhs[2] = mxCreateDoubleMatrix(3, 3, mxREAL);
    double* H_out = mxGetPr(plhs[2]);
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            H_out[row + 3*col] = hess_phi(row, col);
        }
    }
}
