#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

// Include the header that declares:
//  - idcol::ProblemData
//  - idcol::Vector6d, idcol::Matrix6d (or typedefs)
//  - idcol::eval_F_J(...)
#include "core/idcol_kkt.hpp"      // <-- change if your header name differs

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

static void mexErr(const std::string& s) {
    mexErrMsgIdAndTxt("idcol:kktFJ", "%s", s.c_str());
}

static bool isRealDouble(const mxArray* a) {
    return mxIsDouble(a) && !mxIsComplex(a);
}

static Matrix4d getMat4(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetM(a) != 4 || mxGetN(a) != 4)
        mexErr(std::string(name) + " must be 4x4 real double.");
    const double* p = mxGetPr(a);
    Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::ColMajor>> M(p);
    return Matrix4d(M);
}

static int getScalarInt(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetNumberOfElements(a) != 1)
        mexErr(std::string(name) + " must be a real double scalar.");
    double v = mxGetScalar(a);
    if (!mxIsFinite(v)) mexErr(std::string(name) + " must be finite.");
    return static_cast<int>(v);
}

static VectorXd getVec(const mxArray* a, const char* name) {
    if (!isRealDouble(a))
        mexErr(std::string(name) + " must be a real double vector.");
    mwSize n = mxGetNumberOfElements(a);
    const double* p = mxGetPr(a);
    VectorXd v((int)n);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

static Eigen::Matrix<double,6,1> getVec6(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetNumberOfElements(a) != 6)
        mexErr(std::string(name) + " must have 6 elements (real double).");
    const double* p = mxGetPr(a);
    Eigen::Matrix<double,6,1> z;
    for (int i = 0; i < 6; ++i) z(i) = p[i];
    return z;
}

static mxArray* makeVec6(const Eigen::Matrix<double,6,1>& v) {
    mxArray* out = mxCreateDoubleMatrix(6, 1, mxREAL);
    double* p = mxGetPr(out);
    for (int i = 0; i < 6; ++i) p[i] = v(i);
    return out;
}

static mxArray* makeMat6(const Eigen::Matrix<double,6,6>& A) {
    mxArray* out = mxCreateDoubleMatrix(6, 6, mxREAL);
    double* p = mxGetPr(out);
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 6; ++i)
            p[i + 6*j] = A(i,j);
    return out;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        // [F,J] = idcol_kkt_FJ_mex(z, g1, g2, shape_id1, shape_id2, params1, params2)
        if (nrhs != 7)
            mexErr("Expected 7 inputs: z, g1, g2, shape_id1, shape_id2, params1, params2.");
        if (nlhs < 1 || nlhs > 2)
            mexErr("Expected 1 or 2 outputs: F (and optionally J).");

        // Inputs
        const auto z = getVec6(prhs[0], "z");
        const Matrix4d g1 = getMat4(prhs[1], "g1");
        const Matrix4d g2 = getMat4(prhs[2], "g2");
        const int shape_id1 = getScalarInt(prhs[3], "shape_id1");
        const int shape_id2 = getScalarInt(prhs[4], "shape_id2");
        const VectorXd params1 = getVec(prhs[5], "params1");
        const VectorXd params2 = getVec(prhs[6], "params2");

        // Unpack z = [x; alpha; lambda1; lambda2]
        const Vector3d x = z.segment<3>(0);
        const double s = z(3);
        const double lambda1 = z(4);
        const double lambda2 = z(5);

        if (!std::isfinite(s)) mexErrMsgIdAndTxt("idcol:kktFJ", "s (z(4)) must be finite.");

        // Build problem
        idcol::ProblemData P;
        P.g1 = g1;
        P.g2 = g2;
        P.shape_id1 = shape_id1;
        P.shape_id2 = shape_id2;
        P.params1 = params1;
        P.params2 = params2;

        // Evaluate
        idcol::Vector6d F;
        idcol::Matrix6d J;
        idcol::eval_F_J(x, s, lambda1, lambda2, P, F, J);

        // Outputs
        plhs[0] = makeVec6(F);

        if (nlhs >= 2) {
            plhs[1] = makeMat6(J);
        }

    } catch (const std::exception& e) {
        mexErr(std::string("C++ exception: ") + e.what());
    } catch (...) {
        mexErr("Unknown C++ exception.");
    }
}
