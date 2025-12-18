#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

// Your solver header
#include "core/idcol_newton.hpp"

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

static void mexErr(const std::string& s) { mexErrMsgIdAndTxt("idcol:mex", "%s", s.c_str()); }

static bool isRealDouble(const mxArray* a) {
    return mxIsDouble(a) && !mxIsComplex(a);
}

static double getScalarDouble(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetNumberOfElements(a) != 1)
        mexErr(std::string(name) + " must be a real double scalar.");
    return mxGetScalar(a);
}

static int getScalarInt(const mxArray* a, const char* name) {
    double v = getScalarDouble(a, name);
    if (!mxIsFinite(v)) mexErr(std::string(name) + " must be finite.");
    return static_cast<int>(v);
}

static Matrix4d getMat4(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetM(a) != 4 || mxGetN(a) != 4)
        mexErr(std::string(name) + " must be 4x4 real double.");
    const double* p = mxGetPr(a);

    // MATLAB is column-major; Eigen default is column-major -> direct map ok.
    Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::ColMajor>> M(p);
    return Matrix4d(M);
}

static Vector3d getVec3(const mxArray* a, const char* name) {
    if (!isRealDouble(a) || mxGetNumberOfElements(a) != 3)
        mexErr(std::string(name) + " must have 3 elements (real double).");
    const double* p = mxGetPr(a);
    // accept 3x1 or 1x3; just read linear memory
    return Vector3d(p[0], p[1], p[2]);
}

static VectorXd getVec(const mxArray* a, const char* name) {
    if (!isRealDouble(a))
        mexErr(std::string(name) + " must be a real double vector.");
    mwSize n = mxGetNumberOfElements(a);
    const double* p = mxGetPr(a);
    VectorXd v(n);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

// z_opt = [x; alpha; lambda1; lambda2] (6x1)
static mxArray* makeVec6(const Eigen::Matrix<double,6,1>& v) {
    mxArray* out = mxCreateDoubleMatrix(6, 1, mxREAL);
    double* p = mxGetPr(out);
    for (int i = 0; i < 6; ++i) p[i] = v(i);
    return out;
}

static mxArray* makeMat6(const Eigen::Matrix<double,6,6>& A) {
    mxArray* out = mxCreateDoubleMatrix(6, 6, mxREAL);
    double* p = mxGetPr(out);
    // Column-major write (MATLAB layout)
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 6; ++i)
            p[i + 6*j] = A(i,j);
    return out;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        // Expected call:
        // [z_opt, F_opt, J_opt] = idcol_newton_mex( ...
        //   g1, g2, shape_id1, params1, shape_id2, params2,
        //   x0, alpha0, lambda10, lambda20, L, max_iters, tol);

        if (nrhs != 13) {
            mexErr("Expected 13 inputs: g1,g2,shape_id1,params1,shape_id2,params2,x0,alpha0,lambda10,lambda20,L,max_iters,tol");
        }
        if (nlhs < 1 || nlhs > 3) {
            mexErr("Expected 1 to 3 outputs: [z_opt, F_opt, J_opt]");
        }

        // --- Parse inputs ---
        const Matrix4d g1 = getMat4(prhs[0], "g1");
        const Matrix4d g2 = getMat4(prhs[1], "g2");

        const int shape_id1 = getScalarInt(prhs[2], "shape_id1");
        const VectorXd params1 = getVec(prhs[3], "params1");

        const int shape_id2 = getScalarInt(prhs[4], "shape_id2");
        const VectorXd params2 = getVec(prhs[5], "params2");

        const Vector3d x0 = getVec3(prhs[6], "x0");
        const double alpha0  = getScalarDouble(prhs[7],  "alpha0");
        const double lambda10 = getScalarDouble(prhs[8], "lambda10");
        const double lambda20 = getScalarDouble(prhs[9], "lambda20");

        const double L = getScalarDouble(prhs[10], "L");
        const int max_iters = getScalarInt(prhs[11], "max_iters");
        const double tol = getScalarDouble(prhs[12], "tol");

        //const double s_max = getScalarDouble(prhs[13], "s_max");
        //const double s_min = getScalarDouble(prhs[14], "s_min");

        // --- Build problem data ---
        // IMPORTANT: adjust these field names to match your actual ProblemData definition.
        idcol::ProblemData P;
        P.g1 = g1;
        P.g2 = g2;
        P.shape_id1 = shape_id1;
        P.shape_id2 = shape_id2;
        P.params1 = params1;
        P.params2 = params2;

        // --- Options ---
        idcol::NewtonOptions opt;
        opt.L = L;
        opt.max_iters = max_iters;
        //opt.s_max = s_max;
        //opt.s_min = s_min;

        // --- Solve ---
        idcol::NewtonResult res = idcol::solve_idcol_newton(P, x0, alpha0, lambda10, lambda20, opt);

        // z_opt = [x; alpha; lambda1; lambda2]
        Eigen::Matrix<double,6,1> z;
        z.segment<3>(0) = res.x;
        z(3) = res.alpha;
        z(4) = res.lambda1;
        z(5) = res.lambda2;

        // --- Outputs ---
        plhs[0] = makeVec6(z);

        if (nlhs >= 2) {
            plhs[1] = makeVec6(res.F);
        }
        if (nlhs >= 3) {
            plhs[2] = makeMat6(res.J);
        }

    } catch (const std::exception& e) {
        mexErr(std::string("C++ exception: ") + e.what());
    } catch (...) {
        mexErr("Unknown C++ exception.");
    }
}
