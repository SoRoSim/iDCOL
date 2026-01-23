// shape_core_mex.cpp
//
// MATLAB usage:
//   out = shape_core_mex('local_phi_grad', y, shape_id, params)
//   out = shape_core_mex('global_xa_phi_grad', g, x, alpha, shape_id, params)
//   out = shape_core_mex('local', y, shape_id, params)
//   out = shape_core_mex('global_xa', g, x, alpha, shape_id, params)
//
// Returns struct:
//   out.phi
//   out.grad   (3x1 for local*, 4x1 for global_xa*; ordering [dphi/dx; dphi/dalpha])
//   out.H      (only for "local" and "global_xa")
//
// Build example (adjust paths/files):
//   mex -v CXXFLAGS="\$CXXFLAGS -std=c++17 -O3" -I<eigen> -I<include_root> ...
//       shape_core_mex.cpp core/shape_core.cpp

#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <string>
#include <cmath>

// Your header (global namespace functions)
#include "core/shape_core.hpp"

// -------------------- helpers --------------------

static void require(bool cond, const char* id, const char* msg) {
    if (!cond) mexErrMsgIdAndTxt(id, "%s", msg);
}

static bool is_real_double(const mxArray* a) {
    return mxIsDouble(a) && !mxIsComplex(a);
}

static int get_int_scalar(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 1,
            "shape_mex:badArg", (std::string(name) + " must be a real scalar double (used as int).").c_str());
    return (int)std::llround(mxGetScalar(a));
}

static double get_double_scalar(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 1,
            "shape_mex:badArg", (std::string(name) + " must be a real scalar double.").c_str());
    return mxGetScalar(a);
}

static Eigen::Vector3d get_vec3(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 3,
            "shape_mex:badArg", (std::string(name) + " must have 3 elements.").c_str());
    const double* p = mxGetPr(a);
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

static Eigen::VectorXd get_vecN(const mxArray* a, const char* name) {
    require(a && is_real_double(a),
            "shape_mex:badArg", (std::string(name) + " must be a real double array.").c_str());
    const mwSize n = mxGetNumberOfElements(a);
    require(n >= 1, "shape_mex:badArg", (std::string(name) + " must be non-empty.").c_str());
    Eigen::VectorXd v((int)n);
    const double* p = mxGetPr(a);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

static Eigen::Matrix4d get_T44(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetM(a) == 4 && mxGetN(a) == 4,
            "shape_mex:badArg", (std::string(name) + " must be 4x4 real double.").c_str());
    const double* p = mxGetPr(a);
    Eigen::Matrix4d T;
    // MATLAB is column-major; Eigen default is column-major. Fill explicitly.
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            T(r, c) = p[r + 4 * c];
    return T;
}

static mxArray* make_struct_local(double phi, const Eigen::Vector3d& grad, const Eigen::Matrix3d* H) {
    if (H) {
        const char* fields[] = {"phi","grad","H"};
        mxArray* S = mxCreateStructMatrix(1,1,3,fields);
        mxSetField(S,0,"phi", mxCreateDoubleScalar(phi));

        mxArray* gmx = mxCreateDoubleMatrix(3,1,mxREAL);
        double* pg = mxGetPr(gmx);
        pg[0]=grad[0]; pg[1]=grad[1]; pg[2]=grad[2];
        mxSetField(S,0,"grad", gmx);

        mxArray* Hmx = mxCreateDoubleMatrix(3,3,mxREAL);
        double* pH = mxGetPr(Hmx);
        for (int c=0;c<3;++c) for (int r=0;r<3;++r) pH[r+3*c]=(*H)(r,c);
        mxSetField(S,0,"H", Hmx);
        return S;
    } else {
        const char* fields[] = {"phi","grad"};
        mxArray* S = mxCreateStructMatrix(1,1,2,fields);
        mxSetField(S,0,"phi", mxCreateDoubleScalar(phi));

        mxArray* gmx = mxCreateDoubleMatrix(3,1,mxREAL);
        double* pg = mxGetPr(gmx);
        pg[0]=grad[0]; pg[1]=grad[1]; pg[2]=grad[2];
        mxSetField(S,0,"grad", gmx);
        return S;
    }
}

static mxArray* make_struct_global(double phi, const Eigen::Vector4d& grad, const Eigen::Matrix4d* H) {
    if (H) {
        const char* fields[] = {"phi","grad","H"};
        mxArray* S = mxCreateStructMatrix(1,1,3,fields);
        mxSetField(S,0,"phi", mxCreateDoubleScalar(phi));

        mxArray* gmx = mxCreateDoubleMatrix(4,1,mxREAL);
        double* pg = mxGetPr(gmx);
        for (int i=0;i<4;++i) pg[i]=grad[i];
        mxSetField(S,0,"grad", gmx);

        mxArray* Hmx = mxCreateDoubleMatrix(4,4,mxREAL);
        double* pH = mxGetPr(Hmx);
        for (int c=0;c<4;++c) for (int r=0;r<4;++r) pH[r+4*c]=(*H)(r,c);
        mxSetField(S,0,"H", Hmx);
        return S;
    } else {
        const char* fields[] = {"phi","grad"};
        mxArray* S = mxCreateStructMatrix(1,1,2,fields);
        mxSetField(S,0,"phi", mxCreateDoubleScalar(phi));

        mxArray* gmx = mxCreateDoubleMatrix(4,1,mxREAL);
        double* pg = mxGetPr(gmx);
        for (int i=0;i<4;++i) pg[i]=grad[i];
        mxSetField(S,0,"grad", gmx);
        return S;
    }
}

// -------------------- gateway --------------------

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nlhs <= 1, "shape_mex:usage", "One output only.");

    require(nrhs >= 1 && mxIsChar(prhs[0]),
            "shape_mex:usage",
            "First input must be a command string.\n"
            "Commands:\n"
            "  local_phi_grad(y, shape_id, params)\n"
            "  global_xa_phi_grad(g, x, alpha, shape_id, params)\n"
            "  local(y, shape_id, params)\n"
            "  global_xa(g, x, alpha, shape_id, params)\n");

    char cmd_buf[128];
    require(mxGetString(prhs[0], cmd_buf, sizeof(cmd_buf)) == 0,
            "shape_mex:badArg", "Could not read command string.");
    const std::string cmd(cmd_buf);

    if (cmd == "local_phi_grad") {
        require(nrhs == 4, "shape_mex:usage",
                "Usage: out = shape_core_mex('local_phi_grad', y, shape_id, params)");
        Eigen::Vector3d y = get_vec3(prhs[1], "y");
        int shape_id = get_int_scalar(prhs[2], "shape_id");
        Eigen::VectorXd params = get_vecN(prhs[3], "params");

        double phi = 0.0;
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        shape_eval_local_phi_grad(y, shape_id, params, phi, grad);

        plhs[0] = make_struct_local(phi, grad, nullptr);
        return;
    }

    if (cmd == "global_xa_phi_grad") {
        require(nrhs == 6, "shape_mex:usage",
                "Usage: out = shape_core_mex('global_xa_phi_grad', g, x, alpha, shape_id, params)");
        Eigen::Matrix4d g = get_T44(prhs[1], "g");
        Eigen::Vector3d x = get_vec3(prhs[2], "x");
        double alpha = get_double_scalar(prhs[3], "alpha");
        int shape_id = get_int_scalar(prhs[4], "shape_id");
        Eigen::VectorXd params = get_vecN(prhs[5], "params");

        double phi = 0.0;
        Eigen::Vector4d grad = Eigen::Vector4d::Zero();
        // NOTE: call your renamed function:
        shape_eval_global_xa_phi_grad(g, x, alpha, shape_id, params, phi, grad);

        plhs[0] = make_struct_global(phi, grad, nullptr);
        return;
    }

    if (cmd == "local") {
        require(nrhs == 4, "shape_mex:usage",
                "Usage: out = shape_core_mex('local', y, shape_id, params)");
        Eigen::Vector3d y = get_vec3(prhs[1], "y");
        int shape_id = get_int_scalar(prhs[2], "shape_id");
        Eigen::VectorXd params = get_vecN(prhs[3], "params");

        double phi = 0.0;
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        shape_eval_local(y, shape_id, params, phi, grad, H);

        plhs[0] = make_struct_local(phi, grad, &H);
        return;
    }

    if (cmd == "global_xa") {
        require(nrhs == 6, "shape_mex:usage",
                "Usage: out = shape_core_mex('global_xa', g, x, alpha, shape_id, params)");
        Eigen::Matrix4d g = get_T44(prhs[1], "g");
        Eigen::Vector3d x = get_vec3(prhs[2], "x");
        double alpha = get_double_scalar(prhs[3], "alpha");
        int shape_id = get_int_scalar(prhs[4], "shape_id");
        Eigen::VectorXd params = get_vecN(prhs[5], "params");

        double phi = 0.0;
        Eigen::Vector4d grad = Eigen::Vector4d::Zero();
        Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
        // NOTE: call your renamed function:
        shape_eval_global_xa(g, x, alpha, shape_id, params, phi, grad, H);

        plhs[0] = make_struct_global(phi, grad, &H);
        return;
    }

    mexErrMsgIdAndTxt("shape_mex:badCmd", "Unknown command '%s'.", cmd.c_str());
}
