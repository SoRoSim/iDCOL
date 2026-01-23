// idcol_kkt_mex.cpp
//
// MATLAB usage:
//   out = idcol_kkt_mex('F',  x, s, lambda1, lambda2, P)
//   out = idcol_kkt_mex('FJ', x, s, lambda1, lambda2, P)
//
// where:
//   x        : 3x1
//   s        : scalar (alpha = exp(s))
//   lambda1  : scalar
//   lambda2  : scalar
//   P        : struct with fields
//              P.g1 (4x4), P.g2 (4x4),
//              P.shape_id1 (scalar), P.shape_id2 (scalar),
//              P.params1 (vector), P.params2 (vector)
//
// Returns struct:
//   out.F    : 6x1
//   out.J    : 6x6   (only for "FJ")
//
// Build example (adjust include + sources):
//   mex -v CXXFLAGS="\$CXXFLAGS -std=c++17 -O3" -I<eigen> -I<include_root> ...
//       idcol_kkt_mex.cpp core/idcol_kkt.cpp core/shape_core.cpp

#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <string>
#include <cmath>

#include "core/idcol_kkt.hpp"   // your header shown above

namespace {

// ---------- helpers ----------
static void require(bool cond, const char* id, const char* msg) {
    if (!cond) mexErrMsgIdAndTxt(id, "%s", msg);
}

static bool is_real_double(const mxArray* a) {
    return mxIsDouble(a) && !mxIsComplex(a);
}

static int get_int_scalar(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 1,
            "kkt_mex:badArg", (std::string(name) + " must be a real scalar double (used as int).").c_str());
    return (int)std::llround(mxGetScalar(a));
}

static double get_double_scalar(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 1,
            "kkt_mex:badArg", (std::string(name) + " must be a real scalar double.").c_str());
    return mxGetScalar(a);
}

static Eigen::Vector3d get_vec3(const mxArray* a, const char* name) {
    require(a && is_real_double(a) && mxGetNumberOfElements(a) == 3,
            "kkt_mex:badArg", (std::string(name) + " must have 3 elements.").c_str());
    const double* p = mxGetPr(a);
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

static Eigen::VectorXd get_vecN_required_field(const mxArray* s, const char* field) {
    const mxArray* f = mxGetField(s, 0, field);
    require(f != nullptr, "kkt_mex:missingField",
            (std::string("Missing field 'P.") + field + "'.").c_str());
    require(is_real_double(f), "kkt_mex:badField",
            (std::string("Field 'P.") + field + "' must be real double array.").c_str());

    const mwSize n = mxGetNumberOfElements(f);
    require(n >= 1, "kkt_mex:badField",
            (std::string("Field 'P.") + field + "' must be non-empty.").c_str());

    Eigen::VectorXd v((int)n);
    const double* p = mxGetPr(f);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

static Eigen::Matrix4d get_T44_required_field(const mxArray* s, const char* field) {
    const mxArray* f = mxGetField(s, 0, field);
    require(f != nullptr, "kkt_mex:missingField",
            (std::string("Missing field 'P.") + field + "'.").c_str());
    require(is_real_double(f) && mxGetM(f) == 4 && mxGetN(f) == 4,
            "kkt_mex:badField",
            (std::string("Field 'P.") + field + "' must be 4x4 real double.").c_str());

    Eigen::Matrix4d T;
    const double* p = mxGetPr(f);
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            T(r,c) = p[r + 4*c];
    return T;
}

static int get_int_required_field(const mxArray* s, const char* field) {
    const mxArray* f = mxGetField(s, 0, field);
    require(f != nullptr, "kkt_mex:missingField",
            (std::string("Missing field 'P.") + field + "'.").c_str());
    require(is_real_double(f) && mxGetNumberOfElements(f) == 1,
            "kkt_mex:badField",
            (std::string("Field 'P.") + field + "' must be a real scalar double (used as int).").c_str());
    return (int)std::llround(mxGetScalar(f));
}

static idcol::ProblemData parse_problem(const mxArray* Pmx) {
    require(Pmx && mxIsStruct(Pmx), "kkt_mex:badArg", "P must be a struct.");

    idcol::ProblemData P;
    P.g = get_T44_required_field(Pmx, "g");
    P.shape_id1 = get_int_required_field(Pmx, "shape_id1");
    P.shape_id2 = get_int_required_field(Pmx, "shape_id2");
    P.params1 = get_vecN_required_field(Pmx, "params1");
    P.params2 = get_vecN_required_field(Pmx, "params2");
    return P;
}

static mxArray* make_output_F(const idcol::Vector6d& F) {
    const char* fields[] = {"F"};
    mxArray* S = mxCreateStructMatrix(1,1,1,fields);

    mxArray* Fmx = mxCreateDoubleMatrix(6,1,mxREAL);
    double* pF = mxGetPr(Fmx);
    for (int i=0;i<6;++i) pF[i] = F[i];
    mxSetField(S,0,"F",Fmx);

    return S;
}

static mxArray* make_output_FJ(const idcol::Vector6d& F, const idcol::Matrix6d& J) {
    const char* fields[] = {"F","J"};
    mxArray* S = mxCreateStructMatrix(1,1,2,fields);

    mxArray* Fmx = mxCreateDoubleMatrix(6,1,mxREAL);
    double* pF = mxGetPr(Fmx);
    for (int i=0;i<6;++i) pF[i] = F[i];
    mxSetField(S,0,"F",Fmx);

    mxArray* Jmx = mxCreateDoubleMatrix(6,6,mxREAL);
    double* pJ = mxGetPr(Jmx);
    // column-major
    for (int c=0;c<6;++c) {
        for (int r=0;r<6;++r) {
            pJ[r + 6*c] = J(r,c);
        }
    }
    mxSetField(S,0,"J",Jmx);

    return S;
}

} // anonymous namespace

// ---------- gateway ----------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nlhs <= 1, "kkt_mex:usage", "One output only.");

    require(nrhs >= 1 && mxIsChar(prhs[0]),
            "kkt_mex:usage",
            "First input must be a command string.\n"
            "Usage:\n"
            "  out = idcol_kkt_mex('F',  x, s, lambda1, lambda2, P)\n"
            "  out = idcol_kkt_mex('FJ', x, s, lambda1, lambda2, P)\n");

    char cmd_buf[64];
    require(mxGetString(prhs[0], cmd_buf, sizeof(cmd_buf)) == 0,
            "kkt_mex:badArg", "Could not read command string.");
    const std::string cmd(cmd_buf);

    if (cmd == "F") {
        require(nrhs == 6, "kkt_mex:usage",
                "Usage: out = idcol_kkt_mex('F', x, s, lambda1, lambda2, P)");
        Eigen::Vector3d x = get_vec3(prhs[1], "x");
        double s = get_double_scalar(prhs[2], "s");
        double l1 = get_double_scalar(prhs[3], "lambda1");
        double l2 = get_double_scalar(prhs[4], "lambda2");
        idcol::ProblemData P = parse_problem(prhs[5]);

        idcol::Vector6d F;
        idcol::eval_F(x, s, l1, l2, P, F);

        plhs[0] = make_output_F(F);
        return;
    }

    if (cmd == "FJ") {
        require(nrhs == 6, "kkt_mex:usage",
                "Usage: out = idcol_kkt_mex('FJ', x, s, lambda1, lambda2, P)");
        Eigen::Vector3d x = get_vec3(prhs[1], "x");
        double s = get_double_scalar(prhs[2], "s");
        double l1 = get_double_scalar(prhs[3], "lambda1");
        double l2 = get_double_scalar(prhs[4], "lambda2");
        idcol::ProblemData P = parse_problem(prhs[5]);

        idcol::Vector6d F;
        idcol::Matrix6d J;
        idcol::eval_F_J(x, s, l1, l2, P, F, J);

        plhs[0] = make_output_FJ(F, J);
        return;
    }

    mexErrMsgIdAndTxt("kkt_mex:badCmd", "Unknown command '%s'.", cmd.c_str());
}
