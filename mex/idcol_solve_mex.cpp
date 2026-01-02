// mex_idcol_solve.cpp
// Build: mex -v CXXFLAGS="\$CXXFLAGS -std=c++17 -O3" -I<eigen> -I<idcol_include> mex_idcol_solve.cpp <other .cpp files...>

#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>

// ---- Project headers ----
#include "core/idcol_solve.hpp"      // defines: idcol::SolveData, idcol::idcol_solve, SolveResult, SurrogateOptions
#include "core/idcol_kkt.hpp"        // defines: idcol::ProblemData, etc.
#include "core/radial_bounds.hpp"    // defines: RadialBounds
#include "core/idcol_newton.hpp"     // defines: idcol::NewtonOptions, NewtonResult

namespace {

// ------------------------ basic helpers ------------------------

static bool has_field(const mxArray* s, const char* name) {
    return (mxGetField(s, 0, name) != nullptr);
}

static double get_scalar(const mxArray* s, const char* name, double default_val, bool required=false) {
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) {
        if (required) mexErrMsgIdAndTxt("idcol:missingField", "Missing field '%s'.", name);
        return default_val;
    }
    if (!mxIsDouble(f) || mxIsComplex(f) || mxGetNumberOfElements(f) != 1)
        mexErrMsgIdAndTxt("idcol:badField", "Field '%s' must be a real scalar double.", name);
    return mxGetScalar(f);
}

static int get_int_scalar(const mxArray* s, const char* name, int default_val, bool required=false) {
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) {
        if (required) mexErrMsgIdAndTxt("idcol:missingField", "Missing field '%s'.", name);
        return default_val;
    }
    if (!mxIsDouble(f) || mxIsComplex(f) || mxGetNumberOfElements(f) != 1)
        mexErrMsgIdAndTxt("idcol:badField", "Field '%s' must be a real scalar double (used as int).", name);
    return static_cast<int>(std::llround(mxGetScalar(f)));
}

static Eigen::VectorXd get_vec_required(const mxArray* s, const char* name) {
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) mexErrMsgIdAndTxt("idcol:missingField", "Missing field '%s'.", name);
    if (!mxIsDouble(f) || mxIsComplex(f))
        mexErrMsgIdAndTxt("idcol:badField", "Field '%s' must be a real double array.", name);

    const mwSize n = mxGetNumberOfElements(f);
    Eigen::VectorXd v((int)n);
    const double* p = mxGetPr(f);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

static Eigen::Matrix4d get_T44_required(const mxArray* s, const char* name) {
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) mexErrMsgIdAndTxt("idcol:missingField", "Missing field '%s'.", name);
    if (!mxIsDouble(f) || mxIsComplex(f) || mxGetM(f) != 4 || mxGetN(f) != 4)
        mexErrMsgIdAndTxt("idcol:badField", "Field '%s' must be 4x4 real double.", name);

    Eigen::Matrix4d T;
    const double* p = mxGetPr(f);
    // MATLAB is column-major; Eigen default is column-major. Fill explicitly.
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            T(r, c) = p[r + 4 * c];
    return T;
}

static std::vector<int> get_int_vector_optional(const mxArray* s, const char* name) {
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) return {};
    if (mxIsEmpty(f)) return {};
    if (!mxIsDouble(f) || mxIsComplex(f))
        mexErrMsgIdAndTxt("idcol:badField", "Field '%s' must be real double array (used as int vector).", name);
    const mwSize n = mxGetNumberOfElements(f);
    std::vector<int> out; out.reserve((size_t)n);
    const double* p = mxGetPr(f);
    for (mwSize i = 0; i < n; ++i) out.push_back((int)std::llround(p[i]));
    return out;
}

// ------------------------ parse core structs ------------------------

static idcol::ProblemData parse_problem(const mxArray* Pmx) {
    if (!mxIsStruct(Pmx)) mexErrMsgIdAndTxt("idcol:badArg", "S.P must be a struct.");

    idcol::ProblemData P;
    P.g1 = get_T44_required(Pmx, "g1");
    P.g2 = get_T44_required(Pmx, "g2");

    P.shape_id1 = get_int_scalar(Pmx, "shape_id1", 0, /*required=*/true);
    P.shape_id2 = get_int_scalar(Pmx, "shape_id2", 0, /*required=*/true);

    P.params1 = get_vec_required(Pmx, "params1");
    P.params2 = get_vec_required(Pmx, "params2");

    return P;
}

static RadialBounds parse_bounds(const mxArray* Bmx, const char* which_name) {
    if (!mxIsStruct(Bmx)) mexErrMsgIdAndTxt("idcol:badArg", "%s must be a struct.", which_name);

    RadialBounds b;
    b.Rin  = get_scalar(Bmx, "Rin",  0.0, /*required=*/true);
    b.Rout = get_scalar(Bmx, "Rout", 0.0, /*required=*/true);
    return b;
}

static idcol::SolveData parse_solvedata(const mxArray* Smx) {
    if (!mxIsStruct(Smx)) mexErrMsgIdAndTxt("idcol:badArg", "First input S must be a struct.");

    const mxArray* Pmx  = mxGetField(Smx, 0, "P");
    const mxArray* b1mx = mxGetField(Smx, 0, "bounds1");
    const mxArray* b2mx = mxGetField(Smx, 0, "bounds2");

    if (!Pmx)  mexErrMsgIdAndTxt("idcol:missingField", "Missing field 'S.P'.");
    if (!b1mx) mexErrMsgIdAndTxt("idcol:missingField", "Missing field 'S.bounds1'.");
    if (!b2mx) mexErrMsgIdAndTxt("idcol:missingField", "Missing field 'S.bounds2'.");

    idcol::SolveData S;
    S.P = parse_problem(Pmx);
    S.bounds1 = parse_bounds(b1mx, "S.bounds1");
    S.bounds2 = parse_bounds(b2mx, "S.bounds2");
    return S;
}

static idcol::NewtonOptions parse_opt(const mxArray* Omx) {
    if (!mxIsStruct(Omx)) mexErrMsgIdAndTxt("idcol:badArg", "opt must be a struct.");

    idcol::NewtonOptions opt; // start from C++ defaults

    // Only overwrite if provided (keeps compatibility with your evolving NewtonOptions)
    if (has_field(Omx, "max_iters")) opt.max_iters = get_int_scalar(Omx, "max_iters", opt.max_iters);
    if (has_field(Omx, "tol"))       opt.tol       = get_scalar(Omx, "tol", opt.tol);
    if (has_field(Omx, "s_min"))     opt.s_min     = get_scalar(Omx, "s_min", opt.s_min);
    if (has_field(Omx, "s_max"))     opt.s_max     = get_scalar(Omx, "s_max", opt.s_max);

    // Add any other fields you want to expose here as your struct grows
    // e.g. opt.damping = get_scalar(Omx, "damping", opt.damping);

    return opt;
}

static idcol::NewtonOptions parse_opt_optional(const mxArray* Omx) {
    idcol::NewtonOptions opt;                 // defaults
    if (Omx == nullptr || mxIsEmpty(Omx)) return opt;
    return parse_opt(Omx);                    // strict parse
}


static std::optional<idcol::Guess> parse_guess_optional(const mxArray* Gmx) {
    if (Gmx == nullptr || mxIsEmpty(Gmx)) return std::nullopt;
    if (!mxIsStruct(Gmx)) mexErrMsgIdAndTxt("idcol:badArg", "guess must be a struct or [].");

    idcol::Guess g;

    const mxArray* xmx = mxGetField(Gmx, 0, "x");
    if (!xmx || !mxIsDouble(xmx) || mxIsComplex(xmx) || mxGetNumberOfElements(xmx) != 3)
        mexErrMsgIdAndTxt("idcol:badField", "guess.x must be 3x1 real double.");

    const double* px = mxGetPr(xmx);
    g.x = Eigen::Vector3d(px[0], px[1], px[2]);

    g.alpha   = get_scalar(Gmx, "alpha",   1.0);
    g.lambda1 = get_scalar(Gmx, "lambda1", 0.0);
    g.lambda2 = get_scalar(Gmx, "lambda2", 0.0);

    return g;
}

static idcol::SurrogateOptions parse_sopt_optional(const mxArray* Smx_opt) {
    idcol::SurrogateOptions sopt;

    if (Smx_opt == nullptr || mxIsEmpty(Smx_opt)) return sopt;
    if (!mxIsStruct(Smx_opt)) mexErrMsgIdAndTxt("idcol:badArg", "sopt must be a struct or [].");

    if (has_field(Smx_opt, "fS_values")) {
        sopt.fS_values = get_int_vector_optional(Smx_opt, "fS_values");
    }

     // ---- geometric scaling ----
    if (has_field(Smx_opt, "enable_scaling")) {
        // accept logical or scalar double
        const mxArray* f = mxGetField(Smx_opt, 0, "enable_scaling");
        if (mxIsLogicalScalar(f)) {
            sopt.enable_scaling = mxIsLogicalScalarTrue(f);
        } else if (mxIsDouble(f) && !mxIsComplex(f) && mxGetNumberOfElements(f) == 1) {
            sopt.enable_scaling = (mxGetScalar(f) != 0.0);
        } else {
            mexErrMsgIdAndTxt("idcol:badField", "sopt.enable_scaling must be logical or scalar.");
        }
    }

    if (has_field(Smx_opt, "scale_mode")) {
        const mxArray* f = mxGetField(Smx_opt, 0, "scale_mode");
        if (!mxIsChar(f)) mexErrMsgIdAndTxt("idcol:badField", "sopt.scale_mode must be a char array.");
        char* cstr = mxArrayToString(f);
        if (!cstr) mexErrMsgIdAndTxt("idcol:badField", "Failed to read sopt.scale_mode.");
        sopt.scale_mode = std::string(cstr);
        mxFree(cstr);
    }
    mexPrintf("[mex] enable_scaling=%d scale_mode=%s\n",
          (int)sopt.enable_scaling, sopt.scale_mode.c_str());


    return sopt;
}

// ------------------------ MATLAB output ------------------------

static mxArray* make_output(const idcol::SolveResult& out) {
    const char* fields[] = {
        "converged","iters_used","final_F_norm","message",
        "x","alpha","lambda1","lambda2",
        "fS_used","fS_attempts_used","used_surrogate",
        "F","J"
    };
    mxArray* S = mxCreateStructMatrix(1, 1, (int)(sizeof(fields)/sizeof(fields[0])), fields);

    // Scalars / status
    mxSetField(S, 0, "converged",   mxCreateLogicalScalar(out.newton.converged));
    mxSetField(S, 0, "iters_used",  mxCreateDoubleScalar((double)out.newton.iters_used));
    mxSetField(S, 0, "final_F_norm",mxCreateDoubleScalar(out.newton.final_F_norm));
    mxSetField(S, 0, "message",     mxCreateString(out.newton.message.c_str()));

    // x (3x1)
    mxArray* x = mxCreateDoubleMatrix(3, 1, mxREAL);
    double* px = mxGetPr(x);
    px[0] = out.newton.x[0];
    px[1] = out.newton.x[1];
    px[2] = out.newton.x[2];
    mxSetField(S, 0, "x", x);

    // alpha, lambdas
    mxSetField(S, 0, "alpha",   mxCreateDoubleScalar(out.newton.alpha));
    mxSetField(S, 0, "lambda1", mxCreateDoubleScalar(out.newton.lambda1));
    mxSetField(S, 0, "lambda2", mxCreateDoubleScalar(out.newton.lambda2));

    // surrogate metadata
    mxSetField(S, 0, "fS_used",          mxCreateDoubleScalar((double)out.fS_used));
    mxSetField(S, 0, "fS_attempts_used", mxCreateDoubleScalar((double)out.fS_attempts_used));
    mxSetField(S, 0, "used_surrogate",   mxCreateLogicalScalar(out.used_surrogate));

    // F (m x 1)
    {
        const auto& F = out.newton.F; // Eigen vector
        mxArray* Fmx = mxCreateDoubleMatrix((mwSize)F.size(), 1, mxREAL);
        double* pF = mxGetPr(Fmx);
        for (int i = 0; i < F.size(); ++i) pF[i] = F[i];
        mxSetField(S, 0, "F", Fmx);
    }

    // J (m x n)  -- for your case typically 6x6
    {
        const auto& J = out.newton.J; // Eigen matrix
        mxArray* Jmx = mxCreateDoubleMatrix((mwSize)J.rows(), (mwSize)J.cols(), mxREAL);
        double* pJ = mxGetPr(Jmx);

        // Both MATLAB and Eigen are column-major by default.
        for (int c = 0; c < J.cols(); ++c) {
            for (int r = 0; r < J.rows(); ++r) {
                pJ[r + c * J.rows()] = J(r, c);
            }
        }
        mxSetField(S, 0, "J", Jmx);
    }

    return S;
}

} // anonymous namespace

// ------------------------ MEX gateway ------------------------

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

    if (nrhs < 1 || nrhs > 4) {
        mexErrMsgIdAndTxt("idcol:usage",
            "Usage:\n"
            "  out = mex_idcol_solve(S)\n"
            "  out = mex_idcol_solve(S, guess)\n"
            "  out = mex_idcol_solve(S, guess, opt)\n"
            "  out = mex_idcol_solve(S, guess, opt, sopt)\n");
    }

    const mxArray* Smx      = prhs[0];
    const mxArray* Gmx      = (nrhs >= 2 ? prhs[1] : nullptr);
    const mxArray* Omx      = (nrhs >= 3 ? prhs[2] : nullptr);
    const mxArray* Smx_opt  = (nrhs >= 4 ? prhs[3] : nullptr);

    idcol::SolveData S = parse_solvedata(Smx);

    std::optional<idcol::Guess> guess = parse_guess_optional(Gmx);
    idcol::NewtonOptions opt          = parse_opt_optional(Omx);
    idcol::SurrogateOptions sopt      = parse_sopt_optional(Smx_opt);

    idcol::SolveResult out = idcol::idcol_solve(S, guess, opt, sopt);

    plhs[0] = make_output(out);
}

