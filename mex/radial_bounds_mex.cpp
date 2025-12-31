// radial_bounds_mex.cpp
//
// Usage (MATLAB):
//   out = radial_bounds_mex(shape_id, params)
//   out = radial_bounds_mex(shape_id, params, opt)
//
// Returns struct with fields:
//   Rin2, Rout2, Rin, Rout, xin, xout
//
// Build example:
//   mex -v CXXFLAGS="\$CXXFLAGS -std=c++17 -O3" -I<eigen> -I<include_root> ...
//       radial_bounds_mex.cpp core/radial_bounds.cpp core/shape_core.cpp

#include "mex.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <stdexcept>
#include <cmath>

// Your headers
#include "core/radial_bounds.hpp"   // declares RadialBounds, RadialBoundsOptions, compute_radial_bounds_local
#include "core/shape_core.hpp"      // shape_eval_local, etc.

namespace {

// ---------- helpers ----------
static bool is_real_scalar_double(const mxArray* a) {
    return mxIsDouble(a) && !mxIsComplex(a) && mxGetNumberOfElements(a) == 1;
}

static int get_int_scalar_required(const mxArray* a, const char* name) {
    if (!a || !is_real_scalar_double(a))
        mexErrMsgIdAndTxt("rb:badArg", "%s must be a real scalar double.", name);
    return static_cast<int>(std::llround(mxGetScalar(a)));
}

static Eigen::VectorXd get_vec_required(const mxArray* a, const char* name) {
    if (!a || !mxIsDouble(a) || mxIsComplex(a))
        mexErrMsgIdAndTxt("rb:badArg", "%s must be a real double array.", name);
    const mwSize n = mxGetNumberOfElements(a);
    if (n < 1) mexErrMsgIdAndTxt("rb:badArg", "%s must be non-empty.", name);

    Eigen::VectorXd v((int)n);
    const double* p = mxGetPr(a);
    for (mwSize i = 0; i < n; ++i) v((int)i) = p[i];
    return v;
}

static bool has_field(const mxArray* s, const char* name) {
    return mxGetField(s, 0, name) != nullptr;
}

static double get_opt_scalar(const mxArray* opt, const char* field, double default_val) {
    const mxArray* f = mxGetField(opt, 0, field);
    if (!f) return default_val;
    if (!is_real_scalar_double(f))
        mexErrMsgIdAndTxt("rb:badOpt", "opt.%s must be a real scalar double.", field);
    return mxGetScalar(f);
}

static int get_opt_int(const mxArray* opt, const char* field, int default_val) {
    const mxArray* f = mxGetField(opt, 0, field);
    if (!f) return default_val;
    if (!is_real_scalar_double(f))
        mexErrMsgIdAndTxt("rb:badOpt", "opt.%s must be a real scalar double (used as int).", field);
    return static_cast<int>(std::llround(mxGetScalar(f)));
}

static RadialBoundsOptions parse_opt_optional(const mxArray* optmx) {
    RadialBoundsOptions opt; // uses C++ defaults
    if (!optmx || mxIsEmpty(optmx)) return opt;
    if (!mxIsStruct(optmx))
        mexErrMsgIdAndTxt("rb:badOpt", "opt must be a struct or [].");

    // Fill only if present; keep defaults otherwise.
    if (has_field(optmx, "t0"))             opt.t0 = get_opt_scalar(optmx, "t0", opt.t0);
    if (has_field(optmx, "max_grow_steps")) opt.max_grow_steps = get_opt_int(optmx, "max_grow_steps", opt.max_grow_steps);
    if (has_field(optmx, "grow"))           opt.grow = get_opt_scalar(optmx, "grow", opt.grow);
    if (has_field(optmx, "bisect_iters"))   opt.bisect_iters = get_opt_int(optmx, "bisect_iters", opt.bisect_iters);
    if (has_field(optmx, "init_phi_tol"))   opt.init_phi_tol = get_opt_scalar(optmx, "init_phi_tol", opt.init_phi_tol);

    if (has_field(optmx, "newton_iters"))   opt.newton_iters = get_opt_int(optmx, "newton_iters", opt.newton_iters);
    if (has_field(optmx, "F_tol"))          opt.F_tol = get_opt_scalar(optmx, "F_tol", opt.F_tol);

    if (has_field(optmx, "num_starts"))     opt.num_starts = get_opt_int(optmx, "num_starts", opt.num_starts);
    if (has_field(optmx, "rng_seed"))       opt.rng_seed = (unsigned)get_opt_int(optmx, "rng_seed", (int)opt.rng_seed);

    return opt;
}

static mxArray* make_output(const RadialBounds& out) {
    const char* fields[] = {"Rin2","Rout2","Rin","Rout","xin","xout"};
    mxArray* S = mxCreateStructMatrix(1, 1, (int)(sizeof(fields)/sizeof(fields[0])), fields);

    mxSetField(S, 0, "Rin2",  mxCreateDoubleScalar(out.Rin2));
    mxSetField(S, 0, "Rout2", mxCreateDoubleScalar(out.Rout2));
    mxSetField(S, 0, "Rin",   mxCreateDoubleScalar(out.Rin));
    mxSetField(S, 0, "Rout",  mxCreateDoubleScalar(out.Rout));

    mxArray* xin = mxCreateDoubleMatrix(3, 1, mxREAL);
    double* pxin = mxGetPr(xin);
    pxin[0] = out.xin[0]; pxin[1] = out.xin[1]; pxin[2] = out.xin[2];
    mxSetField(S, 0, "xin", xin);

    mxArray* xout = mxCreateDoubleMatrix(3, 1, mxREAL);
    double* pxout = mxGetPr(xout);
    pxout[0] = out.xout[0]; pxout[1] = out.xout[1]; pxout[2] = out.xout[2];
    mxSetField(S, 0, "xout", xout);

    return S;
}

} // anonymous namespace

// ---------- gateway ----------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2 || nrhs > 3) {
        mexErrMsgIdAndTxt("rb:usage",
            "Usage:\n"
            "  out = mex_radial_bounds_local(shape_id, params)\n"
            "  out = mex_radial_bounds_local(shape_id, params, opt)\n");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("rb:usage", "One output only.");
    }

    const int shape_id = get_int_scalar_required(prhs[0], "shape_id");
    const Eigen::VectorXd params = get_vec_required(prhs[1], "params");
    const mxArray* optmx = (nrhs == 3 ? prhs[2] : nullptr);

    try {
        RadialBoundsOptions opt = parse_opt_optional(optmx);
        RadialBounds out = compute_radial_bounds_local(shape_id, params, opt);
        plhs[0] = make_output(out);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("rb:exception", "%s", e.what());
    } catch (...) {
        mexErrMsgIdAndTxt("rb:exception", "Unknown exception.");
    }
}
