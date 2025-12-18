#pragma once
#include <Eigen/Dense>
#include <cstdint>

struct RadialBounds {
    double Rin  = 0.0;
    double Rout = 0.0;
    double Rin2 = 0.0;
    double Rout2 = 0.0;

    Eigen::Vector3d xin  = Eigen::Vector3d::Zero();
    Eigen::Vector3d xout = Eigen::Vector3d::Zero();
};

struct RadialBoundsOptions {
    // multi-start
    int num_starts = 1000; //may need to run a lot if sharp areas are present
    uint32_t rng_seed = 12345;

    // ray-to-surface (initial feasible point)
    double t0 = 1e-3;
    double grow = 2.0;
    int max_grow_steps = 80;
    int bisect_iters = 80;
    double init_phi_tol = 1e-10;

    // Newton on KKT
    int newton_iters = 30;
    double F_tol = 1e-10;   // accept only if ||F|| <= F_tol
};

RadialBounds compute_radial_bounds_local(
    int shape_id,
    const Eigen::VectorXd& params,
    const RadialBoundsOptions& opt = RadialBoundsOptions{});
