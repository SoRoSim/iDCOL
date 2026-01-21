#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "shape_core.hpp" // geometry API

namespace idcol {

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

struct ProblemData {
    Matrix4d g; // relative pose g = g1^{-1} g2
    int shape_id1 = 0;
    int shape_id2 = 0;
    VectorXd params1;
    VectorXd params2;
};

// F and J for z = [x(3); s; lambda1; lambda2], where alpha = exp(s)
void eval_F_J(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F,
    Matrix6d& J);

// F only (cheaper; uses phi+grad only)
void eval_F(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F);

// convenience
inline double merit_from_F(const Vector6d& F) { return 0.5 * F.squaredNorm(); }
inline Vector6d grad_merit(const Matrix6d& J, const Vector6d& F) { return J.transpose() * F; }
inline Matrix6d hess_gn(const Matrix6d& J) { return J.transpose() * J; }

} 
