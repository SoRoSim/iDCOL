#pragma once
#include <Eigen/Dense>

void shape_eval_local_phi_grad( //no Hessian
    const Eigen::Vector3d& y,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector3d& grad_phi);

void shape_eval_global_xa_phi_grad( //no Hessian
    const Eigen::Matrix4d& g,
    const Eigen::Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector4d& grad);

void shape_eval_local(
    const Eigen::Vector3d& y,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector3d& grad_phi,
    Eigen::Matrix3d& hess_phi);

void shape_eval_global_xa(
    const Eigen::Matrix4d& g,
    const Eigen::Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector4d& grad,      // [dphi/dx; dphi/dalpha]
    Eigen::Matrix4d& H);        // Hessian wrt [x; alpha]
