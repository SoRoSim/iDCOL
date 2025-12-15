#include "idcol_kkt.hpp"

namespace idcol {

void eval_F_J(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F,
    Matrix6d& J)
{
    const double alpha = std::exp(s);

    // Evaluate phi1, grad1, H1
    double  phi1;
    Vector4d grad1;
    Matrix4d H1;
    shape_eval_global_ax(P.g1, x, alpha, P.shape_id1, P.params1, phi1, grad1, H1);

    // Evaluate phi2, grad2, H2
    double  phi2;
    Vector4d grad2;
    Matrix4d H2;
    shape_eval_global_ax(P.g2, x, alpha, P.shape_id2, P.params2, phi2, grad2, H2);

    // Split gradients
    const Vector3d g1x = grad1.head<3>();
    const double   g1a = grad1(3);

    const Vector3d g2x = grad2.head<3>();
    const double   g2a = grad2(3);

    // Split Hessians (block views)
    const auto H1_xx = H1.block<3,3>(0,0);
    const auto H1_xa = H1.block<3,1>(0,3);
    const double H1_aa = H1(3,3);

    const auto H2_xx = H2.block<3,3>(0,0);
    const auto H2_xa = H2.block<3,1>(0,3);
    const double H2_aa = H2(3,3);

    // Residual F(z)
    F(0) = phi1;
    F(1) = phi2;
    F.segment<3>(2) = lambda1 * g1x + lambda2 * g2x;
    F(5) = 1.0 + lambda1 * g1a + lambda2 * g2a;

    // Jacobian J
    J.setZero();

    // Row 0: F1 = phi1
    J.block<1,3>(0,0) = g1x.transpose();
    J(0,3) = g1a * alpha;

    // Row 1: F2 = phi2
    J.block<1,3>(1,0) = g2x.transpose();
    J(1,3) = g2a * alpha;

    // Rows 2..4: Fx = λ1 ∇x φ1 + λ2 ∇x φ2
    for (int j = 0; j < 3; ++j) {
        const int row = 2 + j;

        // dF_j/dx
        J.block<1,3>(row, 0) = lambda1 * H1_xx.row(j) + lambda2 * H2_xx.row(j);

        // dF_j/ds
        J(row, 3) = (lambda1 * H1_xa(j) + lambda2 * H2_xa(j)) * alpha;

        // dF_j/dλ1, dF_j/dλ2
        J(row, 4) = g1x(j);
        J(row, 5) = g2x(j);
    }

    // Row 5: F6 = 1 + λ1 g1a + λ2 g2a

    // dF6/dx = λ1 * d(g1a)/dx + λ2 * d(g2a)/dx
    // with symmetry: d(g_a)/dx = (H_xa)^T
    const Eigen::RowVector3d dF6_dx =
        lambda1 * H1_xa.transpose() + lambda2 * H2_xa.transpose();
    J.block<1,3>(5,0) = dF6_dx;

    // dF6/ds = (λ1 H1_aa + λ2 H2_aa) * α
    J(5,3) = (lambda1 * H1_aa + lambda2 * H2_aa) * alpha;

    // dF6/dλ1, dF6/dλ2
    J(5,4) = g1a;
    J(5,5) = g2a;
}

void eval_F(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F)
{
    const double alpha = std::exp(s);

    // φ1, grad1
    double  phi1;
    Vector4d grad1;
    shape_eval_global_ax_phi_grad(P.g1, x, alpha, P.shape_id1, P.params1, phi1, grad1);

    // φ2, grad2
    double  phi2;
    Vector4d grad2;
    shape_eval_global_ax_phi_grad(P.g2, x, alpha, P.shape_id2, P.params2, phi2, grad2);

    const Vector3d g1x = grad1.head<3>();
    const double   g1a = grad1(3);

    const Vector3d g2x = grad2.head<3>();
    const double   g2a = grad2(3);

    F(0) = phi1;
    F(1) = phi2;
    F.segment<3>(2) = lambda1 * g1x + lambda2 * g2x;
    F(5) = 1.0 + lambda1 * g1a + lambda2 * g2a;
}

} // namespace idcol
