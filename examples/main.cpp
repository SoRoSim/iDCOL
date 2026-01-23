#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

#include "core/idcol_implicitfamily.hpp"   // shape builders
#include "core/idcol_contactpair.hpp"      // ContactPair

int main()
{
    using namespace idcol;

    // -------------------------------------------------
    // 1) Define all supported shapes
    // -------------------------------------------------

    // ---- Sphere (shape_id = 1)
    ShapeSpec sphere = make_sphere(0.75);

    // ---- Polytope: Ax <= b (shape_id = 2)
    Eigen::MatrixXd A(8,3);
    A <<  1,  1,  1,
          1, -1, -1,
         -1,  1, -1,
         -1, -1,  1,
         -1, -1, -1,
         -1,  1,  1,
          1, -1,  1,
          1,  1, -1;

    Eigen::VectorXd b(8);
    b << 1.0, 1.0, 1.0, 1.0,
         5.0/3.0, 5.0/3.0, 5.0/3.0, 5.0/3.0;

    double beta = 20.0;
    ShapeSpec poly = make_poly(beta, A, b);

    // ---- Superellipsoid (shape_id = 3)
    int n = 8;
    ShapeSpec se = make_se(n, 0.5, 1.0, 1.5);

    // ---- Superelliptic cylinder (shape_id = 4)
    ShapeSpec sec = make_sec(n, 1.0, 2.0);   // radius, half-height

    // ---- Truncated cone (shape_id = 5)
    ShapeSpec tc = make_tc(beta, 1.0, 1.5, 1.5, 1.5);

    // -------------------------------------------------
    // 2) Pick any contact pair
    // -------------------------------------------------
    ContactPair pair(poly, se);   // poly–superellipsoid contact

    // -------------------------------------------------
    // 3) Relative pose between the two bodies
    // -------------------------------------------------
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g.topRightCorner<3,1>() << -1.8, -2.7, -0.3;

    // -------------------------------------------------
    // 4) Solve contact
    // -------------------------------------------------
    SolveResult out = pair.solve(g);

    if (!out.newton.converged) {
        std::cout << "[iDCOL] did not converge\n";
        return 0;
    }

    // -------------------------------------------------
    // 5) Inspect solution
    // -------------------------------------------------
    std::cout << "[iDCOL] converged\n"
              << "  contact point x = " << out.newton.x.transpose() << "\n"
              << "  alpha           = " << out.newton.alpha << "\n"
              << "  lambda1         = " << out.newton.lambda1 << "\n"
              << "  lambda2         = " << out.newton.lambda2 << "\n"
              << "  iterations      = " << out.newton.iters_used << "\n";

    // -------------------------------------------------
    // 6) Warm start is automatic
    // -------------------------------------------------
    // Calling pair.solve(g_next) will reuse the previous solution.

    return 0;
}