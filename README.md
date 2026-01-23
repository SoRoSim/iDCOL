<p align="center">
  <img src="idcol.png" width="200">
</p>

# iDCOL

iDCOL is a differentiable collision framework for convex contact geometry, designed for gradient-based simulation, planning, and optimization in contact-rich robotic systems.

It provides:

* robust collision detection for strictly convex implicit shapes,
* analytical derivatives of contact kinematics,
* efficient warm-started contact tracking,
* a lightweight C++ API suitable for physics engines and optimizers.

At its core, iDCOL reduces contact computation to a fixed-size nonlinear solve, making it fast, differentiable, and easy to integrate.

---

## Quickstart (C++)

The intended usage is deliberately simple:

1. Create shapes
2. Create a contact pair
3. Solve contact at a relative pose

```cpp
#include <Eigen/Dense>
#include <iostream>

#include "core/idcol_implicitfamily.hpp"
#include "core/idcol_contactpair.hpp"

int main() {
    using namespace idcol;

    // --- 1) Create shapes ---
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
    b << 1,1,1,1, 5.0/3,5.0/3,5.0/3,5.0/3;

    auto poly  = make_poly(20.0, A, b);
    auto ellip = make_se(1.0, 0.5, 1.0, 1.5);

    // --- 2) Create a contact pair ---
    ContactPair pair(poly, ellip);

    // --- 3) Solve contact ---
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g.topRightCorner<3,1>() << 0.2, 0.1, 0.3;

    SolveResult out = pair.solve(g);

    if (!out.newton.converged) {
        std::cout << "Contact solve failed: "
                  << out.newton.message << std::endl;
        return 0;
    }

    std::cout << "Contact solved in "
              << out.newton.iters_used << " iterations\n";
    std::cout << "Contact point x = "
              << out.newton.x.transpose() << "\n";
    std::cout << "alpha = " << out.newton.alpha << "\n";
}
```

That is the entire workflow.

* Warm starting is handled internally by ContactPair
* To force a cold start, call pair.reset_guess()
* Analytical Jacobians are available in out.newton.J

---

## Supported shapes

Shapes are constructed using helper functions:

* make_poly(beta, A, b)
  Smoothed convex polytope Ax <= b

* make_tc(beta, rb, rt, a, b)
  Smooth truncated cone

* make_se(n, a, b, c)
  Superellipsoid (use n = 1 for ellipsoid)

* make_sec(n, r, h)
  Superelliptic cylinder

Each constructor returns a ShapeSpec containing:

* shape identifier,
* packed parameters,
* precomputed radial bounds.

---

## Getting the source

This repository uses Git submodules.

After cloning, initialize dependencies with:

```bash
git submodule update --init --recursive
```
---

## Requirements

* CMake (>= 3.16)
* A C++17-compatible compiler
* Eigen (included as a submodule)
* (Optional) MATLAB with C++17 MEX support

---

## Building the C++ code

### Linux / macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Windows (Visual Studio)

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Executables are generated in:

* build/ on Linux/macOS
* build/Release/ on Windows

---

## Running the C++ example

Windows:

```
build\Release\idcol_ergodic.exe
```

Linux/macOS:

```
./build/idcol_ergodic
```

This example demonstrates warm-started contact tracking over a sequence of configurations.

---

## MATLAB MEX interface (optional)

iDCOL provides MATLAB MEX bindings.

To build:

```
build_mex
```

Compiled MEX files are placed in mex/.

MATLAB usage:

```
out = idcol_solve_mex(S, guess, opt, sopt);
```

The output structure contains the contact solution, residuals, and analytical Jacobians.

---

## Design philosophy

iDCOL is built around three principles:

* Strictly convext implicit premitives
* Fixed-size solvers for predictable performance
* Analytical derivatives first