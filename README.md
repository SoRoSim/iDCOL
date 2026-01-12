# iDCOL

iDCOL is a differentiable collision framework for convex contact geometry, designed to support gradient-based simulation and optimization in contact-rich robotics.

The framework provides efficient collision detection together with analytical derivatives of contact kinematics, and can be integrated as a modular component within physics engines and trajectory optimization frameworks.

# Requirements

 \- CMake (>= 3.16)
\- A C++17-compatible compiler
\- (Optional) MATLAB with C++17 MEX support for the MATLAB interface
# Building and running the C++ code

From the repository root:
 ```bash
# mkdir build

# cmake -S . -B build -DCMAKE\_BUILD\_TYPE=Release

# cmake --build build
```

The executable will be generated in the build directory (or `build/Release` for multi-config generators such as Visual Studio).
# Running the C++ example

On Windows:

 ```bat
# build\Release\idcol_ergodic.exe
```

On Linux/macOS:
 ```bash
./build/idcol_ergodic
 ```

This example demonstrates collision detection over a sequence of configurations.
# MATLAB MEX interface

iDCOL provides MATLAB MEX bindings for collision detection and contact-related computations. To build the MEX functions, run the following command from MATLAB in the repository root:

 ```bash 
build_mex.m
```

The compiled MEX binaries will be placed in the `mex/` directory.
# MATLAB example
After building the MEX files, a minimal example can be run from MATLAB:

 ```bash 
 [z_star, Jc] = idcol_solve_mex(...);
```


