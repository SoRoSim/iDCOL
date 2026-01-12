# \# iDCOL

# 

# iDCOL is a differentiable collision framework for convex contact geometry, designed to support gradient-based simulation and optimization in contact-rich robotics.

# 

# The framework provides efficient collision detection together with analytical derivatives of contact kinematics, and can be integrated as a modular component within physics engines and trajectory optimization frameworks.

# 

# \## Requirements

# 

# \- CMake (>= 3.16)

# \- A C++17-compatible compiler

# \- (Optional) MATLAB with C++17 MEX support for the MATLAB interface

# 

# \## Building and running the C++ code

# 

# From the repository root:

# 

# ```bash

# mkdir build

# cmake -S . -B build -DCMAKE\_BUILD\_TYPE=Release

# cmake --build build

# ```

# 

# The executable will be generated in the build directory (or `build/Release` for multi-config generators such as Visual Studio).

# 

# \### Running the C++ example

# 

# On Windows:

# ```bat

# build\\Release\\idcol\_ergodic.exe

# ```

# 

# On Linux/macOS:

# ```bash

# ./build/idcol\_ergodic

# ```

# 

# This example demonstrates collision detection over a sequence of configurations.

# 

# \## MATLAB MEX interface

# 

# iDCOL provides MATLAB MEX bindings for collision detection and contact-related computations.

# 

# To build the MEX functions, run the following command from MATLAB in the repository root:

# 

# ```matlab

# build\_mex

# ```

# 

# The compiled MEX binaries will be placed in the `mex/` directory.

# 

# \### MATLAB example

# 

# After building the MEX files, a minimal example can be run from MATLAB:

# 

# ```matlab

# % Example usage

# \[z\_star, Jc] = idcol\_solve\_mex(...);

# ```

# 

# \## Notes

# 

# \- Generated build directories and compiled MEX binaries are not tracked in the repository.

# \- The `mex/` directory contains source files (`.cpp`) and build scripts only.

# \- The framework is designed to be modular and extensible for use in contact-aware simulation and optimization pipelines.

# 

