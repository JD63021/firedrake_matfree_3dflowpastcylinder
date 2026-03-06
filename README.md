matfree3dcyl_p2p2.py

This code solves the 3D incompressible Navier-Stokes equations for flow past a cylinder using Firedrake.

Main features:
- Uses a fully matrix-free linear solve by default
- Uses equal-order P2-P2 finite elements
  - quadratic velocity
  - quadratic pressure
- Uses implicit time stepping
  - BDF1 for the first step
  - BDF2 for later steps
- Uses Picard iteration to treat the nonlinear convection term
- Supports SUPG and PSPG stabilization
- Includes LSIC-type divergence stabilization term
- Builds a geometric multigrid hierarchy from a linear base mesh
- Curves only the finest mesh near the cylinder to better match the geometry
- Can optionally fall back to a full LU solve for debugging

Mesh and geometry:
- The mesh is generated using a Netgen-based helper script
- The problem is a 3D channel with a circular cylinder inside it
- Boundary tags are used for inlet, outlet, cylinder, and walls
- The inlet uses a time-dependent parabolic velocity profile

Boundary conditions:
- Prescribed velocity at the inlet
- No-slip condition on the cylinder and channel walls
- Optional pressure Dirichlet condition at the outlet
- A pressure penalty term is included to help regularize the equal-order formulation

Time stepping:
- The time step is chosen from a CFL-like estimate based on mesh size and current velocity
- The first step uses BDF1
- All later steps use BDF2
- The inlet amplitude varies sinusoidally in time

Nonlinear solve:
- Each time step is solved with Picard iterations
- The code monitors the true assembled nonlinear residual
- Dirichlet degrees of freedom are removed from the residual norm before checking convergence
- Solver rebuilds are attempted automatically if convergence fails

Postprocessing:
- Computes force on the cylinder in two ways:
  1. Surface traction integral
  2. Reaction force from the assembled momentum residual
- Writes velocity and pressure output to VTK files
- Writes drag and lift history to a CSV file

Checkpoint and restart:
- Saves checkpoint files at a chosen frequency
- Stores one NPZ file per MPI rank
- Stores metadata in a JSON file
- Can restart from a previous checkpoint

Typical use:
This script is meant for for testing matrix-free solvers, multigrid, stabilization, and force extraction methods.

Citation (Code formulation): 
The code formulation is not mine, it is inspired from the following article: https://doi.org/10.1016/j.jcp.2025.114186
