#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# example:
# mpiexec -n 16 python -u netgen_st3d_curved_mesh.py --iso --h-far 0.4 --grading 0.5 --refine-max 0 --order 2

import os
import sys
import argparse
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from mpi4py import MPI

import netgen
from netgen.occ import Box, Cylinder, OCCGeometry, Pnt, X, Y, Z
from netgen.meshing import MeshingParameters


# ----------------------------
# Core mesh pipeline (import-safe)
# ----------------------------
def mesh_cell_counts(mesh, comm):
    try:
        local = int(mesh.cell_set.size)
    except Exception:
        local = int(mesh.num_cells())
    try:
        global_ = int(mesh.cell_set.total_size)
    except Exception:
        global_ = int(comm.allreduce(local, op=MPI.SUM))
    return local, global_


def _safe_set_mp_attr(mp, name, value, comm, note=None):
    """
    Set MeshingParameters attribute only if it exists on this Netgen build.
    Avoids crashes across versions.
    """
    if hasattr(mp, name):
        try:
            setattr(mp, name, value)
            if comm.rank == 0:
                msg = f"[mp] {name} = {value}"
                if note:
                    msg += f" ({note})"
                print(msg, flush=True)
        except Exception:
            if comm.rank == 0:
                print(f"[mp] WARNING: could not set {name} on this Netgen build.", flush=True)


def build_ngmesh(Lx, Ly, Lz, xc, yc, R, h_near, n_cyl, h_far, grading, iso, comm,
                 curvaturesafety=None, segmentsperedge=None):
    """Build Netgen OCC mesh on rank 0, dummy mesh on other ranks."""
    if comm.rank == 0:
        box = Box(Pnt(0, 0, 0), Pnt(Lx, Ly, Lz))
        cyl = Cylinder(Pnt(xc, yc, 0.0), Z, r=R, h=Lz)
        fluid = box - cyl

        # Name planar faces
        fluid.faces.Min(X).name = "inlet"
        fluid.faces.Max(X).name = "outlet"
        fluid.faces.Min(Y).name = "wall_y0"
        fluid.faces.Max(Y).name = "wall_yL"
        fluid.faces.Min(Z).name = "wall_z0"
        fluid.faces.Max(Z).name = "wall_zL"

        # Identify cylinder face
        target = (xc, yc, 0.5 * Lz)

        def d2(p):
            return (float(p[0]) - target[0])**2 + (float(p[1]) - target[1])**2 + (float(p[2]) - target[2])**2

        cyl_face = min(fluid.faces, key=lambda f: d2(f.center))
        cyl_face.name = "cylinder"

        # Cylinder resolution knob
        if iso:
            cyl_face.maxh = float(h_far)
            print(f"[knob] --iso ON: cyl_face.maxh forced to h_far = {float(h_far):g}", flush=True)
        else:
            h_cyl = float(h_near)
            if n_cyl is not None and int(n_cyl) > 3:
                h_cyl = float(2.0 * np.pi * float(R) / float(n_cyl))
                print(f"[knob] --n-cyl {int(n_cyl)} => cyl_face.maxh ~ {h_cyl:.6g}", flush=True)
            else:
                print(f"[knob] cyl_face.maxh = {h_cyl:.6g} (from --h-near)", flush=True)
            cyl_face.maxh = float(h_cyl)

        # IMPORTANT: do NOT do mp.maxh = ...
        # Some Netgen builds don't expose it as an attribute.
        mp = MeshingParameters(maxh=float(h_far), grading=float(grading))

        # Optional curvature knobs (safe across versions)
        if curvaturesafety is not None:
            _safe_set_mp_attr(mp, "curvaturesafety", float(curvaturesafety), comm,
                              note="lower -> more curvature refinement, higher -> less")
        if segmentsperedge is not None:
            _safe_set_mp_attr(mp, "segmentsperedge", float(segmentsperedge), comm,
                              note="lower -> fewer segments")

        geo = OCCGeometry(fluid, dim=3)
        ngmesh = geo.GenerateMesh(mp=mp)

        # Print boundary IDs
        bnames = ngmesh.GetRegionNames(codim=1)
        print("[netgen] boundary regions (codim=1):", flush=True)
        for i, name in enumerate(bnames, start=1):
            print(f"  id={i:2d}  name={name}", flush=True)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    return ngmesh


def refine_to_distance_field(mesh, xc, yc, R, h_near, h_far, d_min, d_max,
                             refine_factor, refine_max, comm):
    from firedrake import (
        FunctionSpace, Function, SpatialCoordinate, CellDiameter, Constant,
        sqrt, conditional, le, ge, gt, max_value
    )

    for it in range(int(refine_max)):
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W, name="refine_markers")

        x, y, z = SpatialCoordinate(mesh)
        r = sqrt((x - Constant(xc))**2 + (y - Constant(yc))**2 + 1e-16)
        dist = max_value(r - Constant(R), 0.0)

        hnear = Constant(h_near)
        hfar  = Constant(h_far)
        dmin  = Constant(d_min)
        dmax  = Constant(d_max)

        ramp = hnear + (hfar - hnear) * (dist - dmin) / (dmax - dmin + 1e-16)
        h_target = conditional(le(dist, dmin), hnear,
                     conditional(ge(dist, dmax), hfar, ramp))

        h_cell = CellDiameter(mesh)
        flag = conditional(gt(h_cell, Constant(refine_factor) * h_target), 1.0, 0.0)
        markers.interpolate(flag)

        nmark_local = int(np.count_nonzero(markers.dat.data_ro > 0.5))
        nmark_global = comm.allreduce(nmark_local, op=MPI.SUM)

        if comm.rank == 0:
            print(f"[refine] iter {it+1}/{int(refine_max)}: marked cells = {nmark_global}", flush=True)

        if nmark_global == 0:
            break

        mesh = mesh.refine_marked_elements(markers)
        comm.barrier()

    return mesh


def snap_cylinder_p2(mesh, cyl_tag, xc, yc, R, order, comm):
    from firedrake import (
        Mesh, VectorFunctionSpace, FunctionSpace, Function,
        SpatialCoordinate, CellVolume, Constant,
        DirichletBC, as_vector, sqrt
    )

    Vc = VectorFunctionSpace(mesh, "CG", int(order))
    Xc = Function(Vc, name=f"coords_p{order}")

    x, y, z = SpatialCoordinate(mesh)
    Xc.interpolate(as_vector((x, y, z)))

    r = sqrt((x - Constant(xc))**2 + (y - Constant(yc))**2 + 1e-16)
    xnew = Constant(xc) + Constant(R) * (x - Constant(xc)) / r
    ynew = Constant(yc) + Constant(R) * (y - Constant(yc)) / r
    Xproj = as_vector((xnew, ynew, z))

    DirichletBC(Vc, Xproj, (int(cyl_tag),)).apply(Xc)
    mesh_curv = Mesh(Xc)

    W0 = FunctionSpace(mesh_curv, "DG", 0)
    vol = Function(W0).interpolate(CellVolume(mesh_curv))
    vmin_local = float(np.min(vol.dat.data_ro)) if vol.dat.data_ro.size else 1.0
    vmin = comm.allreduce(vmin_local, op=MPI.MIN)

    if comm.rank == 0:
        print(f"[snap] min(CellVolume)={vmin:.3e}", flush=True)

    if vmin <= 0.0:
        raise RuntimeError("Snapped mesh has non-positive cell volume; increase cylinder resolution or reduce refinement aggressiveness.")

    return mesh_curv


def p2p2_dof_report(mesh, comm):
    from firedrake import VectorFunctionSpace, FunctionSpace

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)

    Vg, Qg = int(V.dim()), int(Q.dim())
    Tg = Vg + Qg

    if comm.rank == 0:
        print("---- P2-P2 DOF report (CG2 vel, CG2 pres) ----", flush=True)
        print(f"Velocity V (CG2, 3 comps): global dofs = {Vg:,}", flush=True)
        print(f"Pressure Q (CG2)         : global dofs = {Qg:,}", flush=True)
        print(f"Total (V+Q)              : global dofs = {Tg:,}", flush=True)
        print("----------------------------------------------", flush=True)

    return {"V_global": Vg, "Q_global": Qg, "T_global": Tg}


def generate_st3d_meshes(
    *,
    comm=None,
    # geometry
    Lx=2.5, Ly=0.41, Lz=0.41,
    xc=0.5, yc=0.2, R=0.05,
    # mesh controls
    h_near=0.01, n_cyl=None, h_far=0.08, grading=0.6,
    d_min=0.01, d_max=0.5, refine_max=2, refine_factor=1.25,
    order=2,
    # iso
    iso=False,
    curvaturesafety=None,
    segmentsperedge=None,
    # output
    write_vtk=True, outdir="output_mesh", prefix="st3d_df",
):
    """
    Callable from your NSE script:
      mesh_lin, mesh_curv, tags, dofs = generate_st3d_meshes(...)
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    from firedrake import Mesh, VTKFile

    ngmesh = build_ngmesh(
        Lx, Ly, Lz, xc, yc, R,
        h_near, n_cyl, h_far, grading,
        bool(iso), comm,
        curvaturesafety=curvaturesafety,
        segmentsperedge=segmentsperedge,
    )

    # boundary names -> ids mapping (broadcast)
    if comm.rank == 0:
        bnames = ngmesh.GetRegionNames(codim=1)
    else:
        bnames = None
    bnames = comm.bcast(bnames, root=0)
    tags = {name: i + 1 for i, name in enumerate(bnames)}
    TAG_CYL = tags["cylinder"]

    mesh_lin = Mesh(ngmesh, comm=comm)
    _, gc0 = mesh_cell_counts(mesh_lin, comm)
    if comm.rank == 0:
        print(f"[cells] before refine: global_cells={gc0}", flush=True)

    # effective near size should not be coarser than the cylinder surface size implied by n_cyl
    h_near_eff = float(h_near)
    if (not iso) and (n_cyl is not None) and int(n_cyl) > 3:
        h_cyl = float(2.0 * np.pi * float(R) / float(n_cyl))
        h_near_eff = min(h_near_eff, h_cyl)

    mesh_lin = refine_to_distance_field(
        mesh_lin, xc, yc, R,
        h_near_eff, h_far, d_min, d_max,
        refine_factor, refine_max, comm
    )

    _, gc1 = mesh_cell_counts(mesh_lin, comm)
    if comm.rank == 0:
        print(f"[cells] after  refine: global_cells={gc1} (delta={gc1-gc0})", flush=True)

    mesh_curv = snap_cylinder_p2(mesh_lin, TAG_CYL, xc, yc, R, order, comm)
    dofs = p2p2_dof_report(mesh_curv, comm)

    if write_vtk and outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        VTKFile(os.path.join(outdir, f"{prefix}_linear.pvd")).write(mesh_lin)
        VTKFile(os.path.join(outdir, f"{prefix}_curved_p{order}.pvd")).write(mesh_curv)
        comm.barrier()
        if comm.rank == 0:
            print(f"[done] Wrote:\n  {outdir}/{prefix}_linear.pvd\n  {outdir}/{prefix}_curved_p{order}.pvd", flush=True)

    return mesh_lin, mesh_curv, tags, dofs


# ----------------------------
# CLI (only when run as a script)
# ----------------------------
def main():
    parser = argparse.ArgumentParser("Netgen ST3D mesh + distance-refine + P2 snap + DOF report (MPI-safe)")
    parser.add_argument("--Lx", type=float, default=2.5)
    parser.add_argument("--Ly", type=float, default=0.41)
    parser.add_argument("--Lz", type=float, default=0.41)
    parser.add_argument("--xc", type=float, default=0.5)
    parser.add_argument("--yc", type=float, default=0.2)
    parser.add_argument("--R",  type=float, default=0.05)

    parser.add_argument("--h-near", type=float, default=0.01)
    parser.add_argument("--n-cyl", type=int, default=None)
    parser.add_argument("--h-far", type=float, default=0.08)
    parser.add_argument("--grading", type=float, default=0.6)

    parser.add_argument("--d-min", type=float, default=0.01)
    parser.add_argument("--d-max", type=float, default=0.5)
    parser.add_argument("--refine-max", type=int, default=2)
    parser.add_argument("--refine-factor", type=float, default=1.25)

    parser.add_argument("--order", type=int, default=2)

    parser.add_argument("--iso", action="store_true",
                        help="Force cyl_face.maxh = h_far (attempt at isotropic base mesh near cylinder).")
    parser.add_argument("--curvaturesafety", type=float, default=10,
                        help="Optional Netgen mp.curvaturesafety if supported by your build.")
    parser.add_argument("--segmentsperedge", type=float, default=0.1,
                        help="Optional Netgen mp.segmentsperedge if supported by your build.")

    parser.add_argument("--outdir", type=str, default="output_mesh")
    parser.add_argument("--prefix", type=str, default="st3d_df")
    parser.add_argument("--no-vtk", action="store_true")

    # allow PETSc flags when running this script directly
    args, petsc_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + petsc_argv

    generate_st3d_meshes(
        comm=MPI.COMM_WORLD,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        xc=args.xc, yc=args.yc, R=args.R,
        h_near=args.h_near, n_cyl=args.n_cyl, h_far=args.h_far, grading=args.grading,
        d_min=args.d_min, d_max=args.d_max, refine_max=args.refine_max, refine_factor=args.refine_factor,
        order=args.order,
        iso=bool(args.iso),
        curvaturesafety=args.curvaturesafety,
        segmentsperedge=args.segmentsperedge,
        write_vtk=(not args.no_vtk), outdir=args.outdir, prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
