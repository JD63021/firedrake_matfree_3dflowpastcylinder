#!/usr/bin/env python3
"""
matfree3dcyl_p2p2.py

Example run command:

OMP_NUM_THREADS=1 mpiexec -n 16 python matfree3dcyl_p2p2.py --levels 2 --order 1   --n-cyl 30 --Umax0 2.25 --h-far 0.4 --grading 0.6 --d-min 0.1 --d-max 0.5 --refine-max 0   --steps 1000000 --t-end 8 --dt-max 5e-2 --cfl 10000 --write-freq 2   --picard-maxit 10 --picard-rtol 1e-6 --picard-atol 1e-6   --p-penalty 1e-9 --p-outlet   --checkpoint-prefix cyl3d_checkpoint --checkpoint-freq 25 --qdeg 6    -- -ksp_converged_reason



"""

import os
import sys
import csv
import json
import argparse
import signal
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ----------------------------
# Parse args BEFORE importing Firedrake
# ----------------------------
parser = argparse.ArgumentParser(
    "Netgen ST3D: P2-P2 BDF2 + Picard + TRUE residual + hierarchy MG + curve finest only + checkpoint (symgrad + reactions)"
)

# Geometry (match your netgen file defaults)
parser.add_argument("--Lx", type=float, default=2.5)
parser.add_argument("--Ly", type=float, default=0.41)
parser.add_argument("--Lz", type=float, default=0.41)
parser.add_argument("--xc", type=float, default=0.5)
parser.add_argument("--yc", type=float, default=0.2)
parser.add_argument("--R",  type=float, default=0.05)

# Netgen mesh knobs (passed to generator)
parser.add_argument("--h-near", type=float, default=0.25)
parser.add_argument("--n-cyl", type=int, default=None)
parser.add_argument("--h-far", type=float, default=0.1)
parser.add_argument("--grading", type=float, default=0.1)

# Distance-field refinement (inside generator)
parser.add_argument("--d-min", type=float, default=0.1)
parser.add_argument("--d-max", type=float, default=0.5)
parser.add_argument("--refine-max", type=int, default=0)
parser.add_argument("--refine-factor", type=float, default=1.25)
parser.add_argument("--iso", action="store_true",
                    help="Make the base Netgen mesh isotropic/coarse near the cylinder (cyl_face.maxh=h_far).")

# Geometry coordinate order for curving
parser.add_argument("--order", type=int, default=2)

# Hierarchy controls (THIS enables geometric MG levels)
parser.add_argument("--levels", type=int, default=1, help="MeshHierarchy refinements (>=1 recommended for mg).")
parser.add_argument("--reorder", action="store_true", help="Reorder refined meshes.")

# Optional: also write mesh PVDs from generator
parser.add_argument("--mesh-write-vtk", action="store_true")
parser.add_argument("--mesh-outdir", type=str, default="output_mesh")
parser.add_argument("--mesh-prefix", type=str, default="st3d_df")

# Time stepping
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--t-end", type=float, default=None)
parser.add_argument("--cfl", type=float, default=1.0)
parser.add_argument("--dt-min", type=float, default=1e-6)
parser.add_argument("--dt-max", type=float, default=1e-2)

# Stabilization / quadrature
parser.add_argument("--no-supg", action="store_true")
parser.add_argument("--no-pspg", action="store_true")
parser.add_argument("--no-stab", action="store_true")
parser.add_argument("--qdeg", type=int, default=4)

# Picard control
parser.add_argument("--picard-maxit", type=int, default=5)
parser.add_argument("--picard-rtol", type=float, default=1e-6)
parser.add_argument("--picard-atol", type=float, default=1e-6)
parser.add_argument("--picard-monitor", action="store_true")

# Krylov control
parser.add_argument("--rtol", type=float, default=1e-2)
parser.add_argument("--maxit", type=int, default=100)
parser.add_argument("--divtol", type=float, default=1e30)

# Pressure penalty + optional outlet pressure Dirichlet
parser.add_argument("--p-penalty", type=float, default=1e-15)
parser.add_argument("--p-outlet", action="store_true")

# Output
parser.add_argument("--write-freq", type=int, default=5)
parser.add_argument("--vtk", type=str, default="flow3d.pvd")

# Retry control
parser.add_argument("--solve-retries", type=int, default=10)

# Inlet profile control (paper)
parser.add_argument("--H", type=float, default=0.41)
parser.add_argument("--W", type=float, default=0.41)
parser.add_argument("--Umax0", type=float, default=2.25)
parser.add_argument("--period", type=float, default=8.0)

# CSV + Cd/Cl normalization
parser.add_argument("--csv", type=str, default="forces_cdcl_3d.csv")
parser.add_argument("--rho", type=float, default=1.0)
parser.add_argument("--mu", type=float, default=0.001)
parser.add_argument("--D", type=float, default=0.1)
parser.add_argument("--Umean-ref", type=float, default=1.0)

# Checkpoint / restart
parser.add_argument("--checkpoint-prefix", type=str, default="cyl3d_checkpoint")
parser.add_argument("--checkpoint-freq", type=int, default=25)
parser.add_argument("--restart", action="store_true")

# Debug: force full LU (assembled)
parser.add_argument("--use-lu", action="store_true",
                    help="Assemble matrix and solve with LU(MUMPS). Good for debugging/coarse meshes.")

args, petsc_argv = parser.parse_known_args()
# Feed PETSc args to Firedrake/PETSc
sys.argv = [sys.argv[0]] + petsc_argv

# ----------------------------
# Now import Firedrake + MPI
# ----------------------------
from firedrake import *  # noqa
from firedrake.exceptions import ConvergenceError
from mpi4py import MPI
import numpy as np

dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

comm = COMM_WORLD
rank = comm.rank
mpi_comm = comm

T0 = perf_counter()


def dbg(msg: str):
    if rank == 0:
        dt = perf_counter() - T0
        print(f"[dbg {dt:9.3f}s] {msg}", flush=True)


# ----------------------------
# Graceful stop flag (Ctrl+C / SIGTERM)
# ----------------------------
STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)

# ----------------------------
# Import mesh generator safely
# netgen_st3d_curved_mesh.py parses argv at import time -> sanitize sys.argv temporarily.
# ----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_saved_argv = list(sys.argv)
try:
    sys.argv = [sys.argv[0]]  # hide PETSc flags from the generator's argparse at import time
    from netgen_st3d_curved_mesh import generate_st3d_meshes  # noqa: E402
finally:
    sys.argv = _saved_argv


# ----------------------------
# Helpers
# ----------------------------
def safe_global_hmin(mesh) -> float:
    # robust "h" estimate for curved/high-order geometry: h ~ vol^{1/3}
    V0 = FunctionSpace(mesh, "DG", 0)
    hfun = Function(V0, name="hvol")
    hfun.interpolate((CellVolume(mesh) + 1e-30) ** (1.0 / 3.0))
    local = float(hfun.dat.data_ro.min()) if hfun.dat.data_ro.size else 1e30
    return mpi_comm.allreduce(local, op=MPI.MIN)


def compute_umax(u_fun: Function) -> float:
    arr = u_fun.dat.data_ro
    local = float(np.max(np.linalg.norm(arr, axis=1))) if arr.size else 0.0
    return mpi_comm.allreduce(local, op=MPI.MAX)


def mesh_cell_counts(mesh):
    try:
        local_owned = int(mesh.cell_set.size)
    except Exception:
        local_owned = int(mesh.num_cells())
    try:
        global_total = int(mesh.cell_set.total_size)
    except Exception:
        global_total = int(mpi_comm.allreduce(local_owned, op=MPI.SUM))
    return local_owned, global_total


def _get_ksp_its(lvs) -> int:
    try:
        snes = getattr(lvs, "snes", None)
        if snes is None:
            return 0
        ksp = snes.getKSP()
        if ksp is None:
            return 0
        return int(ksp.getIterationNumber())
    except Exception:
        return 0


def _cofunction_norm2(obj) -> float:
    try:
        with obj.dat.vec_ro as v:
            return float(v.norm())
    except Exception:
        try:
            data = obj.dat.data_ro
        except Exception:
            data = obj.dat.data
        local = float(np.linalg.norm(data))
        return float(mpi_comm.allreduce(local**2, op=MPI.SUM) ** 0.5)


def _zero_dirichlet_dofs_in_assembled(obj, bc_list):
    for bc in bc_list:
        try:
            nodes = bc.nodes
        except Exception:
            continue
        try:
            arr = obj.dat.data_with_halos
        except Exception:
            arr = obj.dat.data
        arr[nodes] = 0.0


def curve_finest_mesh_only(mesh_finest, cyl_tag: int, xc: float, yc: float, R: float, order: int):
    """
    Curve ONLY the finest mesh by snapping cylinder boundary DOFs of a CG(order) coordinate field
    onto the exact cylinder. Returns Mesh(Xc) which should preserve the underlying DM hierarchy.
    """
    Vc = VectorFunctionSpace(mesh_finest, "CG", int(order))
    Xc = Function(Vc, name=f"coords_curved_p{int(order)}")

    x, y, z = SpatialCoordinate(mesh_finest)
    Xc.interpolate(as_vector((x, y, z)))

    xcC = Constant(float(xc))
    ycC = Constant(float(yc))
    RC  = Constant(float(R))
    r = sqrt((x - xcC)**2 + (y - ycC)**2 + 1e-16)

    xnew = xcC + RC * (x - xcC) / r
    ynew = ycC + RC * (y - ycC) / r
    Xproj = as_vector((xnew, ynew, z))

    DirichletBC(Vc, Xproj, (int(cyl_tag),)).apply(Xc)

    #mesh_curv = Mesh(Xc)
    mesh_curv = Mesh(Xc, distribution_parameters=dparams)

    # sanity check
    W0 = FunctionSpace(mesh_curv, "DG", 0)
    vol = Function(W0).interpolate(CellVolume(mesh_curv))
    vmin_local = float(np.min(vol.dat.data_ro)) if vol.dat.data_ro.size else 1.0
    vmin = mpi_comm.allreduce(vmin_local, op=MPI.MIN)
    if rank == 0:
        print(f"[curve-finest] order={int(order)} min(CellVolume)={vmin:.3e}", flush=True)
    if vmin <= 0.0:
        raise RuntimeError("Curved finest mesh has non-positive cell volume. Increase resolution or reduce refinement.")
    return mesh_curv


# ----------------------------
# Checkpointing (NPZ per-rank + JSON meta)
# ----------------------------
def chk_rank_file(prefix: str, r: int) -> str:
    return f"{prefix}.rank{r}.npz"


def chk_meta_file(prefix: str) -> str:
    return f"{prefix}.json"


def _write_meta_atomic(meta_path: str, meta: dict):
    if rank == 0:
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, meta_path)
    mpi_comm.barrier()


def _read_meta(meta_path: str):
    meta = None
    if rank == 0 and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta = mpi_comm.bcast(meta, root=0)
    return meta


def save_checkpoint(prefix: str, step: int, time_val: float, dt_val: float,
                    u_n: Function, p_n: Function, u_nm1: Function, p_nm1: Function):
    meta_path = chk_meta_file(prefix)
    final_file = chk_rank_file(prefix, rank)
    tmp_file = final_file + ".tmp.npz"

    u_n_arr = np.array(u_n.dat.data_ro, copy=True)
    p_n_arr = np.array(p_n.dat.data_ro, copy=True)
    u_nm1_arr = np.array(u_nm1.dat.data_ro, copy=True)
    p_nm1_arr = np.array(p_nm1.dat.data_ro, copy=True)

    if os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except Exception:
            pass
    mpi_comm.barrier()

    np.savez(tmp_file, u_n=u_n_arr, p_n=p_n_arr, u_nm1=u_nm1_arr, p_nm1=p_nm1_arr)
    mpi_comm.barrier()
    os.replace(tmp_file, final_file)
    mpi_comm.barrier()

    meta = {
        "step": int(step),
        "t": float(time_val),
        "dt": float(dt_val),
        "comm_size": int(mpi_comm.size),
        "disc": "P2-P2",
        "hier_levels": int(args.levels),
        "reorder": bool(args.reorder),
        "mesh": {
            "Lx": float(args.Lx), "Ly": float(args.Ly), "Lz": float(args.Lz),
            "xc": float(args.xc), "yc": float(args.yc), "R": float(args.R),
            "n_cyl": args.n_cyl,
            "h_near": float(args.h_near),
            "h_far": float(args.h_far),
            "grading": float(args.grading),
            "d_min": float(args.d_min),
            "d_max": float(args.d_max),
            "refine_max": int(args.refine_max),
            "refine_factor": float(args.refine_factor),
            "curve_order_finest": int(args.order),
        },
        "rho": float(args.rho),
        "mu": float(args.mu),
        "D": float(args.D),
        "H": float(args.H),
        "W": float(args.W),
        "Umax0": float(args.Umax0),
        "period": float(args.period),
        "p_penalty": float(args.p_penalty),
        "p_outlet": bool(args.p_outlet),
        "qdeg": int(args.qdeg),
        "no_supg": bool(args.no_supg),
        "no_pspg": bool(args.no_pspg),
        "no_stab": bool(args.no_stab),
        "V_local_shape": list(u_n_arr.shape),
        "Q_local_shape": list(p_n_arr.shape),
        "viscous_form": "2*nu*inner(sym(grad(u)),sym(grad(v)))",
    }
    _write_meta_atomic(meta_path, meta)
    if rank == 0:
        print(f"[CHK] wrote checkpoint at step={step}, t={time_val:.6e} -> {prefix}.rank*.npz", flush=True)


def load_checkpoint(prefix: str, u_n: Function, p_n: Function, u_nm1: Function, p_nm1: Function):
    meta_path = chk_meta_file(prefix)
    meta = _read_meta(meta_path)
    if meta is None:
        if rank == 0:
            print("[CHK] no metadata file found; cannot restart.", flush=True)
        return None

    fpath = chk_rank_file(prefix, rank)
    if not os.path.exists(fpath):
        if rank == 0:
            print(f"[CHK] missing rank file {fpath}; cannot restart.", flush=True)
        return None
    mpi_comm.barrier()

    data = np.load(fpath)
    u_n.dat.data[:] = data["u_n"]
    p_n.dat.data[:] = data["p_n"]
    u_nm1.dat.data[:] = data["u_nm1"]
    p_nm1.dat.data[:] = data["p_nm1"]

    mpi_comm.barrier()
    if rank == 0:
        print(f"[CHK] restarted from step={int(meta.get('step', 0))}, t={float(meta.get('t', 0.0)):.6e}", flush=True)
    return meta


# ----------------------------
# Main
# ----------------------------
def main():
    dbg("[rank 0] starting")
    dbg(f"COMM size={mpi_comm.size}")
    mpi_comm.barrier()

    # ----------------------------
    # Build base linear mesh from Netgen generator (distance refine optional)
    # ----------------------------
    dbg("Generating Netgen base mesh (linear) via generate_st3d_meshes(...)")
    mesh_lin0, _mesh_curv_unused, tags, _dofs = generate_st3d_meshes(
        comm=MPI.COMM_WORLD,
        write_vtk=bool(args.mesh_write_vtk),
        outdir=args.mesh_outdir,
        prefix=args.mesh_prefix,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        xc=args.xc, yc=args.yc, R=args.R,
        h_near=args.h_near, n_cyl=args.n_cyl,
        h_far=args.h_far, grading=args.grading,
        d_min=args.d_min, d_max=args.d_max,
        refine_max=args.refine_max, refine_factor=args.refine_factor,
        order=1,  # base mesh stays linear here; we curve ONLY mh[-1] later
        iso=bool(args.iso), 
    )
    mpi_comm.barrier()

    # Tags
    TAG_INLET = int(tags["inlet"])
    TAG_OUT   = int(tags["outlet"])
    TAG_CYL   = int(tags["cylinder"])
    TAG_WALLS = (int(tags["wall_y0"]), int(tags["wall_yL"]), int(tags["wall_z0"]), int(tags["wall_zL"]))

    # ----------------------------
    # Build hierarchy from linear base mesh (enables geometric MG levels)
    # ----------------------------
    if int(args.levels) < 0:
        raise ValueError("--levels must be >= 0")
    dbg(f"Building MeshHierarchy(levels={int(args.levels)}, reorder={bool(args.reorder)}) ...")
    #mh = MeshHierarchy(mesh_lin0, int(args.levels), reorder=bool(args.reorder))
    mh = MeshHierarchy(
        mesh_lin0,
        int(args.levels),
        reorder=bool(args.reorder),
        distribution_parameters=dparams,
    )
    mpi_comm.barrier()
    if rank == 0:
        for lev, m in enumerate(mh):
            lc, gc = mesh_cell_counts(m)
            print(f"[hier] level {lev:2d}: global_cells={gc}", flush=True)

    mesh_finest_lin = mh[-1]

    # ----------------------------
    # Curve ONLY the finest mesh (snap cylinder boundary DOFs)
    # ----------------------------
    dbg("Curving ONLY the finest hierarchy mesh (snap cylinder)...")
    mesh = curve_finest_mesh_only(mesh_finest_lin, TAG_CYL, args.xc, args.yc, args.R, args.order)
    mpi_comm.barrier()

    # Measures
    dsq = ds(domain=mesh, metadata={"quadrature_degree": int(args.qdeg)})
    dxq = dx(metadata={"quadrature_degree": int(args.qdeg)})
    ds_ = ds(domain=mesh)

    local_cells, global_cells = mesh_cell_counts(mesh)
    hmin = safe_global_hmin(mesh)
    dbg(f"Finest (curved) mesh: local_cells={local_cells}, global_cells={global_cells}, hmin~{hmin:.3e}")
    mpi_comm.barrier()

    rho = float(args.rho)
    mu = float(args.mu)
    nu_val = mu / rho
    nu = Constant(nu_val)

    dt = Constant(float(args.dt_max))

    # Stabilization toggles
    use_supg = (not args.no_stab) and (not args.no_supg)
    use_pspg = (not args.no_stab) and (not args.no_pspg)
    if rank == 0 and (not use_pspg):
        print("[WARN] PSPG is OFF. For strict P2-P2 this often leads to instability/singularity.", flush=True)

    # ----------------------------
    # STRICT P2-P2 spaces
    # ----------------------------
    dbg("Building function spaces (STRICT P2-P2)...")
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)
    Z = V * Q
    disc = "P2-P2"
    mpi_comm.barrier()

    Vsum = mpi_comm.allreduce(int(V.dof_dset.size), op=MPI.SUM)
    Qsum = mpi_comm.allreduce(int(Q.dof_dset.size), op=MPI.SUM)
    Zsum = mpi_comm.allreduce(int(Z.dof_dset.size), op=MPI.SUM)
    if rank == 0:
        print(f"[disc] {disc}", flush=True)
        print("---- DOF counts (owned sum) ----", flush=True)
        print(f"Velocity V (CG2 vector): {Vsum:,}", flush=True)
        print(f"Pressure Q (CG2)       : {Qsum:,}", flush=True)
        print(f"Total (V+Q)            : {Zsum:,}", flush=True)
        print("--------------------------------", flush=True)

    # Unknowns/tests/trials
    up = Function(Z, name="up")
    v, q = TestFunctions(Z)
    uT, pT = TrialFunctions(Z)
    u_sol, p_sol = up.subfunctions
    u_sol.rename("Velocity")
    p_sol.rename("Pressure")

    # History
    u_n = Function(V, name="u_n");     u_n.assign(Constant((0.0, 0.0, 0.0)))
    u_nm1 = Function(V, name="u_nm1"); u_nm1.assign(u_n)
    p_n = Function(Q, name="p_n");     p_n.assign(0.0)
    p_nm1 = Function(Q, name="p_nm1"); p_nm1.assign(p_n)

    # Picard fields
    u_adv = Function(V, name="u_adv"); u_adv.assign(u_n)
    u_prev_it = Function(V, name="u_prev_it")

    # Cylinder area check
    Scyl = assemble(Constant(1.0) * ds_(TAG_CYL))
    if rank == 0:
        print(f"[check] Cylinder surface area Scyl = {float(Scyl):.6e}", flush=True)

    # ----------------------------
    # BCs
    # ----------------------------
    dbg("Constructing DirichletBCs...")
    x, y, z = SpatialCoordinate(mesh)
    H = Constant(float(args.H))
    Wc = Constant(float(args.W))
    Um = Constant(0.0)

    uin_x = 16.0 * Um * (y * z * (H - y) * (Wc - z)) / ((H**2) * (Wc**2))
    uin = as_vector((uin_x, 0.0, 0.0))

    bc_in = DirichletBC(Z.sub(0), uin, (TAG_INLET,))
    bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (TAG_CYL,) + tuple(TAG_WALLS))
    bc_pout = DirichletBC(Z.sub(1), Constant(0.0), (TAG_OUT,))

    bcs = [bc_in, bc_noslip]
    if args.p_outlet:
        bcs = [bc_in, bc_noslip, bc_pout]

    # Residual-zeroing BCs (for Picard monitoring)
    bc_in_V = DirichletBC(V, uin, (TAG_INLET,))
    bc_noslip_V = DirichletBC(V, Constant((0.0, 0.0, 0.0)), (TAG_CYL,) + tuple(TAG_WALLS))
    bcs_zero_u = [bc_in_V, bc_noslip_V]
    bcs_zero_p = []
    if args.p_outlet:
        bcs_zero_p = [DirichletBC(Q, Constant(0.0), (TAG_OUT,))]

    # Cylinder-only BC for reactions (IMPORTANT: cylinder tag ONLY)
    bc_cyl_only_V = DirichletBC(V, Constant((0.0, 0.0, 0.0)), (TAG_CYL,))

    # Nullspace / penalty
    eps_p = Constant(float(args.p_penalty))
    if float(args.p_penalty) > 0.0:
        nullspace = None
    else:
        # pressure constant nullspace (only meaningful if no p_outlet)
        nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])

    # ----------------------------
    # tau (linearized)
    # ----------------------------
    hcell = (CellVolume(mesh) + 1e-30) ** (1.0 / 3.0)
    u_mag = sqrt(dot(u_adv, u_adv) + 1e-12)
    tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * 2 * u_mag / hcell)**2 + 9*((4.0 * nu * 4 / hcell**2)**2) + 1e-30)
    tau_lsic = 0.5 * u_mag * hcell

    dt_inv = 1.0 / dt
    bdf2_fac = 1.0 / (2.0 * dt)

    # ----------------------------
    # Linearized forms (BDF1/BDF2)
    # Viscous weak form: 2*nu*inner(sym(grad(u)),sym(grad(v)))
    # Strong residual uses: -div(2*nu*sym(grad(u)))
    # ----------------------------
    visc_weak = lambda uu, vv: 2.0 * nu * inner(sym(grad(uu)), sym(grad(vv)))
    visc_strong = lambda uu: div(2.0 * nu * sym(grad(uu)))  # vector

    F1 = (
        dt_inv * inner(uT, v) * dxq
        + inner(dot(grad(uT), u_adv), v) * dxq
        + visc_weak(uT, v) * dxq
         + tau_lsic * inner(div(uT), div(v)) * dxq
        - inner(pT, div(v)) * dxq
        + inner(div(uT), q) * dxq
        + eps_p * inner(pT, q) * dxq
        - dt_inv * inner(u_n, v) * dxq
    )
    if use_supg or use_pspg:
        r1 = dt_inv * uT + dot(grad(uT), u_adv) + grad(pT) - dt_inv * u_n - visc_strong(uT)
        if use_supg:
            F1 += inner(r1, tau * dot(grad(v), u_adv)) * dxq
        if use_pspg:
            F1 += inner(r1, tau * grad(q)) * dxq
    a1, L1 = lhs(F1), rhs(F1)

    F2 = (
        (3.0 * bdf2_fac) * inner(uT, v) * dxq
        + inner(dot(grad(uT), u_adv), v) * dxq
        + visc_weak(uT, v) * dxq
         + tau_lsic * inner(div(uT), div(v)) * dxq
        - inner(pT, div(v)) * dxq
        + inner(div(uT), q) * dxq
        + eps_p * inner(pT, q) * dxq
        - bdf2_fac * inner(4.0 * u_n - u_nm1, v) * dxq
    )
    if use_supg or use_pspg:
        r2 = (3.0 * bdf2_fac) * uT + dot(grad(uT), u_adv) + grad(pT) - bdf2_fac * (4.0 * u_n - u_nm1) - visc_strong(uT)
        if use_supg:
            F2 += inner(r2, tau * dot(grad(v), u_adv)) * dxq
        if use_pspg:
            F2 += inner(r2, tau * grad(q)) * dxq
    a2, L2 = lhs(F2), rhs(F2)

    # ----------------------------
    # Solver parameters
    # ----------------------------
    if args.use_lu:
        sp = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        if rank == 0:
            print("[solve] Using FULL LU(MUMPS) (assembled) for debugging.", flush=True)
    else:
        sp = {
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "ksp_rtol": float(args.rtol),
            "ksp_max_it": int(args.maxit),
            "ksp_divtol": float(args.divtol),
            "ksp_initial_guess_nonzero": True,
            "pc_type": "mg",
            "mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                #"pc_type": "asm",
  	        #"pc_asm_overlap": 1,          # try 1 then 2
  	    	#"sub_ksp_type": "preonly",
  	    	#"sub_pc_type": "lu",
  	    	#"sub_pc_factor_levels": 0,    # try 1 then 2
            },
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "ksp_chebyshev_esteig_steps": 30,
                "pc_type": "jacobi",
                #"pc_type": "python",
    		#"pc_python_type": "firedrake.PatchPC",
    		#"patch": {
        	#    "pc_patch": {
            	#        "construct_type": "star",
            	#        "construct_dim": 0,          # vertex-star
            	#        "sub_mat_type": "seqdense",
            	#        "dense_inverse": True,
            	#        "save_operators": True,
            	#        "precompute_element_tensors": True,
        	#    },
        	#    "sub_ksp_type": "preonly",
        	#    "sub_pc_type": "lu",
                #},
            }
            }

    def build_solver(a_form, L_form):
        prob = LinearVariationalProblem(a_form, L_form, up, bcs=bcs)
        sol = LinearVariationalSolver(prob, solver_parameters=sp, nullspace=nullspace)
        return sol

    # ----------------------------
    # TRUE nonlinear residual forms for Picard stopping
    # (updated to symgrad scaling + strong visc term)
    # ----------------------------
    vR = TestFunction(V)
    qR = TestFunction(Q)

    u_mag_nl = sqrt(dot(u_sol, u_sol) + 1e-12)
    tau_nl = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * 2 * u_mag_nl / hcell)**2 + 9*((4.0 * nu * 4 / hcell**2)**2) + 1e-30)
    tau_lsic_nl = 0.5 * u_mag_nl * hcell

    mom_strong_bdf1 = dt_inv * (u_sol - u_n) + dot(grad(u_sol), u_sol) + grad(p_sol) - visc_strong(u_sol)
    mom_strong_bdf2 = bdf2_fac * (3.0 * u_sol - 4.0 * u_n + u_nm1) + dot(grad(u_sol), u_sol) + grad(p_sol) - visc_strong(u_sol)

    Rv_bdf1 = (
        inner(dt_inv * (u_sol - u_n), vR) * dxq
        + inner(dot(grad(u_sol), u_sol), vR) * dxq
         + tau_lsic_nl * inner(div(u_sol), div(vR)) * dxq
        + visc_weak(u_sol, vR) * dxq
        - inner(p_sol, div(vR)) * dxq
    )
    Rv_bdf2 = (
        inner(bdf2_fac * (3.0 * u_sol - 4.0 * u_n + u_nm1), vR) * dxq
        + inner(dot(grad(u_sol), u_sol), vR) * dxq
         + tau_lsic_nl * inner(div(u_sol), div(vR)) * dxq
        + visc_weak(u_sol, vR) * dxq
        - inner(p_sol, div(vR)) * dxq
    )

    Rq_bdf1 = inner(div(u_sol), qR) * dxq + eps_p * inner(p_sol, qR) * dxq
    Rq_bdf2 = inner(div(u_sol), qR) * dxq + eps_p * inner(p_sol, qR) * dxq

    if use_supg:
        Rv_bdf1 += inner(mom_strong_bdf1, tau_nl * dot(grad(vR), u_sol)) * dxq
        Rv_bdf2 += inner(mom_strong_bdf2, tau_nl * dot(grad(vR), u_sol)) * dxq
    if use_pspg:
        Rq_bdf1 += inner(mom_strong_bdf1, tau_nl * grad(qR)) * dxq
        Rq_bdf2 += inner(mom_strong_bdf2, tau_nl * grad(qR)) * dxq

    Rv_vec = Cofunction(V.dual())
    Rq_vec = Cofunction(Q.dual())
    Rv_react_vec = Cofunction(V.dual())  # for reactions (separate to keep things clean)

    # ----------------------------
    # Forces (KEEP SAME traction formula as your original)
    # ----------------------------
    nrm = FacetNormal(mesh)
    I = Identity(mesh.geometric_dimension())
    sigma = -p_sol * I + nu * (grad(u_sol) + grad(u_sol).T)  # same as 2*nu*sym(grad(u))
    traction = dot(sigma, nrm)

    def cylinder_force_xyz_integral():
        Fx = -assemble(traction[0] * dsq(TAG_CYL))
        Fy = -assemble(traction[1] * dsq(TAG_CYL))
        Fz = -assemble(traction[2] * dsq(TAG_CYL))
        return float(Fx), float(Fy), float(Fz)

    def cylinder_force_xyz_reaction(Rv_form_for_mode):
        """
        
        """
        assemble(Rv_form_for_mode, tensor=Rv_react_vec)

        # owned-only array and cylinder nodes (nodes index data_with_halos, so filter by owned length)
        arr_owned = Rv_react_vec.dat.data_ro
        n_owned = arr_owned.shape[0]
        nodes = np.array(bc_cyl_only_V.nodes, dtype=np.int64)
        nodes_owned = nodes[nodes < n_owned]

        local_sum = arr_owned[nodes_owned, :].sum(axis=0) if nodes_owned.size else np.array([0.0, 0.0, 0.0], dtype=float)

        Fx_loc = float(local_sum[0]); Fy_loc = float(local_sum[1]); Fz_loc = float(local_sum[2])

        Fx = mpi_comm.allreduce(Fx_loc, op=MPI.SUM)
        Fy = mpi_comm.allreduce(Fy_loc, op=MPI.SUM)
        Fz = mpi_comm.allreduce(Fz_loc, op=MPI.SUM)

        # match your integral convention (integral already had leading "-")
        # If signs look flipped, just remove this minus; you'll see it immediately in the printout.
        return float(-Fx), float(-Fy), float(-Fz)

    # Output
    dbg(f"Creating VTKFile: {args.vtk}")
    vtk = VTKFile(args.vtk)
    mpi_comm.barrier()

    # CSV + normalization
    Href = float(args.W)
    Uref = (4.0 / 9.0) * float(args.Umax0)
    den_ref = rho * (Uref**2) * float(args.D) * Href

    csv_fh = None
    csv_writer = None
    if rank == 0:
        csv_fh = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "step", "picard_k", "ksp_its_timestep", "t", "dt",
            "Um_t",
            # Integral forces
            "Fx_int", "Fy_int", "Fz_int", "Cd_int", "Cl_y_int", "Cl_z_int",
            # Reaction forces
            "Fx_rea", "Fy_rea", "Fz_rea", "Cd_rea", "Cl_y_rea", "Cl_z_rea",
            "rho", "mu", "nu", "D", "Umean_ref",
        ])
        csv_fh.flush()

        stab = []
        if use_supg:
            stab.append("SUPG")
        if use_pspg:
            stab.append("PSPG")
        stab_str = "+".join(stab) if stab else "no-stab"
        pout = "p_outlet=ON" if args.p_outlet else "p_outlet=OFF"
        print(f"[post] Writing CSV: {args.csv}", flush=True)
        print(f"[disc] {disc}, {stab_str}, nu={nu_val:g}, p_penalty={args.p_penalty:g}, {pout}", flush=True)
        print(f"[visc] weak=2*nu*inner(sym(grad(u)),sym(grad(v))) ; traction kept as sigma=-pI+nu(grad+grad^T)", flush=True)
        print(f"[MG] levels={int(args.levels)} (coarse linear, finest curved only)", flush=True)
        print(f"[CHK] prefix={args.checkpoint_prefix} freq={int(args.checkpoint_freq)} restart={bool(args.restart)}", flush=True)
        print(f"{'step':>5}  {'mode':>4}  {'t':>12}  {'dt':>11}  {'Um(t)':>11}  {'umax':>11}  {'Picard':>6}  {'KSPits':>8}", flush=True)

    # Restart
    t = 0.0
    start_step = 1
    if args.restart:
        meta = load_checkpoint(args.checkpoint_prefix, u_n, p_n, u_nm1, p_nm1)
        if meta is not None:
            t = float(meta.get("t", 0.0))
            last_step = int(meta.get("step", 0))
            start_step = max(1, last_step + 1)
            if rank == 0:
                print(f"[CHK] continuing from step={start_step} (prev={last_step}) at t={t:.6e}", flush=True)

    # consistent initial state
    u_sol.assign(u_n)
    p_sol.assign(p_n)
    u_adv.assign(u_n)

    last_completed_step = start_step - 1
    last_completed_time = t
    last_completed_dt = float(dt.dat.data_ro[0])

    def update_inlet_amplitude(t_now: float):
        Um_t = float(args.Umax0) * np.sin(np.pi * t_now / float(args.period))
        if Um_t < 0.0 and Um_t > -1e-14:
            Um_t = 0.0
        Um.assign(Um_t)
        return Um_t

    # Build solvers ONCE (reuse)
    dbg("Building BDF1 and BDF2 solvers once (reuse)...")
    solver_bdf1 = build_solver(a1, L1)
    solver_bdf2 = build_solver(a2, L2)
    mpi_comm.barrier()

    wall0 = perf_counter()

    try:
        for nstep in range(start_step, int(args.steps) + 1):
            if args.t_end is not None and t >= float(args.t_end):
                break

            if STOP_REQUESTED:
                if rank == 0:
                    print("[STOP] stop requested; writing checkpoint and exiting.", flush=True)
                if int(args.checkpoint_freq) != 0 and last_completed_step >= 0:
                    save_checkpoint(args.checkpoint_prefix, last_completed_step, last_completed_time, last_completed_dt,
                                    u_n, p_n, u_nm1, p_nm1)
                break

            umax_state = compute_umax(u_n)

            uref_cfl = max(umax_state, 1.0)
            dt_val = float(args.cfl) * float(hmin) / uref_cfl
            dt_val = min(max(dt_val, float(args.dt_min)), float(args.dt_max))
            if args.t_end is not None:
                remaining = float(args.t_end) - t
                if dt_val > remaining:
                    dt_val = remaining

            dt.assign(dt_val)
            last_completed_dt = dt_val
            t += dt_val
            t_now = t

            Um_t = update_inlet_amplitude(t_now)

            if nstep == 1 and start_step == 1:
                mode = "BDF1"
                solver = solver_bdf1
                Rv_form, Rq_form = Rv_bdf1, Rq_bdf1
            else:
                mode = "BDF2"
                solver = solver_bdf2
                Rv_form, Rq_form = Rv_bdf2, Rq_bdf2

            u_adv.assign(u_n)
            u_sol.assign(u_n)
            p_sol.assign(p_n)

            ksp_its_timestep = 0
            res0 = None
            k = 0

            for k in range(1, int(args.picard_maxit) + 1):
                u_prev_it.assign(u_sol)

                solved = False
                last_err = None
                ksp_its_picard = 0

                for attempt in range(1, int(args.solve_retries) + 1):
                    try:
                        if attempt > 1:
                            dbg(f"Retry {attempt}/{int(args.solve_retries)}: rebuilding solver for {mode} ...")
                            if mode == "BDF1":
                                solver_bdf1 = build_solver(a1, L1)
                                solver = solver_bdf1
                            else:
                                solver_bdf2 = build_solver(a2, L2)
                                solver = solver_bdf2

                        solver.solve()
                        ksp_its_picard = _get_ksp_its(solver)
                        solved = True
                        break

                    except ConvergenceError as e:
                        last_err = e
                        ksp_its_picard = max(ksp_its_picard, _get_ksp_its(solver))

                if not solved:
                    raise last_err

                ksp_its_timestep += int(ksp_its_picard)
                u_adv.assign(u_sol)

                # TRUE residual norm monitoring (zero Dirichlet DOFs for the norm)
                assemble(Rv_form, tensor=Rv_vec)
                assemble(Rq_form, tensor=Rq_vec)

                _zero_dirichlet_dofs_in_assembled(Rv_vec, bcs_zero_u)
                _zero_dirichlet_dofs_in_assembled(Rq_vec, bcs_zero_p)

                rvn = _cofunction_norm2(Rv_vec)
                rqn = _cofunction_norm2(Rq_vec)
                res_abs = float((rvn * rvn + rqn * rqn) ** 0.5)

                if res0 is None:
                    res0 = max(res_abs, 1e-30)
                res_rel = res_abs / res0

                du = float(np.sqrt(assemble(inner(u_sol - u_prev_it, u_sol - u_prev_it) * dx)))
                un = float(np.sqrt(assemble(inner(u_sol, u_sol) * dx)))
                rel_du = du / max(un, 1e-16)

                if rank == 0 and args.picard_monitor:
                    print(
                        f"  Picard {k:2d}: ResAbs={res_abs:.3e} ResRel={res_rel:.3e} "
                        f"(|Rv|={rvn:.3e}, |Rq|={rqn:.3e})  "
                        f"|du|={du:.3e} rel_du={rel_du:.3e}  "
                        f"KSPits={ksp_its_picard} KSPcum={ksp_its_timestep}",
                        flush=True
                    )

                if res_abs <= float(args.picard_atol) + float(args.picard_rtol):
                    break
                if rel_du < 1e-14:
                    break

            # ----------------------------
            # Forces: INTEGRAL + REACTION (same timestep, after Picard converged)
            # ----------------------------
            Fx_int, Fy_int, Fz_int = cylinder_force_xyz_integral()
            Cd_int = 2.0 * Fx_int / den_ref
            Cl_y_int = 2.0 * Fy_int / den_ref
            Cl_z_int = 2.0 * Fz_int / den_ref

            #Rv_form_rea = Rv_form  # start from Galerkin momentum residual form

            #if use_supg:
                #if mode == "BDF1":
                   # Rv_form_rea = Rv_form_rea + inner(mom_strong_bdf1, tau_nl * dot(grad(vR), u_sol)) * dxq
               # else:  # BDF2
                  #  Rv_form_rea = Rv_form_rea + inner(mom_strong_bdf2, tau_nl * dot(grad(vR), u_sol)) * dxq
            
            #Fx_rea, Fy_rea, Fz_rea = cylinder_force_xyz_reaction(Rv_form_rea)
            Fx_rea, Fy_rea, Fz_rea = cylinder_force_xyz_reaction(Rv_form)
            
            Cd_rea = 2.0 * Fx_rea / den_ref
            Cl_y_rea = 2.0 * Fy_rea / den_ref
            Cl_z_rea = 2.0 * Fz_rea / den_ref

            # Update history
            u_nm1.assign(u_n)
            u_n.assign(u_sol)
            p_nm1.assign(p_n)
            p_n.assign(p_sol)

            if (nstep % int(args.write_freq) == 0) or (nstep == 1):
                vtk.write(u_sol, p_sol, time=t_now)

            if rank == 0:
                print(
                    f"{nstep:5d}  {mode:>4}  {t_now:12.4e}  {dt_val:11.3e}  {Um_t:11.3e}  {umax_state:11.3e}  {k:6d}  {ksp_its_timestep:8d}",
                    flush=True
                )
                print(
                    f"  [INT] Fx={Fx_int: .6e}  Fy={Fy_int: .6e}  Fz={Fz_int: .6e}   Cd={Cd_int: .6e}  Cl_y={Cl_y_int: .6e}  Cl_z={Cl_z_int: .6e}",
                    flush=True
                )
                print(
                    f"  [REA] Fx={Fx_rea: .6e}  Fy={Fy_rea: .6e}  Fz={Fz_rea: .6e}   Cd={Cd_rea: .6e}  Cl_y={Cl_y_rea: .6e}  Cl_z={Cl_z_rea: .6e}",
                    flush=True
                )

                csv_writer.writerow([
                    nstep, k, ksp_its_timestep, t_now, dt_val,
                    Um_t,
                    Fx_int, Fy_int, Fz_int, Cd_int, Cl_y_int, Cl_z_int,
                    Fx_rea, Fy_rea, Fz_rea, Cd_rea, Cl_y_rea, Cl_z_rea,
                    float(args.rho), float(args.mu), nu_val, float(args.D), float(args.Umean_ref),
                ])
                csv_fh.flush()

            last_completed_step = nstep
            last_completed_time = t_now

            if int(args.checkpoint_freq) > 0 and (nstep % int(args.checkpoint_freq) == 0):
                save_checkpoint(args.checkpoint_prefix, nstep, t_now, dt_val, u_n, p_n, u_nm1, p_nm1)

    except KeyboardInterrupt:
        if rank == 0:
            print("\n[INT] KeyboardInterrupt caught: writing checkpoint and exiting...", flush=True)
        if int(args.checkpoint_freq) != 0 and last_completed_step >= 0:
            save_checkpoint(args.checkpoint_prefix, last_completed_step, last_completed_time, last_completed_dt,
                            u_n, p_n, u_nm1, p_nm1)
    finally:
        if rank == 0 and csv_fh is not None:
            csv_fh.close()

    wall1 = perf_counter()
    if rank == 0:
        print(f"Done. walltime = {wall1 - wall0:.3f} s", flush=True)
        print(f"Wrote VTK: {args.vtk}", flush=True)
        print(f"Wrote CSV: {args.csv}", flush=True)
        if int(args.checkpoint_freq) > 0:
            print(f"Checkpoint prefix: {args.checkpoint_prefix}", flush=True)


if __name__ == "__main__":
    main()
