#!/usr/bin/env python3
"""
IROC‑X  •  Phase‑1 Baseline   (2‑D channel tests only)

What this DOES
--------------
* builds a simple 2‑D rectangular micro‑channel mesh
* solves steady Stokes / Navier‑Stokes for Poiseuille flow
* runs two scalar–transport sanity tests
   ‑ pure diffusion of a Gaussian blob
   ‑ advection‑diffusion of the same blob
* writes minimal artefacts:
      mesh.png, vel_profile.png,
      diffusion_gauss_t*.png,
      advdiff_gauss_t*.png
* prints performance / diagnostic lines

What this DOES NOT do
---------------------
* no reactions, inhibitors, GSH, biosensors
* has optional live matplotlib animations for every species
* no HDF5 checkpoints
"""

#  ── imports ──────────────────────────────────────────────────────────────
import time, pathlib, math
import numpy as np
import matplotlib.pyplot as plt
from dolfin import (
    RectangleMesh, Point, Mesh, VectorFunctionSpace, FunctionSpace, TrialFunction, TrialFunctions,
    TestFunction, TestFunctions, Function, Constant, DirichletBC, Expression, inner, grad, div,
    sym, solve, dot, dx, assemble, plot, parameters, File, project, sqrt, MPI,
    FiniteElement, MixedElement, VectorElement, Measure,
)
from dolfin import set_log_level, LogLevel
from dolfin import CellDiameter
from ufl import nabla_div
plt.rcParams.update({"figure.dpi": 130})

# Silence “Solving linear variational problem.” spam
set_log_level(LogLevel.WARNING)

#  ── configuration ────────────────────────────────────────────────────────
OUT = pathlib.Path("baseline_results"); OUT.mkdir(exist_ok=True)
Lx, Ly = 1.0e-3, 2.0e-4            # 1 mm × 0.2 mm channel
NX, NY = 240, 48                   # mesh resolution (refined for smoother advection)

# --- Level‑3 sensor configuration (moved up to avoid NameError) ---
SENSORS = [
    ("HyPer7",       8.0e5, 0.02),
    ("OxiDiaGreen",  5.0e5, 0.04),
    ("Quencher",     1.2e6, 0.01),
    ("MitoPerOx",    6.0e5, 0.03),
]
SENSOR_NAMES = [n for n, *_ in SENSORS]

# sensor saturation value (fully oxidised)
S_max = 1.0

# Enable interactive live plotting
LIVE_ANIM = True

MU  = 1.0e-3                       # Pa·s        (dynamic viscosity)
RHO = 1000.0                       # kg/m³
U_IN = 1.0e-4                      # m/s at centreline (100 µm/s)

D_SCALAR = 1.0e-9                  # m²/s diffusion coefficient

# --- simple first‑order reaction (Phase‑2) -----------------------------
K_REACT = 0.5                   # 1/s   rate constant (max)
XC_REACT = 0.5*Lx               # m     reaction starts after this x

DT  = 0.01                         # s
N_STEPS_DIFF = 200
N_STEPS_ADV  = 200

parameters["form_compiler"]["optimize"]   = True
parameters["form_compiler"]["cpp_optimize"] = True
rank = MPI.comm_world.rank

# ---- diagnostic dimensionless groups ----
Pe = U_IN*Lx/D_SCALAR                 # Péclet
Da = K_REACT*Lx**2/D_SCALAR           # Damköhler (diffusion‑based)
if rank == 0:
    print(f"[Diagnostics]  Pe ≈ {Pe:.1f}   Da ≈ {Da:.1f}")

#  ── helper: save quick PNG of a FEniCS Function ──────────────────────────
def save_field_png(func, title, fname, cmap="viridis"):
    plt.figure()
    img = plot(func, title=title, cmap=cmap)
    plt.colorbar(img)
    # avoid white “holes” by clamping the lower limit
    img.set_clim(vmin=0.0)
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()

#  ── G‑1 Geometry / mesh ──────────────────────────────────────────────────
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), NX, NY)

# dedicated integration measure tied to this mesh (avoids “missing domain” errors)
dx_mesh = Measure("dx", domain=mesh)

# --- live animation setup ---
if LIVE_ANIM and rank == 0:
    plt.ion()
    fig_H, ax_H = plt.subplots(num="H2O2 live")
    fig_G, ax_G = plt.subplots(num="GSH live")

    # extra live windows for each sensor
    if LIVE_ANIM and rank == 0:
        ax_S = { }
        for name in SENSOR_NAMES:
            fig_tmp, ax_tmp = plt.subplots(num=f"{name} live")
            ax_S[name] = ax_tmp

        fig_O2,  ax_O2  = plt.subplots(num="O2 live")
        fig_GLC, ax_GLC = plt.subplots(num="Glucose live")
        fig_LAC, ax_LAC = plt.subplots(num="Lactate live")

# spatially‑varying first‑order rate  k(x)  (zero on inlet half, ramps to K_REACT)
k_expr = Expression("x[0] <= xc ? 0.0 : k0*(x[0]-xc)/(Lx-xc)",
                    k0=K_REACT, xc=XC_REACT, Lx=Lx, degree=1)

if rank == 0:
    save_field_png(project(Constant(0.0), FunctionSpace(mesh, "P", 1)),
                   "Mesh geometry", "mesh.png")

#  ── G‑2 Steady Poiseuille flow (mixed Taylor–Hood) ───────────────────────
TH = MixedElement([VectorElement("P", mesh.ufl_cell(), 2),
                   FiniteElement("P",  mesh.ufl_cell(), 1)])
W  = FunctionSpace(mesh, TH)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w_     = Function(W)

# Dirichlet BCs on sub‑spaces
inlet  = "near(x[0], 0.0)"
outlet = f"near(x[0], {Lx})"
walls  = f"near(x[1], 0.0) || near(x[1], {Ly})"

bc_in   = DirichletBC(W.sub(0),
                      Expression(("4*U*x[1]*(Ly - x[1])/(Ly*Ly)", "0.0"),
                                 U=U_IN, Ly=Ly, degree=2),
                      inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bc_p_out = DirichletBC(W.sub(1), Constant(0.0), outlet)

a = ( MU*inner(sym(grad(u)), sym(grad(v)))      # viscous term
      - div(v)*p + q*div(u) )*dx
L = Constant(0.0)*q*dx

solve(a == L, w_, [bc_in, bc_walls, bc_p_out],
      solver_parameters={"linear_solver": "mumps"})

u_, p_ = w_.split(True)

# SUPG stabilisation parameter (simple constant estimate)
h_cell   = mesh.hmin()
U_MAG    = U_IN                               # centre‑line speed; used as proxy
tau_supg = Constant(0.5*h_cell/(U_MAG + 1e-12))

if rank == 0:
    save_field_png(sqrt(dot(u_, u_)), "Velocity magnitude", "vel_profile.png")

#  ── G‑3 Phase‑2: coupled H2O2 + GSH + HyPer sensor ──────────────────────
#
#   ∂H/∂t + ∇·(u H)  =  D_H ∇²H   +  S_NOX  – k_GSH H G
#   ∂G/∂t + ∇·(u G)  =  D_G ∇²G   – k_GSH H G
#   ∂S/∂t +          =              k_SENS H (S_max – S)   –  k_RED S
#
#  * volumetric NOX source is confined to the top wall strip  (y > 0.9 Ly)
#  * simple first‑order HyPer (sensor) oxidation / reduction
#
D_H2O2  = 1.4e-9           # m²/s  (free diffusion)
D_GSH   = 0.8e-9
H_inlet = 0.0              # M
G_inlet = 1.0e-3           # M   (1 mM total glutathione)

k_NOX   = 5.0e-7           # mol m⁻³ s⁻¹   (wall‑adjacent volumetric rate)
k_GSH   = 1.0e6            # M⁻¹ s⁻¹        (bi‑molecular)

# --- metabolic layer (Level‑4) ------------------------------------
D_O2   = 2.0e-9   # m²/s
D_GLC  = 1.0e-9
D_LAC  = 1.2e-9

O2_inlet  = 0.21*1.3e-3      # Henry‑law ~0.27 mM at 1 atm
GLC_inlet = 5.0e-3           # 5 mM glucose
LAC_inlet = 0.0

# simple first‑order uptake / production until real kinetics are fitted
k_O2_uptake  = 1.0e-2        # s⁻¹
k_GLC_uptake = 2.0e-2        # s⁻¹
k_LAC_prod   = 2.0e-2        # s⁻¹  (stoichiometric with glucose)

DT      = 0.02             # s
N_STEPS = 400

Vsc = FunctionSpace(mesh, "P", 1)

H,  H_n  = Function(Vsc), Function(Vsc)
G,  G_n  = Function(Vsc), Function(Vsc)
# create one Function slot per sensor
S   = {n: Function(Vsc) for n in SENSOR_NAMES}
S_n = {n: Function(Vsc) for n in SENSOR_NAMES}
dS  = TrialFunction(Vsc)

# metabolic species
O2,  O2_n   = Function(Vsc), Function(Vsc)
GLC, GLC_n  = Function(Vsc), Function(Vsc)
LAC, LAC_n  = Function(Vsc), Function(Vsc)

test = TestFunction(Vsc)
dH   = TrialFunction(Vsc)
dG   = TrialFunction(Vsc)

# inlet Dirichlet BCs
bc_H_in = DirichletBC(Vsc, Constant(H_inlet), inlet)
bc_G_in = DirichletBC(Vsc, Constant(G_inlet), inlet)

bc_O2_in  = DirichletBC(Vsc, Constant(O2_inlet),  inlet)
bc_GLC_in = DirichletBC(Vsc, Constant(GLC_inlet), inlet)
bc_LAC_in = DirichletBC(Vsc, Constant(LAC_inlet), inlet)

# NOX volumetric source   S_NOX = k_NOX * step(y)
nox_src = Expression("x[1]>0.9*Ly ? kN : 0.0", degree=1, Ly=Ly, kN=k_NOX)

# implicit‑diffusion, explicit‑convection scheme
# base Galerkin part
a_H  = (dH*test + DT*D_H2O2*dot(grad(dH), grad(test)) + DT*k_GSH*G_n*dH*test)*dx
a_G  = (dG*test + DT*D_GSH *dot(grad(dG), grad(test)) + DT*k_GSH*H_n*dG*test)*dx
# SUPG stabilisation (streamline‑upwind)
a_H += ( DT*tau_supg*dot(u_, grad(test))*dot(u_, grad(dH)) )*dx
a_G += ( DT*tau_supg*dot(u_, grad(test))*dot(u_, grad(dG)) )*dx

# O2, Glucose, Lactate weak forms (no SUPG yet for simplicity)
a_O2  = (TrialFunction(Vsc)*test + DT*D_O2 * dot(grad(TrialFunction(Vsc)), grad(test))) * dx
a_GLC = (TrialFunction(Vsc)*test + DT*D_GLC * dot(grad(TrialFunction(Vsc)), grad(test))) * dx
a_LAC = (TrialFunction(Vsc)*test + DT*D_LAC * dot(grad(TrialFunction(Vsc)), grad(test))) * dx

# --- aggregate metrics CSV ----------------------------------------
AREA = assemble(Constant(1.0)*dx_mesh)
if rank == 0:
    metrics_f = open(OUT / "metrics.csv", "w")
    metrics_f.write("t,H_mean,G_mean," + ",".join([f"{n}_mean" for n in SENSOR_NAMES]) + ",TEER_norm,LDH_flux,O2_mean,GLC_mean,LAC_mean\n")

for n in range(N_STEPS):
    # ----- update H2O2 ---------------------------------------------------
    L_H = ( (H_n - DT*dot(u_, grad(H_n)) + DT*nox_src) * test )*dx
    solve(a_H == L_H, H, bcs=[bc_H_in])

    # ----- update GSH ----------------------------------------------------
    L_G = ( (G_n - DT*dot(u_, grad(G_n))) * test )*dx
    solve(a_G == L_G, G, bcs=[bc_G_in])

    # ----- update sensors (multi‑analyte) ----------------------------
    for name, k_sens, k_red in SENSORS:
        a_S = (dS*test + DT*k_red*dS*test)*dx
        L_S = ( (S_n[name] + DT*k_sens*H_n*(S_max - S_n[name])) * test )*dx
        solve(a_S == L_S, S[name])
        S_n[name].assign(S[name])

    # ----- update metabolic species ---------------------------------
    L_O2  = ((O2_n  - DT*dot(u_, grad(O2_n))  - DT*k_O2_uptake*O2_n)  * test) * dx
    solve(a_O2  == L_O2,  O2,  bcs=[bc_O2_in])

    L_GLC = ((GLC_n - DT*dot(u_, grad(GLC_n)) - DT*k_GLC_uptake*GLC_n) * test) * dx
    solve(a_GLC == L_GLC, GLC, bcs=[bc_GLC_in])

    L_LAC = ((LAC_n - DT*dot(u_, grad(LAC_n)) + DT*k_LAC_prod*GLC_n) * test) * dx
    solve(a_LAC == L_LAC, LAC, bcs=[bc_LAC_in])

    O2_n.assign(O2)
    GLC_n.assign(GLC)
    LAC_n.assign(LAC)

    # write metrics every 5 steps
    if rank == 0 and n % 5 == 0:
        H_mean = assemble(H*dx_mesh)/AREA
        G_mean = assemble(G*dx_mesh)/AREA
        sensor_means = [assemble(S[name]*dx_mesh)/AREA for name in SENSOR_NAMES]

        O2_mean  = assemble(O2 *dx_mesh)/AREA
        GLC_mean = assemble(GLC*dx_mesh)/AREA
        LAC_mean = assemble(LAC*dx_mesh)/AREA

        # literature‑fit: IC50_TEER ≈ 100 µM H2O2 (Caco‑2, Salvianolic‑B study) citeturn0search8
        IC50_TEER = 1.0e-4   # M
        TEER_norm = 1.0 / (1.0 + (H_mean/IC50_TEER)**2)

        # LDH release: linear above 50 µM peroxide (hippocampal study) citeturn0search11
        LDH_flux = max(0.0, (H_mean - 5.0e-5) * 5.0e6)   # arbitrary scaling to match % release

        metrics_f.write(f"{n*DT:.2f},{H_mean:.6e},{G_mean:.6e}," +
                        ",".join([f"{m:.6e}" for m in sensor_means]) +
                        f",{TEER_norm:.6e},{LDH_flux:.6e},{O2_mean:.6e},{GLC_mean:.6e},{LAC_mean:.6e}\n")

    H_n.assign(H)
    G_n.assign(G)

    # live update every 5 steps
    if LIVE_ANIM and rank == 0 and n % 5 == 0:
        ax_H.clear()
        plt.sca(ax_H)
        plot(H, cmap="viridis")
        ax_H.set_title(f"H2O2  t={n*DT:.1f}s")

        ax_G.clear()
        plt.sca(ax_G)
        plot(G, cmap="viridis")
        ax_G.set_title(f"GSH  t={n*DT:.1f}s")

        for name in SENSOR_NAMES:
            ax_S[name].clear()
            plt.sca(ax_S[name])
            plot(S[name], cmap="viridis")
            ax_S[name].set_title(f"{name}  t={n*DT:.1f}s")

        for ax_, field_, title_ in [(ax_O2,  O2,  "O2"),
                                    (ax_GLC, GLC, "Glc"),
                                    (ax_LAC, LAC, "Lac")]:
            ax_.clear()
            plt.sca(ax_)
            plot(field_, cmap="viridis")
            ax_.set_title(f"{title_}  t={n*DT:.1f}s")

        plt.pause(0.001)

    if n % 40 == 0 and rank == 0:
        save_field_png(H, f"H2O2 t={n*DT:.1f}s", f"h2o2_t{n:03d}.png")
        save_field_png(G, f"GSH  t={n*DT:.1f}s", f"gsh_t{n:03d}.png")
        for name in SENSOR_NAMES:
            save_field_png(S[name], f"{name} t={n*DT:.1f}s", f"{name.lower()}_t{n:03d}.png")

#  ── G‑5/6 Stability + performance printouts ─────────────────────────────
if rank == 0:
    print(f"Scalar max(H)       : {H.vector().max():.3e}")
    print(f"Scalar min(G)       : {G.vector().min():.3e}")
    print(f"Final H2O2 max      : {H.vector().max():.3e}")
    print(f"Final GSH min       : {G.vector().min():.3e}")
    print(f"Sensor mean signal  : {assemble(S['HyPer7']*dx_mesh)/AREA:.3e}")
    print("Results in", OUT.resolve())
    metrics_f.close()
