#!/usr/bin/env python3
"""
Advanced ROS Biosensor Organoid Simulation for IROC-X PRIME
===========================================================

This enhanced simulation package models real-time ROS dynamics in perfused organoid-on-chip
systems with microfluidic flow, advanced biosensors, and AI-ready data generation.

Key Enhancements:
- 2D/3D advection-diffusion with realistic flow profiles
- Enhanced biosensor library (HyPer7, roGFP2-Orp1, genetically encoded variants)
- Microfluidic channel modeling with perfusion
- Antioxidant enzyme kinetics (catalase, GPx, peroxiredoxins)
- NOX source modeling with genetic modulation
- AI training data generation with ground truth labels
- DEI biomarker simulation with genotype-specific parameters
- Modern visualization with interactive plots and export capabilities

Author: IROC-X PRIME Team
Version: 2.0
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve 
from scipy.sparse import eye as speye, kron as spkron
from scipy.optimize import curve_fit
import pandas as pd
import json
import warnings
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Set modern plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)
from pathlib import Path
from datetime import datetime

def stamp(name: str, ext: str, sub=""):
    ts = datetime.now().strftime("%m.%d - %H%M")
    root = Path("/Users/ishaangubbala/Documents/IROCX/results")
    root.mkdir(parents=True, exist_ok=True)
    if sub:
        root /= sub
        root.mkdir(exist_ok=True)
    fname = f"{name} {ts}.{ext}"
    return root / fname

class BiosensorType(Enum):
    """Enhanced biosensor library with latest variants"""
    HYPER = "HyPer"
    HYPER7 = "HyPer7"
    ROGFP2_ORP1 = "roGFP2-Orp1"
    ROGFP2_GRXC1 = "roGFP2-GrxC1"
    HYPERYELLOW = "HyPerYellow"
    OXIDIAEGREEN = "OxiDiaGreen"
    PEROXY_YELLOW1 = "PeroxYellow1"

class GenotypeVariant(Enum):
    """Genetic variants affecting ROS metabolism"""
    WILD_TYPE = "WT"
    CAT_LOW = "CAT_rs1001179"  # Reduced catalase activity
    GPX1_HIGH = "GPX1_rs1050450"  # Enhanced GPx activity
    NOX4_HIGH = "NOX4_rs3017887"  # Enhanced NOX4 expression
    COMBINED_RISK = "Multi-variant"  # Combined high-risk alleles

@dataclass
class AdvancedSensorParameters:
    """Enhanced biosensor parameters with latest kinetic data"""
    k_ox: float              # Oxidation rate constant (µM⁻¹ s⁻¹)
    k_red: float             # Reduction rate constant (s⁻¹)
    kd_h2o2: float          # H₂O₂ dissociation constant (µM)
    dynamic_range: float     # Maximum fold-change in ratio
    response_time: float     # Time to 90% response (s)
    ph_sensitivity: float    # pH dependence factor
    name: str               # Biosensor name
    excitation_peak: float   # Excitation wavelength (nm)
    emission_peak: float     # Emission wavelength (nm)
    quantum_yield: float     # Fluorescence quantum yield
    photobleaching_rate: float  # Photobleaching constant (s⁻¹)
    
    def __post_init__(self):
        """Validate parameters"""
        if self.k_ox <= 0 or self.k_red <= 0:
            raise ValueError("Rate constants must be positive")
        if self.dynamic_range < 1:
            raise ValueError("Dynamic range must be ≥ 1")

@dataclass
class GenotypeParameters:
    """Genotype-specific enzyme parameters"""
    catalase_vmax: float     # µM/s
    catalase_km: float       # µM
    gpx_vmax: float         # µM/s
    gpx_km: float           # µM
    nox_activity: float     # Relative NOX activity (0-2)
    prx_activity: float     # Peroxiredoxin activity
    ethnicity: str          # Population group
    
# Enhanced biosensor library with latest literature values
ADVANCED_BIOSENSOR_PARAMS = {
    BiosensorType.HYPER: AdvancedSensorParameters(
        k_ox=0.76, k_red=0.0023, kd_h2o2=165.0, dynamic_range=6.5,
        response_time=12.0, ph_sensitivity=0.15, name="HyPer",
        excitation_peak=420, emission_peak=516, quantum_yield=0.23,
        photobleaching_rate=0.0012
    ),
    BiosensorType.HYPER7: AdvancedSensorParameters(
        k_ox=1.2, k_red=0.0018, kd_h2o2=95.0, dynamic_range=8.7,
        response_time=8.5, ph_sensitivity=0.08, name="HyPer7",
        excitation_peak=420, emission_peak=516, quantum_yield=0.31,
        photobleaching_rate=0.0008
    ),
    BiosensorType.ROGFP2_ORP1: AdvancedSensorParameters(
        k_ox=0.45, k_red=0.0012, kd_h2o2=87.0, dynamic_range=4.2,
        response_time=25.0, ph_sensitivity=0.12, name="roGFP2-Orp1",
        excitation_peak=405, emission_peak=510, quantum_yield=0.68,
        photobleaching_rate=0.0015
    ),
    BiosensorType.HYPERYELLOW: AdvancedSensorParameters(
        k_ox=0.89, k_red=0.0019, kd_h2o2=125.0, dynamic_range=5.8,
        response_time=10.2, ph_sensitivity=0.09, name="HyPerYellow",
        excitation_peak=514, emission_peak=527, quantum_yield=0.77,
        photobleaching_rate=0.0010
    ),
    BiosensorType.OXIDIAEGREEN: AdvancedSensorParameters(
        k_ox=0.65, k_red=0.0025, kd_h2o2=78.0, dynamic_range=7.1,
        response_time=15.5, ph_sensitivity=0.11, name="OxiDiaGreen",
        excitation_peak=485, emission_peak=510, quantum_yield=0.45,
        photobleaching_rate=0.0020
    )
}

# Genotype-specific parameters based on population genetics
GENOTYPE_PARAMS = {
    GenotypeVariant.WILD_TYPE: GenotypeParameters(
        catalase_vmax=6000, catalase_km=150, gpx_vmax=450, gpx_km=25,
        nox_activity=1.0, prx_activity=1.0, ethnicity="Mixed"
    ),
    GenotypeVariant.CAT_LOW: GenotypeParameters(
        catalase_vmax=3200, catalase_km=180, gpx_vmax=450, gpx_km=25,
        nox_activity=1.0, prx_activity=1.0, ethnicity="European"
    ),
    GenotypeVariant.GPX1_HIGH: GenotypeParameters(
        catalase_vmax=6000, catalase_km=150, gpx_vmax=680, gpx_km=22,
        nox_activity=1.0, prx_activity=1.0, ethnicity="East Asian"
    ),
    GenotypeVariant.NOX4_HIGH: GenotypeParameters(
        catalase_vmax=6000, catalase_km=150, gpx_vmax=450, gpx_km=25,
        nox_activity=1.45, prx_activity=1.0, ethnicity="African"
    ),
    GenotypeVariant.COMBINED_RISK: GenotypeParameters(
        catalase_vmax=3200, catalase_km=180, gpx_vmax=350, gpx_km=28,
        nox_activity=1.35, prx_activity=0.85, ethnicity="Mixed"
    )
}

SENSOR_PANEL = [
    BiosensorType.HYPER7,
    BiosensorType.HYPERYELLOW,
    BiosensorType.OXIDIAEGREEN,
    BiosensorType.ROGFP2_ORP1,
]

def make_sensor_panel(genotype):
    panel = {}
    for sensor in SENSOR_PANEL:
        sim = CoupledIROCXSimulation(
            biosensor_type=sensor,
            geometry="rectangular",
            organoid_size=300e-4,
            n_spatial_points=(100, 60),             # ← finer grid
            flow_rate=0.2,
            genotype=genotype
        )
        sim._set_pk_stimulus(C_max=0.6, t_inf=1800, t_half=7200)
        panel[sensor.value] = sim
    return panel

class AdvancedSensorKinetics:
    """Enhanced biosensor kinetics with realistic cellular environment"""
    
    def __init__(self, biosensor_type: BiosensorType, ph: float = 7.2, 
                 temperature: float = 37.0):
        self.params = ADVANCED_BIOSENSOR_PARAMS[biosensor_type]
        self.ph = ph
        self.temperature = temperature
        self.biosensor_type = biosensor_type
        self.photobleaching_accumulator = 0.0
        
        # Temperature and pH corrections
        self.k_ox_eff = self.params.k_ox * self._temperature_correction() * self._ph_correction()
        self.k_red_eff = self.params.k_red * self._temperature_correction()
        
    def _temperature_correction(self) -> float:
        """Temperature dependence (Q10 = 2.5)"""
        return 2.5 ** ((self.temperature - 25) / 10)
    
    def _ph_correction(self) -> float:
        """pH dependence of sensor response"""
        return 1 - self.params.ph_sensitivity * abs(self.ph - 7.2)
    
    def kinetics_ode(self, t: float, state: np.ndarray, h2o2_conc: float, 
                    gsh_ratio: float = 10.0, illumination: float = 0.0) -> np.ndarray:
        """
        Enhanced ODE system including GSH coupling and photobleaching
        
        Args:
            t: time (s)
            state: [R_ox, active_fraction] - oxidized fraction and active sensor
            h2o2_conc: local H₂O₂ concentration (µM)
            gsh_ratio: GSH/GSSG ratio
            illumination: illumination intensity (W/cm²)
        """
        R_ox, active_fraction = state
        R_red = (1.0 - R_ox) * active_fraction
        
        # Effective H₂O₂ with competitive binding
        effective_h2o2 = h2o2_conc / (1 + h2o2_conc / self.params.kd_h2o2)
        
        # GSH-dependent reduction (for roGFP2 variants)
        if "roGFP2" in self.params.name:
            k_red_gsh = self.k_red_eff * (gsh_ratio / (1 + gsh_ratio))
        else:
            k_red_gsh = self.k_red_eff
        
        # Oxidation-reduction kinetics
        dR_ox_dt = (self.k_ox_eff * effective_h2o2 * R_red - 
                   k_red_gsh * R_ox * active_fraction)
        
        # Photobleaching
        dactive_dt = -self.params.photobleaching_rate * illumination * active_fraction
        
        return np.array([dR_ox_dt, dactive_dt])
    
    def steady_state_response(self, h2o2_conc: float, gsh_ratio: float = 10.0) -> Tuple[float, float]:
        """Calculate steady-state response"""
        effective_h2o2 = h2o2_conc / (1 + h2o2_conc / self.params.kd_h2o2)
        
        if "roGFP2" in self.params.name:
            k_red_gsh = self.k_red_eff * (gsh_ratio / (1 + gsh_ratio))
        else:
            k_red_gsh = self.k_red_eff
        
        R_ox_ss = (self.k_ox_eff * effective_h2o2) / (
            self.k_ox_eff * effective_h2o2 + k_red_gsh
        )
        
        ratio_signal = 1.0 + (self.params.dynamic_range - 1.0) * R_ox_ss
        return R_ox_ss, ratio_signal

class MicrofluidicFlow:
    """Models realistic microfluidic flow patterns in organ-on-chip devices"""
    
    def __init__(self, channel_width: float = 1000e-4, channel_height: float = 100e-4,
                 flow_rate: float = 0.1, viscosity: float = 0.001):
        self.width = channel_width    # cm
        self.height = channel_height  # cm
        self.flow_rate = flow_rate    # µL/min
        self.viscosity = viscosity    # Pa·s
        
        # Calculate characteristic velocity
        cross_section = self.width * self.height  # cm²
        self.v_avg = (flow_rate * 1e-6 / 60) / cross_section  # cm/s
        
        # Reynolds number
        self.Re = (1000 * self.v_avg * self.height) / self.viscosity
        
    def velocity_profile(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate 2D velocity profile in rectangular channel
        
        Args:
            y: y-coordinates (normalized by channel width)
            z: z-coordinates (normalized by channel height)
        """
        # Analytical solution for rectangular channel
        velocity = np.zeros_like(y)
        
        # Fourier series approximation (first 10 terms)
        for n in range(1, 21, 2):  # odd terms only
            term = (1 - np.cosh(n * np.pi * z / 2) / np.cosh(n * np.pi / 2)) * np.cos(n * np.pi * y / 2)
            velocity += (32 / (n**3 * np.pi**3)) * term
        
        return 6 * self.v_avg * velocity
    
    def peclet_number(self, diffusion_coeff: float) -> float:
        """Calculate Peclet number for advection-diffusion balance"""
        return self.v_avg * self.width / diffusion_coeff

class AdvancedOrganoidDiffusion:
    """
    Enhanced 2D/3D diffusion model with microfluidic flow and realistic geometry
    """
    
    def __init__(self, geometry: str = "spherical", 
                 primary_size: float = 250e-4,  # cm
                 secondary_size: float = None,
                 n_points: Tuple[int, ...] = (50,),
                 diffusion_coeff: float = 2.5e-5,
                 flow_field: Optional[MicrofluidicFlow] = None,
                 genotype: GenotypeVariant = GenotypeVariant.WILD_TYPE):
        
        self.geometry = geometry
        self.primary_size = primary_size
        self.secondary_size = secondary_size or primary_size
        self.n_points = n_points
        self.D = diffusion_coeff
        self.flow_field = flow_field
        self.genotype_params = GENOTYPE_PARAMS[genotype]
        
        # Create spatial mesh
        self._create_mesh()
        
        # Initialize fields
        self.C = np.zeros(self.mesh_shape)  # H₂O₂ concentration (µM)
        self.GSH  = np.full(self.mesh_shape,     5.0)  # 5 µM baseline
        self.GSSG = np.full(self.mesh_shape,   0.05)   # GSH:GSSG ≈ 100
        
        # Build operators
        self._build_operators()
        
    def _create_mesh(self):
        """Create spatial mesh based on geometry"""
        if self.geometry == "spherical":
            self.r = np.linspace(0, self.primary_size, self.n_points[0])
            self.mesh_shape = (self.n_points[0],)
            self.dr = self.r[1] - self.r[0]
            
        elif self.geometry == "cylindrical":
            self.r = np.linspace(0, self.primary_size, self.n_points[0])
            self.z = np.linspace(0, self.secondary_size, self.n_points[1])
            self.mesh_shape = self.n_points
            self.dr = self.r[1] - self.r[0]
            self.dz = self.z[1] - self.z[0]
            
        elif self.geometry == "rectangular":
            self.x = np.linspace(0, self.primary_size, self.n_points[0])
            self.y = np.linspace(0, self.secondary_size, self.n_points[1])
            self.mesh_shape = self.n_points
            self.dx = self.x[1] - self.x[0]
            self.dy = self.y[1] - self.y[0]
            
    def _build_operators(self):
        """Build finite difference operators"""
        if self.geometry == "spherical":
            self._build_spherical_operators()
        elif self.geometry == "rectangular":
            self._build_rectangular_operators()
        else:
            raise NotImplementedError(f"Geometry {self.geometry} not implemented")
    
    def _build_spherical_operators(self):
        """Build spherical diffusion operator"""
        n = self.n_points[0]
        dr = self.dr
        
        # Main diagonal
        main_diag = np.zeros(n)
        main_diag[0] = -6 * self.D / dr**2
        main_diag[1:] = -2 * self.D / dr**2
        
        # Add consumption terms
        main_diag -= self._consumption_rate()
        
        # Upper diagonal
        upper_diag = np.zeros(n-1)
        upper_diag[0] = 6 * self.D / dr**2
        for i in range(1, n-1):
            r_i = self.r[i]
            upper_diag[i] = self.D / dr**2 * (1 + dr/(2*r_i))
        
        # Lower diagonal
        lower_diag = np.zeros(n-1)
        for i in range(1, n-1):
            r_i = self.r[i]
            lower_diag[i-1] = self.D / dr**2 * (1 - dr/(2*r_i))
        
        self.diff_operator = diags(
            [lower_diag, main_diag, upper_diag],
            [-1, 0, 1],
            shape=(n, n),
            format='csr'
        )
    
    def _build_rectangular_operators(self):                     # FIXED
        """Build 2-D rectangular diffusion-advection operator (sparse-safe)."""
        nx, ny = self.n_points
        dx, dy = self.dx, self.dy

        # 2-D Laplacian  Δ = ∂²/∂x² + ∂²/∂y²
        Dx = diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx), format="csr") / dx ** 2
        Dy = diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny), format="csr") / dy ** 2

        Ix = speye(nx, format="csr")
        Iy = speye(ny, format="csr")

        Lx = spkron(Iy, Dx, format="csr")
        Ly = spkron(Dy, Ix, format="csr")

        self.diff_operator = -self.D * (Lx + Ly)  # start with pure diffusion

        # Add advection if flow defined
        if self.flow_field:
            self._add_advection_terms()

        # First-pass consumption
        consumption = self._consumption_rate()
        self.diff_operator -= diags(consumption, 0, format="csr")

        # Keep a clean copy for per-timestep refresh
        self.base_operator = self.diff_operator.copy()

    def _add_advection_terms(self):                             # FIXED
        """Central-difference sparse advection in a straight channel."""
        if self.geometry != "rectangular":
            return

        nx, ny = self.n_points
        dx, dy = self.dx, self.dy

        # Velocity field (flattened)
        Y, X = np.meshgrid(np.linspace(-0.5, 0.5, ny), np.linspace(-0.5, 0.5, nx))
        vx = self.flow_field.velocity_profile(Y.flatten(), X.flatten())
        vy = np.zeros_like(vx)  # no transverse component

        Adv_x = diags([-1, 1], [0, 1], shape=(nx, nx), format="csr") / dx
        Adv_y = diags([-1, 0, 1], [-1, 0, 1], shape=(ny, ny), format="csr") / (2 * dy)

        Ix = speye(nx, format="csr")
        Iy = speye(ny, format="csr")

        Advx = spkron(Iy, Adv_x, format="csr")
        Advy = spkron(Adv_y, Ix, format="csr")

        n = nx * ny
        self.diff_operator += (
            diags(vx, 0, shape=(n, n), format="csr") @ Advx
            + diags(vy, 0, shape=(n, n), format="csr") @ Advy
        )

    # ─────────────── Time-step and helper methods (one fix) ────────────────
    def _update_consumption_operator(self):                     # FIXED
        """Refresh diagonal with current consumption each step (no drift)."""
        consumption = self._consumption_rate()
        n = self.base_operator.shape[0]
        self.diff_operator = self.base_operator - diags(consumption, 0, shape=(n, n), format="csr")

    def _consumption_rate(self) -> np.ndarray:
        """Calculate spatially-varying consumption rate"""
        base_consumption = 0.1  # s⁻¹
        
        # Michaelis-Menten kinetics for enzymes
        def mm_rate(C, vmax, km):
            return vmax * C / (km + C)
        
        # Flatten for calculation
        C_flat = self.C.flatten()
        
        # Catalase consumption
        cat_rate = mm_rate(C_flat, self.genotype_params.catalase_vmax, 
                          self.genotype_params.catalase_km)
        
        # GPx consumption
        gpx_rate = mm_rate(C_flat, self.genotype_params.gpx_vmax, 
                          self.genotype_params.gpx_km)
        
        # Peroxiredoxin consumption
        prx_rate = self.genotype_params.prx_activity * 0.05 * C_flat
        
        total_consumption = base_consumption + cat_rate + gpx_rate + prx_rate
        return total_consumption.reshape(self.mesh_shape).flatten()
    
    def step_diffusion(self, dt: float, source_terms: Optional[np.ndarray] = None):
        """Advance diffusion-reaction system"""
        # Flatten for linear algebra
        C_flat = self.C.flatten()
        
        # Update consumption rates
        self._update_consumption_operator()
        
        # Build system matrix
        n = len(C_flat)
        system_matrix = (diags([1] * n, 0, format='csr') - 
                        dt * self.diff_operator)
        
        # Right-hand side
        rhs = C_flat.copy()
        if source_terms is not None:
            rhs += dt * source_terms.flatten()
        
        # Apply boundary conditions
        self._apply_boundary_conditions(system_matrix, rhs)
        
        # Solve
        C_new = spsolve(system_matrix, rhs)
        self.C = np.maximum(C_new.reshape(self.mesh_shape), 0)
        
        # Update GSH/GSSG
        self._update_gsh_gssg(dt)


    def _apply_boundary_conditions(self, system_matrix, rhs):
        """Apply boundary conditions"""
        if self.geometry == "spherical":
            # Surface boundary (last point)
            n = len(rhs)
            system_matrix[n-1, :] = 0
            system_matrix[n-1, n-1] = 1
            rhs[n-1] = 0  # Zero surface concentration
            
        elif self.geometry == "rectangular":
            # Inlet/outlet boundaries for microfluidic flow
            nx, ny = self.n_points
            
            # Inlet (x=0)
            for j in range(ny):
                idx = j * nx
                system_matrix[idx, :] = 0
                system_matrix[idx, idx] = 1
                rhs[idx] = 0  # Zero inlet concentration
    
    def _update_gsh_gssg(self, dt: float):
        """Update GSH/GSSG concentrations"""
        # Simplified GSH oxidation kinetics
        k_gsh_ox   = 0.01
        k_gssg_red = 0.001
        Km_red     = 5.0          # NEW saturation constant (µM)

        dGSH_dt  = -k_gsh_ox * self.GSH * self.C + (k_gssg_red * self.GSSG) / (Km_red + self.GSSG)
        dGSSG_dt =  0.5 * k_gsh_ox * self.GSH * self.C - (k_gssg_red * self.GSSG) / (Km_red + self.GSSG)
        
        self.GSH += dt * dGSH_dt
        self.GSSG += dt * dGSSG_dt
        
        # Ensure non-negative
        self.GSH = np.maximum(self.GSH, 0)
        self.GSSG = np.maximum(self.GSSG, 0)

class NOXSourceModel:
    """Model NOX-mediated ROS production with genetic modulation"""
    
    def __init__(self, genotype: GenotypeVariant = GenotypeVariant.WILD_TYPE,
                 baseline_activity: float = 1.0):
        self.genotype_params = GENOTYPE_PARAMS[genotype]
        self.baseline_activity = baseline_activity
        self.crispr_efficiency = 1.0  # CRISPR knockdown efficiency
        
    def production_rate(self, stimulus: float = 0.0, 
                       agonist_conc: float = 0.0) -> float:
        """
        Calculate ROS production rate
        
        Args:
            stimulus: External stimulus (0-1)
            agonist_conc: Agonist concentration (µM)
        """
        # Basal production
        basal = self.baseline_activity * self.genotype_params.nox_activity
        
        # Stimulus-induced production
        stimulus_response = stimulus * 10.0  # Max 10x increase
        
        # Agonist-induced production (Hill equation)
        agonist_response = (agonist_conc**2) / (50**2 + agonist_conc**2) * 5.0
        
        # CRISPR modification
        total_production = (basal + stimulus_response + agonist_response) * self.crispr_efficiency
        
        return total_production
    
    def apply_crispr_knockdown(self, efficiency: float):
        """Apply CRISPR-mediated knockdown"""
        self.crispr_efficiency = 1.0 - efficiency

class CoupledIROCXSimulation:
    """
    Advanced coupled simulation for IROC-X PRIME with AI data generation
    """
    
    def __init__(self, 
                 biosensor_type: BiosensorType = BiosensorType.HYPER7,
                 geometry: str = "rectangular",
                 organoid_size: float = 250e-4,
                 n_spatial_points: Tuple[int, ...] = (50, 30),
                 flow_rate: float = 0.1,  # µL/min
                 genotype: GenotypeVariant = GenotypeVariant.WILD_TYPE,
                 initial_conditions: Optional[Dict] = None):
        
        self.biosensor_type = biosensor_type
        self.geometry = geometry
        self.genotype = genotype
        self.read_noise_sd = 0.02          # instrument read noise (ratio units)
        self.shot_noise_scale = 0.01       # variance ∝ signal * scale

        # Initialize components
        self.sensor = AdvancedSensorKinetics(biosensor_type)
        
        # Setup flow field
        self.flow_field = MicrofluidicFlow(flow_rate=flow_rate) if geometry == "rectangular" else None
        
        # Setup diffusion model
        self.diffusion = AdvancedOrganoidDiffusion(
            geometry=geometry,
            primary_size=organoid_size,
            secondary_size=organoid_size * 0.6 if geometry == "rectangular" else None,
            n_points=n_spatial_points,
            flow_field=self.flow_field,
            genotype=genotype
        )
        
        # Setup NOX model
        self.nox_model = NOXSourceModel(genotype=genotype)
        
        # Initialize sensor states
        if initial_conditions:
            self.sensor_oxidized = np.full(self.diffusion.mesh_shape, 
                                         initial_conditions.get('oxidized_fraction', 0.1))
            self.sensor_active = np.full(self.diffusion.mesh_shape, 
                                       initial_conditions.get('active_fraction', 1.0))
        else:
            self.sensor_oxidized = np.full(self.diffusion.mesh_shape, 0.1)
            self.sensor_active = np.full(self.diffusion.mesh_shape, 1.0)
        
        # Storage for results
        self.results = {
            'time': [],
            'h2o2_concentration': [],
            'sensor_oxidized': [],
            'sensor_active': [],
            'gsh_ratio': [],
            'flow_field': [],
            'metadata': {
                'biosensor': biosensor_type.value,
                'geometry': geometry,
                'genotype': genotype.value,
                'flow_rate': flow_rate
            }
        }
    
    def add_stimulus_protocol(self, protocol: Dict):
        """Add complex stimulus protocol"""
        self.stimulus_protocol = protocol
    

    def run_simulation(self, 
                      total_time: float = 600.0,
                      dt: float = 0.1,
                      save_interval: float = 2.0,
                      illumination_schedule: Optional[Dict] = None,
                      progress_callback: Optional[callable] = None):
        """Run enhanced simulation with progress tracking"""
        
        print(f"🚀 Starting IROC-X PRIME simulation...")
        print(f"   Biosensor: {self.biosensor_type.value}")
        print(f"   Geometry: {self.geometry}")
        print(f"   Genotype: {self.genotype.value}")
        print(f"   Duration: {total_time:.1f}s")
        
        # Time arrays
        t_points = np.arange(0, total_time + dt, dt)
        save_points = np.arange(0, total_time + save_interval, save_interval)
        
        # Initialize illumination
        if illumination_schedule is None:
            illumination_schedule = {'baseline': 0.01, 'pulses': []}
        
        start_time = time.time()
        
        for i, t in enumerate(t_points):
            # Calculate current illumination
            current_illumination = self._get_illumination(t, illumination_schedule)
            
            # Calculate NOX production
            stimulus = self._get_stimulus(t)
            nox_production = self.nox_model.production_rate(stimulus=stimulus)
            
            # Create source terms for diffusion
            source_terms = np.full(self.diffusion.mesh_shape, nox_production)
            
            # Step diffusion system
            self.diffusion.step_diffusion(dt, source_terms)
            
            # Update sensor states
            self._update_sensor_states(dt, current_illumination)
            
            # Save results at specified intervals
            if t in save_points:
                self._save_timepoint(t)
                
                if progress_callback:
                    progress = t / total_time
                    progress_callback(progress)
                
                # Progress update
                if i % max(1, len(t_points) // 20) == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed * (total_time - t) / t if t > 0 else 0
                    print(f"   Progress: {100*t/total_time:.1f}% | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Remaining: {remaining:.1f}s")
        
        print(f"✅ Simulation completed in {time.time() - start_time:.2f}s")
        return self.results
    
    def _get_illumination(self, t: float, schedule: Dict) -> float:
        """Calculate current illumination intensity"""
        baseline = schedule.get('baseline', 0.01)
        
        for pulse in schedule.get('pulses', []):
            if pulse['start'] <= t <= pulse['end']:
                return pulse['intensity']
        
        return baseline
    def _set_pk_stimulus(self, C_max=1.0, t_inf=1800, t_half=7200):
        """One-shot PK profile: linear rise during infusion then mono-exp decay."""
        self.stimulus_protocol = {
            'type': 'custom_pk',
            'c_max': C_max,
            't_inf': t_inf,
            't_half': t_half
    }
    def _get_stimulus(self, t: float) -> float:
        """Calculate stimulus intensity based on protocol"""
        if hasattr(self, 'stimulus_protocol') and \
            self.stimulus_protocol.get('type') == 'custom_pk':
                inf = self.stimulus_protocol['t_inf']
                cmax = self.stimulus_protocol['c_max']
                half = self.stimulus_protocol['t_half']
                if t <= inf:
                    return cmax * (t / inf)          # linear rise
                return cmax * np.exp(-(t - inf) / half)
        if not hasattr(self, 'stimulus_protocol'):
            return 0.0
        
        protocol = self.stimulus_protocol
        
        # Handle different stimulus types
        if protocol['type'] == 'step':
            return protocol['amplitude'] if t >= protocol['start_time'] else 0.0
        
        elif protocol['type'] == 'pulse':
            pulses = protocol['pulses']
            for pulse in pulses:
                if pulse['start'] <= t <= pulse['end']:
                    return pulse['amplitude']
            return 0.0
        
        elif protocol['type'] == 'sine':
            amplitude = protocol['amplitude']
            frequency = protocol['frequency']
            phase = protocol.get('phase', 0)
            return amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        elif protocol['type'] == 'ramp':
            start_time = protocol['start_time']
            end_time = protocol['end_time']
            if t < start_time:
                return 0.0
            elif t > end_time:
                return protocol['final_amplitude']
            else:
                progress = (t - start_time) / (end_time - start_time)
                return protocol['final_amplitude'] * progress
        
        return 0.0
    
    def _update_sensor_states(self, dt: float, illumination: float):
        """Vectorised sensor update across entire grid."""
        h2o2 = self.diffusion.C.ravel()
        gsh_ratio = (self.diffusion.GSH / (self.diffusion.GSSG + 1e-9)).ravel()

        R = self.sensor_oxidized.ravel()
        A = self.sensor_active.ravel()

        # Cache constants
        k_ox = self.sensor.k_ox_eff
        k_red = self.sensor.k_red_eff
        kd = self.sensor.params.kd_h2o2
        bleach = self.sensor.params.photobleaching_rate

        # Vectorised derivative
        eff_h2o2 = h2o2 / (1 + h2o2 / kd)
        dR = (k_ox * eff_h2o2 * (1 - R) * A -
              k_red * R * A) * dt
        dA = (-bleach * illumination * A) * dt

        # Euler step and clamp
        R += dR
        A += dA
        np.clip(R, 0, 1, out=R)
        np.clip(A, 0, 1, out=A)

        # Reshape back
        self.sensor_oxidized[:] = R.reshape(self.diffusion.mesh_shape)
        self.sensor_active[:] = A.reshape(self.diffusion.mesh_shape)
    def _add_sensor_noise(self, ratio_frame: np.ndarray) -> np.ndarray:
        """Return a noisy copy of a ratio image (shot + read noise)."""
        shot_sd  = np.sqrt(ratio_frame * self.shot_noise_scale)
        noise    = np.random.normal(0.0, shot_sd) + \
                np.random.normal(0.0, self.read_noise_sd, ratio_frame.shape)
        return np.clip(ratio_frame + noise, 0, None)


    def _save_timepoint(self, t: float):
        """Save current simulation state"""
        self.results['time'].append(t)
        self.results['h2o2_concentration'].append(self.diffusion.C.copy())
        self.results['sensor_oxidized'].append(self.sensor_oxidized.copy())
        # Store the *noisy* ratio signal for ML / plotting
        params = self.sensor.params
        ratio_clean = 1.0 + (params.dynamic_range - 1.0) * self.sensor_oxidized
        ratio_noisy = self._add_sensor_noise(ratio_clean)
        self.results.setdefault('sensor_ratio_noisy', []).append(ratio_noisy)
        self.results['sensor_active'].append(self.sensor_active.copy())
        self.results['gsh_ratio'].append(
            self.diffusion.GSH / (self.diffusion.GSSG + 1e-9)
        )
        # Add TEER drop surrogate: inverse of mean ROS
        teer = 1 / (np.mean(self.diffusion.C) + 1e-6)
        self.results.setdefault('teer', []).append(teer)

        # Add LDH release surrogate: cumulative damaged fraction
        damage_rate = np.mean(self.diffusion.C > 0.3) * 0.002  # 0.2 % per 0.5 s if high ROS
        cumulative = self.results.get('ldh', [0])[-1] + damage_rate
        self.results.setdefault('ldh', []).append(cumulative)
        # Save flow field if applicable
        if self.flow_field:
            self.results['flow_field'].append(self._compute_flow_field())
    
    def _compute_flow_field(self) -> np.ndarray:
        """Compute flow field for visualization"""
        if self.geometry != "rectangular":
            return np.array([])
        
        nx, ny = self.diffusion.n_points
        Y, X = np.meshgrid(np.linspace(-0.5, 0.5, ny), np.linspace(-0.5, 0.5, nx))
        
        velocities = self.flow_field.velocity_profile(Y.flatten(), X.flatten())
        return velocities.reshape(nx, ny)
    
    def generate_ai_training_data(self, 
                                 n_samples: int = 1000,
                                 parameter_ranges: Optional[Dict] = None,
                                 noise_level: float = 0.02) -> pd.DataFrame:
        """Generate AI training dataset with ground truth labels"""
        
        print(f"🧠 Generating AI training data ({n_samples} samples)...")
        
        if parameter_ranges is None:
            parameter_ranges = {
                'h2o2_conc': (0.1, 100.0),  # µM
                'gsh_ratio': (1.0, 50.0),
                'ph': (6.8, 7.6),
                'temperature': (35.0, 39.0),
                'illumination': (0.001, 0.1)
            }
        
        # Generate parameter combinations
        data = []
        
        for i in range(n_samples):
            # Random parameter sampling
            h2o2 = np.random.uniform(*parameter_ranges['h2o2_conc'])
            gsh_ratio = np.random.uniform(*parameter_ranges['gsh_ratio'])
            ph = np.random.uniform(*parameter_ranges['ph'])
            temp = np.random.uniform(*parameter_ranges['temperature'])
            illumination = np.random.uniform(*parameter_ranges['illumination'])
            
            # Create temporary sensor with these conditions
            temp_sensor = AdvancedSensorKinetics(
                self.biosensor_type, ph=ph, temperature=temp
            )
            
            # Calculate ground truth response
            R_ox_true, ratio_true = temp_sensor.steady_state_response(h2o2, gsh_ratio)
            
            # Add realistic noise
            ratio_measured = ratio_true * (1 + np.random.normal(0, noise_level))
            R_ox_measured = R_ox_true * (1 + np.random.normal(0, noise_level))
            
            # Add photobleaching effects
            bleach_factor = np.exp(-temp_sensor.params.photobleaching_rate * illumination * 300)
            ratio_measured *= bleach_factor
            
            sample = {
                'h2o2_concentration': h2o2,
                'gsh_ratio': gsh_ratio,
                'ph': ph,
                'temperature': temp,
                'illumination': illumination,
                'ratio_signal_measured': ratio_measured,
                'oxidized_fraction_measured': R_ox_measured,
                'ratio_signal_true': ratio_true,
                'oxidized_fraction_true': R_ox_true,
                'biosensor_type': self.biosensor_type.value,
                'genotype': self.genotype.value,
                'photobleach_factor': bleach_factor,
                'noise_level': noise_level
            }
            
            data.append(sample)
            
            if (i + 1) % (n_samples // 10) == 0:
                print(f"   Generated {i+1}/{n_samples} samples")
        
        df = pd.DataFrame(data)
        print(f"✅ AI training data generated: {len(df)} samples")
        return df
    
    def export_results(self, filename: str, format: str = 'hdf5'):
        """Export simulation results in multiple formats"""
        
        if format == 'hdf5':
            # Export to HDF5 for efficient storage
            import h5py
            
            with h5py.File(filename, 'w') as f:
                # Metadata
                meta_grp = f.create_group('metadata')
                for key, value in self.results['metadata'].items():
                    meta_grp.attrs[key] = value
                
                # Time series data
                f.create_dataset('time', data=np.array(self.results['time']))
                
                # Spatial data (4D: time, spatial dimensions)
                h2o2_array = np.array(self.results['h2o2_concentration'])
                f.create_dataset('h2o2_concentration', data=h2o2_array)
                
                sensor_ox_array = np.array(self.results['sensor_oxidized'])
                f.create_dataset('sensor_oxidized', data=sensor_ox_array)
                
                sensor_act_array = np.array(self.results['sensor_active'])
                f.create_dataset('sensor_active', data=sensor_act_array)
                
                gsh_array = np.array(self.results['gsh_ratio'])
                f.create_dataset('gsh_ratio', data=gsh_array)
                
                # Flow field if present
                if self.results['flow_field']:
                    flow_array = np.array(self.results['flow_field'])
                    f.create_dataset('flow_field', data=flow_array)
        
        elif format == 'json':
            # Export to JSON (smaller datasets)
            export_data = {
                'metadata': self.results['metadata'],
                'time': self.results['time'],
                'spatial_averages': {
                    'h2o2': [np.mean(frame) for frame in self.results['h2o2_concentration']],
                    'sensor_oxidized': [np.mean(frame) for frame in self.results['sensor_oxidized']],
                    'sensor_active': [np.mean(frame) for frame in self.results['sensor_active']],
                    'gsh_ratio': [np.mean(frame) for frame in self.results['gsh_ratio']]
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to {filename}")

class AdvancedVisualization:
    """Enhanced visualization suite for IROC-X PRIME data"""
    
    def __init__(self, simulation: CoupledIROCXSimulation):
        self.sim = simulation
        self.results = simulation.results
        
        # Set up modern plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_temporal_dynamics(self, 
                              variables: List[str] = ['h2o2', 'sensor_ratio', 'gsh_ratio'],
                              spatial_reduction: str = 'mean',
                              save_path: Optional[str] = None):
        """Plot temporal dynamics with multiple variables"""
        
        fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4*len(variables)))
        if len(variables) == 1:
            axes = [axes]
        
        time_points = np.array(self.results['time'])
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            if var == 'h2o2':
                data = self.results['h2o2_concentration']
                ylabel = 'H₂O₂ Concentration (µM)'
                color = 'red'
            elif var == 'sensor_ratio':
                # Calculate ratio signal
                ox_data = self.results['sensor_oxidized']
                params = self.sim.sensor.params
                ratio_data = [1.0 + (params.dynamic_range - 1.0) * frame 
                             for frame in ox_data]
                data = self.results['sensor_ratio_noisy']
                ylabel = f'{params.name} Ratio Signal'
                color = 'green'
            elif var == 'gsh_ratio':
                data = self.results['gsh_ratio']
                ax.set_yscale('log')
                ylabel = 'GSH/GSSG (log scale)'
                color = 'blue'
            elif var == 'teer':
                data   = self.results['teer']
                ylabel = 'Relative TEER (Ω·cm²)'
                color  = 'purple'
            elif var == 'ldh':
                data   = self.results['ldh']
                ylabel = 'Cumulative LDH (a.u.)'
                color  = 'orange'
            else:
                continue
            
            # Apply spatial reduction
            if spatial_reduction == 'mean':
                y_values = [np.mean(frame) for frame in data]
                y_std = [np.std(frame) for frame in data]
            elif spatial_reduction == 'max':
                y_values = [np.max(frame) for frame in data]
                y_std = [0] * len(data)
            elif spatial_reduction == 'center':
                # Center point value
                center_idx = tuple(s//2 for s in data[0].shape)
                y_values = [frame[center_idx] for frame in data]
                y_std = [0] * len(data)
            
            # Plot with error bars
            ax.plot(time_points, y_values, color=color, linewidth=2, label=var)
            ax.fill_between(time_points, 
                           np.array(y_values) - np.array(y_std),
                           np.array(y_values) + np.array(y_std),
                           alpha=0.3, color=color)
            
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add stimulus overlay if present
            if hasattr(self.sim, 'stimulus_protocol'):
                self._add_stimulus_overlay(ax, time_points[-1])
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spatial_distribution(self, 
                                 timepoint_idx: int = -1,
                                 variables: List[str] = ['h2o2', 'sensor_ratio'],
                                 save_path: Optional[str] = None):
        """Plot spatial distribution at specific timepoint"""
        
        if self.sim.geometry == "rectangular":
            self._plot_2d_spatial(timepoint_idx, variables, save_path)
        elif self.sim.geometry == "spherical":
            self._plot_1d_radial(timepoint_idx, variables, save_path)
    
    def _plot_2d_spatial(self, timepoint_idx: int, variables: List[str], save_path: Optional[str]):
        """Plot 2D spatial distribution"""
        
        n_vars = len(variables)
        fig, axes = plt.subplots(1, n_vars, figsize=(6*n_vars, 5))
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            if var == 'h2o2':
                data = self.results['h2o2_concentration'][timepoint_idx]
                title = 'H₂O₂ Concentration (µM)'
                cmap = 'Reds'
            elif var == 'sensor_ratio':
                ox_data = self.results['sensor_oxidized'][timepoint_idx]
                params = self.sim.sensor.params
                data = 1.0 + (params.dynamic_range - 1.0) * ox_data
                title = f'{params.name} Ratio Signal'
                cmap = 'Greens'
            elif var == 'gsh_ratio':
                data = self.results['gsh_ratio'][timepoint_idx]
                title = 'GSH/GSSG Ratio'
                cmap = 'Blues'
            else:
                continue
            
            # Create heatmap
            im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(title)
            
            # Add flow field overlay if available
            if var == 'h2o2' and self.results['flow_field']:
                self._add_flow_overlay(ax, timepoint_idx)
        
        time_val = self.results['time'][timepoint_idx]
        fig.suptitle(f'Spatial Distribution at t = {time_val:.1f}s', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_1d_radial(self, timepoint_idx: int, variables: List[str], save_path: Optional[str]):
        """Plot 1D radial distribution"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r_values = self.sim.diffusion.r * 1e4  # Convert to µm
        
        for var in variables:
            if var == 'h2o2':
                data = self.results['h2o2_concentration'][timepoint_idx]
                label = 'H₂O₂ Concentration (µM)'
                color = 'red'
            elif var == 'sensor_ratio':
                ox_data = self.results['sensor_oxidized'][timepoint_idx]
                params = self.sim.sensor.params
                data = 1.0 + (params.dynamic_range - 1.0) * ox_data
                label = f'{params.name} Ratio Signal'
                color = 'green'
            elif var == 'gsh_ratio':
                data = self.results['gsh_ratio'][timepoint_idx]
                label = 'GSH/GSSG Ratio'
                color = 'blue'
            else:
                continue
            
            ax.plot(r_values, data, color=color, linewidth=2, label=label)
        
        ax.set_xlabel('Radial Distance (µm)')
        ax.set_ylabel('Concentration / Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        time_val = self.results['time'][timepoint_idx]
        ax.set_title(f'Radial Distribution at t = {time_val:.1f}s')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _add_flow_overlay(self, ax, timepoint_idx: int):
        """Add flow field overlay to 2D plot"""
        if not self.results['flow_field']:
            return
            
        flow_data = self.results['flow_field'][timepoint_idx]
        nx, ny = flow_data.shape
        
        # Create arrow grid (subsample for clarity)
        skip = max(1, min(nx, ny) // 10)
        X, Y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
        U = flow_data[::skip, ::skip]
        V = np.zeros_like(U)  # No y-component for channel flow
        
        ax.quiver(X, Y, U, V, color='black', alpha=0.5, scale=None)
    
    def _add_stimulus_overlay(self, ax, max_time: float):
        """Add stimulus protocol overlay"""
        if not hasattr(self.sim, 'stimulus_protocol'):
            return
            
        protocol = self.sim.stimulus_protocol
        
        # Create secondary y-axis for stimulus
        ax2 = ax.twinx()
        
        # Generate stimulus time series
        t_stim = np.linspace(0, max_time, 1000)
        stimulus_values = [self.sim._get_stimulus(t) for t in t_stim]
        
        ax2.plot(t_stim, stimulus_values, 'k--', alpha=0.7, linewidth=1)
        ax2.set_ylabel('Stimulus', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, max(stimulus_values) * 1.1 if max(stimulus_values) > 0 else 1)
    
    def create_animation(self, 
                        variable: str = 'h2o2',
                        save_path: Optional[str] = None,
                        fps: int = 10):
        """Create animation of spatial dynamics"""
        
        if self.sim.geometry != "rectangular":
            print("Animation only supported for rectangular geometry")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Setup data
        if variable == 'h2o2':
            data_frames = self.results['h2o2_concentration']
            title_base = 'H₂O₂ Concentration'
            cmap = 'Reds'
        elif variable == 'sensor_ratio':
            ox_frames = self.results['sensor_oxidized']
            params = self.sim.sensor.params
            data_frames = [1.0 + (params.dynamic_range - 1.0) * frame 
                          for frame in ox_frames]
            title_base = f'{params.name} Ratio Signal'
            cmap = 'Greens'
        else:
            print(f"Variable {variable} not supported for animation")
            return
        
        # Find global min/max for consistent colorbar
        vmin = min(np.min(frame) for frame in data_frames)
        vmax = max(np.max(frame) for frame in data_frames)
        
        # Initialize plot
        im = ax.imshow(data_frames[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', origin='lower')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title_base)
        
        title = ax.set_title(f'{title_base} at t = 0.0s')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        def animate(frame_idx):
            im.set_array(data_frames[frame_idx])
            time_val = self.results['time'][frame_idx]
            title.set_text(f'{title_base} at t = {time_val:.1f}s')
            return [im, title]
        
        anim = FuncAnimation(fig, animate, frames=len(data_frames), 
                           interval=1000//fps, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps, dpi=150)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim

# Example usage and demonstration
def run_iroc_x_demo():
    """Demonstrate IROC-X PRIME capabilities"""
    
    print("=" * 60)
    print("IROC-X PRIME Demonstration")
    print("=" * 60)
    
    # Create simulation with HyPer7 biosensor
    sim = CoupledIROCXSimulation(
        biosensor_type=BiosensorType.OXIDIAEGREEN,
        geometry="rectangular",
        organoid_size=300e-4,  # 300 µm
        n_spatial_points=(100, 60),
        flow_rate=0.2,  # µL/min
        genotype=GenotypeVariant.WILD_TYPE
    )
    sim._set_pk_stimulus(C_max=0.6, t_inf=1800, t_half=7200)
    

    stimulus_protocol = {
        'type': 'pulse',
        'pulses': [
            {'start': 60, 'end': 120, 'amplitude': 0.5},
            {'start': 300, 'end': 360, 'amplitude': 0.8},
            {'start': 480, 'end': 540, 'amplitude': 0.3},
            {'start': 700, 'end': 960, 'amplitude': -0.6}
        ]
    }
    sim.add_stimulus_protocol(stimulus_protocol)
    
    # Run simulation
    results = sim.run_simulation(
        total_time=1200.0,
        dt=0.2,
        save_interval=5.0,
        illumination_schedule={
            'baseline': 0.01,
            'pulses': [
                {'start': 200, 'end': 250, 'intensity': 0.05}
            ]
        }
    )
    
    # Generate AI training data
    ai_data = sim.generate_ai_training_data(n_samples=500)
    
    # Create visualizations
    viz = AdvancedVisualization(sim)
    plot_path = stamp("irocx", "png")                      # e.g. /results/irocx 07.12 - 1130.png
    viz.plot_temporal_dynamics(
        variables=['h2o2', 'sensor_ratio', 'gsh_ratio', 'teer', 'ldh'],
        spatial_reduction='mean',
        save_path=plot_path
    )

    # 2. spatial plot at final time point
    sp_path = stamp("irocx spatial", "png")
    viz.plot_spatial_distribution(timepoint_idx=-1, save_path=sp_path)

    # 3. export hdf5 + csv
    data_h5 = stamp("results", "h5", sub="results")
    sim.export_results(str(data_h5), format="hdf5")

    csv_path = stamp("data", "csv", sub="results")
    pd.DataFrame({
        'time': sim.results['time'],
        'mean_h2o2': [np.mean(f) for f in sim.results['h2o2_concentration']],
        'mean_ratio': [np.mean(f) for f in sim.results['sensor_ratio_noisy']],
        'teer': sim.results['teer'],
        'ldh': sim.results['ldh']
    }).to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Files generated:")
    print("- iroc_x_demo_results.h5 (simulation data)")
    print("- iroc_x_ai_training_data.csv (AI training data)")
    print("=" * 60)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def _single_run(args):
    sensor_type, genotype = args
    sim = CoupledIROCXSimulation(
        biosensor_type=sensor_type,
        geometry="rectangular",
        organoid_size=300e-4,
        n_spatial_points=(100, 60),
        flow_rate=0.2,
        genotype=genotype
    )
    sim._set_pk_stimulus(C_max=0.8, t_inf=1800, t_half=7200)
    sim.run_simulation(total_time=3600, dt=1.0, save_interval=60)
    return {
        'sensor': sensor_type.value,
        'genotype': genotype.value,
        'peak_h2o2': np.max([np.mean(f) for f in sim.results['h2o2_concentration']]),
        'final_ldh': sim.results['ldh'][-1]
    }

def monte_carlo_runs(n=20):
    sensors = [BiosensorType.HYPER7, BiosensorType.OXIDIAEGREEN]
    genotypes = np.random.choice(list(GenotypeVariant), n)

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as ex:
        out = list(ex.map(_single_run,
                          [(s, g) for g in genotypes for s in sensors]))
    return pd.DataFrame(out)

from scipy.stats import kruskal, mannwhitneyu

def stats_report(df: pd.DataFrame):
    out = []
    # overall Kruskal-Wallis
    kw_p = kruskal(*[df[df.genotype==g]['peak_h2o2'] for g in df.genotype.unique()]).pvalue
    out.append({'test':'Kruskal', 'p':kw_p})

    # pairwise U-tests
    genos = df.genotype.unique()
    for i in range(len(genos)):
        for j in range(i+1, len(genos)):
            g1, g2 = genos[i], genos[j]
            p = mannwhitneyu(df[df.genotype==g1]['peak_h2o2'],
                             df[df.genotype==g2]['peak_h2o2']).pvalue
            out.append({'test':f'{g1} vs {g2}', 'p':p})
    stats_df = pd.DataFrame(out)
    stats_df.to_csv(stamp("stats", "csv", sub="results"), index=False)
    return stats_df

def plot_equity_violins(df):
    import seaborn as sns, matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    sns.violinplot(
        data=df, x='genotype', y='peak_h2o2', inner='quartile', palette='husl'
    )
    plt.ylabel('Peak H₂O₂ (µM)')
    plt.title('Genotype-stratified ROS burden (n=50 runs)')
    plt.tight_layout()
    plt.savefig('irocx_equity_violin.png', dpi=300)
    plt.show()

def live_view(sim, variable='h2o2', total_time=1800, dt=1.0):
    """Inline real-time visualiser (matplotlib)"""
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6,4))
    data = sim.diffusion.C
    im = ax.imshow(data, cmap='plasma', origin='lower', vmin=0, vmax=0.6)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('H₂O₂ (µM)')
    title = ax.set_title('t = 0 s')

    def step(frame):
        t = frame * dt
        sim.run_simulation(total_time=dt, dt=dt, save_interval=dt)  # one slice
        im.set_array(sim.diffusion.C)
        title.set_text(f't = {t:.0f} s')
        return [im, title]

    frames = int(total_time/dt)
    anim = FuncAnimation(fig, step, frames=frames, blit=True, interval=50)
    plt.show()


if __name__ == "__main__":
    run_iroc_x_demo()
    mc_df = monte_carlo_runs()
    mc_df.to_csv("irocx_montecarlo_10.csv", index=False)
    plot_equity_violins(mc_df)
