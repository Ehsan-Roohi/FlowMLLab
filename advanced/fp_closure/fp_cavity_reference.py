#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instructor reference implementation for the guided Fokker-Planck cavity project.

Students should normally run generate_fp_cavity_dataset.py and
run_fp_cavity_test.py rather than executing this file directly.  The core
particle-FP equations, exact 9x9 closure, GPU-native neural forward pass,
high-order diagnostics, and entropy proxy are retained from the instructor
research code.
"""

FP_REFERENCE_VERSION = "1.0-course"

import numpy as np
import cupy as cp
import time
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ============================================================================
# 1. Constants and Parameters (2D Cavity)
# ============================================================================

# Physical Constants
PI = 2.0 * np.arcsin(1.0)
K_B = 1.380e-23  # Boltzmann constant
MASS_AR = 66.3e-27  # Argon mass
VIS0 = 2.117e-5  # Reference viscosity
VISP = 1.0  # Viscosity power (Maxwell molecules)
RATSH = 5.0 / 3.0  # Specific heat ratio

RHO_IN_BASE = (266.644 / 10.0) * MASS_AR / K_B / 273.15  # (Kn ~ 0.15)
RHO_IN = RHO_IN_BASE 

# --- 2D Boundary Conditions (HIGH SPEED) ---
UW_LID = 800.0   # High speed test (extrapolation)
TW_ALL = 273.15  # Wall temperature
T_IN_BASE = 273.15  # Initial gas temperature

THETA_IN = np.sqrt(K_B * T_IN_BASE / MASS_AR)
THETA_W = np.sqrt(K_B * TW_ALL / MASS_AR) 

# --- 2D Geometry ---
LX = 0.001  # X-length
LY = 0.001  # Y-length

# --- 2D Discretization ---
NX = 50   # Cells in X
NY = 50   # Cells in Y
NC = NX * NY  # Total cells
PARTICLES_PER_CELL_TARGET = 1000 
NP = PARTICLES_PER_CELL_TARGET * NC  

# --- 2D Time Step ---
DX = LX / float(NX)
DY = LY / float(NY)
MIN_DIM = min(DX, DY)
# --- DT calculated based on 800m/s velocity ---
DT = 0.2 * MIN_DIM / max(UW_LID, THETA_IN) 
print(f"INFO: Running comparison at UW_LID = {UW_LID} m/s")
print(f"INFO: Calculated DT = {DT:.2e} s")

NTSS = 2000  # Steps to reach steady state
N_STEPS_PER_RUN = 20000 # Total steps

W_PARTICLE = (LX * LY * RHO_IN_BASE) / float(NP)  # 2D Weight
if W_PARTICLE <= 0:
    raise ValueError("Calculated W_PARTICLE is zero or negative.")

EPSILON = 1e-30 

# Discrete entropy proxy settings. ENTROPY_EVERY=0 disables entropy monitoring.
ENTROPY_EVERY = int(os.environ.get('ENTROPY_EVERY', '200'))
ENTROPY_BINS_PER_DIM = int(os.environ.get('ENTROPY_BINS_PER_DIM', '9'))
ENTROPY_CMAX = float(os.environ.get('ENTROPY_CMAX', '4.0'))
ENTROPY_OUTDIR = os.environ.get('ENTROPY_OUTDIR', 'cavity77_entropy')

# High-moment diagnostics for R13/R26-style nonequilibrium analysis.
# These are averaged only every HIGH_MOMENTS_EVERY post-transient steps to keep the cost controlled.
HIGH_MOMENTS_EVERY = int(os.environ.get("HIGH_MOMENTS_EVERY", "20"))
HIGH_MOMENT_KEYS = [
    'M3sym', 'm3_stf',
    'Rij_raw', 'Rij_dev',
    'Delta4', 'Delta4_norm',
    'DM6', 'DM6_norm',
    'sigma_norm', 'q_norm', 'm3_norm', 'Rij_norm'
]

# ============================================================================
# 2. Utility and Initialization Functions
# ============================================================================

def rfn_gpu(n_samples):
    return cp.random.normal(0.0, 1.0, n_samples)

def initialize_particles_cupy(np_val, lx_val, ly_val, theta0_val, w_val):
    p_x = lx_val * cp.random.rand(np_val)
    p_y = ly_val * cp.random.rand(np_val) 
    p_z = cp.zeros(np_val, dtype=cp.float64)
    p_vx = theta0_val * rfn_gpu(np_val)
    p_vy = theta0_val * rfn_gpu(np_val)
    p_vz = theta0_val * rfn_gpu(np_val)
    p_vp_x = p_vx.copy()
    p_vp_y = p_vy.copy()
    p_vp_z = p_vz.copy()
    p_xl = p_x.copy() 
    p_yl = p_y.copy() 
    p_zl = cp.zeros(np_val, dtype=cp.float64)
    p_weight = cp.full(np_val, w_val, dtype=cp.float64)
    p_ind = cp.zeros(np_val, dtype=cp.int32) 
    return (
        p_x, p_y, p_z, p_vx, p_vy, p_vz, p_vp_x, p_vp_y, p_vp_z,
        p_xl, p_yl, p_zl, p_weight, p_ind
    )

def initialize_grid_cupy(nx_val, ny_val, lx_val, ly_val):
    nc_val = nx_val * ny_val
    dx = lx_val / float(nx_val)
    dy = ly_val / float(ny_val)
    cell_vol = dx * dy * 1.0 
    
    x_coords = cp.linspace(dx/2.0, lx_val - dx/2.0, nx_val)
    y_coords = cp.linspace(dy/2.0, ly_val - dy/2.0, ny_val)
    
    grid_gpu = {
        'x_coords': x_coords, 
        'y_coords': y_coords, 
        'vol': cp.full(nc_val, cell_vol, dtype=cp.float64),
        'N': cp.zeros(nc_val, dtype=cp.float64),
        'rho': cp.full(nc_val, RHO_IN, dtype=cp.float64), 
        'T': cp.full(nc_val, T_IN_BASE, dtype=cp.float64),   
        'U': cp.zeros((nc_val, 3), dtype=cp.float64),
        'PIJ': cp.zeros((nc_val, 6), dtype=cp.float64),
        'Q': cp.zeros((nc_val, 3), dtype=cp.float64),
        'M3': cp.zeros((nc_val, 10), dtype=cp.float64),
        'M4': cp.zeros((nc_val, 6), dtype=cp.float64),
        'M5': cp.zeros((nc_val, 3), dtype=cp.float64),
        'DM2': cp.zeros(nc_val, dtype=cp.float64),
        'DM4': cp.zeros(nc_val, dtype=cp.float64),
        'nu': cp.zeros(nc_val, dtype=cp.float64),
        'nubol': cp.zeros(nc_val, dtype=cp.float64),
        'lam': cp.zeros(nc_val, dtype=cp.float64),
        'Diff': cp.zeros(nc_val, dtype=cp.float64),
        # R13/R26-style high-moment diagnostics.
        # M3sym/m3_stf columns: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
        # Rij_raw/Rij_dev columns: xx, xy, xz, yy, yz, zz.
        'M3sym': cp.zeros((nc_val, 10), dtype=cp.float64),
        'm3_stf': cp.zeros((nc_val, 10), dtype=cp.float64),
        'Rij_raw': cp.zeros((nc_val, 6), dtype=cp.float64),
        'Rij_dev': cp.zeros((nc_val, 6), dtype=cp.float64),
        'Delta4': cp.zeros(nc_val, dtype=cp.float64),
        'Delta4_norm': cp.zeros(nc_val, dtype=cp.float64),
        'DM6': cp.zeros(nc_val, dtype=cp.float64),
        'DM6_norm': cp.zeros(nc_val, dtype=cp.float64),
        'sigma_norm': cp.zeros(nc_val, dtype=cp.float64),
        'q_norm': cp.zeros(nc_val, dtype=cp.float64),
        'm3_norm': cp.zeros(nc_val, dtype=cp.float64),
        'Rij_norm': cp.zeros(nc_val, dtype=cp.float64),
    }
    coeffs_gpu = {
        'A': cp.zeros((nc_val, 6), dtype=cp.float64),
        'B': cp.zeros((nc_val, 3), dtype=cp.float64),
        'C': cp.zeros(nc_val, dtype=cp.float64),
    }
    linsys_gpu = {
        'lhs': cp.zeros((nc_val, 9, 9), dtype=cp.float64),
        'rhs': cp.zeros((nc_val, 9), dtype=cp.float64),
    }
    return grid_gpu, coeffs_gpu, linsys_gpu 

# ============================================================================
# 3. Core Simulation Functions
# ============================================================================

def apply_boundary_cavity_cupy(p_data, lx_val, ly_val, dt_val):
    p_x, p_y, p_vx, p_vy, p_vz = p_data[0], p_data[1], p_data[3], p_data[4], p_data[5]
    
    # Top wall (Moving Lid, y > LY)
    idx_top = cp.where(p_y > ly_val)[0]
    n_top = len(idx_top)
    if n_top > 0:
        seed1 = cp.maximum(cp.random.rand(n_top), EPSILON)
        xi = rfn_gpu((n_top, 2))
        p_vx[idx_top] = UW_LID + THETA_W * xi[:, 0]
        p_vy[idx_top] = -THETA_W * cp.sqrt(-2.0 * cp.log(seed1)) 
        p_vz[idx_top] = THETA_W * xi[:, 1]
        p_y[idx_top] = ly_val + p_vy[idx_top] * dt_val * cp.random.rand(n_top) 
        p_x[idx_top] = cp.clip(p_x[idx_top], 0.0, lx_val) 

    # Bottom wall (Stationary, y < 0)
    idx_bot = cp.where(p_y < 0.0)[0]
    n_bot = len(idx_bot)
    if n_bot > 0:
        seed1 = cp.maximum(cp.random.rand(n_bot), EPSILON)
        xi = rfn_gpu((n_bot, 2))
        p_vx[idx_bot] = THETA_W * xi[:, 0] 
        p_vy[idx_bot] = THETA_W * cp.sqrt(-2.0 * cp.log(seed1)) 
        p_vz[idx_bot] = THETA_W * xi[:, 1]
        p_y[idx_bot] = p_vy[idx_bot] * dt_val * cp.random.rand(n_bot)
        p_x[idx_bot] = cp.clip(p_x[idx_bot], 0.0, lx_val)

    # Left wall (Stationary, x < 0)
    idx_left = cp.where(p_x < 0.0)[0]
    n_left = len(idx_left)
    if n_left > 0:
        seed1 = cp.maximum(cp.random.rand(n_left), EPSILON)
        xi = rfn_gpu((n_left, 2))
        p_vx[idx_left] = THETA_W * cp.sqrt(-2.0 * cp.log(seed1)) 
        p_vy[idx_left] = THETA_W * xi[:, 0] 
        p_vz[idx_left] = THETA_W * xi[:, 1]
        p_x[idx_left] = p_vx[idx_left] * dt_val * cp.random.rand(n_left)
        p_y[idx_left] = cp.clip(p_y[idx_left], 0.0, ly_val)

    # Right wall (Stationary, x > LX)
    idx_right = cp.where(p_x > lx_val)[0]
    n_right = len(idx_right)
    if n_right > 0:
        seed1 = cp.maximum(cp.random.rand(n_right), EPSILON)
        xi = rfn_gpu((n_right, 2))
        p_vx[idx_right] = -THETA_W * cp.sqrt(-2.0 * cp.log(seed1)) 
        p_vy[idx_right] = THETA_W * xi[:, 0] 
        p_vz[idx_right] = THETA_W * xi[:, 1]
        p_x[idx_right] = lx_val + p_vx[idx_right] * dt_val * cp.random.rand(n_right)
        p_y[idx_right] = cp.clip(p_y[idx_right], 0.0, ly_val)

def get_1d_cell_indices(p_x, p_y, nx, ny, lx, ly):
    """Helper function to convert 2D particle positions to 1D cell index."""
    dx = lx / float(nx)
    dy = ly / float(ny)
    ix = cp.clip(cp.floor(p_x / dx).astype(cp.int32), 0, nx - 1)
    iy = cp.clip(cp.floor(p_y / dy).astype(cp.int32), 0, ny - 1)
    return ix + iy * nx

def sort_and_calc_moments_cupy_LITE(p_data, grid, nc_val, nx_val, ny_val, lx_val, ly_val):
    """
    LITE 2D moment calculation (for ML solver).
    Only calculates the 16 features needed for ML input.
    """
    p_x, p_y = p_data[0], p_data[1]
    p_vx, p_vy, p_vz = p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]
    
    cell_indices = get_1d_cell_indices(p_x, p_y, nx_val, ny_val, lx_val, ly_val)
    p_ind[:] = cell_indices

    grid['N'] = cp.bincount(cell_indices, weights=p_weight, minlength=nc_val)
    N_safe = cp.maximum(grid['N'], EPSILON)
    vol_safe = cp.maximum(grid['vol'], EPSILON)
    grid['rho'] = grid['N'] / vol_safe
    grid['U'][:, 0] = cp.bincount(cell_indices, weights=p_vx * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 1] = cp.bincount(cell_indices, weights=p_vy * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 2] = cp.bincount(cell_indices, weights=p_vz * p_weight, minlength=nc_val) / N_safe
    
    U_particles = grid['U'][cell_indices]
    p_vp_x[:] = p_vx - U_particles[:, 0]
    p_vp_y[:] = p_vy - U_particles[:, 1]
    p_vp_z[:] = p_vz - U_particles[:, 2]
    
    vp_x = p_vp_x
    vp_y = p_vp_y
    vp_z = p_vp_z
    vp_sq = vp_x**2 + vp_y**2 + vp_z**2
    
    grid['DM2'] = cp.bincount(cell_indices, weights=vp_sq * p_weight, minlength=nc_val) / N_safe
    grid['T'] = MASS_AR * grid['DM2'] / (3.0 * K_B)
    grid['T'] = cp.maximum(grid['T'], 1.0)
    grid['PIJ'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 3] = cp.bincount(cell_indices, weights=vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 4] = cp.bincount(cell_indices, weights=vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 5] = cp.bincount(cell_indices, weights=vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq*p_weight, minlength=nc_val) / N_safe
    
    number_density = grid['rho'] / MASS_AR
    number_density = cp.maximum(number_density, EPSILON)
    P = number_density * K_B * grid['T']
    vis = cp.maximum(VIS0 * (grid['T'] / 273.15)**VISP, EPSILON)
    grid['nubol'] = P / vis
    grid['nu'] = grid['nubol'] * 0.5
    grid['Diff'] = 2.0 * K_B * grid['nu'] * grid['T'] / MASS_AR

def sort_and_calc_moments_cupy_FULL(p_data, grid, nc_val, nx_val, ny_val, lx_val, ly_val):
    """Full 2D moment calculation (for physics solver)."""
    p_x, p_y = p_data[0], p_data[1]
    p_vx, p_vy, p_vz = p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]

    cell_indices = get_1d_cell_indices(p_x, p_y, nx_val, ny_val, lx_val, ly_val)
    p_ind[:] = cell_indices

    grid['N'] = cp.bincount(cell_indices, weights=p_weight, minlength=nc_val)
    N_safe = cp.maximum(grid['N'], EPSILON)
    vol_safe = cp.maximum(grid['vol'], EPSILON)
    grid['rho'] = grid['N'] / vol_safe
    grid['U'][:, 0] = cp.bincount(cell_indices, weights=p_vx * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 1] = cp.bincount(cell_indices, weights=p_vy * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 2] = cp.bincount(cell_indices, weights=p_vz * p_weight, minlength=nc_val) / N_safe
    
    U_particles = grid['U'][cell_indices]
    p_vp_x[:] = p_vx - U_particles[:, 0]
    p_vp_y[:] = p_vy - U_particles[:, 1]
    p_vp_z[:] = p_vz - U_particles[:, 2]
    
    vp_x = p_vp_x
    vp_y = p_vp_y
    vp_z = p_vp_z
    vp_sq = vp_x**2 + vp_y**2 + vp_z**2
    grid['DM2'] = cp.bincount(cell_indices, weights=vp_sq * p_weight, minlength=nc_val) / N_safe
    grid['T'] = MASS_AR * grid['DM2'] / (3.0 * K_B)
    grid['T'] = cp.maximum(grid['T'], 1.0)
    grid['PIJ'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 3] = cp.bincount(cell_indices, weights=vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 4] = cp.bincount(cell_indices, weights=vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 5] = cp.bincount(cell_indices, weights=vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq*p_weight, minlength=nc_val) / N_safe
    
    grid['M3'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 3] = cp.bincount(cell_indices, weights=vp_x*vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 4] = cp.bincount(cell_indices, weights=vp_x*vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 5] = cp.bincount(cell_indices, weights=vp_x*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 6] = cp.bincount(cell_indices, weights=vp_y*vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 7] = cp.bincount(cell_indices, weights=vp_y*vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 8] = cp.bincount(cell_indices, weights=vp_y*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 9] = cp.bincount(cell_indices, weights=vp_z*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 0] = cp.bincount(cell_indices, weights=(vp_x*vp_x)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 1] = cp.bincount(cell_indices, weights=(vp_x*vp_y)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 2] = cp.bincount(cell_indices, weights=(vp_x*vp_z)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 3] = cp.bincount(cell_indices, weights=(vp_y*vp_y)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 4] = cp.bincount(cell_indices, weights=(vp_y*vp_z)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 5] = cp.bincount(cell_indices, weights=(vp_z*vp_z)*vp_sq*p_weight, minlength=nc_val) / N_safe
    vp_sq_sq = vp_sq * vp_sq
    grid['M5'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    grid['M5'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    grid['M5'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    
    grid['DM4'] = grid['M4'][:, 0] + grid['M4'][:, 3] + grid['M4'][:, 5]
    
    number_density = grid['rho'] / MASS_AR
    number_density = cp.maximum(number_density, EPSILON)
    P = number_density * K_B * grid['T']
    vis = cp.maximum(VIS0 * (grid['T'] / 273.15)**VISP, EPSILON)
    grid['nubol'] = P / vis
    grid['nu'] = grid['nubol'] * 0.5
    grid['Diff'] = 2.0 * K_B * grid['nu'] * grid['T'] / MASS_AR
    PIJdev_1 = grid['PIJ'][:, 0] - (1./3.) * grid['DM2']
    PIJdev_4 = grid['PIJ'][:, 3] - (1./3.) * grid['DM2']
    PIJdev_6 = grid['PIJ'][:, 5] - (1./3.) * grid['DM2']
    lam_sq = (PIJdev_1**2 + PIJdev_4**2 + PIJdev_6**2 + 2.0 * (grid['PIJ'][:, 1]**2 + grid['PIJ'][:, 2]**2 + grid['PIJ'][:, 4]**2))
    DM2_pow_3p_safe = cp.maximum(grid['DM2']**3.5, EPSILON)
    grid['lam'] = -lam_sq * grid['nu'] / DM2_pow_3p_safe



def calc_high_moments_R13_R26_cupy(p_data, grid, nc_val):
    """
    Compute diagnostic high moments used in R13/R26-style analysis.

    The normalization follows the existing code convention: moments are cell averages
    of peculiar-velocity powers, weighted by p_weight and divided by grid['N'].
    To obtain density-weighted dimensional moments, multiply by rho if needed.

    Stored moments:
      M3sym    : <c_i c_j c_k>, symmetric 10-component form
      m3_stf   : trace-free third-order moment, useful in R26 (7 independent components)
      Rij_raw  : <c^2 c_i c_j>, 6-component symmetric tensor
      Rij_dev  : <c^2 c_<i c_j>>, trace-free fourth-order contraction, useful in R13/R26
      Delta4   : <c^4> - 15 theta^2, scalar fourth-order deviation; zero for Maxwellian
      DM6      : <c^6>
      *_norm   : dimensionless nonequilibrium indicators for plotting/comparison
    """
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, cell_indices = p_data[12], p_data[13]
    N_safe = cp.maximum(grid['N'], EPSILON)

    cx = p_vp_x
    cy = p_vp_y
    cz = p_vp_z
    cx2 = cx * cx
    cy2 = cy * cy
    cz2 = cz * cz
    c2 = cx2 + cy2 + cz2
    c4 = c2 * c2
    c6 = c4 * c2

    def avg(w):
        return cp.bincount(cell_indices, weights=w * p_weight, minlength=nc_val) / N_safe

    # Symmetric third-order tensor: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
    M3 = grid['M3sym']
    M3[:, 0] = avg(cx2 * cx)
    M3[:, 1] = avg(cx2 * cy)
    M3[:, 2] = avg(cx2 * cz)
    M3[:, 3] = avg(cx * cy2)
    M3[:, 4] = avg(cx * cy * cz)
    M3[:, 5] = avg(cx * cz2)
    M3[:, 6] = avg(cy2 * cy)
    M3[:, 7] = avg(cy2 * cz)
    M3[:, 8] = avg(cy * cz2)
    M3[:, 9] = avg(cz2 * cz)

    # Trace-free third-order part:
    # m_ijk = <c_i c_j c_k> - (1/5)(Q_i delta_jk + Q_j delta_ik + Q_k delta_ij).
    Qx = grid['Q'][:, 0]
    Qy = grid['Q'][:, 1]
    Qz = grid['Q'][:, 2]
    m3 = grid['m3_stf']
    m3[:, 0] = M3[:, 0] - (3.0/5.0) * Qx       # xxx
    m3[:, 1] = M3[:, 1] - (1.0/5.0) * Qy       # xxy
    m3[:, 2] = M3[:, 2] - (1.0/5.0) * Qz       # xxz
    m3[:, 3] = M3[:, 3] - (1.0/5.0) * Qx       # xyy
    m3[:, 4] = M3[:, 4]                         # xyz
    m3[:, 5] = M3[:, 5] - (1.0/5.0) * Qx       # xzz
    m3[:, 6] = M3[:, 6] - (3.0/5.0) * Qy       # yyy
    m3[:, 7] = M3[:, 7] - (1.0/5.0) * Qz       # yyz
    m3[:, 8] = M3[:, 8] - (1.0/5.0) * Qy       # yzz
    m3[:, 9] = M3[:, 9] - (3.0/5.0) * Qz       # zzz

    # Fourth-order contracted tensor R_ij = <c^2 c_i c_j> and its trace-free part.
    Rij = grid['Rij_raw']
    Rij[:, 0] = avg(c2 * cx2)       # xx
    Rij[:, 1] = avg(c2 * cx * cy)   # xy
    Rij[:, 2] = avg(c2 * cx * cz)   # xz
    Rij[:, 3] = avg(c2 * cy2)       # yy
    Rij[:, 4] = avg(c2 * cy * cz)   # yz
    Rij[:, 5] = avg(c2 * cz2)       # zz

    DM4 = Rij[:, 0] + Rij[:, 3] + Rij[:, 5]
    theta = cp.maximum(grid['DM2'] / 3.0, EPSILON)

    Rdev = grid['Rij_dev']
    one_third_DM4 = DM4 / 3.0
    Rdev[:, 0] = Rij[:, 0] - one_third_DM4
    Rdev[:, 1] = Rij[:, 1]
    Rdev[:, 2] = Rij[:, 2]
    Rdev[:, 3] = Rij[:, 3] - one_third_DM4
    Rdev[:, 4] = Rij[:, 4]
    Rdev[:, 5] = Rij[:, 5] - one_third_DM4

    # Scalar fourth/sixth-order deviations. For a Maxwellian: <c^4>=15 theta^2, <c^6>=105 theta^3.
    grid['Delta4'] = DM4 - 15.0 * theta**2
    grid['Delta4_norm'] = grid['Delta4'] / cp.maximum(15.0 * theta**2, EPSILON)
    grid['DM6'] = avg(c6)
    grid['DM6_norm'] = grid['DM6'] / cp.maximum(105.0 * theta**3, EPSILON) - 1.0

    # Dimensionless nonequilibrium indicators.
    P = grid['PIJ']
    sigma_xx = P[:, 0] - theta
    sigma_xy = P[:, 1]
    sigma_xz = P[:, 2]
    sigma_yy = P[:, 3] - theta
    sigma_yz = P[:, 4]
    sigma_zz = P[:, 5] - theta
    sigma_norm_sq = sigma_xx**2 + sigma_yy**2 + sigma_zz**2 + 2.0*(sigma_xy**2 + sigma_xz**2 + sigma_yz**2)
    grid['sigma_norm'] = cp.sqrt(cp.maximum(sigma_norm_sq, 0.0)) / cp.maximum(grid['DM2'], EPSILON)

    q_norm_sq = Qx**2 + Qy**2 + Qz**2
    grid['q_norm'] = cp.sqrt(cp.maximum(q_norm_sq, 0.0)) / cp.maximum(grid['DM2']**1.5, EPSILON)

    # Symmetric third-order norm with multiplicities.
    m3_norm_sq = (m3[:, 0]**2 + 3*m3[:, 1]**2 + 3*m3[:, 2]**2 + 3*m3[:, 3]**2 +
                  6*m3[:, 4]**2 + 3*m3[:, 5]**2 + m3[:, 6]**2 + 3*m3[:, 7]**2 +
                  3*m3[:, 8]**2 + m3[:, 9]**2)
    grid['m3_norm'] = cp.sqrt(cp.maximum(m3_norm_sq, 0.0)) / cp.maximum(grid['DM2']**1.5, EPSILON)

    Rnorm_sq = Rdev[:, 0]**2 + Rdev[:, 3]**2 + Rdev[:, 5]**2 + 2.0*(Rdev[:, 1]**2 + Rdev[:, 2]**2 + Rdev[:, 4]**2)
    grid['Rij_norm'] = cp.sqrt(cp.maximum(Rnorm_sq, 0.0)) / cp.maximum(grid['DM2']**2, EPSILON)


def average_high_moments_cupy(avg_grid, grid_gpu, sample_count):
    """Separate averaging for high moments sampled every HIGH_MOMENTS_EVERY steps."""
    nave = 1.0 / float(sample_count)
    for key in HIGH_MOMENT_KEYS:
        if key in avg_grid and key in grid_gpu:
            avg_grid[key] = nave * grid_gpu[key] + (1.0 - nave) * avg_grid[key]

def evolve_velocities_cupy(p_data, grid, coeffs, dt_val, nc_val):
    p_vx, p_vy, p_vz = p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]
    n_particles = len(p_vx)
    cell_indices = p_ind
    p_U = grid['U'][cell_indices]
    p_DM2 = grid['DM2'][cell_indices]
    p_Q = grid['Q'][cell_indices]
    p_nu = grid['nu'][cell_indices]
    p_Diff = grid['Diff'][cell_indices]
    p_A = coeffs['A'][cell_indices]
    p_B = coeffs['B'][cell_indices]
    p_C = coeffs['C'][cell_indices] 
    
    gforce = cp.zeros((n_particles, 3))
    DM2_safe = p_DM2 + EPSILON
    Q_safe = p_Q + EPSILON
    gforce[:, 0] = p_B[:, 0] * (-DM2_safe) + p_C * (-Q_safe[:, 0])
    gforce[:, 1] = p_B[:, 1] * (-DM2_safe) + p_C * (-Q_safe[:, 1])
    gforce[:, 2] = p_B[:, 2] * (-DM2_safe) + p_C * (-Q_safe[:, 2])
    
    nu_safe = cp.maximum(p_nu, EPSILON)
    s = nu_safe * dt_val
    A1 = cp.exp(-s)
    taus = 1.0 / nu_safe
    A2 = taus * (1.0 - A1)
    diff_term = cp.abs(taus * p_Diff * (1.0 - cp.exp(-2.*s))/2.0)
    A3 = cp.sqrt(diff_term)
    
    vp_x = p_vp_x
    vp_y = p_vp_y
    vp_z = p_vp_z
    vp_sq = vp_x**2 + vp_y**2 + vp_z**2
    vp_sq_safe = vp_sq + EPSILON
    
    nvA = cp.zeros((n_particles, 3))
    nvA[:, 0] = p_A[:, 0]*vp_x + p_A[:, 1]*vp_y + p_A[:, 2]*vp_z
    nvA[:, 1] = p_A[:, 1]*vp_x + p_A[:, 3]*vp_y + p_A[:, 4]*vp_z
    nvA[:, 2] = p_A[:, 2]*vp_x + p_A[:, 4]*vp_y + p_A[:, 5]*vp_z
    
    nq = cp.zeros((n_particles, 3))
    nq[:, 0] = p_B[:, 0]*vp_sq_safe + p_C*vp_x*vp_sq_safe
    nq[:, 1] = p_B[:, 1]*vp_sq_safe + p_C*vp_y*vp_sq_safe
    nq[:, 2] = p_B[:, 2]*vp_sq_safe + p_C*vp_z*vp_sq_safe
    
    xi_raw = rfn_gpu((n_particles, 3))
    sxi = cp.zeros((nc_val, 3))
    vxi_calc = cp.zeros((nc_val, 3))
    nloc_safe = cp.maximum(grid['N'], EPSILON)
    
    cp.add.at(sxi, cell_indices, xi_raw*p_weight[:, cp.newaxis])
    sxi /= nloc_safe[:, cp.newaxis]
    xi_mean_corrected = xi_raw - sxi[cell_indices]
    cp.add.at(vxi_calc, cell_indices, xi_mean_corrected**2*p_weight[:, cp.newaxis])
    vxi_term = cp.maximum(vxi_calc / nloc_safe[:, cp.newaxis], 0.0)
    vxi_stddev = cp.sqrt(vxi_term)
    vxi_stddev_safe = cp.maximum(vxi_stddev, EPSILON)
    xi_normalized = xi_mean_corrected / cp.maximum(vxi_stddev_safe[cell_indices], EPSILON)
    
    e_frac1 = cp.bincount(cell_indices, weights=vp_sq*p_weight, minlength=nc_val)
    vp_x_new = (A1*vp_x + A2*(nvA[:, 0]+nq[:, 0]+gforce[:, 0]) + A3*xi_normalized[:, 0])
    vp_y_new = (A1*vp_y + A2*(nvA[:, 1]+nq[:, 1]+gforce[:, 1]) + A3*xi_normalized[:, 1])
    vp_z_new = (A1*vp_z + A2*(nvA[:, 2]+nq[:, 2]+gforce[:, 2]) + A3*xi_normalized[:, 2])
    
    vp_sq_new = vp_x_new**2 + vp_y_new**2 + vp_z_new**2
    e_frac2 = cp.bincount(cell_indices, weights=vp_sq_new*p_weight, minlength=nc_val)
    e_frac1_safe = cp.maximum(e_frac1, EPSILON)
    e_frac2_safe = cp.maximum(e_frac2, EPSILON)
    e_frac = cp.sqrt(e_frac1_safe / e_frac2_safe)
    e_frac_particles = e_frac[cell_indices]
    
    p_vp_x[:] = e_frac_particles * vp_x_new
    p_vp_y[:] = e_frac_particles * vp_y_new
    p_vp_z[:] = e_frac_particles * vp_z_new
    
    U_full_particles = grid['U'][cell_indices]
    p_vx[:] = U_full_particles[:, 0] + p_vp_x
    p_vy[:] = U_full_particles[:, 1] + p_vp_y
    p_vz[:] = U_full_particles[:, 2] + p_vp_z



# ============================================================================
# Entropy / H-theorem diagnostic proxy
# ============================================================================

def _entropy_maxwellian_bin_prob_cupy(nbins, cmax):
    # Discrete standard-normal Maxwellian probabilities over a cubic velocity box.
    dc = 2.0 * cmax / float(nbins)
    centers = cp.linspace(-cmax + 0.5*dc, cmax - 0.5*dc, nbins)
    q1 = cp.exp(-0.5 * centers**2)
    q1 = q1 / cp.sum(q1)
    q3 = (q1[:, None, None] * q1[None, :, None] * q1[None, None, :]).ravel()
    q3 = q3 / cp.sum(q3)
    return cp.maximum(q3, EPSILON)


def compute_entropy_proxy_cupy(p_data, grid, nc_val, nbins=None, cmax=None):
    # Per-cell discrete relative entropy D_KL(f_hist || Maxwellian_hist).
    # This is a diagnostic proxy, not a proof of an H-theorem.
    if nbins is None:
        nbins = ENTROPY_BINS_PER_DIM
    if cmax is None:
        cmax = ENTROPY_CMAX

    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, cell_indices = p_data[12], p_data[13]

    theta = cp.maximum(grid['DM2'] / 3.0, EPSILON)
    thermal = cp.sqrt(theta[cell_indices])

    cx = cp.clip(p_vp_x / thermal, -cmax, cmax)
    cy = cp.clip(p_vp_y / thermal, -cmax, cmax)
    cz = cp.clip(p_vp_z / thermal, -cmax, cmax)

    scale = float(nbins) / (2.0 * cmax)
    ix = cp.clip(cp.floor((cx + cmax) * scale).astype(cp.int32), 0, nbins - 1)
    iy = cp.clip(cp.floor((cy + cmax) * scale).astype(cp.int32), 0, nbins - 1)
    iz = cp.clip(cp.floor((cz + cmax) * scale).astype(cp.int32), 0, nbins - 1)

    nb3 = nbins**3
    vbin = ix + nbins * (iy + nbins * iz)
    joint = cell_indices.astype(cp.int64) * nb3 + vbin.astype(cp.int64)

    hist = cp.bincount(joint, weights=p_weight, minlength=nc_val * nb3).reshape(nc_val, nb3)
    mass = cp.sum(hist, axis=1)
    mass_safe = cp.maximum(mass, EPSILON)
    p_hist = hist / mass_safe[:, None]

    q_hist = _entropy_maxwellian_bin_prob_cupy(nbins, cmax)[None, :]
    p_safe = cp.maximum(p_hist, EPSILON)

    dkl_cells = cp.sum(cp.where(p_hist > 0.0, p_hist * cp.log(p_safe / q_hist), 0.0), axis=1)
    entropy_cells = -cp.sum(cp.where(p_hist > 0.0, p_hist * cp.log(p_safe), 0.0), axis=1)

    cell_weights = mass / cp.maximum(cp.sum(mass), EPSILON)
    global_dkl = cp.sum(cell_weights * dkl_cells)
    global_entropy = cp.sum(cell_weights * entropy_cells)

    return {
        'global_dkl': float(cp.asnumpy(global_dkl)),
        'mean_dkl': float(cp.asnumpy(cp.mean(dkl_cells))),
        'p95_dkl': float(cp.asnumpy(cp.percentile(dkl_cells, 95))),
        'max_dkl': float(cp.asnumpy(cp.max(dkl_cells))),
        'global_entropy': float(cp.asnumpy(global_entropy)),
        'mean_entropy': float(cp.asnumpy(cp.mean(entropy_cells))),
        'min_entropy': float(cp.asnumpy(cp.min(entropy_cells))),
        'empty_cells': int(cp.asnumpy(cp.sum(mass <= EPSILON))),
        'nonfinite_cells': int(cp.asnumpy(cp.sum(~cp.isfinite(dkl_cells)))),
    }


def _ensure_entropy_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'step', 'label', 'global_dkl', 'mean_dkl', 'p95_dkl', 'max_dkl',
            'global_entropy', 'mean_entropy', 'min_entropy',
            'empty_cells', 'nonfinite_cells',
            'nbins', 'cmax'
        ])


def append_entropy_csv(path, step, label, stats):
    _ensure_entropy_csv(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            step, label,
            stats['global_dkl'], stats['mean_dkl'], stats['p95_dkl'], stats['max_dkl'],
            stats['global_entropy'], stats['mean_entropy'], stats['min_entropy'],
            stats['empty_cells'], stats['nonfinite_cells'],
            ENTROPY_BINS_PER_DIM, ENTROPY_CMAX
        ])

def average_results_cupy(avg_grid, grid_gpu, nt, ntss):
    """
    2D Averaging (Corrected, Stable Version).
    """
    nave = 1.0 / float(nt - ntss + 1)
    
    for key in avg_grid.keys():
        if key in grid_gpu and isinstance(avg_grid[key], cp.ndarray):
            if key not in ['x_coords', 'y_coords', 'vol'] and key not in HIGH_MOMENT_KEYS:
                avg_grid[key] = nave * grid_gpu[key] + (1.0 - nave) * avg_grid[key]
    
    avg_grid['T'] = cp.maximum(avg_grid['T'], 1.0)


# ============================================================================
# 4. SOLVER FUNCTIONS (Both Versions)
# ============================================================================

def build_linear_systems_cupy(grid, linsys):
    lhs = linsys['lhs']
    rhs = linsys['rhs']
    PIJ = grid['PIJ']
    Q = grid['Q']
    M3 = grid['M3']
    M4 = grid['M4']
    M5 = grid['M5']
    DM2 = grid['DM2']
    DM4 = grid['DM4']
    lam = grid['lam']
    nu = grid['nu']
    nubol = grid['nubol']
    
    lhs.fill(0)
    rhs.fill(0)
    
    lhs[:, 0, 0] = 2.*PIJ[:, 0]
    lhs[:, 0, 1] = 2.*PIJ[:, 1]
    lhs[:, 0, 2] = 2.*PIJ[:, 2]
    lhs[:, 1, 0] = PIJ[:, 1]
    lhs[:, 1, 1] = PIJ[:, 0] + PIJ[:, 3]
    lhs[:, 1, 2] = PIJ[:, 4]
    lhs[:, 1, 3] = PIJ[:, 1]
    lhs[:, 1, 4] = PIJ[:, 2]
    lhs[:, 2, 0] = PIJ[:, 2]
    lhs[:, 2, 1] = PIJ[:, 4]
    lhs[:, 2, 2] = PIJ[:, 0] + PIJ[:, 5]
    lhs[:, 2, 4] = PIJ[:, 1]
    lhs[:, 2, 5] = PIJ[:, 2]
    lhs[:, 3, 1] = 2.*PIJ[:, 1]
    lhs[:, 3, 3] = 2.*PIJ[:, 3]
    lhs[:, 3, 4] = 2.*PIJ[:, 4]
    lhs[:, 4, 1] = PIJ[:, 2]
    lhs[:, 4, 2] = PIJ[:, 1]
    lhs[:, 4, 3] = PIJ[:, 4]
    lhs[:, 4, 4] = PIJ[:, 3] + PIJ[:, 5]
    lhs[:, 4, 5] = PIJ[:, 4]
    lhs[:, 5, 2] = 2.*PIJ[:, 2]
    lhs[:, 5, 4] = 2.*PIJ[:, 4]
    lhs[:, 5, 5] = 2.*PIJ[:, 5]
    lhs[:, 0, 6] = 2.*Q[:, 0]
    lhs[:, 1, 6] = Q[:, 1]
    lhs[:, 1, 7] = Q[:, 0]
    lhs[:, 2, 6] = Q[:, 2]
    lhs[:, 2, 8] = Q[:, 0]
    lhs[:, 3, 7] = 2.*Q[:, 1]
    lhs[:, 4, 7] = Q[:, 2]
    lhs[:, 4, 8] = Q[:, 1]
    lhs[:, 5, 8] = 2.*Q[:, 2]
    lhs[:, 6, 0] = Q[:, 0] + 2.*M3[:, 0]
    lhs[:, 7, 0] = 2.*M3[:, 1]
    lhs[:, 8, 0] = 2.*M3[:, 2]
    lhs[:, 6, 1] = Q[:, 1] + 4.*M3[:, 1]
    lhs[:, 7, 1] = Q[:, 0] + 4.*M3[:, 3]
    lhs[:, 8, 1] = 4.*M3[:, 4]
    lhs[:, 6, 2] = Q[:, 2] + 4.*M3[:, 2]
    lhs[:, 7, 2] = 4.*M3[:, 4]
    lhs[:, 8, 2] = Q[:, 0] + 4.*M3[:, 5]
    lhs[:, 6, 3] = 2.*M3[:, 3]
    lhs[:, 7, 3] = Q[:, 1] + 2.*M3[:, 6]
    lhs[:, 8, 3] = 2.*M3[:, 7]
    lhs[:, 6, 4] = 4.*M3[:, 4]
    lhs[:, 7, 4] = Q[:, 2] + 4.*M3[:, 7]
    lhs[:, 8, 4] = Q[:, 1] + 4.*M3[:, 8]
    lhs[:, 6, 5] = 2.*M3[:, 5]
    lhs[:, 7, 5] = 2.*M3[:, 8]
    lhs[:, 8, 5] = Q[:, 2] + 2.*M3[:, 9]
    
    DM4_term = DM4 - DM2**2
    lhs[:, 6, 6] = DM4_term + 2.*M4[:, 0] - 2.*DM2*PIJ[:, 0]
    lhs[:, 6, 7] = 2.*M4[:, 1] - 2.*DM2*PIJ[:, 1]
    lhs[:, 6, 8] = 2.*M4[:, 2] - 2.*DM2*PIJ[:, 2]
    lhs[:, 7, 6] = 2.*M4[:, 1] - 2.*DM2*PIJ[:, 1]
    lhs[:, 7, 7] = DM4_term + 2.*M4[:, 3] - 2.*DM2*PIJ[:, 3]
    lhs[:, 7, 8] = 2.*M4[:, 4] - 2.*DM2*PIJ[:, 4]
    lhs[:, 8, 6] = 2.*M4[:, 2] - 2.*DM2*PIJ[:, 2]
    lhs[:, 8, 7] = 2.*M4[:, 4] - 2.*DM2*PIJ[:, 4]
    lhs[:, 8, 8] = DM4_term + 2.*M4[:, 5] - 2.*DM2*PIJ[:, 5]
    
    diag_boost = 1e-10 * cp.identity(9)
    lhs += diag_boost[cp.newaxis, :, :] 
    
    rhs[:, 0] = lam * (-2.*M4[:, 0])
    rhs[:, 1] = lam * (-2.*M4[:, 1])
    rhs[:, 2] = lam * (-2.*M4[:, 2])
    rhs[:, 3] = lam * (-2.*M4[:, 3])
    rhs[:, 4] = lam * (-2.*M4[:, 4])
    rhs[:, 5] = lam * (-2.*M4[:, 5])
    rhs[:, 6] = -lam * (3.*M5[:, 0] - DM2*Q[:, 0] - 2.*(PIJ[:, 0]*Q[:, 0] + PIJ[:, 1]*Q[:, 1] + PIJ[:, 2]*Q[:, 2]))
    rhs[:, 7] = -lam * (3.*M5[:, 1] - DM2*Q[:, 1] - 2.*(PIJ[:, 1]*Q[:, 0] + PIJ[:, 3]*Q[:, 1] + PIJ[:, 4]*Q[:, 2]))
    rhs[:, 8] = -lam * (3.*M5[:, 2] - DM2*Q[:, 2] - 2.*(PIJ[:, 2]*Q[:, 0] + PIJ[:, 4]*Q[:, 1] + PIJ[:, 5]*Q[:, 2]))
    
    nu_term = (3.*nu - 2./3.*nubol)
    rhs[:, 6] += nu_term*Q[:, 0]
    rhs[:, 7] += nu_term*Q[:, 1]
    rhs[:, 8] += nu_term*Q[:, 2]

def solve_linear_systems_cupy(linsys, coeffs):
    try:
        X = cp.linalg.solve(linsys['lhs'], linsys['rhs'])
    except cp.linalg.LinAlgError:
        print(f"Warning: Batched solve failed.")
        X = cp.zeros((NC, 9))
    coeffs['A'][:, 0:6] = X[:, 0:6]
    coeffs['B'][:, 0:3] = X[:, 6:9]

def relu_gpu(x):
    return cp.maximum(x, 0)

def predict_coeffs_cupy_native(grid_gpu, coeffs_gpu, ml_params):
    """
    MODIFIED: This now loads the 5-layer ROBUST model
    """
    input_features_gpu = cp.stack([
        grid_gpu['rho'],
        grid_gpu['T'],
        grid_gpu['U'][:, 0], grid_gpu['U'][:, 1], grid_gpu['U'][:, 2],
        grid_gpu['PIJ'][:, 0], grid_gpu['PIJ'][:, 1], grid_gpu['PIJ'][:, 2],
        grid_gpu['PIJ'][:, 3], grid_gpu['PIJ'][:, 4], grid_gpu['PIJ'][:, 5],
        grid_gpu['Q'][:, 0], grid_gpu['Q'][:, 1], grid_gpu['Q'][:, 2],
        grid_gpu['DM2'],
        grid_gpu['nu']
    ], axis=1) 

    X_scaled = (input_features_gpu - ml_params['X_mean']) / ml_params['X_scale']
    
    # NEW 5-LAYER Architecture
    L1 = relu_gpu(cp.dot(X_scaled, ml_params['W1']) + ml_params['b1'])
    L2 = relu_gpu(cp.dot(L1, ml_params['W2']) + ml_params['b2'])
    L3 = relu_gpu(cp.dot(L2, ml_params['W3']) + ml_params['b3'])
    L4 = relu_gpu(cp.dot(L3, ml_params['W4']) + ml_params['b4'])
    Out_scaled = cp.dot(L4, ml_params['W5']) + ml_params['b5']
    
    Out_unscaled = (Out_scaled * ml_params['y_scale']) + ml_params['y_mean']
    
    coeffs_gpu['A'][:] = Out_unscaled[:, 0:6]
    coeffs_gpu['B'][:] = Out_unscaled[:, 6:9]
    coeffs_gpu['C'].fill(0) 

# ============================================================================
# 5. Plotting Function
# ============================================================================

def plot_cavity_comparison_results(avg_grid_phys, avg_grid_ml, nx, ny, rho_base, plot_filename="cavity_comparison.jpg"):
    """
    Plots 2D contour comparisons from 2D cavity data.
    ML = Filled Contour
    Physics = Black Solid Lines
    ML = Red Dashed Lines
    """
    print(f"Creating 2D comparison plot and saving to {plot_filename}...")
    
    font_settings = {'fontsize': 22}
    plt.rcParams.update({'font.size': 22, 'axes.labelsize': 22, 'axes.titlesize': 22, 'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 18})

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f'2D Cavity (UW_LID={UW_LID}m/s): Physics (Black Lines) vs. AI (Red/Filled)', fontsize=28)

    x_coords_np = cp.asnumpy(avg_grid_phys['x_coords'])
    y_coords_np = cp.asnumpy(avg_grid_phys['y_coords'])
    X, Y = np.meshgrid(x_coords_np, y_coords_np)
    
    U_phys_2d = avg_grid_phys['U'][:, 0].reshape((ny, nx))
    V_phys_2d = avg_grid_phys['U'][:, 1].reshape((ny, nx))
    Speed_phys = np.sqrt(U_phys_2d**2 + V_phys_2d**2)
    
    U_ml_2d = avg_grid_ml['U'][:, 0].reshape((ny, nx))
    V_ml_2d = avg_grid_ml['U'][:, 1].reshape((ny, nx))
    Speed_ml = np.sqrt(U_ml_2d**2 + V_ml_2d**2)

    T_phys_2d = avg_grid_phys['T'].reshape((ny, nx))
    T_ml_2d = avg_grid_ml['T'].reshape((ny, nx))
    
    Rho_phys_2d = avg_grid_phys['rho'].reshape((ny, nx)) / rho_base
    Rho_ml_2d = avg_grid_ml['rho'].reshape((ny, nx)) / rho_base

    line_phys = mlines.Line2D([], [], color='black', linestyle='solid', label='Physics', linewidth=2)
    line_ml = mlines.Line2D([], [], color='red', linestyle='dashed', label='Fast ML', linewidth=2)
    
    # 1. Speed Contour Comparison
    levels = np.linspace(min(Speed_phys.min(), Speed_ml.min()), max(Speed_phys.max(), Speed_ml.max()), 15)
    cf_ml = axs[0].contourf(X, Y, Speed_ml, levels=levels, cmap='viridis', extend='both')
    fig.colorbar(cf_ml, ax=axs[0], label='Speed (m/s)')
    axs[0].contour(X, Y, Speed_phys, levels=levels, colors='black', linestyles='solid', linewidths=1.5)
    axs[0].contour(X, Y, Speed_ml, levels=levels, colors='red', linestyles='dashed', linewidths=1.5)
    
    axs[0].set_title('Speed Comparison')
    axs[0].set_xlabel('X Position (m)', **font_settings)
    axs[0].set_ylabel('Y Position (m)', **font_settings)
    axs[0].legend(handles=[line_phys, line_ml], loc='upper left')
    axs[0].set_aspect('equal')
    axs[0].set_xlim(0, LX)
    axs[0].set_ylim(0, LY)

    # 2. Temperature Contour Comparison
    levels = np.linspace(min(T_phys_2d.min(), T_ml_2d.min()), max(T_phys_2d.max(), T_ml_2d.max()), 15)
    cf_t = axs[1].contourf(X, Y, T_ml_2d, levels=levels, cmap='inferno', extend='both')
    fig.colorbar(cf_t, ax=axs[1], label='Temperature (K)')
    axs[1].contour(X, Y, T_phys_2d, levels=levels, colors='black', linestyles='solid', linewidths=1.5)
    axs[1].contour(X, Y, T_ml_2d, levels=levels, colors='red', linestyles='dashed', linewidths=1.5)

    axs[1].set_title('Temperature Comparison')
    axs[1].set_xlabel('X Position (m)', **font_settings)
    axs[1].set_ylabel('Y Position (m)', **font_settings)
    axs[1].legend(handles=[line_phys, line_ml], loc='upper left')
    axs[1].set_aspect('equal')
    axs[1].set_xlim(0, LX)
    axs[1].set_ylim(0, LY)

    # 3. Density Contour Comparison
    levels = np.linspace(min(Rho_phys_2d.min(), Rho_ml_2d.min()), max(Rho_phys_2d.max(), Rho_ml_2d.max()), 15)
    cf_r = axs[2].contourf(X, Y, Rho_ml_2d, levels=levels, cmap='cividis', extend='both')
    fig.colorbar(cf_r, ax=axs[2], label='Normalized Density')
    axs[2].contour(X, Y, Rho_phys_2d, levels=levels, colors='black', linestyles='solid', linewidths=1.5)
    axs[2].contour(X, Y, Rho_ml_2d, levels=levels, colors='red', linestyles='dashed', linewidths=1.5)

    axs[2].set_title('Density Comparison')
    axs[2].set_xlabel('X Position (m)', **font_settings)
    axs[2].set_ylabel('Y Position (m)', **font_settings)
    axs[2].legend(handles=[line_phys, line_ml], loc='upper left')
    axs[2].set_aspect('equal')
    axs[2].set_xlim(0, LX)
    axs[2].set_ylim(0, LY)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=font_settings['fontsize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    try:
        plt.savefig(plot_filename, format='jpeg', dpi=150)
        print("2D comparison plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    plt.close(fig)

# ============================================================================
# 6. Main Execution
# ============================================================================

def main():
    
    print("="*60)
    print(f"  2D Cavity Comparison (PHYSICS vs. ROBUST ML) @ {UW_LID} m/s")
    print("="*60)
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"  Grid: {NX}x{NY} ({NC} cells), Particles: {NP} ({PARTICLES_PER_CELL_TARGET}/cell)")
    print(f"  Steps: {N_STEPS_PER_RUN}, Averaging starts at: {NTSS}")
    print(f"  High-moment R13/R26 diagnostics sampled every {HIGH_MOMENTS_EVERY} post-transient steps")

    # Load ML Parameters
    print("Loading ROBUST ML model parameters from .npz file...")
    try:
        params_npz = np.load('model_params_robust_cavity_for_cupy.npz')
        
        ml_params_gpu = {}
        for key, val in params_npz.items():
            ml_params_gpu[key] = cp.asarray(val)
        params_npz.close()
        print("Model parameters transferred to GPU.")
    except FileNotFoundError:
        print("Error: 'model_params_robust_cavity_for_cupy.npz' file not found.")
        print("Please run the 'extract_params_cavity.py' script first.")
        return
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        return

    # CPU buffers for storing final results
    avg_grid_cpu_phys = {
        'x_coords': np.linspace(DX/2.0, LX - DX/2.0, NX),
        'y_coords': np.linspace(DY/2.0, LY - DY/2.0, NY),
        'vol': np.zeros(NC, dtype=np.float64), 
        'U': np.zeros((NC, 3), dtype=np.float64),
        'N': np.zeros(NC, dtype=np.float64),
        'T': np.zeros(NC, dtype=np.float64),
        'rho': np.zeros(NC, dtype=np.float64),
        'PIJ': np.zeros((NC, 6), dtype=np.float64),
        'Q': np.zeros((NC, 3), dtype=np.float64),
        'DM2': np.zeros(NC, dtype=np.float64),
        'nu': np.zeros(NC, dtype=np.float64),
        # High-moment diagnostics; see HIGH_MOMENT_KEYS above.
        'M3sym': np.zeros((NC, 10), dtype=np.float64),
        'm3_stf': np.zeros((NC, 10), dtype=np.float64),
        'Rij_raw': np.zeros((NC, 6), dtype=np.float64),
        'Rij_dev': np.zeros((NC, 6), dtype=np.float64),
        'Delta4': np.zeros(NC, dtype=np.float64),
        'Delta4_norm': np.zeros(NC, dtype=np.float64),
        'DM6': np.zeros(NC, dtype=np.float64),
        'DM6_norm': np.zeros(NC, dtype=np.float64),
        'sigma_norm': np.zeros(NC, dtype=np.float64),
        'q_norm': np.zeros(NC, dtype=np.float64),
        'm3_norm': np.zeros(NC, dtype=np.float64),
        'Rij_norm': np.zeros(NC, dtype=np.float64),
    }
    
    avg_grid_cpu_ml = {key: val.copy() for key, val in avg_grid_cpu_phys.items()}

    # =================================================
    # === Run 1: PHYSICS BASELINE
    # =================================================
    
    print("\n" + "="*60)
    print(f"Starting 2D PHYSICS BASELINE run ({N_STEPS_PER_RUN} steps)...")
    
    grid_gpu, coeffs_gpu, linsys_gpu = initialize_grid_cupy(NX, NY, LX, LY)
    p_data = initialize_particles_cupy(NP, LX, LY, THETA_IN, W_PARTICLE)
    avg_grid_gpu, _, _ = initialize_grid_cupy(NX, NY, LX, LY)
    
    for key in avg_grid_gpu:
        if key not in ['x_coords', 'y_coords', 'vol']:
            if isinstance(avg_grid_gpu[key], cp.ndarray):
                avg_grid_gpu[key].fill(0)
    
    start_time_physics = time.time()
    high_count_physics = 0
    
    for nt in range(1, N_STEPS_PER_RUN + 1):
        
        # 1. Move
        p_data[9][:] = p_data[0]
        p_data[10][:] = p_data[1]
        p_data[0][:] = p_data[9] + p_data[3] * DT
        p_data[1][:] = p_data[10] + p_data[4] * DT
        
        # 2. Collide
        apply_boundary_cavity_cupy(p_data, LX, LY, DT)
        
        # 3. Calculate FULL moments
        sort_and_calc_moments_cupy_FULL(p_data, grid_gpu, NC, NX, NY, LX, LY)
        # Entropy / H-theorem diagnostic proxy
        if ENTROPY_EVERY > 0 and nt > NTSS and (nt % ENTROPY_EVERY == 0):
            stats = compute_entropy_proxy_cupy(p_data, grid_gpu, NC)
            append_entropy_csv(os.path.join(ENTROPY_OUTDIR, 'entropy_history_physics.csv'), nt, 'physics', stats)
        if nt > NTSS and (nt % HIGH_MOMENTS_EVERY == 0):
            calc_high_moments_R13_R26_cupy(p_data, grid_gpu, NC)
            high_count_physics += 1
            average_high_moments_cupy(avg_grid_gpu, grid_gpu, high_count_physics)
        
        # 4. Average
        if nt > NTSS:
            average_results_cupy(avg_grid_gpu, grid_gpu, nt, NTSS)
        
        # 5. Report
        if nt % 1000 == 0 or nt == N_STEPS_PER_RUN:
            avg_T_cpu = cp.asnumpy(cp.mean(grid_gpu['T']))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_time_physics
            time_per_step = elapsed / nt
            eta = (N_STEPS_PER_RUN - nt) * time_per_step
            print(f"\r  Physics Step {nt}/{N_STEPS_PER_RUN} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s | Avg T: {avg_T_cpu:.2f} K", end="")

        # 6. Solve
        build_linear_systems_cupy(grid_gpu, linsys_gpu)
        solve_linear_systems_cupy(linsys_gpu, coeffs_gpu)
        
        # 7. Evolve
        evolve_velocities_cupy(p_data, grid_gpu, coeffs_gpu, DT, NC)
    
    cp.cuda.Stream.null.synchronize() 
    end_time_physics = time.time()
    time_physics = end_time_physics - start_time_physics
    print(f"\nPHYSICS run finished in {time_physics:.2f} seconds.")

    # Save physics results to CPU buffer
    output_filename_phys = "cavity_data_PHYSICS.npz"
    try:
        for key in avg_grid_cpu_phys.keys():
            if key in avg_grid_gpu and key not in ['x_coords', 'y_coords']:
                avg_grid_cpu_phys[key] = cp.asnumpy(avg_grid_gpu[key])
        np.savez_compressed(output_filename_phys, **avg_grid_cpu_phys)
        print(f"Physics results saved to {output_filename_phys}")
    except Exception as e:
        print(f"Error saving data file: {e}")


    # =================================================
    # === Run 2: FAST ML
    # =================================================
    
    print("\n" + "="*60)
    print(f"Starting 2D FAST ML run ({N_STEPS_PER_RUN} steps)...")

    # Reset simulation
    grid_gpu, coeffs_gpu, _ = initialize_grid_cupy(NX, NY, LX, LY)
    p_data = initialize_particles_cupy(NP, LX, LY, THETA_IN, W_PARTICLE)
    avg_grid_gpu, _, _ = initialize_grid_cupy(NX, NY, LX, LY)
    
    for key in avg_grid_gpu:
        if key not in ['x_coords', 'y_coords', 'vol']:
            if isinstance(avg_grid_gpu[key], cp.ndarray):
                avg_grid_gpu[key].fill(0)

    start_time_ml = time.time()
    high_count_ml = 0
    
    for nt in range(1, N_STEPS_PER_RUN + 1):
        # 1. Move
        p_data[9][:] = p_data[0]
        p_data[10][:] = p_data[1]
        p_data[0][:] = p_data[9] + p_data[3] * DT
        p_data[1][:] = p_data[10] + p_data[4] * DT
        
        # 2. Collide
        apply_boundary_cavity_cupy(p_data, LX, LY, DT)

        # 3. Calculate LITE moments
        sort_and_calc_moments_cupy_LITE(p_data, grid_gpu, NC, NX, NY, LX, LY)
        # Entropy / H-theorem diagnostic proxy
        if ENTROPY_EVERY > 0 and nt > NTSS and (nt % ENTROPY_EVERY == 0):
            stats = compute_entropy_proxy_cupy(p_data, grid_gpu, NC)
            append_entropy_csv(os.path.join(ENTROPY_OUTDIR, 'entropy_history_ml.csv'), nt, 'ml', stats)
        if nt > NTSS and (nt % HIGH_MOMENTS_EVERY == 0):
            calc_high_moments_R13_R26_cupy(p_data, grid_gpu, NC)
            high_count_ml += 1
            average_high_moments_cupy(avg_grid_gpu, grid_gpu, high_count_ml)
        
        # 4. Average
        if nt > NTSS:
            average_results_cupy(avg_grid_gpu, grid_gpu, nt, NTSS)
        
        # 5. Report
        if nt % 1000 == 0 or nt == N_STEPS_PER_RUN:
            avg_T_cpu = cp.asnumpy(cp.mean(grid_gpu['T']))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_time_ml
            time_per_step = elapsed / nt
            eta = (N_STEPS_PER_RUN - nt) * time_per_step
            print(f"\r  FAST ML Step {nt}/{N_STEPS_PER_RUN} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s | Avg T: {avg_T_cpu:.2f} K", end="")

        # 6. Solve (ML)
        predict_coeffs_cupy_native(grid_gpu, coeffs_gpu, ml_params_gpu)
        
        # 7. Evolve
        evolve_velocities_cupy(p_data, grid_gpu, coeffs_gpu, DT, NC)
    
    cp.cuda.Stream.null.synchronize() 
    end_time_ml = time.time()
    time_ml = end_time_ml - start_time_ml
    print(f"\nFAST ML run finished in {time_ml:.2f} seconds.")

    # Save ML results to CPU buffer
    output_filename_ml = "cavity_data_FAST_ML.npz"
    try:
        for key in avg_grid_cpu_ml.keys():
            if key in avg_grid_gpu and key not in ['x_coords', 'y_coords']:
                avg_grid_cpu_ml[key] = cp.asnumpy(avg_grid_gpu[key])
        np.savez_compressed(output_filename_ml, **avg_grid_cpu_ml)
        print(f"ML results saved to {output_filename_ml}")
    except Exception as e:
        print(f"Error saving data file: {e}")


    # =================================================
    # === 3. Final Report & Plot
    # =================================================
    
    print("\n" + "="*60)
    print("Creating 2D comparison plot...")
    plot_cavity_comparison_results(avg_grid_cpu_phys, avg_grid_cpu_ml, NX, NY, RHO_IN_BASE, f"cavity_comparison_robust_{int(UW_LID)}ms.jpg")

    print("\n" + "="*60)
    print(f"           Final Performance Report (2D Cavity @ {UW_LID} m/s)")
    print("="*60)
    print(f"  Grid Size: {NX}x{NY} ({NC} cells)")
    print(f"  Physics Solver (Baseline) Time: {time_physics:.2f} seconds")
    print(f"  Fast ML Solver Time:            {time_ml:.2f} seconds")
    print("-" * 60)
    if time_ml > 0:
        speedup = time_physics / time_ml
        print(f"  Speedup:                        {speedup:.2f}x")
    else:
        print("  Speedup:                        N/A")
    print("="*60)

if __name__ == "__main__":
    main()