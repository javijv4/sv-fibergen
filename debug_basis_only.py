#!/usr/bin/env python
"""Debug script to check if the basis vectors themselves are different."""

import os
import numpy as np
import pyvista as pv
from src.FibGenOO import FibGenBayer, axis, orient
from src import quat_utils as qu

# Load the Laplace results
laplace_results_file = "example/truncated/output_b_oo/result_001.vtu"

# Parameters for Bayer method
params_deg = {
    "ALFA_END": 60.0,
    "ALFA_EPI": -60.0,
    "BETA_END": 0.0,
    "BETA_EPI": 0.0,
}

# Convert to radians
params = {k: np.deg2rad(v) for k, v in params_deg.items()}

# Initialize fiber generator
fib_gen = FibGenBayer()
fib_gen.load_laplace_results(laplace_results_file)

#Test on a few cells
n_test = 5
test_cells = [350652, 203975, 208707, 611991, 505537]  # Cells with large differences

print("Testing basis generation for cells with largest differences...\n")

for i, iCell in enumerate(test_cells):
    print(f"Cell {iCell}:")
    
    # Get gradients for this cell
    gPhiLV = fib_gen.grad['Trans_LV'][iCell, :]
    gPhiRV = fib_gen.grad['Trans_RV'][iCell, :]
    gPhiAB = fib_gen.grad['Long_AB'][iCell, :]
    gPhiEP = fib_gen.grad['Trans_EPI'][iCell, :]
    
    # Get Laplace values
    phiLV = fib_gen.lap['Trans_LV'][iCell]
    phiRV = fib_gen.lap['Trans_RV'][iCell]
    phiEP = fib_gen.lap['Trans_EPI'][iCell]
    
    d = phiRV / (phiLV + phiRV)
    alfaS = params['ALFA_END'] * (1 - d) - params['ALFA_END'] * d
    betaS = params['BETA_END'] * (1 - d) - params['BETA_END'] * d
    
    # Old method (scalar)
    Q_LV_old = axis(gPhiAB, -gPhiLV)
    Q_LV_old = orient(Q_LV_old, alfaS, betaS)
    
    Q_RV_old = axis(gPhiAB, gPhiRV)
    Q_RV_old = orient(Q_RV_old, alfaS, betaS)
    
    # New method (vectorized, single element)
    Q_LV_new = fib_gen.calculate_basis(gPhiAB[None, :], -gPhiLV[None, :])
    Q_LV_new = fib_gen.rotate_basis_matrix(Q_LV_new, np.array([alfaS]), np.array([betaS]))[0]
    
    Q_RV_new = fib_gen.calculate_basis(gPhiAB[None, :], gPhiRV[None, :])
    Q_RV_new = fib_gen.rotate_basis_matrix(Q_RV_new, np.array([alfaS]), np.array([betaS]))[0]
    
    print(f"  Q_LV difference: {np.linalg.norm(Q_LV_new - Q_LV_old):.6e}")
    print(f"  Q_RV difference: {np.linalg.norm(Q_RV_new - Q_RV_old):.6e}")
    
    # Check quaternions
    q_LV_new = qu.rotm_to_quat_batch(Q_LV_new[None, :])[0]
    q_RV_new = qu.rotm_to_quat_batch(Q_RV_new[None, :])[0]
    
    from src.FibGenOO import rot2quat
    q_LV_old = rot2quat(Q_LV_old)
    q_RV_old = rot2quat(Q_RV_old)
    
    print(f"  q_LV_new: {q_LV_new}")
    print(f"  q_LV_old: {q_LV_old}")
    print(f"  q_LV dot: {np.dot(q_LV_new, q_LV_old):.6f}")
    
    print(f"  q_RV_new: {q_RV_new}")
    print(f"  q_RV_old: {q_RV_old}")
    print(f"  q_RV dot: {np.dot(q_RV_new, q_RV_old):.6f}")
    print()
