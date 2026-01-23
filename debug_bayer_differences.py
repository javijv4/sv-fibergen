#!/usr/bin/env python
"""Debug script to identify differences between generate_fibers and generate_fibers_old."""

import os
import numpy as np
import pyvista as pv
from src.FibGenOO import FibGenBayer

# Load the Laplace results
laplace_results_file = "example/truncated/output_b_oo/result_001.vtu"

# Parameters for Bayer method
params = {
    "ALFA_END": 60.0,
    "ALFA_EPI": -60.0,
    "BETA_END": 0.0,
    "BETA_EPI": 0.0,
}

# Initialize two fiber generators
fib_gen_new = FibGenBayer()
fib_gen_old = FibGenBayer()

# Load Laplace results for both
fib_gen_new.load_laplace_results(laplace_results_file)
fib_gen_old.load_laplace_results(laplace_results_file)

# Generate fibers with both methods
print("Generating fibers with NEW method...")
F_new, S_new, T_new = fib_gen_new.generate_fibers(params)

print("\nGenerating fibers with OLD method...")
F_old, S_old, T_old = fib_gen_old.generate_fibers_old(params)

# Compare results
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

# Fiber directions
diff_F = np.linalg.norm(F_new - F_old, axis=1)
print(f"\nFiber direction differences:")
print(f"  Max difference: {np.max(diff_F):.6e}")
print(f"  Mean difference: {np.mean(diff_F):.6e}")
print(f"  Median difference: {np.median(diff_F):.6e}")
print(f"  Cells with diff > 1e-6: {np.sum(diff_F > 1e-6)} / {len(diff_F)}")

# Sheet directions
diff_S = np.linalg.norm(S_new - S_old, axis=1)
print(f"\nSheet normal differences:")
print(f"  Max difference: {np.max(diff_S):.6e}")
print(f"  Mean difference: {np.mean(diff_S):.6e}")

# Sheet normal directions  
diff_T = np.linalg.norm(T_new - T_old, axis=1)
print(f"\nSheet direction differences:")
print(f"  Max difference: {np.max(diff_T):.6e}")
print(f"  Mean difference: {np.mean(diff_T):.6e}")

# Check basis vectors
if 'eC_LV' in fib_gen_new.mesh.cell_data and 'eC_LV' in fib_gen_old.mesh.cell_data:
    eC_LV_new = fib_gen_new.mesh.cell_data['eC_LV']
    eC_LV_old = fib_gen_old.mesh.cell_data['eC_LV']
    # Handle shape mismatch - old version might store differently
    if eC_LV_new.shape != eC_LV_old.shape:
        print(f"\nShape mismatch: eC_LV_new {eC_LV_new.shape} vs eC_LV_old {eC_LV_old.shape}")
        print("Skipping basis vector comparison.")
    else:
        diff_eC_LV = np.linalg.norm(eC_LV_new - eC_LV_old, axis=1)
        print(f"\neC_LV (LV circumferential basis) differences:")
        print(f"  Max difference: {np.max(diff_eC_LV):.6e}")
        print(f"  Mean difference: {np.mean(diff_eC_LV):.6e}")

        eC_RV_new = fib_gen_new.mesh.cell_data['eC_RV']
        eC_RV_old = fib_gen_old.mesh.cell_data['eC_RV']
        diff_eC_RV = np.linalg.norm(eC_RV_new - eC_RV_old, axis=1)
        print(f"\neC_RV (RV circumferential basis) differences:")
        print(f"  Max difference: {np.max(diff_eC_RV):.6e}")
        print(f"  Mean difference: {np.mean(diff_eC_RV):.6e}")

# Find cells with largest differences
print(f"\nCells with largest fiber direction differences:")
worst_indices = np.argsort(diff_F)[-5:][::-1]
for idx in worst_indices:
    print(f"  Cell {idx}: diff = {diff_F[idx]:.6e}")
    print(f"    F_new: {F_new[idx]}")
    print(f"    F_old: {F_old[idx]}")
    print(f"    Dot product: {np.dot(F_new[idx], F_old[idx]):.6f}")
