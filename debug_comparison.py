#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Debug script to compare single element vs full problem results."""

import numpy as np
import pyvista as pv

element = 530439

# Load the full problem results
print("Loading full problem results from check.vtu...")
full_mesh = pv.read("check.vtu")

# Load single element results
print("Loading single element results from single_element_fibers.vtu...")
single_mesh = pv.read("single_element_fibers.vtu")

# Extract data from full problem for this element
print(f"\n{'='*60}")
print(f"COMPARISON FOR ELEMENT {element}")
print(f"{'='*60}")

# Compare Laplace fields
print("\nLaplace Fields:")
print(f"  Trans_EPI - Full: {full_mesh.cell_data['Trans_EPI'][element]:.6f}, Single: {single_mesh.cell_data['Trans_EPI'][0]:.6f}")
print(f"  Trans_LV  - Full: {full_mesh.cell_data['Trans_LV'][element]:.6f}, Single: {single_mesh.cell_data['Trans_LV'][0]:.6f}")
print(f"  Trans_RV  - Full: {full_mesh.cell_data['Trans_RV'][element]:.6f}, Single: {single_mesh.cell_data['Trans_RV'][0]:.6f}")
print(f"  Long_AB   - Full: {full_mesh.cell_data['Long_AB'][element]:.6f}, Single: {single_mesh.cell_data['Long_AB'][0]:.6f}")

# Compare d values
print(f"\nd value - Full: {full_mesh.cell_data['d'][element]:.6f}, Single: (computed in script)")

# Compare angles
print("\nAngles:")
print(f"  alfaS - Full: {np.rad2deg(full_mesh.cell_data['alfaS'][element]):.6f} deg")
print(f"  betaS - Full: {np.rad2deg(full_mesh.cell_data['betaS'][element]):.6f} deg")
print(f"  alfaW - Full: {np.rad2deg(full_mesh.cell_data['alfaW'][element]):.6f} deg")
print(f"  betaW - Full: {np.rad2deg(full_mesh.cell_data['betaW'][element]):.6f} deg")

# Compare basis vectors
print("\neC_END (circumferential endocardial):")
eC_END_full = full_mesh.cell_data['eC_END'][element]
eC_END_single = single_mesh.cell_data['eC_END'][0]
print(f"  Full:   {eC_END_full}")
print(f"  Single: {eC_END_single}")
print(f"  Diff:   {np.linalg.norm(eC_END_full - eC_END_single):.6e}")

print("\neC_EPI (circumferential epicardial):")
eC_EPI_full = full_mesh.cell_data['eC_EPI'][element]
eC_EPI_single = single_mesh.cell_data['eC_EPI'][0]
print(f"  Full:   {eC_EPI_full}")
print(f"  Single: {eC_EPI_single}")
print(f"  Diff:   {np.linalg.norm(eC_EPI_full - eC_EPI_single):.6e}")

# Compare final fiber directions
print("\nFiber Direction (F):")
F_full = full_mesh.cell_data['F'][element]
F_single = single_mesh.cell_data['Fiber'][0]
print(f"  Full:   {F_full}")
print(f"  Single: {F_single}")
print(f"  Diff:   {np.linalg.norm(F_full - F_single):.6e}")

print("\nSheet Direction (S):")
S_full = full_mesh.cell_data['S'][element]
S_single = single_mesh.cell_data['Sheet'][0]
print(f"  Full:   {S_full}")
print(f"  Single: {S_single}")
print(f"  Diff:   {np.linalg.norm(S_full - S_single):.6e}")

print("\nNormal Direction (T):")
T_full = full_mesh.cell_data['T'][element]
T_single = single_mesh.cell_data['Normal'][0]
print(f"  Full:   {T_full}")
print(f"  Single: {T_single}")
print(f"  Diff:   {np.linalg.norm(T_full - T_single):.6e}")

print(f"\n{'='*60}")

# Check if results are significantly different
tolerance = 1e-6
if (np.linalg.norm(F_full - F_single) > tolerance or
    np.linalg.norm(S_full - S_single) > tolerance or
    np.linalg.norm(T_full - T_single) > tolerance):
    print("WARNING: Results differ significantly!")
    print("\nChecking intermediate values...")
    
    # Compare LV and RV basis before rotation
    print("\neC_LV0 (before rotation):")
    print(f"  Full:   {full_mesh.cell_data['eC_LV0'][element]}")
    
    print("\neC_LV (after rotation):")
    print(f"  Full:   {full_mesh.cell_data['eC_LV'][element]}")
    
    print("\neC_RV0 (before rotation):")
    print(f"  Full:   {full_mesh.cell_data['eC_RV0'][element]}")
    
    print("\neC_RV (after rotation):")
    print(f"  Full:   {full_mesh.cell_data['eC_RV'][element]}")
else:
    print("Results match within tolerance!")
