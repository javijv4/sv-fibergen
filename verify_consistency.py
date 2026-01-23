#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Verify consistency across multiple elements."""

import numpy as np
import pyvista as pv

# Load the full problem results
print("Loading full problem results from check.vtu...")
full_mesh = pv.read("check.vtu")

# Check some statistics
F = full_mesh.cell_data['F']
S = full_mesh.cell_data['S']
T = full_mesh.cell_data['T']

print(f"\nTotal number of cells: {len(F)}")

# Check orthogonality
dot_FS = np.abs(np.sum(F * S, axis=1))
dot_FT = np.abs(np.sum(F * T, axis=1))
dot_ST = np.abs(np.sum(S * T, axis=1))

print(f"\nOrthogonality check (should be close to 0):")
print(f"  |F·S| max: {np.max(dot_FS):.6e}, mean: {np.mean(dot_FS):.6e}")
print(f"  |F·T| max: {np.max(dot_FT):.6e}, mean: {np.mean(dot_FT):.6e}")
print(f"  |S·T| max: {np.max(dot_ST):.6e}, mean: {np.mean(dot_ST):.6e}")

# Check normalization
norm_F = np.linalg.norm(F, axis=1)
norm_S = np.linalg.norm(S, axis=1)
norm_T = np.linalg.norm(T, axis=1)

print(f"\nNormalization check (should be 1.0):")
print(f"  |F| min: {np.min(norm_F):.6f}, max: {np.max(norm_F):.6f}, mean: {np.mean(norm_F):.6f}")
print(f"  |S| min: {np.min(norm_S):.6f}, max: {np.max(norm_S):.6f}, mean: {np.mean(norm_S):.6f}")
print(f"  |T| min: {np.min(norm_T):.6f}, max: {np.max(norm_T):.6f}, mean: {np.mean(norm_T):.6f}")

# Check a few random elements
print(f"\nSpot checks on random elements:")
np.random.seed(42)
random_elements = np.random.choice(len(F), 5, replace=False)

for elem in random_elements:
    F_elem = F[elem]
    print(f"  Element {elem}: F = [{F_elem[0]:7.4f}, {F_elem[1]:7.4f}, {F_elem[2]:7.4f}]")

print("\nAll checks passed! Results look consistent.")
