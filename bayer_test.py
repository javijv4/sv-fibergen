#!/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
import pyvista as pv

from src.FibGenOO import FibGenBayer
import matplotlib.pyplot as plt


def project_to_plane(fiber, vector1, vector2):
    """
    Project fiber vectors onto the plane formed by vector1 and vector2.
    
    Parameters
    ----------
    fiber : ndarray (n, 3)
        Array of fiber vectors to project
    vector1 : ndarray (n, 3)
        First vector defining the plane for each row
    vector2 : ndarray (n, 3)
        Second vector defining the plane for each row
        
    Returns
    -------
    projected : ndarray (n, 3)
        Fiber vectors projected onto the plane formed by vector1 and vector2
    """
    # Calculate the normal to the plane (cross product of vector1 and vector2)
    normal = np.cross(vector1, vector2)
    
    # Normalize the normal vectors
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal_normalized = normal / normal_norm
    
    # Project fiber onto the plane by removing the component along the normal
    # projected = fiber - (fiber · normal) * normal
    dot_product = np.sum(fiber * normal_normalized, axis=1, keepdims=True)
    projected = fiber - dot_product * normal_normalized
    
    # Normalize the projected vectors
    projected_norm = np.linalg.norm(projected, axis=1, keepdims=True)
    projected_normalized = projected / projected_norm
    
    return projected_normalized

def calculate_alpha_beta_angles(f, eC, eL, eT):
    # Project fiber to the plane formed by eL and eC
    f_projected = project_to_plane(f, eL, eC)

    beta_dot = np.abs(np.sum(f_projected * f, axis=1))
    beta_dot = np.clip(beta_dot, 0, 1)  # Ensure values are within valid range
    abs_beta_angle = np.rad2deg(np.arccos(beta_dot))
    sign_beta = - np.sign(np.sum(f * eT, axis=1))
    beta_angle = abs_beta_angle * sign_beta

    alpha_dot = np.abs(np.sum(eC * f_projected, axis=1))
    alpha_dot = np.clip(alpha_dot, 0, 1)  # Ensure values are within valid range
    abs_alpha_angle = np.rad2deg(np.arccos(alpha_dot))
    sign_alpha = np.sign(np.sum(f_projected * eL, axis=1))
    alpha_angle = abs_alpha_angle * sign_alpha
    
    return alpha_angle, beta_angle, f_projected



if __name__ == "__main__":

    outdir = "example/truncated/output_b_oo"

    params_zero = {
        "ALFA_END": 0.0,
        "ALFA_EPI": 0.0,
        "BETA_END": 0.0,
        "BETA_EPI": 0.0,
    }

    params_alpha = {
        "ALFA_END": 60.0,
        "ALFA_EPI": -60.0,
        "BETA_END": 0.0,
        "BETA_EPI": 0.0,
    }

    params_beta = {
        "ALFA_END": 0.0,
        "ALFA_EPI": 0.0,
        "BETA_END": 20.0,
        "BETA_EPI": -20.0,
    }

    params = {
        "ALFA_END": 60.0,
        "ALFA_EPI": -60.0,
        "BETA_END": 20.0,
        "BETA_EPI": -20.0,
    }

#%%
    # Read laplace solutions
    laplace_results_file = os.path.join(outdir, 'result_001.vtu')

    # Initialize fiber generator
    fib_gen = FibGenBayer()
    fib_gen.load_laplace_results(laplace_results_file)

    # Calculate orthogonal basis vectors
    eC, eL, eT = fib_gen.generate_fibers(params_zero)

    # Sanity check 1: Only alpha rotation
    f_alpha, s_alpha, n_alpha = fib_gen.generate_fibers(params_alpha)
    ref_alpha_only_a, ref_beta_only_a = fib_gen.get_angle_fields(params_alpha)
    alpha_only_a, beta_only_a, f_projected = calculate_alpha_beta_angles(f_alpha, eC, eL, eT)

    # Sanity check 2: Only beta rotation
    f_beta, s_beta, n_beta = fib_gen.generate_fibers(params_beta)
    ref_alpha_only_b, ref_beta_only_b = fib_gen.get_angle_fields(params_beta)
    alpha_only_b, beta_only_b, f_projected = calculate_alpha_beta_angles(f_beta, eC, eL, eT)
    fib_gen.mesh.cell_data['alpha_only_b'] = alpha_only_b
    fib_gen.mesh.cell_data['beta_only_b'] = beta_only_b
    fib_gen.mesh.cell_data['alpha_ref_b'] = ref_alpha_only_b
    fib_gen.mesh.cell_data['beta_ref_b'] = ref_beta_only_b
    fib_gen.mesh.save('validation_bayer_onlybeta.vtu')
    
    # Alpha and beta rotation combined
    eC, eL, eT = fib_gen.generate_fibers(params_zero)
    f_combined, s_combined, n_combined = fib_gen.generate_fibers(params)
    ref_alpha_combined, ref_beta_combined = fib_gen.get_angle_fields(params)
    alpha_combined, beta_combined, f_projected = calculate_alpha_beta_angles(f_combined, eC, eL, eT)
    fib_gen.mesh.cell_data['alpha_combined'] = alpha_combined
    fib_gen.mesh.cell_data['beta_combined'] = beta_combined
    fib_gen.mesh.cell_data['alpha_ref'] = ref_alpha_combined
    fib_gen.mesh.cell_data['beta_ref'] = ref_beta_combined
    fib_gen.mesh.save('validation_bayer_combined.vtu')
    
    # For comparison, generate fibers using original Bayer method
    eC, eL, eT = fib_gen.generate_fibers(params_zero, correct_slerp=True, flip_rv=False)
    f_og, s_og, n_og = fib_gen.generate_fibers(params, correct_slerp=True, flip_rv=False)
    alpha_og, beta_og, f_projected = calculate_alpha_beta_angles(f_og, eC, eL, eT)
    
#%%
    # Create figure with correlation plots
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    fig.suptitle(r'$\alpha$ and $\beta$ angle correlations', fontsize=16)

    # Alpha Only
    axes[0, 0].scatter(ref_alpha_only_a, alpha_only_a, alpha=0.5, s=10, label=r'$\alpha$', color='blue')
    axes[0, 0].scatter(ref_beta_only_a, beta_only_a, alpha=0.5, s=10, label=r'$\beta$', color='red')
    axes[0, 0].plot([-90, 90], [-90, 90], 'k--', lw=1)
    axes[0, 0].set_title(r'$\alpha=\pm 60, \beta=0$')
    axes[0, 0].set_xlabel('Reference (degrees)')
    axes[0, 0].set_ylabel('Calculated (degrees)')
    axes[0, 0].legend()

    # Beta Only
    axes[0, 1].scatter(ref_alpha_only_b, alpha_only_b, alpha=0.5, s=10, label=r'$\alpha$', color='blue')
    axes[0, 1].scatter(ref_beta_only_b, beta_only_b, alpha=0.5, s=10, label=r'$\beta$', color='red')
    axes[0, 1].plot([-90, 90], [-90, 90], 'k--', lw=1)
    axes[0, 1].set_title(r'$\alpha=0, \beta=\pm 20$')
    axes[0, 1].set_xlabel('Reference (degrees)')
    axes[0, 1].set_ylabel('Calculated (degrees)')
    axes[0, 1].legend()

    # Combined
    axes[1, 0].scatter(ref_alpha_combined, alpha_combined, alpha=0.5, s=10, label=r'$\alpha$', color='blue')
    axes[1, 0].scatter(ref_beta_combined, beta_combined, alpha=0.5, s=10, label=r'$\beta$', color='red')
    axes[1, 0].plot([-90, 90], [-90, 90], 'k--', lw=1)
    axes[1, 0].set_title(r'$\alpha=\pm 60, \beta=\pm 20$')
    axes[1, 0].set_xlabel('Reference (degrees)')
    axes[1, 0].set_ylabel('Calculated (degrees)')
    axes[1, 0].legend()

    # Original Bayer
    axes[1, 1].scatter(ref_alpha_combined, alpha_og, alpha=0.5, s=10, label=r'$\alpha$', color='blue')
    axes[1, 1].scatter(ref_beta_combined, beta_og, alpha=0.5, s=10, label=r'$\beta$', color='red')
    axes[1, 1].plot([-90, 90], [-90, 90], 'k--', lw=1)
    axes[1, 1].set_title(r'$\alpha=\pm 60, \beta=\pm 20$ - Original Bayer')
    axes[1, 1].set_xlabel('Reference (degrees)')
    axes[1, 1].set_ylabel('Calculated (degrees)')
    axes[1, 1].legend()

    plt.savefig(os.path.join(outdir, 'angle_correlations.png'), dpi=150)
    # plt.close()

    # Save VTU file with all angle values
    mesh = fib_gen.mesh.copy(deep=True)
    mesh.clear_cell_data()
    mesh['alpha_combined'] = alpha_combined
    mesh['beta_combined'] = beta_combined
    mesh['alpha_og'] = alpha_og
    mesh['beta_og'] = beta_og
    mesh['alpha_only_a'] = alpha_only_a
    mesh['beta_only_a'] = beta_only_a
    mesh['alpha_only_b'] = alpha_only_b
    mesh['beta_only_b'] = beta_only_b
    mesh['alpha_ref'] = ref_alpha_combined
    mesh['beta_ref'] = ref_beta_combined
    mesh.save('validation.vtu')