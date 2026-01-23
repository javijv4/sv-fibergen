#!/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
import pyvista as pv

from src.FibGenOO import FibGenBayer


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

if __name__ == "__main__":

    # Original parameters used in main_bayer_oo.py
    params = {
        "ALFA_END": 60.0,
        "ALFA_EPI": -60.0,
        "BETA_END": 20.0,
        "BETA_EPI": -20.0,
    }

    outdir = "example/truncated/output_b_oo"
    fiber_results = pv.read(outdir + "/results_bayer_old.vtu")
    f = fiber_results.cell_data['fiber']
    s = fiber_results.cell_data['sheet']
    n = fiber_results.cell_data['sheet-normal']

    # Read laplace solutions
    laplace_results_file = os.path.join(outdir, 'result_001.vtu')

    # Generate "fibers" using all params set to zero
    # This gives the local eC, eL, eT
    fib_gen = FibGenBayer()
    fib_gen.load_laplace_results(laplace_results_file)
    eC, eL, eT = fib_gen.generate_fibers({
        "ALFA_END": 0.0,
        "ALFA_EPI": 0.0,
        "BETA_END": 0.0,
        "BETA_EPI": 0.0,
    })

    fproj, eLproj, eTproj = fib_gen.generate_fibers({
        "ALFA_END": 60.0,
        "ALFA_EPI": -60.0,
        "BETA_END": 0.0,
        "BETA_EPI": 0.0,
    })

    #%% Project fiber to the plane formed by eL and eC
    f_projected = project_to_plane(f, eL, eC)

    beta_dot = np.abs(np.sum(f_projected * f, axis=1))
    beta_dot = np.clip(beta_dot, 0, 1)  # Ensure values are within valid range
    abs_beta_angle = np.rad2deg(np.arccos(beta_dot))
    sign_beta = - np.sign(np.sum(f * eT, axis=1))
    beta_angle = abs_beta_angle * sign_beta

    alpha_dot = np.abs(np.sum(eC * f_projected, axis=1))
    alpha_dot = np.clip(alpha_dot, 0, 1)  # Ensure values are within valid range
    abs_alpha_angle = np.rad2deg(np.arccos(alpha_dot))
    sign_alpha = np.sign(np.sum(fproj * eL, axis=1))
    alpha_angle = abs_alpha_angle * sign_alpha

    # Compare angles
    global_alpha, global_beta = fib_gen.get_angle_fields(params)

    #%% Print statistics
    print("Alpha angle comparison:")
    print(f"  Mean absolute error: {np.mean(np.abs(global_alpha - alpha_angle)):.3f} degrees")
    print(f"  Max absolute error: {np.max(np.abs(global_alpha - alpha_angle)):.3f} degrees")
    print()
    print("Beta angle comparison:")
    print(f"  Mean absolute error: {np.mean(np.abs(global_beta - beta_angle)):.3f} degrees")
    print(f"  Max absolute error: {np.max(np.abs(global_beta - beta_angle)):.3f} degrees")
    #%% Save validation results
    validation_mesh = fiber_results.copy(deep=True)
    validation_mesh.clear_field_data()

    validation_mesh.cell_data['alpha_angle'] = alpha_angle
    validation_mesh.cell_data['beta_angle'] = beta_angle
    validation_mesh.cell_data['global_alpha'] = global_alpha
    validation_mesh.cell_data['global_beta'] = global_beta
    validation_mesh.cell_data['f'] = f
    validation_mesh.cell_data['s'] = s
    validation_mesh.cell_data['n'] = n
    validation_mesh.cell_data['fprojected'] = f_projected
    validation_mesh.cell_data['eC'] = eC
    validation_mesh.cell_data['eL'] = eL
    validation_mesh.cell_data['eT'] = eT
    validation_mesh.cell_data['fproj'] = fproj
    validation_mesh.cell_data['eLproj'] = eLproj
    validation_mesh.cell_data['eTproj'] = eTproj

    validation_mesh.save("validation_bayer_old.vtu")



