#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Main script for generating biventricular fibers using the Bayer method.

This module implements fiber generation for biventricular heart models using
the Laplace-Dirichlet rule-based method described in:
Bayer et al. 2012, "A Novel Rule-Based Algorithm for Assigning Myocardial 
Fiber Orientation to Computational Heart Models"
https://doi.org/10.1007/s10439-012-0593-5

The script supports command-line arguments for customization of mesh paths,
output directories, and solver executables.
"""

import os
import pyvista as pv
from src.LaplaceSolver import LaplaceSolver
from src.FibGenOO import FibGenBayer
from src.SurfaceNames import SurfaceName
from time import time


if __name__ == "__main__":

    ###########################################################
    ############  USER INPUTS  ################################
    ###########################################################

    run_flag = False
    svfsi_exec = "svmultiphysics "

    mesh_path = "example/truncated/VOLUME.vtu"
    outdir = "example/truncated/output_b_oo"

    surface_paths = {SurfaceName.EPI: 'example/truncated/mesh-surfaces/EPI.vtp',
                    SurfaceName.EPI_APEX: 'example/truncated/mesh-surfaces/EPI_APEX.vtp',
                    SurfaceName.BASE: 'example/truncated/mesh-surfaces/BASE.vtp',
                    SurfaceName.ENDO_LV: 'example/truncated/mesh-surfaces/LV.vtp',
                    SurfaceName.ENDO_RV: 'example/truncated/mesh-surfaces/RV.vtp'}

    # Parameters for the Bayer et al. method https://doi.org/10.1007/s10439-012-0593-5. 
    params = {
        "ALFA_END": 60.0,
        "ALFA_EPI": -60.0,
        "BETA_END": 0., #20.0,
        "BETA_EPI": 0. #-20.0,
    }


    ###########################################################
    ############  FIBER GENERATION  ###########################
    ###########################################################

    # Make sure the paths are full paths
    mesh_path = os.path.abspath(mesh_path)
    outdir = os.path.abspath(outdir)

    start = time()

    # Create output directory if needed
    os.makedirs(outdir, exist_ok=True)

    # Initialize Laplace solver
    solver = LaplaceSolver(mesh_path, surface_paths, svfsi_exec)

    # Run the Laplace solver
    if run_flag:
        print("Running Laplace solver...")
        laplace_results_file = solver.run("bayer", outdir)
    else:
        laplace_results_file = os.path.join(outdir, 'result_001.vtu')

    # Initialize fiber generator
    print("\nGenerating fibers using Bayer method...")
    fib_gen = FibGenBayer()

    # Load Laplace results
    fib_gen.load_laplace_results(laplace_results_file)

    # Generate fiber directions
    F, S, T = fib_gen.generate_fibers_new(params)
    fib_gen.write_fibers(outdir)

    print(f"generate fibers (Bayer method) elapsed time: {time() - start:.3f} s")

    # Save the result mesh
    result_mesh_path = os.path.join(outdir, "results_bayer.vtu")
    fib_gen.mesh.save(result_mesh_path)
    print(fib_gen.mesh.cell_data['alfaS'][811948])
    print(f"\nResults saved to: {result_mesh_path}")
