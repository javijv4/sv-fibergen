#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Main script for generating biventricular fibers using the Doste method.

This module implements fiber generation for biventricular heart models using
the Laplace-Dirichlet rule-based method described in:
Doste et al. 2019, "A rule-based method to model myocardial fiber orientation
in cardiac biventricular geometries with outflow tracts"
https://doi.org/10.1002/cnm.3185

The script supports command-line arguments for customization of mesh paths,
output directories, and solver executables.
"""

import os
import argparse
import pyvista as pv
from src.LaplaceSolver import LaplaceSolver
from src.FibGen import FibGenDoste
from src.SurfaceNames import SurfaceName
from time import time

###########################################################
############  USER INPUTS  ################################
###########################################################

run_flag = False
svfsi_exec = "svmultiphysics "

mesh_path = "example/ot/mesh-complete.mesh.vtu"
surfaces_dir = None  # default computed from mesh_path below
outdir = "example/ot/output_d_oo"

surface_paths = {
    SurfaceName.EPI: 'example/ot/mesh-surfaces/epi.vtp',
    SurfaceName.EPI_APEX: 'example/ot/mesh-surfaces/epi_apex.vtp',
    SurfaceName.AV: 'example/ot/mesh-surfaces/av.vtp',
    SurfaceName.MV: 'example/ot/mesh-surfaces/mv.vtp',
    SurfaceName.TV: 'example/ot/mesh-surfaces/tv.vtp',
    SurfaceName.PV: 'example/ot/mesh-surfaces/pv.vtp',
    SurfaceName.ENDO_LV: 'example/ot/mesh-surfaces/endo_lv.vtp',
    SurfaceName.ENDO_RV: 'example/ot/mesh-surfaces/endo_rv.vtp'
}

# Parameters from the Doste paper https://doi.org/10.1002/cnm.3185
params = {
    # A = alpha angle
    'AENDORV': 90,
    'AEPIRV': -25,
    'AENDOLV': 60,
    'AEPILV': -60,

    'AOTENDOLV': 90, 
    'AOTENDORV': 90,
    'AOTEPILV': 0,
    'AOTEPIRV': 0,

    # B = beta angle (this have an opposite sign to the Doste paper, 
    # but it's because the longitudinal direction is opposite)
    'BENDORV': 0, #20,
    'BEPIRV': 0, #-20,
    'BENDOLV': 0, #20,
    'BEPILV': 0, #-20,
}


###########################################################
############  FIBER GENERATION  ###########################
###########################################################

# Optional CLI overrides
parser = argparse.ArgumentParser(description="Generate fibers using the Doste method (OO implementation).")
parser.add_argument("--svfsi-exec", default=svfsi_exec, help="svMultiPhysics executable/command (default: %(default)s)")
parser.add_argument("--mesh-path", default=mesh_path, help="Path to the volumetric mesh .vtu (default: %(default)s)")
parser.add_argument("--outdir", default=outdir, help="Output directory (default: %(default)s)")
args = parser.parse_args()

svfsi_exec = args.svfsi_exec
if not svfsi_exec.endswith(" "):
    svfsi_exec = svfsi_exec + " "

mesh_path = args.mesh_path
outdir = args.outdir

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
    laplace_results_file = solver.run("doste", outdir)
else:
    laplace_results_file = os.path.join(outdir, 'result_001.vtu')

# Initialize fiber generator
print("\nGenerating fibers using Doste method (OO)...")
fib_gen = FibGenDoste()

# Load Laplace results
fib_gen.load_laplace_results(laplace_results_file)

# Generate fiber directions
F, S, T = fib_gen.generate_fibers(params)

# Add fiber fields to the mesh
fib_gen.mesh.cell_data['fibersLong'] = F
fib_gen.mesh.cell_data['fibersSheet'] = S
fib_gen.mesh.cell_data['fibersNormal'] = T

print(f"generate fibers (Doste method OO) elapsed time: {time() - start:.3f} s")

# Save the result mesh
result_mesh_path = os.path.join(outdir, "results_doste.vtu")
fib_gen.mesh.save(result_mesh_path)
print(f"\nResults saved to: {result_mesh_path}")
