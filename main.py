#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import src.FibGen as fg
from time import time

run_flag = True
method = 'bayer'
svfsi_exec = "svmultiphysics "

mesh_path = "example/truncated/VOLUME.vtu"
surfaces_dir = f"example/truncated/mesh-surfaces"
outdir = "example/truncated/output"

surface_names = {'epi': 'EPI.vtp',
                 'epi_apex': 'EPI_APEX2.vtp',    # New surface
                 'base': 'BASE.vtp',
                 'endo_lv': 'LV.vtp',
                 'endo_rv': 'RV.vtp'}

params = {
    "ALFA_END": 60.0,
    "ALFA_EPI": -60.0,
    "BETA_END": 20.0,
    "BETA_EPI": -20.0,
}

# Make sure the paths are full paths
mesh_path = os.path.abspath(mesh_path)
surfaces_dir = os.path.abspath(surfaces_dir)
outdir = os.path.abspath(outdir)

# # Generate the apex surface
start = time()
fg.generate_epi_apex(surfaces_dir, surface_names)


# Run the Laplace solver
run_flag = False
if run_flag:
    if method == 'bayer':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
    laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
laplace_results_file = outdir + '/result_020.vtu'

# Generate the fiber directions
result_mesh = fg.generate_fibers_BiV_Bayer_cells(laplace_results_file, params)


print(f"generate fibers elapsed: {time() - start:.3f} s")

########## NEW CODE FOR JJV APEX SURFACE ##########
start = time()
fg.generate_epi_apex_jjv(mesh_path, surfaces_dir, surface_names)

# Run the Laplace solver
run_flag = False
outdir = os.path.join(os.path.dirname(outdir), "output_jjv")
if run_flag:
    if method == 'bayer':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer_jjv.xml")
    laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
laplace_results_file = outdir + '/result_001.vtu'

# Generate the fiber directions
result_mesh_jjv = fg.generate_fibers_BiV_Bayer_cells_jjv(laplace_results_file, params)

print(f"generate fibers jjv elapsed: {time() - start:.3f} s")