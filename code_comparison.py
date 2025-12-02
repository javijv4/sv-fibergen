#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import src.FibGen as fg
import src.FibGenOld as fg_old
from time import time

run_flag = True
method = 'bayer'
svfsi_exec = "svmultiphysics "

mesh_path = "example/truncated/VOLUME.vtu"
surfaces_dir = f"example/truncated/mesh-surfaces"
outdir = "example/truncated/output"

surface_names = {'epi': 'EPI.vtp',
                 'epi_apex': 'EPI_APEX.vtp',    # New surface
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

# Generate the apex surface
# start = time()
# fg_old.generate_epi_apex(surfaces_dir, surface_names)


# ########## OLD CODE  ##########
# run_flag = False
# outdir = os.path.join(os.path.dirname(outdir), "output_old")
# if run_flag:
#     if method == 'bayer':
#         template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer_old.xml")
#     laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
# laplace_results_file = outdir + '/result_020.vtu'

# # # Generate the fiber directions
# # result_mesh = fg_old.generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params)
# result_mesh = fg_old.generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params, fix=True)


# print(f"generate fibers (old code) elapsed time: {time() - start:.3f} s")

########## NEW CODE  ##########
start = time()
fg.generate_epi_apex(mesh_path, surfaces_dir, surface_names)

# Run the Laplace solver
run_flag = False
outdir = os.path.join(os.path.dirname(outdir), "output")
if run_flag:
    if method == 'bayer':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
    laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
laplace_results_file = outdir + '/result_001.vtu'

# Generate the fiber directions
result_mesh = fg.generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params, return_angles=True, return_intermediate=True)

result_mesh.save(os.path.join(outdir, "check.vtu"))

print(f"generate fibers (new code) elapsed time: {time() - start:.3f} s")

#%% Load results and compare
import meshio as io
import numpy as np

mesh = io.read(os.path.join(outdir, f"fibersLong.vtu"))
new_mesh = io.Mesh(points=mesh.points, cells=mesh.cells)

family = 'Long'

cell_data={}
for family in ['Long', 'Sheet', 'Normal']:
    outdir = os.path.join(os.path.dirname(outdir), "output")
    fibers_new = io.read(os.path.join(outdir, f"fibers{family}.vtu"))
    fibers_new = fibers_new.cell_data['FIB_DIR'][0]

    outdir = os.path.join(os.path.dirname(outdir), "output_old")
    fibers_old = io.read(os.path.join(outdir, f"fibers{family}.vtu"))
    fibers_old = fibers_old.cell_data['FIB_DIR'][0]

    outdir = os.path.join(os.path.dirname(outdir), "output_old_fix")
    fibers_old_fix = io.read(os.path.join(outdir, f"fibers{family}.vtu"))
    fibers_old_fix = fibers_old_fix.cell_data['FIB_DIR'][0]

    # per-row dot products, absolute values, then L2 norms of those absolute values
    dots_old = np.einsum('ij,ij->i', fibers_new, fibers_old)
    dots_old_fix = np.einsum('ij,ij->i', fibers_new, fibers_old_fix)

    abs_dots_old = np.clip(np.abs(dots_old), 0.0, 1.0)
    abs_dots_old_fix = np.clip(np.abs(dots_old_fix), 0.0, 1.0)

    angles_old = np.rad2deg(np.arccos(abs_dots_old))
    angles_old_fix = np.rad2deg(np.arccos(abs_dots_old_fix))

    norm_old = np.linalg.norm(angles_old)
    norm_old_fix = np.linalg.norm(angles_old_fix)

    print(f"{family} norm(|dot(fibers_new, fibers_old)|) = {norm_old}")
    print(f"{family} norm(|dot(fibers_new, fibers_old_fix)|) = {norm_old_fix}")
        

    cell_data.update({f"{family}_DIFF_OLD": [angles_old], 
                f"{family}_ANGLE_DIFF_OLD_FIX": [angles_old_fix],
                f"{family}_FIB_NEW": [fibers_new],
                f"{family}_FIB_OLD": [fibers_old],
                f"{family}_FIB_OLD_FIX": [fibers_old_fix],
                })

new_mesh.cell_data.update(cell_data)
io.write('check.vtu', new_mesh)

