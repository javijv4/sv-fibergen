#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import src.FibGen as fg
from time import time

run_flag = False
method = 'doste'
svfsi_exec = "svmultiphysics "

mesh_path = "example/ot/mesh-complete.mesh.vtu"
surfaces_dir = f"example/ot/mesh-surfaces"
outdir = "example/ot/output"

surface_names = {'epi': 'epi.vtp',
                 'epi_apex': 'epi_apex.vtp',    # New surface
                 'av': 'av.vtp',
                 'mv': 'mv.vtp',
                 'tv': 'tv.vtp',
                 'pv': 'pv.vtp',
                 'base': 'top.vtp',
                 'endo_lv': 'endo_lv.vtp',
                 'endo_rv': 'endo_rv.vtp'}

params = {
    # A = alpha angle
    'AENDORV' : 90,
    'AEPIRV' : -25,
    'AENDOLV' : 60,
    'AEPILV' : -60,

    # AOTENDOLV = 90
    'AOTENDOLV' : 0, 
    'AOTENDORV' : 90,
    'AOTEPILV' : 0,
    'AOTEPIRV' : 0,
    'ATRIENDO' : 0, 

    # B = beta angle
    'BENDORV' : 0,
    'BEPIRV' : 0,
    'BENDOLV' : 0,
    'BEPILV' : 0,
}

# Make sure the paths are full paths
mesh_path = os.path.abspath(mesh_path)
surfaces_dir = os.path.abspath(surfaces_dir)
outdir = os.path.abspath(outdir)

# Generate the apex surface
start = time()

start = time()
fg.generate_epi_apex(mesh_path, surfaces_dir, surface_names)

# Run the Laplace solver
if run_flag:
    if method == 'bayer':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
    elif method == 'doste':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_doste.xml")
    else:
        raise ValueError(f"Unknown method: {method}")
    laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
laplace_results_file = outdir + '/result_001.vtu'

# Generate the fiber directions
result_mesh = fg.generate_fibers_BiV_Doste_cells(outdir, laplace_results_file, params)

# print(f"generate fibers (new code) elapsed time: {time() - start:.3f} s")