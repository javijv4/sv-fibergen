#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:43:23

@author: Javiera Jilberto Vallejos 
'''

import os
import sys
import re
import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util import numpy_support as vtknp
import time
import copy



def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid


def generate_epi_apex_jjv(mesh_path, surfaces_dir, surface_names):
    '''
    Generate the epi apex and epi mid surfaces from the epi surface of the BiV.
    
    Parameters:
    -----------
    surfaces_dir : str
        Directory containing surface meshes
    surface_names : list of str
        List of surface mesh filenames
    '''
    
    # Load the epi surface
    epi_name = os.path.join(surfaces_dir, surface_names['epi'])
    epi_mesh = pv.read(epi_name)
    epi_points = epi_mesh.points
    epi_cells = epi_mesh.faces
    epi_eNoN = epi_cells[0]
    epi_cells = epi_cells.reshape((-1, epi_eNoN + 1))
    epi_cells = epi_cells[:, 1:]
    epi_global_node_id = epi_mesh.point_data['GlobalNodeID']
    epi_global_cell_id = epi_mesh.cell_data['GlobalElementID']

    # Load the base surface
    base_name = os.path.join(surfaces_dir, surface_names['base'])
    base_mesh = pv.read(base_name)
    base_global_node_id = base_mesh.point_data['GlobalNodeID']

    # Extract the boundary of the epi surface (at the top) to find the apex point
    epi_base_global_node_id = np.intersect1d(epi_global_node_id, base_global_node_id)
    epi_base_nodes = np.where(np.isin(epi_global_node_id, epi_base_global_node_id))[0]

    # # Get normal
    base_normal, base_centroid = get_normal_plane_svd(epi_points[epi_base_nodes, :])

    # Find the index of the apex point of the epi surface
    distance = np.abs(base_normal@(epi_points - base_centroid).T)
    epi_apex_point_index = np.argmax(distance)

    # Find elements containing the apex point
    epi_apex_cell_index = np.where(epi_cells == epi_apex_point_index)[0]

    # Create epi_apex mesh
    submesh_cells = epi_cells[epi_apex_cell_index]
    submesh_xyz = np.zeros([len(np.unique(submesh_cells)), epi_points.shape[1]])
    map_mesh_submesh = np.ones(epi_points.shape[0], dtype=int)*-1
    map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
    child_elems_new = np.zeros(submesh_cells.shape, dtype=int)

    cont = 0
    for e in range(submesh_cells.shape[0]):
        for i in range(submesh_cells.shape[1]):
            if map_mesh_submesh[submesh_cells[e,i]] == -1:
                child_elems_new[e,i] = cont
                submesh_xyz[cont] = epi_points[submesh_cells[e,i]]
                map_mesh_submesh[submesh_cells[e,i]] = cont
                map_submesh_mesh[cont] = submesh_cells[e,i]
                cont += 1
            else:
                child_elems_new[e,i] = map_mesh_submesh[submesh_cells[e,i]]

    epi_apex_cells_type = np.full((child_elems_new.shape[0], 1), epi_eNoN)
    epi_apex_cells = np.hstack((epi_apex_cells_type, child_elems_new))
    epi_apex_cells = np.hstack(epi_apex_cells) 

    # Get global IDs
    epi_apex_global_node_id = epi_global_node_id[map_submesh_mesh]
    epi_apex_global_cell_id = epi_global_cell_id[epi_apex_cell_index]

    # Create and save mesh
    epi_apex_mesh = pv.PolyData(submesh_xyz, epi_apex_cells)
    epi_apex_mesh.point_data.set_array(epi_apex_global_node_id, 'GlobalNodeID')
    epi_apex_mesh.cell_data.set_array(epi_apex_global_cell_id, 'GlobalElementID')

    epi_apex_name = os.path.join(surfaces_dir, surface_names['epi_apex'])
    epi_apex_mesh.save(epi_apex_name)


def loadLaplaceSoln_jjv(fileName):
    '''
    Load a solution to a Laplace-Dirichlet problem from a .vtu file and extract
    the solution and its gradients at the cells.

    ARGS:
    fileName : str
        Path to the .vtu file with the Laplace solution. The solution should be
        defined at the nodes. The Laplace fields should be named as follows:
        - Phi_BiV_EPI: Laplace field for the endocardium
        - Phi_BiV_LV: Laplace field for the left ventricle
        - Phi_BiV_RV: Laplace field for the right ventricle
        - Phi_BiV_AB: Laplace field for the apex to base direction
    '''

    DATASTR1 = 'Phi_BiV_EPI'
    DATASTR2 = 'Phi_BiV_LV'
    DATASTR3 = 'Phi_BiV_RV'
    DATASTR4 = 'Phi_BiV_AB'

    print("   Loading Laplace solution <---   %s" % (fileName))

    # Read mesh with pyvista
    result_mesh = pv.read(fileName)

    # Convert point-data to cell-data (keep point data passed to cells)
    mesh_cells = result_mesh.point_data_to_cell_data()

    print("   Extracting solution and estimating gradients at cells")

    # Get cell centers (Nx3) and scalar cell arrays (N,)
    cPhiEP = np.asarray(mesh_cells.cell_data[DATASTR1])
    cPhiLV = np.asarray(mesh_cells.cell_data[DATASTR2])
    cPhiRV = np.asarray(mesh_cells.cell_data[DATASTR3])
    cPhiAB = np.asarray(mesh_cells.cell_data[DATASTR4])

    # Use pyvista's compute_derivative to get cell gradients
    # compute_derivative will add arrays named '<DATA>_grad' to the cell_data
    grad_mesh = mesh_cells.compute_derivative(scalars=DATASTR1, gradient=True, preference='cell')
    cGPhiEP = np.asarray(grad_mesh.cell_data['gradient'])

    grad_mesh = mesh_cells.compute_derivative(scalars=DATASTR2, gradient=True, preference='cell')
    cGPhiLV = np.asarray(grad_mesh.cell_data['gradient'])

    grad_mesh = mesh_cells.compute_derivative(scalars=DATASTR3, gradient=True, preference='cell')
    cGPhiRV = np.asarray(grad_mesh.cell_data['gradient'])

    grad_mesh = mesh_cells.compute_derivative(scalars=DATASTR4, gradient=True, preference='cell')
    cGPhiAB = np.asarray(grad_mesh.cell_data['gradient'])

    # Use the mesh with cell-data (but without the large scalar arrays) as result_mesh
    mesh_cells.cell_data[DATASTR1 + '_grad'] = cGPhiEP
    mesh_cells.cell_data[DATASTR2 + '_grad'] = cGPhiLV
    mesh_cells.cell_data[DATASTR3 + '_grad'] = cGPhiRV
    mesh_cells.cell_data[DATASTR4 + '_grad'] = cGPhiAB

    return mesh_cells, cPhiEP, cPhiLV, cPhiRV, cPhiAB, \
        cGPhiEP, cGPhiLV, cGPhiRV, cGPhiAB


def bislerp_jjv(Q1, Q2, interp_func):
    """
    Vectorized spherical interpolation between batches of rotation matrices.
    Q1, Q2: (N, 3, 3)
    interp_func: (N,) values in [0,1]
    Returns Q: (N, 3, 3)
    Notes:
      - Uses wxyz quaternion convention internally, matching quat2rot below.
      - Avoids per-element Python/Scipy objects for performance.
    """
    def rotm_to_quat_batch(R):
        # R: (N,3,3) -> q: (N,4) [w,x,y,z]
        t = np.einsum('nii->n', R)  # trace
        q = np.zeros((R.shape[0], 4), dtype=float)
        # Branch where trace is positive
        mask_t = t > 0.0
        if np.any(mask_t):
            S = np.sqrt(t[mask_t] + 1.0) * 2.0
            q[mask_t, 0] = 0.25 * S
            q[mask_t, 1] = (R[mask_t, 2, 1] - R[mask_t, 1, 2]) / S
            q[mask_t, 2] = (R[mask_t, 0, 2] - R[mask_t, 2, 0]) / S
            q[mask_t, 3] = (R[mask_t, 1, 0] - R[mask_t, 0, 1]) / S
        # For remaining, choose major diagonal
        mask_f = ~mask_t
        if np.any(mask_f):
            Rf = R[mask_f]
            m00 = Rf[:, 0, 0]
            m11 = Rf[:, 1, 1]
            m22 = Rf[:, 2, 2]
            idx = np.argmax(np.stack([m00, m11, m22], axis=1), axis=1)
            # Case idx==0
            m0 = idx == 0
            if np.any(m0):
                S = np.sqrt(1.0 + Rf[m0, 0, 0] - Rf[m0, 1, 1] - Rf[m0, 2, 2]) * 2.0
                q[mask_f][m0, 0] = (Rf[m0, 2, 1] - Rf[m0, 1, 2]) / S
                q[mask_f][m0, 1] = 0.25 * S
                q[mask_f][m0, 2] = (Rf[m0, 0, 1] + Rf[m0, 1, 0]) / S
                q[mask_f][m0, 3] = (Rf[m0, 0, 2] + Rf[m0, 2, 0]) / S
            # Case idx==1
            m1 = idx == 1
            if np.any(m1):
                S = np.sqrt(1.0 + Rf[m1, 1, 1] - Rf[m1, 0, 0] - Rf[m1, 2, 2]) * 2.0
                q[mask_f][m1, 0] = (Rf[m1, 0, 2] - Rf[m1, 2, 0]) / S
                q[mask_f][m1, 1] = (Rf[m1, 0, 1] + Rf[m1, 1, 0]) / S
                q[mask_f][m1, 2] = 0.25 * S
                q[mask_f][m1, 3] = (Rf[m1, 1, 2] + Rf[m1, 2, 1]) / S
            # Case idx==2
            m2 = idx == 2
            if np.any(m2):
                S = np.sqrt(1.0 + Rf[m2, 2, 2] - Rf[m2, 0, 0] - Rf[m2, 1, 1]) * 2.0
                q[mask_f][m2, 0] = (Rf[m2, 1, 0] - Rf[m2, 0, 1]) / S
                q[mask_f][m2, 1] = (Rf[m2, 0, 2] + Rf[m2, 2, 0]) / S
                q[mask_f][m2, 2] = (Rf[m2, 1, 2] + Rf[m2, 2, 1]) / S
                q[mask_f][m2, 3] = 0.25 * S
        # Normalize for numerical safety
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q

    def quat_to_rotm_batch(q):
        # q: (N,4) [w,x,y,z] -> R: (N,3,3)
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        R = np.zeros((q.shape[0], 3, 3), dtype=float)
        R[:, 0, 0] = 1.0 - 2.0 * y2 - 2.0 * z2
        R[:, 1, 0] = 2.0 * xy + 2.0 * wz
        R[:, 2, 0] = 2.0 * xz - 2.0 * wy
        R[:, 0, 1] = 2.0 * xy - 2.0 * wz
        R[:, 1, 1] = 1.0 - 2.0 * x2 - 2.0 * z2
        R[:, 2, 1] = 2.0 * yz + 2.0 * wx
        R[:, 0, 2] = 2.0 * xz + 2.0 * wy
        R[:, 1, 2] = 2.0 * yz - 2.0 * wx
        R[:, 2, 2] = 1.0 - 2.0 * x2 - 2.0 * y2
        return R

    # Prepare inputs
    t = np.clip(np.asarray(interp_func, dtype=float), 0.0, 1.0)
    q1 = rotm_to_quat_batch(np.asarray(Q1, dtype=float))
    q2 = rotm_to_quat_batch(np.asarray(Q2, dtype=float))

    # Ensure shortest path on the unit 4-sphere
    dot = np.sum(q1 * q2, axis=1)
    neg_mask = dot < 0.0
    if np.any(neg_mask):
        q2[neg_mask] = -q2[neg_mask]
        dot[neg_mask] = -dot[neg_mask]

    # SLERP weights
    dot_clipped = np.clip(dot, -1.0, 1.0)
    theta0 = np.arccos(dot_clipped)
    sin_theta0 = np.sin(theta0)

    # Threshold for linear interpolation
    lin_mask = sin_theta0 < 1e-6
    q = np.empty_like(q1)

    if np.any(~lin_mask):
        theta = theta0[~lin_mask] * t[~lin_mask]
        s0 = np.sin(theta0[~lin_mask] - theta) / sin_theta0[~lin_mask]
        s1 = np.sin(theta) / sin_theta0[~lin_mask]
        q[~lin_mask] = (s0[:, None] * q1[~lin_mask]) + (s1[:, None] * q2[~lin_mask])

    if np.any(lin_mask):
        # Nearly identical orientations: keep source orientation (avoid linear interpolation)
        q[lin_mask] = q1[lin_mask]

    # Normalize and convert back to rotation matrices
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quat_to_rotm_batch(q)

def axis_jjv(u, v):
    """
    u, v: (nelems, 3)
    return Q: (nelems, 3, 3) where columns are [e0, e1, e2] per element
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ne = u.shape[0]

    # e1 = normalize rows of u
    e1 = u/np.linalg.norm(u, axis=1)[:, None]

    # e2 = v - proj_{e1}(v)
    proj = np.sum(e1 * v, axis=1)[:, None] * e1
    e2 = v - proj
    e2 = e2 / np.linalg.norm(e2, axis=1)[:, None]

    # e0 = cross(e1, e2) normalized
    e0 = np.cross(e1, e2, axisa=1, axisb=1)
    e0 = e0 / np.linalg.norm(e0, axis=1)[:, None]

    Q = np.zeros((ne, 3, 3), dtype=float)
    Q[:, :, 0] = e0
    Q[:, :, 1] = e1
    Q[:, :, 2] = e2

    return Q

def orient_jjv(Q, alpha, beta):
    """
    Given an orthogonal matrix Q (ne,3,3), rotate each Q[i] by alpha[i] about
    the z-axis and then by beta[i] about the x-axis. alpha and beta are arrays 
    of length ne.
    """
    Q = np.asarray(Q, dtype=float)
    ne = Q.shape[0]

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    # Rotation about z (Ra) and x (Rb) for each element
    Ra = np.zeros((ne, 3, 3), dtype=float)
    Ra[:, 0, 0] = ca
    Ra[:, 0, 1] = -sa
    Ra[:, 1, 0] = sa
    Ra[:, 1, 1] = ca
    Ra[:, 2, 2] = 1.0

    Rb = np.zeros((ne, 3, 3), dtype=float)
    Rb[:, 0, 0] = 1.0
    Rb[:, 1, 1] = cb
    Rb[:, 1, 2] = sb
    Rb[:, 2, 1] = -sb
    Rb[:, 2, 2] = cb

    # Compose rotations and apply to Q per element
    RaRb = np.einsum('nij,njk->nik', Ra, Rb)
    Qt = np.einsum('nij,njk->nik', Q, RaRb)

    return Qt

def getFiberDirections_jjv(Phi_EP, Phi_LV, Phi_RV, \
                        gPhi_EP, gPhi_LV, gPhi_RV, gPhi_AB, \
                        params):
    '''
    Compute the fiber directions at the center of each cell
    '''

    # Unpack parameters
    ALFA_END = params["ALFA_END"]
    ALFA_EPI = params["ALFA_EPI"]
    BETA_END = params["BETA_END"]
    BETA_EPI = params["BETA_EPI"]

    print("   Computing fiber directions at cells")

    d = Phi_RV / (Phi_LV + Phi_RV)
    alfaS = ALFA_END * (1 - d) - ALFA_END * d
    betaS = BETA_END * (1 - d) - BETA_END * d
    alfaW = ALFA_END * (1 - Phi_EP) + ALFA_EPI * Phi_EP
    betaW = BETA_END * (1 - Phi_EP) + BETA_EPI * Phi_EP
    Q_LV = axis_jjv(gPhi_AB, -gPhi_LV)
    Q_LV = orient_jjv(Q_LV, alfaS, betaS)

    Q_RV = axis_jjv(gPhi_AB, gPhi_RV)
    Q_RV = orient_jjv(Q_RV, alfaS, betaS)

    Q_END = bislerp_jjv(Q_LV, Q_RV, d)
    Q_END[d > 0.5,:,0] = -Q_END[d > 0.5,:,0]
    Q_END[d > 0.5,:,2] = -Q_END[d > 0.5,:,2]

    Q_EPI = axis_jjv(gPhi_AB, gPhi_EP)
    Q_EPI = orient_jjv(Q_EPI, alfaW, betaW)

    FST = bislerp_jjv(Q_END, Q_EPI, Phi_EP)

    F = FST[:, :, 0]
    S = FST[:, :, 1]
    T = FST[:, :, 2]
    

    return F, S, T

def generate_fibers_BiV_Bayer_cells_jjv(laplace_results_file, params):
    '''
    Generate fiber directions on a truncated BiV ventricular geometry using the
    Laplace-Dirichlet rule-based method of Bayer et al. 2012

    ARGS:
    laplace_results_file : str
        Path to the .vtu mesh with Laplace fields defined at nodes
    params : dict
        Dictionary of parameters for fiber generation
    '''
    
    t1 = time.time()
    print("========================================================")

    # Get directory of mesh (Laplace solution is in {mesh_dir}/lps_thickness/result_020.vtu)
    # So we go up one level from lps_thickness to get mesh_dir
    mesh_dir = os.path.dirname(os.path.dirname(laplace_results_file))
    
    result_mesh, Phi_EPI, Phi_LV, Phi_RV, Phi_AB, \
    gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB = loadLaplaceSoln_jjv(laplace_results_file)


    # Generate fiber directions
    F, S, T = getFiberDirections_jjv(Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 params)
    
    result_mesh.cell_data.set_array(F, 'F')
    result_mesh.cell_data.set_array(S, 'S')
    result_mesh.cell_data.set_array(T, 'T')

    # Save a copy of the mesh with Laplace fields for inspection
    check_fname = os.path.join(mesh_dir, "check_jjv.vtu")
    print(f"   Writing check file   --->   {check_fname}")
    result_mesh.save(check_fname)

    # print("   Writing domains and fibers to VTK data structure")

    # # Write the fiber directions to a vtu files
    # output_mesh = copy.deepcopy(result_mesh)
    # #output_mesh.cell_data.remove('Proc_ID')
    # #output_mesh.point_data.remove(DATASTR1)
    # #output_mesh.point_data.remove(DATASTR2)
    # #output_mesh.point_data.remove(DATASTR3)
    # #output_mesh.point_data.remove(DATASTR4)

    # fname1 = os.path.join(mesh_dir, "fibersLong.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.set_array(F, 'FIB_DIR')
    # output_mesh.save(fname1)

    # fname1 = os.path.join(mesh_dir, "fibersSheet.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.remove('FIB_DIR')
    # output_mesh.cell_data.set_array(T, 'FIB_DIR')
    # output_mesh.save(fname1)

    # fname1 = os.path.join(mesh_dir, "fibersNormal.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.remove('FIB_DIR')
    # output_mesh.cell_data.set_array(S, 'FIB_DIR')
    # output_mesh.save(fname1)

    # t2 = time.time()
    # print('\n   Total time: %.3fs' % (t2-t1))
    # print("========================================================")

#################### OLD FUNCTION BELOW ####################

def distance_to_surface(point, surface_normal, surface_point):
    '''
    Calculate the distance from a point to a point on a surface
    '''
    # Ensure that the input vectors are NumPy arrays
    point = np.array(point)
    surface_normal = np.array(surface_normal)
    surface_point = np.array(surface_point)

    # Calculate the vector from the surface point to the target point
    v = point - surface_point

    # Calculate the dot product between the vector v and the surface normal
    dot_product = np.dot(v, surface_normal)

    # Calculate the absolute value of the dot product to get the distance
    distance = abs(dot_product)

    return distance

def generate_epi_apex(surfaces_dir, surface_names):
    '''
    Generate the epi apex and epi mid surfaces from the epi surface of the BiV.
    
    Parameters:
    -----------
    surfaces_dir : str
        Directory containing surface meshes
    surface_names : list of str
        List of surface mesh filenames
    '''

    # Generate epi_apex and epi_mid from epi surface
    # top_name = os.path.join(surfaces_dir, "top.vtp")
    # top_mesh = pv.read(top_name)
    # top_mesh = top_mesh.compute_normals()
    # #top_normal_rep = top_mesh['Normals'][0, :]
    # top_normal_rep = np.mean(top_mesh['Normals'], axis=0)
    # top_normal_rep = top_normal_rep / np.linalg.norm(top_normal_rep)
    # #top_point_rep = top_mesh.points[0, :]
    # top_point_rep = np.mean(top_mesh.points, axis=0)
    
    # Load the epi surface
    epi_name = os.path.join(surfaces_dir, surface_names['epi'])
    epi_mesh = pv.read(epi_name)
    epi_points = epi_mesh.points
    epi_cells = epi_mesh.faces
    epi_global_node_id = epi_mesh.point_data['GlobalNodeID']
    epi_global_cell_id = epi_mesh.cell_data['GlobalElementID']
    epi_eNoN = epi_cells[0]
    epi_cells = epi_cells.reshape((-1, epi_eNoN + 1))
    epi_cells = epi_cells[:, 1:]

    # Extract the boundary of the epi surface (at the top) to find the apex point
    epi_boundary = epi_mesh.extract_feature_edges(non_manifold_edges=False, feature_edges=False, manifold_edges=False)

    # Triangulate the boundary
    epi_boundary_triangulated = epi_boundary.delaunay_2d()

    # Compute the center and average normal vector of the triangulated boundary
    top_point_rep = np.mean(epi_boundary_triangulated.points, axis=0)
    epi_boundary_triangulated = epi_boundary_triangulated.compute_normals()
    top_normal_rep = np.mean(epi_boundary_triangulated['Normals'], axis=0)
    top_normal_rep = top_normal_rep / np.linalg.norm(top_normal_rep)

    # # Show the normal vector and centroid
    # p = pv.Plotter()
    # p.add_mesh(epi_boundary_triangulated, color='white', show_edges=True)
    # p.add_arrows(top_point_rep, top_normal_rep, mag=20, color='r')
    # p.show()

    # Find the index of the apex point of the epi surface
    epi_apex_point_index = 0
    for i in range(epi_points.shape[0]):
        epi_point_dis_to_top = distance_to_surface(epi_points[i, :], top_normal_rep, top_point_rep)
        if i == 0:
            epi_apex_point_dis = epi_point_dis_to_top
        elif epi_point_dis_to_top > epi_apex_point_dis:
            epi_apex_point_index = i
            epi_apex_point_dis = epi_point_dis_to_top

    # # Generate epi_apex mesh
    epi_apex_cell_index = np.where(epi_cells == epi_apex_point_index)[0]
    epi_apex_name = os.path.join(surfaces_dir, "epi_apex.vtp")
    epi_apex_cells_surf = epi_cells[epi_apex_cell_index, :]
    epi_apex_cells = np.zeros_like(epi_apex_cells_surf)
    epi_apex_points_index = np.unique(np.hstack(epi_apex_cells_surf))
    epi_apex_points = epi_points[epi_apex_points_index, :]
    for j in range(epi_apex_cells_surf.shape[0]):
        for k in range(epi_apex_cells_surf.shape[1]):
            temp_index = np.where(epi_apex_points_index == epi_apex_cells_surf[j, k])
            temp_index = temp_index[0]
            epi_apex_cells[j, k] = temp_index

    epi_apex_cells_type = np.full((epi_apex_cells.shape[0], 1), epi_eNoN)
    epi_apex_cells = np.hstack((epi_apex_cells_type, epi_apex_cells))
    epi_apex_cells = np.hstack(epi_apex_cells)

    epi_apex_global_node_id = epi_global_node_id[epi_apex_points_index]
    epi_apex_global_cell_id = epi_global_cell_id[epi_apex_cell_index]

    epi_apex_mesh = pv.PolyData(epi_apex_points, epi_apex_cells)

    epi_apex_mesh.point_data.set_array(epi_apex_global_node_id, 'GlobalNodeID')
    epi_apex_mesh.cell_data.set_array(epi_apex_global_cell_id, 'GlobalElementID')

    epi_apex_mesh.save(epi_apex_name)

    # # Generate epi_mid mesh (epi_mid = epi - epi_apex)
    # epi_mid_cell_index = np.where(~np.any(epi_cells == epi_apex_point_index, axis=1))[0]
    # epi_mid_name = os.path.join(surfaces_dir, "epi_mid.vtp")
    # epi_mid_cells_surf = epi_cells[epi_mid_cell_index, :]
    # epi_mid_cells = np.zeros_like(epi_mid_cells_surf)
    # epi_mid_points_index = np.unique(np.hstack(epi_mid_cells_surf))
    # epi_mid_points = epi_points[epi_mid_points_index, :]
    # for j in range(epi_mid_cells_surf.shape[0]):
    #     for k in range(epi_mid_cells_surf.shape[1]):
    #         temp_index = np.where(epi_mid_points_index == epi_mid_cells_surf[j, k])
    #         temp_index = temp_index[0]
    #         epi_mid_cells[j, k] = temp_index

    # epi_mid_cells_type = np.full((epi_mid_cells.shape[0], 1), epi_eNoN)
    # epi_mid_cells = np.hstack((epi_mid_cells_type, epi_mid_cells))
    # epi_mid_cells = np.hstack(epi_mid_cells)

    # epi_mid_global_node_id = epi_global_node_id[epi_mid_points_index]
    # epi_mid_global_cell_id = epi_global_cell_id[epi_mid_cell_index]

    # epi_mid_mesh = pv.PolyData(epi_mid_points, epi_mid_cells)

    # epi_mid_mesh.point_data.set_array(epi_mid_global_node_id, 'GlobalNodeID')
    # epi_mid_mesh.cell_data.set_array(epi_mid_global_cell_id, 'GlobalElementID')

    # epi_mid_mesh.save(epi_mid_name)


def runLaplaceSolver(mesh_dir, surfaces_dir, mesh_file, exec_svfsi, template_file, outdir, surface_names):
    xml_template_path = template_file
    out_name = os.path.join(surfaces_dir, "../svFSI_BiV.xml")
    
    with open(xml_template_path, 'r') as svFile:
        xml_content = svFile.read()
    
    # Update mesh file path using regex
    mesh_pattern = r'(<Mesh_file_path>)\s+[^\s<]+[^<]*(</Mesh_file_path>)'
    xml_content = re.sub(mesh_pattern, r'\1 ' + mesh_file + r' \2', xml_content)
    
    # Update face file paths - need to identify which face by checking context
    # Read lines to determine context
    lines = xml_content.split('\n')
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has a face name
        face_match = re.search(r'name="([^"]+)"', line)
        face_name = face_match.group(1) if face_match else None
        
        # Look ahead for Face_file_path
        if face_name and i + 1 < len(lines) and "<Face_file_path>" in lines[i + 1]:
            # Determine which file to use based on face name
            if face_name == "epi":
                new_path = os.path.join(surfaces_dir, surface_names['epi'])
            elif face_name == "epi_top":
                new_path = os.path.join(surfaces_dir, surface_names['base'])
            elif face_name == "epi_apex":
                new_path = os.path.join(surfaces_dir, surface_names['epi_apex'])
            elif face_name == "endo_lv":
                new_path = os.path.join(surfaces_dir, surface_names['endo_lv'])
            elif face_name == "endo_rv":
                new_path = os.path.join(surfaces_dir, surface_names['endo_rv'])
            else:
                new_path = None
            
            if new_path:
                # Add current line
                updated_lines.append(line)
                # Replace the path in the next line
                i += 1
                face_pattern = r'(<Face_file_path>)\s+[^\s<]+[^<]*(</Face_file_path>)'
                updated_line = re.sub(face_pattern, r'\1 ' + new_path + r' \2', lines[i])
                updated_lines.append(updated_line)
                i += 1
                continue
        
        # Add line as-is
        updated_lines.append(line)
        i += 1
    
    xml_content = '\n'.join(updated_lines)
    
    # Update save results folder using regex
    save_pattern = r'(<Save_results_in_folder>)\s+[^\s<]+[^<]*(</Save_results_in_folder>)'
    xml_content = re.sub(save_pattern, r'\1 ' + outdir + r' \2', xml_content)

    with open(out_name, 'w') as svFileNew:
        svFileNew.write(xml_content)

    print("   Running svFSI solver")
    print(f"   {exec_svfsi + out_name}")
    os.system(exec_svfsi + out_name)

    return outdir + '/results_001.vtu'


def loadLaplaceSoln(fileName):
    '''
    Load a solution to a Laplace-Dirichlet problem from a .vtu file and extract
    the solution and its gradients at the cells.

    ARGS:
    fileName : str
        Path to the .vtu file with the Laplace solution. The solution should be
        defined at the nodes. The Laplace fields should be named as follows:
        - Phi_BiV_EPI: Laplace field for the endocardium
        - Phi_BiV_LV: Laplace field for the left ventricle
        - Phi_BiV_RV: Laplace field for the right ventricle
        - Phi_BiV_AB: Laplace field for the apex to base direction
    '''

    DATASTR1 = 'Phi_BiV_EPI'
    DATASTR2 = 'Phi_BiV_LV'
    DATASTR3 = 'Phi_BiV_RV'
    DATASTR4 = 'Phi_BiV_AB'

    print("   Loading Laplace solution   <---   %s" % (fileName))
    vtuReader = vtk.vtkXMLUnstructuredGridReader()
    vtuReader.SetFileName(fileName)
    vtuReader.Update()

    result_mesh = pv.read(fileName)

    print("   Extracting solution and its gradients at cells")

    pt2Cell = vtk.vtkPointDataToCellData()
    pt2Cell.SetInputConnection(vtuReader.GetOutputPort())
    pt2Cell.PassPointDataOn()

    gradFilter = vtk.vtkGradientFilter()
    gradFilter.SetInputConnection(pt2Cell.GetOutputPort())

    print(f"      Reading {DATASTR1} into cPhiEP") 
    gradFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,\
        DATASTR1)
    gradFilter.SetResultArrayName(DATASTR1 + '_grad')
    gradFilter.Update()
    vtuMesh = gradFilter.GetOutput()
    cPhiEP  = vtuMesh.GetCellData().GetArray(DATASTR1)
    cGPhiEP = vtuMesh.GetCellData().GetArray(DATASTR1+'_grad')

    print(f"      Reading {DATASTR2} into cPhiLV")
    gradFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,\
        DATASTR2)
    gradFilter.SetResultArrayName(DATASTR2 + '_grad')
    gradFilter.Update()
    vtuMesh = gradFilter.GetOutput()
    cPhiLV  = vtuMesh.GetCellData().GetArray(DATASTR2)
    cGPhiLV = vtuMesh.GetCellData().GetArray(DATASTR2+'_grad')

    print(f"      Reading {DATASTR3} into cPhiRV")
    gradFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,\
        DATASTR3)
    gradFilter.SetResultArrayName(DATASTR3 + '_grad')
    gradFilter.Update()
    vtuMesh = gradFilter.GetOutput()
    cPhiRV  = vtuMesh.GetCellData().GetArray(DATASTR3)
    cGPhiRV = vtuMesh.GetCellData().GetArray(DATASTR3+'_grad')

    print(f"      Reading {DATASTR4} into cPhiAB")
    gradFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,\
        DATASTR4)
    gradFilter.SetResultArrayName(DATASTR4 + '_grad')
    gradFilter.Update()
    vtuMesh = gradFilter.GetOutput()
    cPhiAB  = vtuMesh.GetCellData().GetArray(DATASTR4)
    cGPhiAB = vtuMesh.GetCellData().GetArray(DATASTR4+'_grad')

    # Clean unnecessary arrays
    vtuMesh.GetPointData().RemoveArray(DATASTR1)
    vtuMesh.GetCellData().RemoveArray(DATASTR1)
    vtuMesh.GetCellData().RemoveArray(DATASTR1+'_grad')

    vtuMesh.GetPointData().RemoveArray(DATASTR2)
    vtuMesh.GetCellData().RemoveArray(DATASTR2)
    vtuMesh.GetCellData().RemoveArray(DATASTR2+'_grad')

    vtuMesh.GetPointData().RemoveArray(DATASTR3)
    vtuMesh.GetCellData().RemoveArray(DATASTR3)
    vtuMesh.GetCellData().RemoveArray(DATASTR3+'_grad')

    vtuMesh.GetPointData().RemoveArray(DATASTR4)
    vtuMesh.GetCellData().RemoveArray(DATASTR4)
    vtuMesh.GetCellData().RemoveArray(DATASTR4+'_grad')

    cPhiEP = vtknp.vtk_to_numpy(cPhiEP)
    cPhiLV = vtknp.vtk_to_numpy(cPhiLV)
    cPhiRV = vtknp.vtk_to_numpy(cPhiRV)
    cPhiAB = vtknp.vtk_to_numpy(cPhiAB)
    cGPhiEP = vtknp.vtk_to_numpy(cGPhiEP)
    cGPhiLV = vtknp.vtk_to_numpy(cGPhiLV)
    cGPhiRV = vtknp.vtk_to_numpy(cGPhiRV)
    cGPhiAB = vtknp.vtk_to_numpy(cGPhiAB)

    # Use the mesh with cell-data (but without the large scalar arrays) as result_mesh
    result_mesh.cell_data[DATASTR1 + '_grad'] = cGPhiEP
    result_mesh.cell_data[DATASTR2 + '_grad'] = cGPhiLV
    result_mesh.cell_data[DATASTR3 + '_grad'] = cGPhiRV
    result_mesh.cell_data[DATASTR4 + '_grad'] = cGPhiAB

    return result_mesh, cPhiEP, cPhiLV, cPhiRV, cPhiAB, \
        cGPhiEP, cGPhiLV, cGPhiRV, cGPhiAB
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def axis (u, v):
    '''
    Given two vectors u and v, compute an orthogonal matrix Q whose first
    column is u, second column is othogonal to u in the direction of v, and
    third column is orthogonal to both u and v.
    '''

    e1 = normalize(u)

    e2 = v - (e1.dot(v)) * e1
    e2 = normalize(e2)

    e0 = np.cross(e1, e2)
    e0 = normalize(e0)

    Q  = np.zeros((3,3))
    Q[:,0] = e0
    Q[:,1] = e1
    Q[:,2] = e2

    return Q
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def orient(Q, alpha, beta):
    '''
    Given an orthogonal matrix Q, rotate it by alpha about the z-axis and
    then by beta about the x-axis.
    '''
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    Ra = np.array([ [ ca,  -sa,  0.0],
                    [ sa,   ca,  0.0],
                    [0.0,  0.0,  1.0]])

    # Rb = np.array([ [1.0,  0.0,  0.0],
    #                 [0.0,   cb,   sb],
    #                 [0.0,  -sb,   cb]])
    
    Rb = np.array([ [1.0,  0.0,  0.0],
                    [0.0,   cb,   -sb],
                    [0.0,  sb,   cb]])

    Qt = np.matmul(Q, np.matmul(Ra, Rb))

    return Qt


#----------------------------------------------------------------------

#---------------------------------------------------------------------
def normalize(u):
    '''
    Calculate the normalized vector of a given vector
    '''
    u_norm = np.linalg.norm(u)
    if u_norm > 0.0:
        return u / u_norm
    return u

#----------------------------------------------------------------------
def rot2quat(R):
    """
    ROT2QUAT - Transform Rotation matrix into normalized quaternion.
    Usage: q = rot2quat(R)
    Input:
    R - 3-by-3 Rotation matrix
    Output:
    q - 4-by-1 quaternion, with form [w x y z], where w is the scalar term.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return normalize(np.array([qw, qx, qy, qz]))
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def quat2rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix

    Parameters
    ----------
    q : np.ndarray
        Quaternion

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    R = np.zeros((3, 3))
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z

    yz = y * z

    R[0][0] = 1.0 - 2.0 * y2 - 2.0 * z2
    R[1][0] = 2.0 * xy + 2.0 * wz
    R[2][0] = 2.0 * xz - 2.0 * wy
    R[0][1] = 2.0 * xy - 2.0 * wz
    R[1][1] = 1.0 - 2.0 * x2 - 2.0 * z2
    R[2][1] = 2.0 * yz + 2.0 * wx
    R[0][2] = 2.0 * xz + 2.0 * wy
    R[1][2] = 2.0 * yz - 2.0 * wx
    R[2][2] = 1.0 - 2.0 * x2 - 2.0 * y2

    return R
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation from `q1` to `q2` at `t`

    Parameters
    ----------
    q1 : np.ndarray
        Source quaternion
    q2 : np.ndarray
        Target quaternion
    t : float
        Interpolation factor, between 0 and 1

    Returns
    -------
    np.ndarray
        The spherical linear interpolation between `q1` and `q2` at `t`
    """
    dot = q1.dot(q2)
    q3 = q2
    if dot < 0.0:
        dot = -dot
        q3 = -q2

    if dot < 0.9999:
        angle = np.arccos(dot)
        a = np.sin(angle * (1 - t)) / np.sin(angle)
        b = np.sin(angle * t) / np.sin(angle)
        return a * q1 + b * q3

    # Angle is close to zero - do linear interpolation
    return q1 * (1 - t) + q3 * t
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def bislerp(Q_A, Q_B, t):
    '''
    :param Q_A: ndarray
    :param Q_B: ndarray
    :param t: float
    :return: ndarray
    Linear interpolation of two orthogonal matrices.
    '''
    qa = rot2quat(Q_A)
    qb = rot2quat(Q_B)

    quat_array = np.array([
        [-qa[1], qa[0], qa[3], -qa[2]],
        [-qa[2], -qa[3], qa[0], qa[1]],
        [-qa[3], qa[2], -qa[1], qa[0]],
    ])

    qm = qa
    max_dot = abs(qm.dot(qb))

    for v in quat_array[0:]:
        dot = abs(v.dot(qb))
        if dot > max_dot:
            max_dot = dot
            qm = v

    qm_slerp = slerp(qm, qb, t)

    return quat2rot(qm_slerp)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def getFiberDirections(vtuMesh, Phi_EP, Phi_LV, Phi_RV, \
                        gPhi_EP, gPhi_LV, gPhi_RV, gPhi_AB, \
                        ALFA_END, ALFA_EPI, BETA_END, BETA_EPI):
    '''
    Compute the fiber directions at the center of each cell
    '''
    EPS = sys.float_info.epsilon

    numCells = vtuMesh.GetNumberOfCells()

    print("   Computing fiber directions at cells")
    F = np.zeros((numCells, 3))
    S = np.zeros((numCells, 3))
    T = np.zeros((numCells, 3))

    j = 1
    k = 1
    print ("      Progress "),
    sys.stdout.flush()
    for iCell in range(0, numCells):
        phiEP = Phi_EP[iCell]
        phiLV = Phi_LV[iCell]
        phiRV = Phi_RV[iCell]

        gPhiEP = gPhi_EP[iCell, :]
        gPhiLV = gPhi_LV[iCell, :]
        gPhiRV = gPhi_RV[iCell, :]
        gPhiAB = gPhi_AB[iCell, :]

        d = phiRV / max(EPS, phiLV + phiRV)
        alfaS = ALFA_END * (1 - d) - ALFA_END * d
        betaS = BETA_END * (1 - d) - BETA_END * d
        alfaW = ALFA_END * (1 - phiEP) + ALFA_EPI * phiEP
        betaW = BETA_END * (1 - phiEP) + BETA_EPI * phiEP

        Q_LV = axis(gPhiAB, - gPhiLV)
        Q_LV = orient(Q_LV, alfaS, betaS)

        Q_RV = axis(gPhiAB, gPhiRV)
        Q_RV = orient(Q_RV, alfaS, betaS)

        Q_END = bislerp(Q_LV, Q_RV, d)

        Q_EPI = axis(gPhiAB, gPhiEP)
        Q_EPI = orient(Q_EPI, alfaW, betaW)

        FST = bislerp(Q_END, Q_EPI, phiEP)

        F[iCell, :] = np.array([FST[0, 0], FST[1, 0], FST[2, 0]])
        S[iCell, :] = np.array([FST[0, 1], FST[1, 1], FST[2, 1]])
        T[iCell, :] = np.array([FST[0, 2], FST[1, 2], FST[2, 2]])
        if iCell==j:
            print ("%d%%  " % ((k-1)*10)),
            sys.stdout.flush()
            k = k + 1
            j = int(float((k-1)*numCells)/10.0)
    print("[Done!]")

    return F, S, T


def generate_fibers_BiV_Bayer_cells(laplace_results_file, params):
    '''
    Generate fiber directions on a truncated BiV ventricular geometry using the
    Laplace-Dirichlet rule-based method of Bayer et al. 2012

    ARGS:
    laplace_results_file : str
        Path to the .vtu mesh with Laplace fields defined at nodes
    params : dict
        Dictionary of parameters for fiber generation
    '''
    
    t1 = time.time()
    print("========================================================")

    # Get directory of mesh (Laplace solution is in {mesh_dir}/lps_thickness/result_020.vtu)
    # So we go up one level from lps_thickness to get mesh_dir
    mesh_dir = os.path.dirname(os.path.dirname(laplace_results_file))
    
    result_mesh, Phi_EPI, Phi_LV, Phi_RV, Phi_AB, \
    gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB = loadLaplaceSoln(laplace_results_file)

    # Unpack the input data
    ALFA_END = params['ALFA_END']
    ALFA_EPI = params['ALFA_EPI']
    BETA_END = params['BETA_END']
    BETA_EPI = params['BETA_EPI']

    # Generate fiber directions
    F, S, T = getFiberDirections(result_mesh, Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 ALFA_END, ALFA_EPI, BETA_END, BETA_EPI)


    result_mesh.cell_data.set_array(F, 'F')
    result_mesh.cell_data.set_array(S, 'S')
    result_mesh.cell_data.set_array(T, 'T')

    # Save a copy of the mesh with Laplace fields for inspection
    check_fname = os.path.join(mesh_dir, "check.vtu")
    print(f"   Writing check file   --->   {check_fname}")
    result_mesh.save(check_fname)

    # print("   Writing domains and fibers to VTK data structure")

    # # Write the fiber directions to a vtu files
    # output_mesh = copy.deepcopy(result_mesh)
    # #output_mesh.cell_data.remove('Proc_ID')
    # #output_mesh.point_data.remove(DATASTR1)
    # #output_mesh.point_data.remove(DATASTR2)
    # #output_mesh.point_data.remove(DATASTR3)
    # #output_mesh.point_data.remove(DATASTR4)

    # fname1 = os.path.join(mesh_dir, "fibersLong.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.set_array(F, 'FIB_DIR')
    # output_mesh.save(fname1)

    # fname1 = os.path.join(mesh_dir, "fibersSheet.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.remove('FIB_DIR')
    # output_mesh.cell_data.set_array(T, 'FIB_DIR')
    # output_mesh.save(fname1)

    # fname1 = os.path.join(mesh_dir, "fibersNormal.vtu")
    # print("   Writing to vtu file   --->   %s" % (fname1))
    # output_mesh.cell_data.remove('FIB_DIR')
    # output_mesh.cell_data.set_array(S, 'FIB_DIR')
    # output_mesh.save(fname1)

    # t2 = time.time()
    # print('\n   Total time: %.3fs' % (t2-t1))
    # print("========================================================")