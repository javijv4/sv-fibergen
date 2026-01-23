#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Fiber generation module for biventricular heart models.

This module provides functions to generate myocardial fiber orientations for
biventricular heart models using Laplace-Dirichlet rule-based methods. It supports
both the Bayer et al. (2012) and Doste et al. (2019) fiber generation algorithms.

The module includes utilities for:
- Generating epicardial apex surfaces
- Running Laplace-Dirichlet solvers
- Computing fiber directions based on Laplace field gradients
- Handling coordinate transformations and rotations

References:
    Bayer et al. 2012: https://doi.org/10.1007/s10439-012-0593-5
    Doste et al. 2019: https://doi.org/10.1002/cnm.3185
"""

import os
import re
import numpy as np
import pyvista as pv
import time
import copy


def normalize(x):
    """Normalize each row of an (N, 3) array.
    
    Zero-length rows remain zero after normalization.
    
    Args:
        x: Array-like of shape (N, 3).
    
    Returns:
        np.ndarray of shape (N, 3) with row-wise normalized vectors.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("normalize expects an array of shape (N, 3)")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    out = a / safe_norms
    zero_rows = (norms.squeeze() == 0.0)
    if np.any(zero_rows):
        out[zero_rows] = 0.0
    return out



def get_normal_plane_svd(points):
    """Find the plane that best fits a set of points using SVD.
    
    Args:
        points: Array of shape (N, 3) representing 3D points.
    
    Returns:
        tuple: A tuple containing:
            - normal (np.ndarray): Unit normal vector to the fitted plane.
            - centroid (np.ndarray): Centroid of the input points.
    """
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid


def generate_epi_apex(surfaces_dir, surface_names):
    """Generate the epicardial apex surface from the epicardial surface of the BiV.
    
    This function identifies the apex point of the epicardium and creates a surface
    mesh containing elements that include the apex point. The surface is saved with
    global node and element IDs.
    
    Args:
        surfaces_dir: Directory containing surface mesh files.
        surface_names: Dictionary mapping surface types to filenames.
            Expected keys: 'epi', 'epi_apex', 'base'.
    """
    
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



def runLaplaceSolver(surfaces_dir, mesh_file, exec_svmultiphysics, template_file, outdir, surface_names):
    """Run the svMultiPhysics Laplace solver for fiber generation.
    
    This function creates an XML configuration file from a template, updates paths,
    and executes the Laplace-Dirichlet solver to compute the Laplace fields needed
    for fiber direction generation.
    
    Args:
        surfaces_dir: Directory containing surface mesh files.
        mesh_file: Path to the volumetric mesh file.
        exec_svmultiphysics: Command to execute svMultiPhysics (e.g., "svmultiphysics ").
        template_file: Path to the XML template file for the solver.
        outdir: Output directory for solver results.
        surface_names: Dictionary mapping surface types to filenames.
    
    Returns:
        str: Path to the result file containing Laplace solutions.
    """
    xml_template_path = template_file
    out_name = os.path.join(surfaces_dir, "../svFibers_BiV.xml")
    
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
            elif face_name == "mv":
                new_path = os.path.join(surfaces_dir, surface_names['mv'])
            elif face_name == "tv":
                new_path = os.path.join(surfaces_dir, surface_names['tv'])
            elif face_name == "av":
                new_path = os.path.join(surfaces_dir, surface_names['av'])
            elif face_name == "pv":
                new_path = os.path.join(surfaces_dir, surface_names['pv'])
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

    print("   Running svMultiPhysics solver")
    print(f"   {exec_svmultiphysics + out_name}")
    os.system(exec_svmultiphysics + out_name)

    return outdir + '/results_001.vtu'


def loadLaplaceSolnBayer(fileName):
    """Load Laplace-Dirichlet solution for Bayer method from a VTU file.
    
    Extracts the Laplace solution and gradients at cell centers. The solution
    is normalized to [0, 1] range before gradient computation.
    
    Args:
        fileName: Path to the .vtu file with the Laplace solution. The solution
            should be defined at the nodes. Expected field names:
            - Trans_EPI: Endocardial-epicardial transmural field
            - Trans_LV: Left ventricle field
            - Trans_RV: Right ventricle field
            - Long_AB: Apex-to-base field
    
    Returns:
        tuple: A tuple containing:
            - mesh_cells: PyVista mesh with cell-centered data
            - cPhiEP: Transmural field at cells (N,)
            - cPhiLV: LV field at cells (N,)
            - cPhiRV: RV field at cells (N,)
            - cPhiAB: Apex-base field at cells (N,)
            - cGPhiEP: Gradient of transmural field (N, 3)
            - cGPhiLV: Gradient of LV field (N, 3)
            - cGPhiRV: Gradient of RV field (N, 3)
            - cGPhiAB: Gradient of apex-base field (N, 3)
    """

    DATASTR1 = 'Trans_EPI'
    DATASTR2 = 'Trans_LV'
    DATASTR3 = 'Trans_RV'
    DATASTR4 = 'Long_AB'

    print("   Loading Laplace solution <---   %s" % (fileName))

    # Read mesh with pyvista
    result_mesh = pv.read(fileName)

    print("   Extracting solution and estimating gradients at points (then averaging to cells)")

    # Helper: min-max scale array to [0,1]; if flat, return 0.5
    def _minmax01(arr):
        arr = np.asarray(arr, dtype=float)
        amin = np.min(arr)
        amax = np.max(arr)
        if amax > amin:
            return (arr - amin) / (amax - amin)
        else:
            return np.ones_like(arr) * 0.5

    # Compute point-wise gradient. Smoother than cell-averaged gradients.
    for data_name in (DATASTR1, DATASTR2, DATASTR3, DATASTR4):
        if data_name not in result_mesh.point_data:
            raise KeyError(
                f"Expected point_data array '{data_name}' not found in {fileName}. "
                f"Available point_data: {list(result_mesh.point_data.keys())}"
            )

        # Scale to [0,1] before gradient computation
        scaled = _minmax01(result_mesh.point_data[data_name])
        result_mesh.point_data[data_name] = scaled

        gmesh = result_mesh.compute_derivative(scalars=data_name, gradient=True, preference='point')
        # PyVista stores the gradient vector in an array named 'gradient'
        result_mesh.point_data[data_name + "_grad"] = np.asarray(gmesh.point_data["gradient"])

    # Convert point-data to cell-data (includes the newly computed *_grad arrays)
    mesh_cells = result_mesh.point_data_to_cell_data()

    # Get cell centers (Nx3) and scalar cell arrays (N,)
    cPhiEP = np.asarray(mesh_cells.cell_data[DATASTR1])
    cPhiLV = np.asarray(mesh_cells.cell_data[DATASTR2])
    cPhiRV = np.asarray(mesh_cells.cell_data[DATASTR3])
    cPhiAB = np.asarray(mesh_cells.cell_data[DATASTR4])

    # Gradient cell arrays (Ncells, 3)
    cGPhiEP = np.asarray(mesh_cells.cell_data[DATASTR1 + "_grad"])
    cGPhiLV = np.asarray(mesh_cells.cell_data[DATASTR2 + "_grad"])
    cGPhiRV = np.asarray(mesh_cells.cell_data[DATASTR3 + "_grad"])
    cGPhiAB = np.asarray(mesh_cells.cell_data[DATASTR4 + "_grad"])

    return mesh_cells, cPhiEP, cPhiLV, cPhiRV, cPhiAB, cGPhiEP, cGPhiLV, cGPhiRV, cGPhiAB


def bislerp(Q1, Q2, interp_func):
    """Vectorized spherical linear interpolation between batches of rotation matrices.
    
    Performs SLERP on rotation matrices represented as quaternions internally.
    Uses wxyz quaternion convention to match quat2rot.
    
    Args:
        Q1: Array of shape (N, 3, 3) containing starting rotation matrices.
        Q2: Array of shape (N, 3, 3) containing ending rotation matrices.
        interp_func: Array of shape (N,) with interpolation values in [0, 1].
    
    Returns:
        np.ndarray: Array of shape (N, 3, 3) containing interpolated rotation matrices.
    
    Note:
        Avoids per-element Python/Scipy objects for performance.
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
            mf_idx = np.nonzero(mask_f)[0]
            # Case idx==0
            m0 = idx == 0
            if np.any(m0):
                S = np.sqrt(1.0 + Rf[m0, 0, 0] - Rf[m0, 1, 1] - Rf[m0, 2, 2]) * 2.0
                rows = mf_idx[m0]
                q[rows, 0] = (Rf[m0, 2, 1] - Rf[m0, 1, 2]) / S
                q[rows, 1] = 0.25 * S
                q[rows, 2] = (Rf[m0, 0, 1] + Rf[m0, 1, 0]) / S
                q[rows, 3] = (Rf[m0, 0, 2] + Rf[m0, 2, 0]) / S
            # Case idx==1
            m1 = idx == 1
            if np.any(m1):
                S = np.sqrt(1.0 + Rf[m1, 1, 1] - Rf[m1, 0, 0] - Rf[m1, 2, 2]) * 2.0
                rows = mf_idx[m1]
                q[rows, 0] = (Rf[m1, 0, 2] - Rf[m1, 2, 0]) / S
                q[rows, 1] = (Rf[m1, 0, 1] + Rf[m1, 1, 0]) / S
                q[rows, 2] = 0.25 * S
                q[rows, 3] = (Rf[m1, 1, 2] + Rf[m1, 2, 1]) / S
            # Case idx==2
            m2 = idx == 2
            if np.any(m2):
                S = np.sqrt(1.0 + Rf[m2, 2, 2] - Rf[m2, 0, 0] - Rf[m2, 1, 1]) * 2.0
                rows = mf_idx[m2]
                q[rows, 0] = (Rf[m2, 1, 0] - Rf[m2, 0, 1]) / S
                q[rows, 1] = (Rf[m2, 0, 2] + Rf[m2, 2, 0]) / S
                q[rows, 2] = (Rf[m2, 1, 2] + Rf[m2, 2, 1]) / S
                q[rows, 3] = 0.25 * S

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
        # Nearly identical orientations: perform linear interpolation and normalize later
        tl = t[lin_mask][:, None]
        q[lin_mask] = (1.0 - tl) * q1[lin_mask] + tl * q2[lin_mask]

    # Normalize and convert back to rotation matrices
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quat_to_rotm_batch(q)

def axis(u, v):
    """Construct orthogonal coordinate systems from two vector fields.
    
    Creates an orthonormal basis [e0, e1, e2] for each element where:
    - e1 is aligned with u (normalized)
    - e2 is orthogonal to e1 and in the plane of v
    - e0 is the cross product of e1 and e2
    
    Args:
        u: Array of shape (nelems, 3) representing the longitudinal direction.
        v: Array of shape (nelems, 3) representing a secondary direction.
    
    Returns:
        np.ndarray: Array of shape (nelems, 3, 3) where columns are
            [e0 (circumferential), e1 (longitudinal), e2 (transmural)].
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ne = u.shape[0]

    # e1 = normalize rows of u
    e1 = normalize(u)

    # e2 = v - proj_{e1}(v)
    proj = np.sum(e1 * v, axis=1)[:, None] * e1
    e2 = v - proj
    e2 = normalize(e2)

    # e0 = cross(e1, e2) normalized
    e0 = np.cross(e1, e2, axisa=1, axisb=1)
    e0 = normalize(e0)

    Q = np.zeros((ne, 3, 3), dtype=float)
    Q[:, :, 0] = e0
    Q[:, :, 1] = e1
    Q[:, :, 2] = e2

    return Q

def orient(Q, alpha, beta):
    """Apply alpha and beta rotations to orthogonal matrices.
    
    For each element, rotates Q[i] by alpha[i] about the z-axis and then
    by beta[i] about the y-axis (transmural direction).
    
    Args:
        Q: Array of shape (ne, 3, 3) containing orthogonal matrices.
        alpha: Array of shape (ne,) with rotation angles (radians) about z-axis.
        beta: Array of shape (ne,) with rotation angles (radians) about y-axis.
    
    Returns:
        np.ndarray: Array of shape (ne, 3, 3) containing rotated matrices.
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

    # Rb = np.zeros((ne, 3, 3), dtype=float)
    # Rb[:, 0, 0] = 1.0
    # Rb[:, 1, 1] = cb
    # Rb[:, 1, 2] = sb
    # Rb[:, 2, 1] = -sb
    # Rb[:, 2, 2] = cb

    # Rb = np.zeros((ne, 3, 3), dtype=float)
    # Rb[:, 0, 0] = 1.0
    # Rb[:, 1, 1] = cb
    # Rb[:, 1, 2] = -sb
    # Rb[:, 2, 1] = sb
    # Rb[:, 2, 2] = cb

    Rb = np.zeros((ne, 3, 3), dtype=float)
    Rb[:, 0, 0] = cb
    Rb[:, 0, 2] = sb
    Rb[:, 1, 1] = 1.0
    Rb[:, 2, 0] = -sb
    Rb[:, 2, 2] = cb

    # Compose rotations and apply to Q per element
    RaRb = np.einsum('nij,njk->nik', Ra, Rb)
    Qt = np.einsum('nij,njk->nik', Q, RaRb)

    return Qt

def getFiberDirectionsBayer(Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 params, intermediate=False):
    """Compute fiber directions using the Bayer et al. method.
    
    Implements the rule-based algorithm from Bayer et al. (2012) to compute
    fiber, sheet, and normal directions from Laplace field gradients.
    
    Args:
        Phi_EPI: Transmural field values at cells (N,).
        Phi_LV: Left ventricle field values at cells (N,).
        Phi_RV: Right ventricle field values at cells (N,).
        gPhi_EPI: Gradient of transmural field (N, 3).
        gPhi_LV: Gradient of LV field (N, 3).
        gPhi_RV: Gradient of RV field (N, 3).
        gPhi_AB: Gradient of apex-base field (N, 3).
        params: Dictionary with keys 'ALFA_END', 'ALFA_EPI', 'BETA_END', 'BETA_EPI'
            (all in degrees).
        intermediate: If True, also return intermediate basis vectors.
    
    Returns:
        tuple: If intermediate=False, returns (F, S, T) where:
            - F: Fiber directions (N, 3)
            - S: Sheet normal directions (N, 3)
            - T: Sheet directions (N, 3)
        If intermediate=True, also returns intermediate circumferential vectors.
    """

    # Unpack parameters
    ALFA_END = np.deg2rad(params["ALFA_END"])
    ALFA_EPI = np.deg2rad(params["ALFA_EPI"])
    BETA_END = np.deg2rad(params["BETA_END"])
    BETA_EPI = np.deg2rad(params["BETA_EPI"])

    print("   Computing fiber directions at cells")

    d = Phi_RV / (Phi_LV + Phi_RV)
    alfaS = ALFA_END * (1 - d) - ALFA_END * d
    betaS = BETA_END * (1 - d) - BETA_END * d
    alfaW = ALFA_END * (1 - Phi_EPI) + ALFA_EPI * Phi_EPI
    betaW = BETA_END * (1 - Phi_EPI) + BETA_EPI * Phi_EPI

    Q_LV0 = axis(gPhi_AB, -gPhi_LV)
    Q_LV = orient(Q_LV0, alfaS, betaS)
    Q_RV0 = axis(gPhi_AB, gPhi_RV)  # Note that gPhi_RV points the other way
    Q_RV = orient(Q_RV0, alfaS, -betaS)  # Therefore, we need a minus in betaS

    # Flipping vectors for consistent bislerp interpolation
    Q_END = bislerp(Q_LV, Q_RV, d)
    Q_END[d > 0.5,:,0] = -Q_END[d > 0.5,:,0]
    Q_END[d > 0.5,:,2] = -Q_END[d > 0.5,:,2]

    Q_EPI0 = axis(gPhi_AB, gPhi_EPI)
    Q_EPI = orient(Q_EPI0, alfaW, betaW)

    FST = bislerp(Q_END, Q_EPI, Phi_EPI)

    F = FST[:, :, 0]
    S = FST[:, :, 1]
    T = FST[:, :, 2]

    if intermediate:
        return F, S, T, Q_LV[:,:,0], Q_RV[:,:,0], Q_END[:,:,0], Q_EPI[:,:,0]

    return F, S, T


def get_alpha_beta_angles_Bayer(F, Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 params):
    """Compute alpha and beta angles from fiber directions (Bayer method).
    
    This is a sanity check routine that computes the helix and transverse angles
    from given fiber directions and Laplace field gradients.
    
    Args:
        F: Fiber directions at cells (N, 3).
        Phi_EPI: Transmural field values at cells (N,).
        Phi_LV: Left ventricle field values at cells (N,).
        Phi_RV: Right ventricle field values at cells (N,).
        gPhi_EPI: Gradient of transmural field (N, 3).
        gPhi_LV: Gradient of LV field (N, 3).
        gPhi_RV: Gradient of RV field (N, 3).
        gPhi_AB: Gradient of apex-base field (N, 3).
        params: Dictionary with keys 'ALFA_END', 'ALFA_EPI' (in degrees).
    
    Returns:
        tuple: A tuple containing:
            - alpha_angle: Helix angles in degrees (N,)
            - beta_angle: Transverse angles in degrees (N,)
            - C: Circumferential vectors before rotation (N, 3)
            - Cr: Circumferential vectors after alpha rotation (N, 3)
    """

    ALFA_END = np.deg2rad(params["ALFA_END"])
    ALFA_EPI = np.deg2rad(params["ALFA_EPI"])

    d = Phi_RV / (Phi_LV + Phi_RV)
    alfaS = ALFA_END * (1 - d) - ALFA_END * d
    alfaW = ALFA_END * (1 - Phi_EPI) + ALFA_EPI * Phi_EPI

    # Alpha angle 
    Q_LV = axis(gPhi_AB, -gPhi_LV)          # (N,3,3)
    Q_RV = axis(gPhi_AB, gPhi_RV)            # (N,3,3)
    Q = np.copy(Q_LV)
    Q[d > 0.5,:,0] = -Q_RV[d > 0.5,:,0]
    Q[d > 0.5,:,2] = -Q_RV[d > 0.5,:,2]
    C = Q[:, :, 0]                       # (N,3)
    L = Q[:, :, 1]                       # (N,3)

    # Angle in radians between F and circumferential vector
    # alpha is positive in the direction of the longitudinal vector
    cosang = np.clip(np.sum(F * C, axis=1), -1.0, 1.0)
    sinang = np.clip(np.sum(F * L, axis=1), -1.0, 1.0)
    alpha_angle = np.sign(sinang) * np.arccos(np.abs(cosang))

    # Beta angle
    Q_LV = orient(axis(gPhi_AB, -gPhi_LV), 
                    alfaS,
                    0)
    Q_RV = orient(axis(gPhi_AB, gPhi_RV), 
                    alfaS,
                    0)
    
    Q_END = bislerp(Q_LV, Q_RV, d)
    Q_END[d > 0.5,:,0] = -Q_END[d > 0.5,:,0]
    Q_END[d > 0.5,:,2] = -Q_END[d > 0.5,:,2]

    Q_EPI0 = axis(gPhi_AB, gPhi_EPI)
    Q_EPI = orient(Q_EPI0, alfaW, 0.0)

    Q = bislerp(Q_END, Q_EPI, Phi_EPI)
    Cr = Q[:, :, 0]
    Tr = Q[:, :, 2]

    # Angle in radians between F and rotated circumferential vector
    # beta is negative in the direction of the transmural vector
    cosang = np.clip(np.sum(F * Cr, axis=1), -1.0, 1.0)
    sinang = np.clip(np.sum(F * Tr, axis=1), -1.0, 1.0)
    beta_angle = - np.sign(sinang) * np.arccos(np.abs(cosang)) # Note the minus sign to match definition

    return np.rad2deg(alpha_angle), np.rad2deg(beta_angle), C, Cr

def generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params, return_angles=False, return_intermediate=False):
    """Generate fiber directions for BiV geometry using the Bayer method.
    
    Implements the Laplace-Dirichlet rule-based method of Bayer et al. (2012)
    to generate fiber directions on a truncated biventricular heart geometry.
    
    Args:
        outdir: Output directory for fiber files.
        laplace_results_file: Path to the .vtu mesh with Laplace fields defined at nodes.
        params: Dictionary of parameters with keys:
            - ALFA_END: Endocardial helix angle in degrees
            - ALFA_EPI: Epicardial helix angle in degrees
            - BETA_END: Endocardial transverse angle in degrees
            - BETA_EPI: Epicardial transverse angle in degrees
        return_angles: If True, compute and include alpha/beta angles in output.
        return_intermediate: If True, include intermediate basis vectors in output.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh with computed fiber directions and optional
            angle/intermediate data in cell_data.
    """
    
    t1 = time.time()
    print("========================================================")

    # Load Laplace solution    
    result_mesh, Phi_EPI, Phi_LV, Phi_RV, Phi_AB, \
    gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB = loadLaplaceSolnBayer(laplace_results_file)

    # Generating output mesh
    output_mesh = copy.deepcopy(result_mesh)
    # Ensure only FIB_DIR is present
    for k in list(output_mesh.cell_data.keys()):
        output_mesh.cell_data.remove(k)
    for k in list(output_mesh.point_data.keys()):
        output_mesh.point_data.remove(k)
        
    # Generate fiber directions
    out = getFiberDirectionsBayer(Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 params, intermediate=return_intermediate)
    
    if return_intermediate:
        F, S, T, eC_LV, eC_RV, eC_END, eC_EPI = out
        result_mesh.cell_data['F'] = F
        result_mesh.cell_data['S'] = S
        result_mesh.cell_data['T'] = T
        result_mesh.cell_data['eC_LV'] = eC_LV
        result_mesh.cell_data['eC_RV'] = eC_RV
        result_mesh.cell_data['eC_END'] = eC_END
        result_mesh.cell_data['eC_EPI'] = eC_EPI
    else:
        F, S, T = out
        result_mesh.cell_data['F'] = F
        result_mesh.cell_data['S'] = S
        result_mesh.cell_data['T'] = T

    t2 = time.time()
    print('\n   Total time: %.3fs' % (t2-t1))
    print("========================================================")
    
    # Write the fiber directions to a vtu files
    print("   Writing domains and fibers to VTK data structure")
    
    fname1 = os.path.join(outdir, "fibersLong.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(F, 'FIB_DIR')
    output_mesh.save(fname1)

    fname1 = os.path.join(outdir, "fibersSheet.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(T, 'FIB_DIR')
    output_mesh.save(fname1)

    fname1 = os.path.join(outdir, "fibersNormal.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(S, 'FIB_DIR')
    output_mesh.save(fname1)

    t2 = time.time()
    print('\n   Total time: %.3fs' % (t2-t1))
    print("========================================================")

    if return_angles:
        alpha_angle, beta_angle, eC, eCr = get_alpha_beta_angles_Bayer(F, Phi_EPI, Phi_LV, Phi_RV,
                                 gPhi_EPI, gPhi_LV, gPhi_RV, gPhi_AB, 
                                 params)
        result_mesh.cell_data['Alpha_Angle'] = alpha_angle
        result_mesh.cell_data['Beta_Angle'] = beta_angle
        result_mesh.cell_data['eC'] = eC
        result_mesh.cell_data['eCr'] = eCr
    

    return result_mesh



def loadLaplaceSolnDoste(fileName):
    """Load Laplace-Dirichlet solution for Doste method from a VTU file.
    
    Extracts the Laplace solution and gradients at cell centers. The solution
    is normalized to [0, 1] range before gradient computation.
    
    Args:
        fileName: Path to the .vtu file with the Laplace solution. Expected fields:
            - Trans_BiV: Ventricular transmural coordinate
            - Long_AV, Long_MV, Long_PV, Long_TV: Longitudinal fields from valves
            - Weight_LV, Weight_RV: Ventricle weight fields
            - Trans_EPI, Trans_LV, Trans_RV: Transmural fields
    
    Returns:
        tuple: A tuple containing:
            - mesh_cells: PyVista mesh with cell-centered data
            - lap: Dictionary of Laplace solution values at cells
            - grad: Dictionary of gradient arrays at cells (N, 3)
    """

    varnames = ['Trans_BiV', 'Long_AV', 'Long_MV', 'Long_PV', 'Long_TV', 'Weight_LV', 
                'Weight_RV', 'Trans_EPI', 'Trans_LV', 'Trans_RV']

    print("   Loading Laplace solution <---   %s" % (fileName))

    # Read mesh with pyvista
    result_mesh = pv.read(fileName)

    print("   Extracting solution and estimating gradients at points (then averaging to cells)")

    # Map VTU array names to internal keys expected by downstream code
    name_map = {
        'Trans_BiV': 'ven_trans',
        'Long_AV': 'lv_av_long',
        'Long_MV': 'lv_mv_long',
        'Long_PV': 'rv_pv_long',
        'Long_TV': 'rv_tv_long',
        'Weight_LV': 'lv_weight',
        'Weight_RV': 'rv_weight',
        'Weight_RV_OP': 'rv_op_weight',
        'Trans_EPI': 'epi_trans',
        'Trans_LV': 'lv_trans',
        'Trans_RV': 'rv_trans',
    }

    lap = {}
    grad = {}

    # Helper: min-max scale array to [0,1]; if flat, return 0.5
    def _minmax01(arr):
        arr = np.asarray(arr, dtype=float)
        amin = np.min(arr)
        amax = np.max(arr)
        if amax > amin:
            return (arr - amin) / (amax - amin)
        else:
            return np.ones_like(arr) * 0.5

    # Compute gradients at points (smoother than cell-averaged gradients)
    for vname in varnames:
        if vname not in result_mesh.point_data:
            raise KeyError(
                f"Expected point_data array '{vname}' not found in {fileName}. "
                f"Available point_data: {list(result_mesh.point_data.keys())}"
            )

        # Scale to [0,1] before gradient computation
        result_mesh.point_data[vname] = _minmax01(result_mesh.point_data[vname])

        gmesh = result_mesh.compute_derivative(scalars=vname, gradient=True, preference='point')
        # PyVista stores the gradient vector in an array named 'gradient'
        result_mesh.point_data[vname + "_grad"] = np.asarray(gmesh.point_data["gradient"])

    # Convert point-data to cell-data (includes the newly computed *_grad arrays)
    mesh_cells = result_mesh.point_data_to_cell_data()

    # Extract cell-centered Laplace values and gradients with expected key names
    for vname in varnames:
        key = name_map[vname]
        lap[key] = np.asarray(mesh_cells.cell_data[vname])
        grad[key] = np.asarray(mesh_cells.cell_data[vname + "_grad"])

    return mesh_cells, lap, grad


def compute_basis_vectors(lap, grad):
    """Compute local orthogonal basis vectors for LV and RV.
    
    Constructs circumferential, longitudinal, and transmural directions for
    both left and right ventricles from Laplace field gradients.
    
    Args:
        lap: Dictionary of Laplace solution values at cells.
        grad: Dictionary of gradient arrays at cells (N, 3).
    
    Returns:
        dict: Dictionary with keys 'eC_lv', 'eT_lv', 'eL_lv', 'eC_rv', 'eT_rv',
            'eL_rv', 'eC' containing basis vectors (N, 3).
    """
    # LV
    # longitudinal
    lv_glong = grad['lv_mv_long']*lap['lv_weight'][:,None] + grad['lv_av_long']*(1 - lap['lv_weight'][:,None])
    eL_lv = normalize(lv_glong)

    # transmural
    lv_gtrans = grad['lv_trans'] - (eL_lv*grad['lv_trans'])*eL_lv
    eT_lv = normalize(lv_gtrans)

    # circumferential
    eC_lv = np.cross(eL_lv, eT_lv, axisa=1, axisb=1)
    eC_lv = normalize(eC_lv)

    # Ensuring orthogonality
    eT_lv = np.cross(eC_lv, eL_lv, axisa=1, axisb=1)
    eT_lv = normalize(eT_lv)

    # RV
    # longitudinal
    rv_glong = grad['rv_tv_long']*lap['rv_weight'][:,None]  + grad['rv_pv_long']*(1 - lap['rv_weight'][:,None] )
    eL_rv = normalize(rv_glong)

    # transmural
    rv_gtrans = grad['rv_trans'] - (eL_rv*grad['rv_trans'])*eL_rv
    eT_rv = normalize(rv_gtrans)

    # circumferential
    eC_rv = np.cross(eL_rv, eT_rv, axisa=1, axisb=1)
    eC_rv = normalize(eC_rv)

    # Ensuring orthogonality
    eT_rv = np.cross(eC_rv, eL_rv, axisa=1, axisb=1)
    eT_rv = normalize(eT_rv)

    # Write out global circumferential vector
    eC = eC_rv*(1-lap['ven_trans'][:,None]) + eC_lv*lap['ven_trans'][:,None]
    eC = normalize(eC)

    basis = {'eC_lv': eC_lv,
                    'eT_lv': eT_lv,
                    'eL_lv': eL_lv,
                    'eC_rv': eC_rv,
                    'eT_rv': eT_rv,
                    'eL_rv': eL_rv,
                    'eC': eC}

    return basis


def redistribute_weight(weight, up, low, strategy='centre'):
    """Redistribute weight values to center or flip their distribution.
    
    Args:
        weight: Array of weight values to redistribute.
        up: Upper quantile threshold (e.g., 0.7).
        low: Lower quantile threshold (e.g., 0.01).
        strategy: Redistribution strategy. Options:
            - 'centre': Clip tails and renormalize to [0, 1]
            - 'flip': Flip the distribution and handle tails
    
    Returns:
        np.ndarray: Redistributed weight values normalized to [0, 1].
    """
    new_weight = weight.copy()

    if strategy == 'flip':
        # Shift all weights
        new_mean = 1 - np.mean(weight)
        shift = new_mean - np.mean(weight)
        new_weight = new_weight + shift

        # Cut off values outside of range 0 - 1
        new_weight[new_weight > 1] = 1
        new_weight[new_weight < 0] = 0

        # Redistribute new tail
        new_weight = (new_weight - np.min(new_weight)) / (np.max(new_weight) - np.min(new_weight))
        tmp = new_weight.copy()

        if shift > 0:
            tmp[tmp >= new_mean] = np.nan
            tmp = (tmp - np.nanmin(tmp)) / (new_mean - np.nanmin(tmp))
        elif shift < 0:
            tmp[tmp <= new_mean] = np.nan
            tmp = (tmp - new_mean) / (np.nanmax(tmp) - new_mean)

        tmp[np.isnan(tmp)] = new_weight[np.isnan(tmp)]
        new_weight = tmp

    else:  # cut off tails so that weights are centered
        # Find upper and lower limits
        upper_lim = np.quantile(weight, up)
        while upper_lim == 0:
            print('Upper limit is 0, increasing upper limit')
            up += 0.1
            upper_lim = np.quantile(weight, up)
        
        lower_lim = np.quantile(weight, low)

        # Set upper and lower values to limits
        new_weight[new_weight > upper_lim] = upper_lim
        new_weight[new_weight < lower_lim] = lower_lim

        # Redistribute/normalize values
        new_weight = (new_weight - np.min(new_weight)) / (np.max(new_weight) - np.min(new_weight))

    return new_weight


def compute_alpha_beta_angles(lap, params):
    """Compute spatially-varying alpha and beta angles for Doste method.
    
    Calculates helix and transverse angles for LV, RV, and septum regions
    based on transmural position and valve weights.
    
    Args:
        lap: Dictionary of Laplace solution values at cells.
        params: Dictionary with angle parameters (in degrees):
            - AENDOLV, AEPILV: LV endo/epi helix angles
            - AENDORV, AEPIRV: RV endo/epi helix angles
            - AOTENDOLV, AOTEPILV: LV outflow tract endo/epi helix angles
            - AOTENDORV, AOTEPIRV: RV outflow tract endo/epi helix angles
            - BENDOLV, BEPILV: LV endo/epi transverse angles
            - BENDORV, BEPIRV: RV endo/epi transverse angles
    
    Returns:
        dict: Dictionary containing computed angle arrays for each region.
    """
    # Modify weights so the effect of outflow tracts is localized
    lv_weight = redistribute_weight(lap['lv_weight'], 0.7, 0.01)
    rv_weight = redistribute_weight(lap['rv_weight'], 0.1, 0.001)

    # LV
    alpha_lv_endo_long = params['AENDOLV'] * lv_weight + params['AOTENDOLV'] * (1 - lv_weight)  # Endo
    alpha_lv_epi_long = params['AEPILV'] * lv_weight + params['AOTEPILV'] * (1 - lv_weight)
    
    alpha_wall_lv = alpha_lv_endo_long * (1 - lap['epi_trans']) + alpha_lv_epi_long * lap['epi_trans']
    beta_wall_lv = (params['BENDOLV'] * (1 - lap['epi_trans']) + params['BEPILV'] * lap['epi_trans']) * lv_weight

    # RV
    alpha_rv_endo_long = params['AENDORV'] * rv_weight + params['AOTENDORV'] * (1 - rv_weight)
    alpha_rv_epi_long = params['AEPIRV'] * rv_weight + params['AOTEPIRV'] * (1 - rv_weight)

    alpha_wall_rv = alpha_rv_endo_long * (1 - lap['epi_trans']) + alpha_rv_epi_long * lap['epi_trans']
    beta_wall_rv = (params['BENDORV'] * (1 - lap['epi_trans']) + params['BEPIRV'] * lap['epi_trans']) * rv_weight

    # Septum
    sep = np.abs(lap['ven_trans'] - 0.5)
    sep = (sep - np.min(sep)) / (np.max(sep) - np.min(sep))
    alpha_septum = (alpha_lv_endo_long * sep * lap['lv_trans']) + (alpha_rv_endo_long * sep * lap['rv_trans'])
    beta_septum = (params['BENDOLV'] * lap['lv_trans'] * lv_weight) + (params['BENDORV'] * lap['rv_trans'] * rv_weight)

    angles = {'alpha_lv_endo_long': alpha_lv_endo_long,
            'alpha_lv_epi_long': alpha_lv_epi_long,
            'alpha_wall_lv': alpha_wall_lv,
            'beta_wall_lv': beta_wall_lv,
            'alpha_rv_endo_long': alpha_rv_endo_long,
            'alpha_rv_epi_long': alpha_rv_epi_long,
            'alpha_wall_rv': alpha_wall_rv,
            'beta_wall_rv': beta_wall_rv,
            'alpha_septum': alpha_septum,
            'beta_septum': beta_septum
            }

    return angles


def rotate_basis(eC, eL, eT, alpha, beta):
    """Rotate local basis vectors by alpha and beta angles.
    
    Applies two successive rotations: first by alpha about the transmural axis,
    then by beta about the rotated longitudinal axis.
    
    Args:
        eC: Circumferential vectors (N, 3).
        eL: Longitudinal vectors (N, 3).
        eT: Transmural vectors (N, 3).
        alpha: Helix rotation angles in radians (N,).
        beta: Transverse rotation angles in radians (N,).
    
    Returns:
        np.ndarray: Rotated basis matrices of shape (N, 3, 3).
    """
    eC = normalize(eC)
    eT = normalize(eT)
    eL = normalize(eL)

    # Matrix of directional vectors
    Q = np.stack([eC, eL, eT], axis=-1)
    Q = np.transpose(Q, (2, 1, 0))

    # Create rotation matrix - from Doste code
    axis = eT
    R = np.array([[np.cos(alpha) + (axis[:, 0]**2)*(1 - np.cos(alpha)), axis[:,0] * axis[:,1]*(1 - np.cos(alpha)) - axis[:,2]*np.sin(alpha), axis[:,0]*axis[:,2]*(1 - np.cos(alpha)) + axis[:,1]*np.sin(alpha)],
                            [axis[:,1]*axis[:,0]*(1 - np.cos(alpha)) + axis[:,2]*np.sin(alpha), np.cos(alpha) + (axis[:,1]**2)*(1 - np.cos(alpha)), axis[:,1]*axis[:, 2]*(1 - np.cos(alpha)) - axis[:, 0]*np.sin(alpha)],
                            [axis[:,2]*axis[:,0]*(1 - np.cos(alpha)) - axis[:,1]*np.sin(alpha), axis[:,2]*axis[:,1]*(1 - np.cos(alpha)) + axis[:, 0]*np.sin(alpha), np.cos(alpha)+(axis[:, 2]**2)*(1 - np.cos(alpha))]])

    # Rotate the circumferential direction around the transmural direction
    QX = np.zeros_like(R)
    for i in range(len(eC)):
        QX[:, :, i] = np.matmul(Q[:, :, i], R[:, :, i])

    # Second rotation (beta) about QX
    axis2 = QX[1, :, :].T
    R2 = np.array([
        [np.cos(beta) + (axis2[:,0]**2)*(1 - np.cos(beta)), axis2[:,0]*axis2[:, 1]*(1 - np.cos(beta)) - axis2[:,2] * np.sin(beta), axis2[:,0] * axis2[:,2] * (1 - np.cos(beta)) + axis2[:,1] * np.sin(beta)],
        [axis2[:,1] * axis2[:,0]*(1 - np.cos(beta)) + axis2[:,2]*np.sin(beta), np.cos(beta) + (axis2[:,1]**2)*(1 - np.cos(beta)), axis2[:,1] * axis2[:,2] * (1 - np.cos(beta)) - axis2[:,0] * np.sin(beta)],
        [axis2[:,2] * axis2[:,0]*(1 - np.cos(beta)) - axis2[:,1]*np.sin(beta), axis2[:, 2] * axis2[:,1] * (1 - np.cos(beta)) + axis2[:,0] * np.sin(beta), np.cos(beta) + (axis2[:,2]**2) * (1 - np.cos(beta))]
    ])

    QX2 = np.zeros((R.shape[2], 3, 3), dtype=float)
    for i in range(len(eC)):
        QX2[i] = np.matmul(QX[:, :, i], R2[:, :, i]).T

    return QX2


def compute_local_basis(basis, angles):
    """Compute rotated local basis for septum and free wall regions.
    
    Args:
        basis: Dictionary of basis vectors for LV and RV.
        angles: Dictionary of angle arrays for each region.
    
    Returns:
        dict: Dictionary with keys 'Qlv_septum', 'Qrv_septum', 'Qlv_epi', 'Qrv_epi'
            containing rotated basis matrices (N, 3, 3).
    """
    Qlv_septum = rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'], angles['alpha_septum'], angles['beta_septum'])
    Qrv_septum = rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'], angles['alpha_septum'], angles['beta_septum'])
    Qlv_epi = rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'], angles['alpha_wall_lv'], angles['beta_wall_lv'])
    Qrv_epi = rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'], angles['alpha_wall_rv'], angles['beta_wall_rv'])

    local_basis = {'Qlv_septum': Qlv_septum,
                    'Qrv_septum': Qrv_septum,
                    'Qlv_epi': Qlv_epi,
                    'Qrv_epi': Qrv_epi,
                    }

    return local_basis


def interpolate_local_basis(lap, local_basis):
    """Interpolate local basis across septum and transmural direction.
    
    Uses spherical linear interpolation (SLERP) to smoothly blend basis vectors
    between LV and RV, and between endocardium and epicardium.
    
    Args:
        lap: Dictionary of Laplace solution values at cells.
        local_basis: Dictionary of rotated basis matrices.
    
    Returns:
        tuple: A tuple containing:
            - Q: Interpolated basis matrices (N, 3, 3)
            - Qepi: Epicardial basis matrices (N, 3, 3)
    """

    epi_trans = lap['epi_trans']

    Qrv_septum = local_basis['Qrv_septum']
    Qlv_septum = local_basis['Qlv_septum']
    Qrv_epi = local_basis['Qrv_epi']
    Qlv_epi = local_basis['Qlv_epi']

    Qepi = bislerp(Qrv_epi, Qlv_epi, lap['ven_trans'])
    Qendo = bislerp(Qrv_septum, Qlv_septum, lap['ven_trans'])
    Q = bislerp(Qendo, Qepi, epi_trans)

    return Q, Qepi


def getFiberDirectionsDoste(lap, grad, params, intermediate=False):
    """Compute fiber directions using the Doste et al. method.
    
    Implements the rule-based algorithm from Doste et al. (2019) to compute
    fiber, sheet, and normal directions from Laplace field gradients.
    
    Args:
        lap: Dictionary of Laplace solution values at cells.
        grad: Dictionary of gradient arrays at cells (N, 3).
        params: Dictionary of angle parameters (in degrees).
        intermediate: If True, also return intermediate basis vectors and angles.
    
    Returns:
        tuple: If intermediate=False, returns (f, s, n) where:
            - f: Fiber directions (N, 3)
            - s: Sheet normal directions (N, 3)
            - n: Sheet directions (N, 3)
        If intermediate=True, also returns basis, angles, local_basis, and Qepi.
    """
    # Convert parameters from degrees to radians
    for key in params:
        params[key] = np.deg2rad(params[key])

    print('Computing basis vectors')
    basis = compute_basis_vectors(lap, grad)

    print('Computing angles')
    angles = compute_alpha_beta_angles(lap, params)

    print('Computing local basis')
    local_basis = compute_local_basis(basis, angles)

    print('Interpolating basis')
    Q, Qepi = interpolate_local_basis(lap, local_basis)

    print('Done!')
    f = Q[:, :, 0]
    s = Q[:, :, 1]
    n = Q[:, :, 2]

    if intermediate:
        return f, s, n, basis, angles, local_basis, Qepi[:,:,0]

    return f, s, n


def generate_fibers_BiV_Doste_cells(outdir, laplace_results_file, params, return_angles=False, return_intermediate=False):
    """Generate fiber directions for BiV geometry using the Doste method.
    
    Implements the Laplace-Dirichlet rule-based method of Doste et al. (2019)
    to generate fiber directions on a biventricular heart geometry with outflow tracts.
    
    Args:
        outdir: Output directory for fiber files.
        laplace_results_file: Path to the .vtu mesh with Laplace fields defined at nodes.
        params: Dictionary of angle parameters (in degrees). See compute_alpha_beta_angles
            for required keys.
        return_angles: If True, compute and include alpha/beta angles in output.
        return_intermediate: If True, include intermediate basis vectors and angles in output.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh with computed fiber directions and optional
            angle/intermediate data in cell_data.
    """
    
    t1 = time.time()
    print("========================================================")

    # Load Laplace solution    
    result_mesh,lap, grad = loadLaplaceSolnDoste(laplace_results_file)
    
    # Generating output mesh
    output_mesh = copy.deepcopy(result_mesh)
    # Ensure only FIB_DIR is present
    for k in list(output_mesh.cell_data.keys()):
        output_mesh.cell_data.remove(k)
    for k in list(output_mesh.point_data.keys()):
        output_mesh.point_data.remove(k)

    # Generate fiber directions
    out = getFiberDirectionsDoste(lap, grad, params, intermediate=return_intermediate)

    if return_intermediate:
        F, S, T, basis, angles, local_basis, Qepi = out
        result_mesh.cell_data['F'] = F
        result_mesh.cell_data['S'] = S
        result_mesh.cell_data['T'] = T
        result_mesh.cell_data['Qepi'] = Qepi
        for k, v in basis.items():
            result_mesh.cell_data[k] = v
        for k, v in angles.items():
            result_mesh.cell_data[k] = v
        for k, v in local_basis.items():
            # Flatten local basis matrices to store
            flattened = v[:, :, 0]
            result_mesh.cell_data[k] = flattened
    else:
        F, S, T = out
        result_mesh.cell_data['F'] = F
        result_mesh.cell_data['S'] = S
        result_mesh.cell_data['T'] = T

    t2 = time.time()
    print('\n   Total time: %.3fs' % (t2-t1))
    print("========================================================")
    
    # Write the fiber directions to a vtu files
    print("   Writing domains and fibers to VTK data structure")

    fname1 = os.path.join(outdir, "fibersLong.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(F, 'FIB_DIR')
    output_mesh.save(fname1)

    fname1 = os.path.join(outdir, "fibersSheet.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(T, 'FIB_DIR')
    output_mesh.save(fname1)

    fname1 = os.path.join(outdir, "fibersNormal.vtu")
    print("   Writing to vtu file   --->   %s" % (fname1))
    output_mesh.cell_data.set_array(S, 'FIB_DIR')
    output_mesh.save(fname1)

    if return_angles:
        alpha_angle, beta_angle, eC, eCr = get_alpha_beta_angles_Doste(F, lap, grad, params)
        result_mesh.cell_data['Alpha_Angle'] = alpha_angle
        result_mesh.cell_data['Beta_Angle'] = beta_angle
        result_mesh.cell_data['eC'] = eC
        result_mesh.cell_data['eCr'] = eCr


    return result_mesh



def get_alpha_beta_angles_Doste(F, lap, grad, params):
    """Compute alpha and beta angles from fiber directions (Doste method).
    
    This is a sanity check routine that computes the helix and transverse angles
    from given fiber directions using the Doste method's Laplace fields and basis.
    
    Args:
        F: Fiber directions at cells (N, 3).
        lap: Dictionary of Laplace solution values at cells.
        grad: Dictionary of gradient arrays at cells (N, 3).
        params: Dictionary of angle parameters (in degrees).
    
    Returns:
        tuple: A tuple containing:
            - alpha_angle_deg: Helix angles in degrees (N,)
            - beta_angle_deg: Transverse angles in degrees (N,)
            - eC_ref: Reference circumferential vector before rotations (N, 3)
            - Cr_ref: Circumferential vector after alpha rotation only (N, 3)
    """

    # Reconstruct base vectors used by Doste
    basis = compute_basis_vectors(lap, grad)
    eC_global = basis['eC']  # blended circumferential

    # Build a blended longitudinal direction for alpha sign (LV/RV mix)
    ven = lap['ven_trans'][:, None]
    eL_blend = normalize(basis['eL_rv'] * (1.0 - ven) + basis['eL_lv'] * ven)

    # Alpha: signed angle between F and circumferential in the tangent plane,
    # sign taken along longitudinal direction (consistent with Bayer routine, 
    # but negative because the longitudinal direction is opposite to the Doste paper)
    cos_a = np.clip(np.sum(F * eC_global, axis=1), -1.0, 1.0)
    sin_a = np.clip(np.sum(F * eL_blend, axis=1), -1.0, 1.0)
    alpha_angle = -np.sign(sin_a) * np.arccos(np.abs(cos_a))

    # Build reference frame after ONLY alpha rotation (beta = 0)
    angles = compute_alpha_beta_angles(lap, params)  # radians
    Qlv_septum_a = rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'],
                                angles['alpha_septum'], 0.0)
    Qrv_septum_a = rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'],
                                angles['alpha_septum'], 0.0)
    Qlv_epi_a = rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'],
                             angles['alpha_wall_lv'], 0.0)
    Qrv_epi_a = rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'],
                             angles['alpha_wall_rv'], 0.0)

    Qepi_a = bislerp(Qrv_epi_a, Qlv_epi_a, lap['ven_trans'])
    Qendo_a = bislerp(Qrv_septum_a, Qlv_septum_a, lap['ven_trans'])
    Qa = bislerp(Qendo_a, Qepi_a, lap['epi_trans'])

    Cr = Qa[:, :, 0]  # circumferential after alpha-only rotation
    Tr = Qa[:, :, 2]  # transmural after alpha-only rotation

    # Beta: signed angle between F and Cr, sign w.r.t. Tr (negative by convention)
    cos_b = np.clip(np.sum(F * Cr, axis=1), -1.0, 1.0)
    sin_b = np.clip(np.sum(F * Tr, axis=1), -1.0, 1.0)
    beta_angle = np.sign(sin_b) * np.arccos(np.abs(cos_b))

    return np.rad2deg(alpha_angle), np.rad2deg(beta_angle), eC_global, Cr

