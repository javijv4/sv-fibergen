#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Object-oriented fiber generation module for biventricular heart models.

This module provides classes to generate myocardial fiber orientations for
biventricular heart models using Laplace-Dirichlet rule-based methods.

Supports both:
    - Bayer et al. (2012): Truncated BiV geometry
    - Doste et al. (2019): BiV geometry with outflow tracts

References:
    Bayer et al. 2012: https://doi.org/10.1007/s10439-012-0593-5
    Doste et al. 2019: https://doi.org/10.1002/cnm.3185
"""

import os
import numpy as np
import pyvista as pv

import src.quat_utils as qu


class FibGen:
    """Base class for fiber generation.
    
    Provides common utilities for computing fiber orientations from
    Laplace field solutions.
    
    Attributes:
        mesh: PyVista mesh with cell-centered data.
        lap: Dictionary of Laplace solution values at cells.
        grad: Dictionary of gradient arrays at cells (N, 3).
    """
    
    def __init__(self):
        """Initialize the FibGen base class."""
        self.mesh = None
        self.lap = None
        self.grad = None
    
    def normalize(self, x):
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
    
    def _minmax01(self, arr):
        """Scale array to [0, 1] range."""
        arr = np.asarray(arr, dtype=float)
        amin = np.min(arr)
        amax = np.max(arr)
        if amax > amin:
            return (arr - amin) / (amax - amin)
        else:
            return np.ones_like(arr) * 0.5
    
    def _compute_gradients(self, mesh, field_names):
        """Compute gradients for specified fields at points.
        
        Args:
            mesh: PyVista mesh with point data.
            field_names: List of field names to compute gradients for.
            
        Returns:
            PyVista mesh with gradient arrays added to point_data.
        """
        for name in field_names:
            if name not in mesh.point_data:
                raise KeyError(f"Field '{name}' not found in mesh point_data")
            
            # Scale to [0, 1] before gradient computation
            mesh.point_data[name] = self._minmax01(mesh.point_data[name])
            
            gmesh = mesh.compute_derivative(scalars=name, gradient=True, preference='point')
            mesh.point_data[name + "_grad"] = np.asarray(gmesh.point_data["gradient"])
        
        return mesh
    
    def calculate_basis(self, gL, gT):
        """Construct orthogonal coordinate systems from two gradient fields.
        
        Creates an orthonormal basis [eC, eL, eT] for each element where:
        - eL is aligned with gL (normalized longitudinal)
        - eT is orthogonal to eL and in the plane of gT (transmural)
        - eC is the cross product of eL and eT (circumferential)
        
        Args:
            gL: Array of shape (N, 3) representing the longitudinal gradient.
            gT: Array of shape (N, 3) representing the transmural gradient.
        
        Returns:
            np.ndarray: Array of shape (N, 3, 3) where columns are
                [eC (circumferential), eL (longitudinal), eT (transmural)].
        """
        gL = np.asarray(gL, dtype=float)
        gT = np.asarray(gT, dtype=float)
        ne = gL.shape[0]
        
        # eL = normalized longitudinal
        eL = self.normalize(gL)
        
        # eT = gT - proj_{eL}(gT), orthogonal to eL
        proj = np.sum(eL * gT, axis=1)[:, None] * eL
        eT = gT - proj
        eT = self.normalize(eT)
        
        # eC = cross(eL, eT), circumferential
        eC = np.cross(eL, eT, axisa=1, axisb=1)
        eC = self.normalize(eC)
        
        # Build basis matrix Q = [eC, eL, eT]
        Q = np.zeros((ne, 3, 3), dtype=float)
        Q[:, :, 0] = eC
        Q[:, :, 1] = eL
        Q[:, :, 2] = eT
        
        return Q
    
    def calculate_angle(self, trans, endo_value, epi_value):
        """Compute angle varying linearly from endo to epi.
        
        Args:
            trans: Transmural coordinate array (N,), values in [0, 1].
            endo_value: Angle value at endocardium (scalar).
            epi_value: Angle value at epicardium (scalar).
        
        Returns:
            np.ndarray: Angle values at each point (N,).
        """
        return endo_value * (1 - trans) + epi_value * trans
    
    def rotate_basis_matrix(self, Q, alpha, beta):
        """Apply alpha and beta rotations to orthogonal matrices.
        
        Rotates Q by alpha about the z-axis (transmural) and then
        by beta about the y-axis (longitudinal direction).
        
        Args:
            Q: Array of shape (N, 3, 3) containing orthogonal matrices.
            alpha: Array of shape (N,) with rotation angles (radians) about z-axis.
            beta: Array of shape (N,) with rotation angles (radians) about y-axis.
        
        Returns:
            np.ndarray: Array of shape (N, 3, 3) containing rotated matrices.
        """
        Q = np.asarray(Q, dtype=float)
        ne = Q.shape[0]
        
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # Rotation about z-axis (Ra)
        Ra = np.zeros((ne, 3, 3), dtype=float)
        Ra[:, 0, 0] = ca
        Ra[:, 0, 1] = -sa
        Ra[:, 1, 0] = sa
        Ra[:, 1, 1] = ca
        Ra[:, 2, 2] = 1.0
        
        # Rotation about y-axis (Rb)
        Rb = np.zeros((ne, 3, 3), dtype=float)
        Rb[:, 0, 0] = cb
        Rb[:, 0, 2] = sb
        Rb[:, 1, 1] = 1.0
        Rb[:, 2, 0] = -sb
        Rb[:, 2, 2] = cb
        
        # Compose rotations and apply to Q
        RaRb = np.einsum('nij,njk->nik', Ra, Rb)
        Qt = np.einsum('nij,njk->nik', Q, RaRb)
        
        return Qt

    def rotate_basis_rodriguez(self, Q, alpha, beta):
        """Rotate basis using Rodriguez rotation formula (Doste method).
        
        Applies two successive rotations using Rodriguez formula:
        1. Rotate by alpha about the transmural axis (eT)
        2. Rotate by beta about the rotated longitudinal axis
        
        Args:
            Q: Array of shape (N, 3, 3) containing basis matrices.
                Columns are [eC (circumferential), eL (longitudinal), eT (transmural)].
            alpha: Array of shape (N,) with rotation angles (radians) about transmural axis.
            beta: Array of shape (N,) with rotation angles (radians) about rotated longitudinal axis.
        
        Returns:
            np.ndarray: Array of shape (N, 3, 3) containing rotated basis matrices.
        """
        Q = np.asarray(Q, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        
        n = Q.shape[0]
        
        # Extract basis vectors
        eC = Q[:, :, 0]  # Circumferential
        eL = Q[:, :, 1]  # Longitudinal
        eT = Q[:, :, 2]  # Transmural
        
        # Normalize basis vectors
        eC = self.normalize(eC)
        eL = self.normalize(eL)
        eT = self.normalize(eT)
        
        # First rotation: alpha about transmural axis (eT)
        axis = eT
        ca, sa = np.cos(alpha), np.sin(alpha)
        R1 = np.zeros((3, 3, n), dtype=float)
        R1[0, 0] = ca + axis[:, 0]**2 * (1 - ca)
        R1[0, 1] = axis[:, 0] * axis[:, 1] * (1 - ca) - axis[:, 2] * sa
        R1[0, 2] = axis[:, 0] * axis[:, 2] * (1 - ca) + axis[:, 1] * sa
        R1[1, 0] = axis[:, 1] * axis[:, 0] * (1 - ca) + axis[:, 2] * sa
        R1[1, 1] = ca + axis[:, 1]**2 * (1 - ca)
        R1[1, 2] = axis[:, 1] * axis[:, 2] * (1 - ca) - axis[:, 0] * sa
        R1[2, 0] = axis[:, 2] * axis[:, 0] * (1 - ca) - axis[:, 1] * sa
        R1[2, 1] = axis[:, 2] * axis[:, 1] * (1 - ca) + axis[:, 0] * sa
        R1[2, 2] = ca + axis[:, 2]**2 * (1 - ca)
        
        # Apply first rotation
        # Stack as [eC, eL, eT] and transpose to (3, 3, n)
        Q_stack = np.stack([eC, eL, eT], axis=-1).transpose(2, 1, 0)
        QX = np.zeros_like(R1)
        for i in range(n):
            QX[:, :, i] = Q_stack[:, :, i] @ R1[:, :, i]
        
        # Second rotation: beta about rotated longitudinal axis
        axis2 = QX[1, :, :].T  # Rotated longitudinal axis
        cb, sb = np.cos(beta), np.sin(beta)
        R2 = np.zeros((3, 3, n), dtype=float)
        R2[0, 0] = cb + axis2[:, 0]**2 * (1 - cb)
        R2[0, 1] = axis2[:, 0] * axis2[:, 1] * (1 - cb) - axis2[:, 2] * sb
        R2[0, 2] = axis2[:, 0] * axis2[:, 2] * (1 - cb) + axis2[:, 1] * sb
        R2[1, 0] = axis2[:, 1] * axis2[:, 0] * (1 - cb) + axis2[:, 2] * sb
        R2[1, 1] = cb + axis2[:, 1]**2 * (1 - cb)
        R2[1, 2] = axis2[:, 1] * axis2[:, 2] * (1 - cb) - axis2[:, 0] * sb
        R2[2, 0] = axis2[:, 2] * axis2[:, 0] * (1 - cb) - axis2[:, 1] * sb
        R2[2, 1] = axis2[:, 2] * axis2[:, 1] * (1 - cb) + axis2[:, 0] * sb
        R2[2, 2] = cb + axis2[:, 2]**2 * (1 - cb)
        
        # Apply second rotation
        result = np.zeros((n, 3, 3), dtype=float)
        for i in range(n):
            result[i] = (QX[:, :, i] @ R2[:, :, i]).T
        
        return result

    
    def interpolate_basis(self, Q1, Q2, t, correct_slerp=False):
        """Spherical linear interpolation between batches of rotation matrices.
        
        Performs SLERP on rotation matrices represented as quaternions internally.
        
        Args:
            Q1: Array of shape (N, 3, 3) containing starting rotation matrices.
            Q2: Array of shape (N, 3, 3) containing ending rotation matrices.
            t: Array of shape (N,) with interpolation values in [0, 1].
        
        Returns:
            np.ndarray: Array of shape (N, 3, 3) containing interpolated rotation matrices.
        """
        
        # Prepare inputs
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        
        # Ensure shortest path on the unit 4-sphere
        if correct_slerp:
            q1 = np.zeros((len(t), 4), dtype=float)
            q2 = np.zeros((len(t), 4), dtype=float)
            q1, q2 = qu.find_best_quaternions_old(Q1, Q2)
            dot = np.einsum('ni,ni->n', q1, q2)
        else:
            q1 = qu.rotm_to_quat_batch(Q1)
            q2 = qu.rotm_to_quat_batch(Q2)
            dot = np.einsum('ni,ni->n', q1, q2)
            if np.any(dot < 0.0):
                neg_mask = dot < 0.0
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
            tl = t[lin_mask][:, None]
            q[lin_mask] = (1.0 - tl) * q1[lin_mask] + tl * q2[lin_mask]
        
        # Normalize and convert back to rotation matrices
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return qu.quat_to_rotm_batch(q)
    
    def generate_fibers(self, params):
        """Generate fiber directions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate_fibers()")


class FibGenBayer(FibGen):
    """Fiber generator using the Bayer et al. (2012) method.
    
    Suitable for truncated biventricular geometries with and without outflow tracts.
    """
    
    # Field names in Laplace solution
    FIELD_NAMES = ['Trans_EPI', 'Trans_LV', 'Trans_RV', 'Long_AB']
    
    def __init__(self):
        """Initialize the Bayer fiber generator."""
        super().__init__()
    
    def load_laplace_results(self, file_path):
        """Load Laplace-Dirichlet solution for Bayer method.
        
        Args:
            file_path: Path to the .vtu file with Laplace solution.
        
        Returns:
            tuple: (lap, grad) dictionaries with Laplace values and gradients.
        """
        print(f"   Loading Laplace solution <--- {file_path}")
        result_mesh = pv.read(file_path)
        
        print("   Computing gradients at points")
        result_mesh = self._compute_gradients(result_mesh, self.FIELD_NAMES)
        
        # Convert point-data to cell-data
        mesh_cells = result_mesh.point_data_to_cell_data()
        self.mesh = mesh_cells
        
        # Extract Laplace values and gradients 
        self.lap = {}
        self.grad = {}
        
        for key in self.FIELD_NAMES:
            self.lap[key] = np.asarray(mesh_cells.cell_data[key])
            self.grad[key] = np.asarray(mesh_cells.cell_data[key + "_grad"])
        
        
        return self.lap, self.grad
    
    def generate_fibers(self, params):
        """Generate fiber directions using the Bayer method.
        
        Args:
            params: Dictionary with keys:
                - ALFA_END: Endocardial helix angle (degrees)
                - ALFA_EPI: Epicardial helix angle (degrees)
                - BETA_END: Endocardial transverse angle (degrees)
                - BETA_EPI: Epicardial transverse angle (degrees)
        
        Returns:
            tuple: (F, S, T) fiber, sheet, and normal directions (N, 3) each.
        """
        if self.lap is None or self.grad is None:
            raise ValueError("Must call load_laplace_results() first")
        
        # Convert parameters to radians (consistent with Doste method)
        params = {k: np.deg2rad(v) for k, v in params.items()}
                
        print("   Computing fiber directions at cells")
        
        # Interpolation factor between LV and RV
        d = self.lap['Trans_RV'] / (self.lap['Trans_LV'] + self.lap['Trans_RV'])
        
        # Septum angles (interpolated between LV and RV)
        alfaS = self.calculate_angle(d, params['ALFA_END'], -params['ALFA_END'])
        betaS = self.calculate_angle(d, params['BETA_END'], -params['BETA_END'])
        
        # Wall angles (interpolated from endo to epi)
        alfaW = self.calculate_angle(self.lap['Trans_EPI'], params['ALFA_END'], params['ALFA_EPI'])
        betaW = self.calculate_angle(self.lap['Trans_EPI'], params['BETA_END'], params['BETA_EPI'])
        
        # Build LV and RV basis
        Q_LV0 = self.calculate_basis(self.grad['Long_AB'], -self.grad['Trans_LV'])
        Q_LV = self.rotate_basis_matrix(Q_LV0, alfaS, betaS)
        
        Q_RV0 = self.calculate_basis(self.grad['Long_AB'], self.grad['Trans_RV'])
        Q_RV = self.rotate_basis_matrix(Q_RV0, alfaS, betaS)
        
        # Interpolate between LV and RV (endocardial layer)
        Q_END = self.interpolate_basis(Q_LV, Q_RV, d, correct_slerp=False)

        # Flip circumferential and transmural directions in RV
        Q_END[d > 0.5,:,0] = -Q_END[d > 0.5,:,0]
        Q_END[d > 0.5,:,2] = -Q_END[d > 0.5,:,2]
        
        # Build epicardial basis
        Q_EPI0 = self.calculate_basis(self.grad['Long_AB'], self.grad['Trans_EPI'])
        Q_EPI = self.rotate_basis_matrix(Q_EPI0, alfaW, betaW)
        
        # Interpolate from endo to epi
        FST = self.interpolate_basis(Q_END, Q_EPI, self.lap['Trans_EPI'], correct_slerp=False)
        
        F = FST[:, :, 0]  # Fiber direction
        S = FST[:, :, 1]  # Sheet normal
        T = FST[:, :, 2]  # Sheet direction

        # Save to mesh cell data
        self.mesh.cell_data['fiber'] = F
        self.mesh.cell_data['sheet'] = T
        self.mesh.cell_data['sheet-normal'] = S

        
        self.mesh.cell_data['d'] = d
        self.mesh.cell_data['alfaS'] = alfaS
        self.mesh.cell_data['betaS'] = betaS
        self.mesh.cell_data['alfaW'] = alfaW
        self.mesh.cell_data['betaW'] = betaW
        
        self.mesh.cell_data['eC_LV0'] = Q_LV0[:, :, 0]
        self.mesh.cell_data['eL_LV0'] = Q_LV0[:, :, 1]
        self.mesh.cell_data['eT_LV0'] = Q_LV0[:, :, 2]
        
        self.mesh.cell_data['eC_RV0'] = Q_RV0[:, :, 0]
        self.mesh.cell_data['eL_RV0'] = Q_RV0[:, :, 1]
        self.mesh.cell_data['eT_RV0'] = Q_RV0[:, :, 2]
        
        self.mesh.cell_data['eC_LV'] = Q_LV[:, :, 0]
        self.mesh.cell_data['eL_LV'] = Q_LV[:, :, 1]
        self.mesh.cell_data['eT_LV'] = Q_LV[:, :, 2]
        
        self.mesh.cell_data['eC_RV'] = Q_RV[:, :, 0]
        self.mesh.cell_data['eL_RV'] = Q_RV[:, :, 1]
        self.mesh.cell_data['eT_RV'] = Q_RV[:, :, 2]
        
        self.mesh.cell_data['eC_END'] = Q_END[:, :, 0]
        self.mesh.cell_data['eL_END'] = Q_END[:, :, 1]
        self.mesh.cell_data['eT_END'] = Q_END[:, :, 2]
        
        self.mesh.cell_data['eC_EPI0'] = Q_EPI0[:, :, 0]
        self.mesh.cell_data['eL_EPI0'] = Q_EPI0[:, :, 1]
        self.mesh.cell_data['eT_EPI0'] = Q_EPI0[:, :, 2]
        
        self.mesh.cell_data['eC_EPI'] = Q_EPI[:, :, 0]
        self.mesh.cell_data['eL_EPI'] = Q_EPI[:, :, 1]
        self.mesh.cell_data['eT_EPI'] = Q_EPI[:, :, 2]
        
        self.mesh.cell_data['F'] = F
        self.mesh.cell_data['S'] = S
        self.mesh.cell_data['T'] = T
        
        print("   Writing mesh to check.vtu")
        self.mesh.save('check.vtu')
    
        return F, S, T
        

    def generate_fibers_new(self, params):
        """Generate fiber directions using the Bayer method.
        
        Args:
            params: Dictionary with keys:
                - ALFA_END: Endocardial helix angle (degrees)
                - ALFA_EPI: Epicardial helix angle (degrees)
                - BETA_END: Endocardial transverse angle (degrees)
                - BETA_EPI: Epicardial transverse angle (degrees)
        
        Returns:
            tuple: (F, S, T) fiber, sheet, and normal directions (N, 3) each.
        """
        if self.lap is None or self.grad is None:
            raise ValueError("Must call load_laplace_results() first")
        
        # Convert parameters to radians (consistent with Doste method)
        params = {k: np.deg2rad(v) for k, v in params.items()}
                
        print("   Computing fiber directions at cells")
        
        # Interpolation factor between LV and RV
        trans_biv = self.lap['Trans_RV'] / (self.lap['Trans_LV'] + self.lap['Trans_RV'])
        sep = np.abs(trans_biv - 0.5)
        sep = (sep - np.min(sep)) / (np.max(sep) - np.min(sep))
        alpha_septum = -self.lap['Trans_RV']*trans_biv + self.lap['Trans_LV']*(1-trans_biv)
        print(alpha_septum[811948])
        betaS = params['BETA_END'] * sep * self.lap['Trans_LV'] + params['BETA_EPI'] * sep * self.lap['Trans_RV']
                
        # Wall angles (interpolated from endo to epi)
        alfaW = self.calculate_angle(self.lap['Trans_EPI'], params['ALFA_END'], params['ALFA_EPI'])
        betaW = self.calculate_angle(self.lap['Trans_EPI'], params['BETA_END'], params['BETA_EPI'])
        
        # Build LV and RV basis
        Q_LV0 = self.calculate_basis(self.grad['Long_AB'], -self.grad['Trans_LV'])
        Q_LV = self.rotate_basis_matrix(Q_LV0, alpha_septum, betaS)
        
        Q_RV0 = self.calculate_basis(self.grad['Long_AB'], self.grad['Trans_RV'])
        Q_RV = self.rotate_basis_matrix(Q_RV0, alpha_septum, betaS)
        
        # Interpolate between LV and RV (endocardial layer)
        Q_END = self.interpolate_basis(Q_LV, Q_RV, trans_biv, correct_slerp=False)

        # Flip circumferential and transmural directions in RV
        Q_END[trans_biv > 0.5,:,0] = -Q_END[trans_biv > 0.5,:,0]
        Q_END[trans_biv > 0.5,:,2] = -Q_END[trans_biv > 0.5,:,2]
        
        # Build epicardial basis
        Q_EPI0 = self.calculate_basis(self.grad['Long_AB'], self.grad['Trans_EPI'])
        Q_EPI = self.rotate_basis_matrix(Q_EPI0, alfaW, betaW)
        
        # Interpolate from endo to epi
        FST = self.interpolate_basis(Q_END, Q_EPI, self.lap['Trans_EPI'], correct_slerp=False)
        
        F = FST[:, :, 0]  # Fiber direction
        S = FST[:, :, 1]  # Sheet normal
        T = FST[:, :, 2]  # Sheet direction

        # Save to mesh cell data
        self.mesh.cell_data['fiber'] = F
        self.mesh.cell_data['sheet'] = T
        self.mesh.cell_data['sheet-normal'] = S

        
        print(alpha_septum[811948])
        self.mesh.cell_data['trans_biv'] = trans_biv
        self.mesh.cell_data['sep'] = sep
        self.mesh.cell_data['alfaS'] = alpha_septum
        self.mesh.cell_data['betaS'] = betaS
        self.mesh.cell_data['alfaW'] = alfaW
        self.mesh.cell_data['betaW'] = betaW
        
        self.mesh.cell_data['eC_LV0'] = Q_LV0[:, :, 0]
        self.mesh.cell_data['eL_LV0'] = Q_LV0[:, :, 1]
        self.mesh.cell_data['eT_LV0'] = Q_LV0[:, :, 2]
        
        self.mesh.cell_data['eC_RV0'] = Q_RV0[:, :, 0]
        self.mesh.cell_data['eL_RV0'] = Q_RV0[:, :, 1]
        self.mesh.cell_data['eT_RV0'] = Q_RV0[:, :, 2]
        
        self.mesh.cell_data['eC_LV'] = Q_LV[:, :, 0]
        self.mesh.cell_data['eL_LV'] = Q_LV[:, :, 1]
        self.mesh.cell_data['eT_LV'] = Q_LV[:, :, 2]
        
        self.mesh.cell_data['eC_RV'] = Q_RV[:, :, 0]
        self.mesh.cell_data['eL_RV'] = Q_RV[:, :, 1]
        self.mesh.cell_data['eT_RV'] = Q_RV[:, :, 2]
        
        self.mesh.cell_data['eC_END'] = Q_END[:, :, 0]
        self.mesh.cell_data['eL_END'] = Q_END[:, :, 1]
        self.mesh.cell_data['eT_END'] = Q_END[:, :, 2]
        
        self.mesh.cell_data['eC_EPI0'] = Q_EPI0[:, :, 0]
        self.mesh.cell_data['eL_EPI0'] = Q_EPI0[:, :, 1]
        self.mesh.cell_data['eT_EPI0'] = Q_EPI0[:, :, 2]
        
        self.mesh.cell_data['eC_EPI'] = Q_EPI[:, :, 0]
        self.mesh.cell_data['eL_EPI'] = Q_EPI[:, :, 1]
        self.mesh.cell_data['eT_EPI'] = Q_EPI[:, :, 2]
        
        self.mesh.cell_data['F'] = F
        self.mesh.cell_data['S'] = S
        self.mesh.cell_data['T'] = T
        
        print("   Writing mesh to check.vtu")
        self.mesh.save('check.vtu')
    
        return F, S, T
        


    #----------------------------------------------------------------------
    def generate_fibers_old(self, params):
        '''
        Compute the fiber directions at the center of each cell
        '''
        params = {k: np.deg2rad(v) for k, v in params.items()}

        numCells = self.mesh.GetNumberOfCells()

        print("   Computing fiber directions at cells")
        F = np.zeros((numCells, 3))
        S = np.zeros((numCells, 3))
        T = np.zeros((numCells, 3))

        j = 1
        k = 1
        print ("      Progress "),

        Q_LV_arr = np.zeros((numCells, 3))
        Q_RV_arr = np.zeros((numCells, 3))
        Q_END_arr = np.zeros((numCells, 3))
        Q_EPI_arr = np.zeros((numCells, 3))
        for iCell in range(0, numCells):
            phiEP = self.lap['Trans_EPI'][iCell]
            phiLV = self.lap['Trans_LV'][iCell]
            phiRV = self.lap['Trans_RV'][iCell]

            gPhiEP = self.grad['Trans_EPI'][iCell, :]
            gPhiLV = self.grad['Trans_LV'][iCell, :]
            gPhiRV = self.grad['Trans_RV'][iCell, :]
            gPhiAB = self.grad['Long_AB'][iCell, :]

            d = phiRV / (phiLV + phiRV)
            alfaS = params['ALFA_END'] * (1 - d) - params['ALFA_END'] * d
            betaS = params['BETA_END'] * (1 - d) - params['BETA_END'] * d
            alfaW = params['ALFA_END'] * (1 - phiEP) + params['ALFA_EPI'] * phiEP
            betaW = params['BETA_END'] * (1 - phiEP) + params['BETA_EPI'] * phiEP

            Q_LV = axis(gPhiAB, - gPhiLV)
            Q_LV = orient(Q_LV, alfaS, betaS)
            Q_LV_arr[iCell, :] = Q_LV[:, 0]

            Q_RV = axis(gPhiAB, gPhiRV)
            Q_RV = orient(Q_RV, alfaS, betaS)
            Q_RV_arr[iCell, :] = Q_RV[:, 0]
            Q_END = bislerp(Q_LV, Q_RV, d)
            Q_END_arr[iCell, :] = Q_END[:, 0]

            Q_EPI = axis(gPhiAB, gPhiEP)
            Q_EPI = orient(Q_EPI, alfaW, betaW)
            Q_EPI_arr[iCell, :] = Q_EPI[:, 0]
            FST = bislerp(Q_END, Q_EPI, phiEP)

            F[iCell, :] = np.array([FST[0, 0], FST[1, 0], FST[2, 0]])
            S[iCell, :] = np.array([FST[0, 1], FST[1, 1], FST[2, 1]])
            T[iCell, :] = np.array([FST[0, 2], FST[1, 2], FST[2, 2]])
            if iCell==j:
                print ("%d%%  " % ((k-1)*10)),
                k = k + 1
                j = int(float((k-1)*numCells)/10.0)
        print("[Done!]")

        self.mesh.cell_data['fiber'] = F
        self.mesh.cell_data['sheet'] = T
        self.mesh.cell_data['sheet-normal'] = S

        
        self.mesh.cell_data['d'] = d
        self.mesh.cell_data['alfaS'] = alfaS
        self.mesh.cell_data['betaS'] = betaS
        self.mesh.cell_data['alfaW'] = alfaW
        self.mesh.cell_data['betaW'] = betaW
        
        self.mesh.cell_data['eC_LV'] = Q_LV_arr[:, 0]
        self.mesh.cell_data['eL_LV'] = Q_LV_arr[:, 1]
        self.mesh.cell_data['eT_LV'] = Q_LV_arr[:, 2]
        
        self.mesh.cell_data['eC_RV'] = Q_RV_arr[:, 0]
        self.mesh.cell_data['eL_RV'] = Q_RV_arr[:, 1]
        self.mesh.cell_data['eT_RV'] = Q_RV_arr[:, 2]
        
        self.mesh.cell_data['eC_END'] = Q_END_arr[:, 0]
        self.mesh.cell_data['eL_END'] = Q_END_arr[:, 1]
        self.mesh.cell_data['eT_END'] = Q_END_arr[:, 2]
        
        self.mesh.cell_data['eC_EPI'] = Q_EPI_arr[:, 0]
        self.mesh.cell_data['eL_EPI'] = Q_EPI_arr[:, 1]
        self.mesh.cell_data['eT_EPI'] = Q_EPI_arr[:, 2]
        
        self.mesh.cell_data['F'] = F
        self.mesh.cell_data['S'] = S
        self.mesh.cell_data['T'] = T

        return F, S, T
    

    def get_angle_fields(self, params):
        "Helper function to compute a global alpha and beta angle fields."

        # Interpolation factor between LV and RV
        d = self.lap['Trans_RV'] / (self.lap['Trans_LV'] + self.lap['Trans_RV'])
        
        # Septum angles (interpolated between LV and RV)
        alfaS = self.calculate_angle(d, params['ALFA_END'], -params['ALFA_END'])
        betaS = self.calculate_angle(d, params['BETA_END'], -params['BETA_END'])
        alfaS = np.abs(alfaS)   # Note this is doing the same as flipping the sign
        betaS = np.abs(betaS)   # Note this is doing the same as flipping the sign
        
        # Wall angles (interpolated from endo to epi)
        alfaW = self.calculate_angle(self.lap['Trans_EPI'], params['ALFA_END'], params['ALFA_EPI'])
        betaW = self.calculate_angle(self.lap['Trans_EPI'], params['BETA_END'], params['BETA_EPI'])

        alfa = alfaS * (1 - self.lap['Trans_EPI']) + alfaW * self.lap['Trans_EPI']
        beta = betaS * (1 - self.lap['Trans_EPI']) + betaW * self.lap['Trans_EPI']

        return alfa, beta
        


    def write_fibers(self, outdir):
        # Create a copy of the mesh without any data
        mesh_out = self.mesh.copy(deep=True)
        mesh_out.clear_data()

        # Fiber direction
        mesh_out.cell_data['FIB_DIR'] = self.mesh.cell_data['fiber']
        mesh_out.save(os.path.join(outdir, "fiber.vtu"))

        # Sheet direction
        mesh_out.cell_data['FIB_DIR'] = self.mesh.cell_data['sheet']
        mesh_out.save(os.path.join(outdir, "sheet.vtu"))

        # Normal direction
        mesh_out.cell_data['FIB_DIR'] = self.mesh.cell_data['sheet-normal']
        mesh_out.save(os.path.join(outdir, "normal.vtu"))


def normalize(u):
    '''
    Calculate the normalized vector of a given vector
    '''
    u_norm = np.linalg.norm(u)
    if u_norm > 0.0:
        return u / u_norm
    return u

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
    
    Rb = np.array([ [cb,  0.0,  -sb],
                    [0.0, 1.0,  0.0],
                    [sb,  0.0,   cb]])

    Qt = np.matmul(Q, np.matmul(Ra, Rb))

    return Qt


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

class FibGenDoste(FibGen):
    """Fiber generator using the Doste et al. (2019) method.
    
    Suitable for biventricular geometries with outflow tracts.
    """
    
    # Field names in Laplace solution
    FIELD_NAMES = ['Trans_BiV', 'Long_AV', 'Long_MV', 'Long_PV', 'Long_TV',
                   'Weight_LV', 'Weight_RV', 'Trans_EPI', 'Trans_LV', 'Trans_RV']
    
    def __init__(self):
        """Initialize the Doste fiber generator."""
        super().__init__()
    
    def load_laplace_results(self, file_path):
        """Load Laplace-Dirichlet solution for Doste method.
        
        Args:
            file_path: Path to the .vtu file with Laplace solution.
        
        Returns:
            tuple: (lap, grad) dictionaries with Laplace values and gradients.
        """
        print(f"   Loading Laplace solution <--- {file_path}")
        result_mesh = pv.read(file_path)
        
        print("   Computing gradients at points")
        result_mesh = self._compute_gradients(result_mesh, self.FIELD_NAMES)
        
        # Convert point-data to cell-data
        mesh_cells = result_mesh.point_data_to_cell_data()
        self.mesh = mesh_cells
        
        # Extract Laplace values and gradients using mapped names
        self.lap = {}
        self.grad = {}
        
        for key in self.FIELD_NAMES:
            self.lap[key] = np.asarray(mesh_cells.cell_data[key])
            self.grad[key] = np.asarray(mesh_cells.cell_data[key + "_grad"])
        
        return self.lap, self.grad
    
    def _redistribute_weight(self, weight, up, low):
        """Redistribute weight values to center their distribution.
        
        Args:
            weight: Array of weight values.
            up: Upper quantile threshold.
            low: Lower quantile threshold.
        
        Returns:
            np.ndarray: Redistributed weight values in [0, 1].
        """
        new_weight = weight.copy()
        
        upper_lim = np.quantile(weight, up)
        while upper_lim == 0:
            up += 0.1
            upper_lim = np.quantile(weight, up)
        
        lower_lim = np.quantile(weight, low)
        
        new_weight[new_weight > upper_lim] = upper_lim
        new_weight[new_weight < lower_lim] = lower_lim
        
        return (new_weight - np.min(new_weight)) / (np.max(new_weight) - np.min(new_weight))
    
    def _compute_basis_vectors(self):
        """Compute local orthogonal basis vectors for LV and RV.
        
        Returns:
            dict: Dictionary with basis vectors for LV, RV, and global.
        """
        lap, grad = self.lap, self.grad
        
        # Calculate combined LV longitudinal
        lv_glong = (grad['Long_MV'] * lap['Weight_LV'][:, None] + 
                   grad['Long_AV'] * (1 - lap['Weight_LV'][:, None]))

        # Calculate LV basis
        Q_lv = self.calculate_basis(lv_glong, grad['Trans_LV'])
        eC_lv = Q_lv[:, :, 0]  # Circumferential
        eL_lv = Q_lv[:, :, 1]  # Longitudinal
        eT_lv = Q_lv[:, :, 2]  # Transmural
        
        # Calculate combined RV longitudinal
        rv_glong = (grad['Long_TV'] * lap['Weight_RV'][:, None] + 
                   grad['Long_PV'] * (1 - lap['Weight_RV'][:, None]))
        Q_rv = self.calculate_basis(rv_glong, grad['Trans_RV'])
        eC_rv = Q_rv[:, :, 0]  # Circumferential
        eL_rv = Q_rv[:, :, 1]  # Longitudinal
        eT_rv = Q_rv[:, :, 2]  # Transmural
        
        # Global circumferential (blended)
        eC = eC_rv * (1 - lap['Trans_BiV'][:, None]) + eC_lv * lap['Trans_BiV'][:, None]
        eC = self.normalize(eC)
        
        return {
            'eC_lv': eC_lv, 'eT_lv': eT_lv, 'eL_lv': eL_lv,
            'eC_rv': eC_rv, 'eT_rv': eT_rv, 'eL_rv': eL_rv,
            'eC': eC
        }
    
    def _compute_angles(self, params):
        """Compute spatially-varying alpha and beta angles.
        
        Args:
            params: Dictionary with angle parameters (in radians).
        
        Returns:
            dict: Dictionary of angle arrays.
        """
        lap = self.lap
        
        # Redistribute weights
        lv_weight = self._redistribute_weight(lap['Weight_LV'], 0.7, 0.01)
        rv_weight = self._redistribute_weight(lap['Weight_RV'], 0.1, 0.001)
        
        # LV angles
        alpha_lv_endo = params['AENDOLV'] * lv_weight + params['AOTENDOLV'] * (1 - lv_weight)
        alpha_lv_epi = params['AEPILV'] * lv_weight + params['AOTEPILV'] * (1 - lv_weight)
        alpha_wall_lv = self.calculate_angle(lap['Trans_EPI'], alpha_lv_endo, alpha_lv_epi)
        beta_wall_lv = self.calculate_angle(lap['Trans_EPI'], params['BENDOLV'], params['BEPILV']) * lv_weight
        
        # RV angles
        alpha_rv_endo = params['AENDORV'] * rv_weight + params['AOTENDORV'] * (1 - rv_weight)
        alpha_rv_epi = params['AEPIRV'] * rv_weight + params['AOTEPIRV'] * (1 - rv_weight)
        alpha_wall_rv = self.calculate_angle(lap['Trans_EPI'], alpha_rv_endo, alpha_rv_epi)
        beta_wall_rv = self.calculate_angle(lap['Trans_EPI'], params['BENDORV'], params['BEPIRV']) * rv_weight
        
        # Septum angles
        sep = np.abs(lap['Trans_BiV'] - 0.5)
        sep = (sep - np.min(sep)) / (np.max(sep) - np.min(sep))
        alpha_septum = alpha_lv_endo * sep * lap['Trans_LV'] + alpha_rv_endo * sep * lap['Trans_RV']
        beta_septum = params['BENDOLV'] * lap['Trans_LV'] * lv_weight + params['BENDORV'] * lap['Trans_RV'] * rv_weight
        
        return {
            'alpha_wall_lv': alpha_wall_lv, 'beta_wall_lv': beta_wall_lv,
            'alpha_wall_rv': alpha_wall_rv, 'beta_wall_rv': beta_wall_rv,
            'alpha_septum': alpha_septum, 'beta_septum': beta_septum
        }
    
    
    def generate_fibers(self, params):
        """Generate fiber directions using the Doste method.
        
        Args:
            params: Dictionary with angle parameters (in degrees):
                - AENDOLV, AEPILV: LV endo/epi helix angles
                - AENDORV, AEPIRV: RV endo/epi helix angles
                - AOTENDOLV, AOTEPILV: LV outflow tract angles
                - AOTENDORV, AOTEPIRV: RV outflow tract angles
                - BENDOLV, BEPILV: LV endo/epi transverse angles
                - BENDORV, BEPIRV: RV endo/epi transverse angles
        
        Returns:
            tuple: (F, S, T) fiber, sheet, and normal directions (N, 3) each.
        """
        if self.lap is None or self.grad is None:
            raise ValueError("Must call load_laplace_results() first")
        
        # Convert parameters to radians (consistent with Bayer method)
        params_rad = {k: np.deg2rad(v) for k, v in params.items()}
        
        print("   Computing basis vectors")
        basis = self._compute_basis_vectors()
        
        print("   Computing angles")
        angles = self._compute_angles(params_rad)
        
        print("   Computing local basis")
        # Build basis matrices from vectors
        Q_lv = np.stack([basis['eC_lv'], basis['eL_lv'], basis['eT_lv']], axis=-1)
        Q_rv = np.stack([basis['eC_rv'], basis['eL_rv'], basis['eT_rv']], axis=-1)
        
        # Septum basis
        Qlv_sep = self.rotate_basis_rodriguez(
            Q_lv, angles['alpha_septum'], angles['beta_septum']
        )
        Qrv_sep = self.rotate_basis_rodriguez(
            Q_rv, angles['alpha_septum'], angles['beta_septum']
        )
        
        # Wall basis
        Qlv_wall = self.rotate_basis_rodriguez(
            Q_lv, angles['alpha_wall_lv'], angles['beta_wall_lv']
        )
        Qrv_wall = self.rotate_basis_rodriguez(
            Q_rv, angles['alpha_wall_rv'], angles['beta_wall_rv']
        )
        
        print("   Interpolating basis")

        # Get discontinous septal fibers
        Qsep = Qrv_sep.copy()
        Qsep[self.lap['Trans_BiV'] > 0.5] = Qlv_sep[self.lap['Trans_BiV'] > 0.5]
        
        # Interpolate across ventricles
        Qepi = self.interpolate_basis(Qrv_wall, Qlv_wall, self.lap['Trans_BiV'])
        
        # Interpolate from endo to epi
        Q = self.interpolate_basis(Qsep, Qepi, self.lap['Trans_EPI'])
        
        print("   Done!")
        F = Q[:, :, 0]  # Fiber direction
        S = Q[:, :, 1]  # Sheet normal
        T = Q[:, :, 2]  # Sheet direction
        
        for k, v in basis.items():
            self.mesh.cell_data[k] = v
        for k, v in angles.items():
            self.mesh.cell_data[k] = v
        for i in range(Q_lv.shape[2]):
            self.mesh.cell_data[f'Q_lv_{i}'] = Q_lv[:, :, i]
            self.mesh.cell_data[f'Q_rv_{i}'] = Q_rv[:, :, i]
            self.mesh.cell_data[f'Qlv_sep_{i}'] = Qlv_sep[:, :, i]
            self.mesh.cell_data[f'Qrv_sep_{i}'] = Qrv_sep[:, :, i]
            self.mesh.cell_data[f'Qlv_wall_{i}'] = Qlv_wall[:, :, i]
            self.mesh.cell_data[f'Qrv_wall_{i}'] = Qrv_wall[:, :, i]
            self.mesh.cell_data[f'Qepi_{i}'] = Qepi[:, :, i]
            self.mesh.cell_data[f'Qsep_{i}'] = Qsep[:, :, i]
            self.mesh.cell_data[f'Q_{i}'] = Q[:, :, i]  

        self.mesh.cell_data['F'] = F
        self.mesh.cell_data['S'] = S
        self.mesh.cell_data['T'] = T
        
        print("   Writing mesh to check.vtu")
        self.mesh.save('check.vtu')
        
        return F, S, T
