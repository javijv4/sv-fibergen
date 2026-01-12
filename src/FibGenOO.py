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

import numpy as np
import pyvista as pv


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
    
    @staticmethod
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
    
    @staticmethod
    def _minmax01(arr):
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
    
    @staticmethod
    def calculate_basis(gL, gT):
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
        eL = FibGen.normalize(gL)
        
        # eT = gT - proj_{eL}(gT), orthogonal to eL
        proj = np.sum(eL * gT, axis=1)[:, None] * eL
        eT = gT - proj
        eT = FibGen.normalize(eT)
        
        # eC = cross(eL, eT), circumferential
        eC = np.cross(eL, eT, axisa=1, axisb=1)
        eC = FibGen.normalize(eC)
        
        # Build basis matrix Q = [eC, eL, eT]
        Q = np.zeros((ne, 3, 3), dtype=float)
        Q[:, :, 0] = eC
        Q[:, :, 1] = eL
        Q[:, :, 2] = eT
        
        return Q
    
    @staticmethod
    def calculate_angle(trans, endo_value, epi_value):
        """Compute angle varying linearly from endo to epi.
        
        Args:
            trans: Transmural coordinate array (N,), values in [0, 1].
            endo_value: Angle value at endocardium (scalar).
            epi_value: Angle value at epicardium (scalar).
        
        Returns:
            np.ndarray: Angle values at each point (N,).
        """
        return endo_value * (1 - trans) + epi_value * trans
    
    @staticmethod
    def rotate_basis_matrix(Q, alpha, beta):
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

    @staticmethod
    def rotate_basis_rodriguez(Q, alpha, beta):
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
        eC = FibGen.normalize(eC)
        eL = FibGen.normalize(eL)
        eT = FibGen.normalize(eT)
        
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

    
    @staticmethod
    def interpolate_basis(Q1, Q2, t):
        """Spherical linear interpolation between batches of rotation matrices.
        
        Performs SLERP on rotation matrices represented as quaternions internally.
        
        Args:
            Q1: Array of shape (N, 3, 3) containing starting rotation matrices.
            Q2: Array of shape (N, 3, 3) containing ending rotation matrices.
            t: Array of shape (N,) with interpolation values in [0, 1].
        
        Returns:
            np.ndarray: Array of shape (N, 3, 3) containing interpolated rotation matrices.
        """
        def rotm_to_quat_batch(R):
            # R: (N,3,3) -> q: (N,4) [w,x,y,z]
            trace = np.einsum('nii->n', R)
            q = np.zeros((R.shape[0], 4), dtype=float)
            
            # Branch where trace is positive
            mask_t = trace > 0.0
            if np.any(mask_t):
                S = np.sqrt(trace[mask_t] + 1.0) * 2.0
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
                
                for case_idx, (i, j, k) in enumerate([(0, 1, 2), (1, 0, 2), (2, 0, 1)]):
                    mask_case = idx == case_idx
                    if np.any(mask_case):
                        S = np.sqrt(1.0 + Rf[mask_case, i, i] - Rf[mask_case, j, j] - Rf[mask_case, k, k]) * 2.0
                        rows = mf_idx[mask_case]
                        if case_idx == 0:
                            q[rows, 0] = (Rf[mask_case, 2, 1] - Rf[mask_case, 1, 2]) / S
                            q[rows, 1] = 0.25 * S
                            q[rows, 2] = (Rf[mask_case, 0, 1] + Rf[mask_case, 1, 0]) / S
                            q[rows, 3] = (Rf[mask_case, 0, 2] + Rf[mask_case, 2, 0]) / S
                        elif case_idx == 1:
                            q[rows, 0] = (Rf[mask_case, 0, 2] - Rf[mask_case, 2, 0]) / S
                            q[rows, 1] = (Rf[mask_case, 0, 1] + Rf[mask_case, 1, 0]) / S
                            q[rows, 2] = 0.25 * S
                            q[rows, 3] = (Rf[mask_case, 1, 2] + Rf[mask_case, 2, 1]) / S
                        else:
                            q[rows, 0] = (Rf[mask_case, 1, 0] - Rf[mask_case, 0, 1]) / S
                            q[rows, 1] = (Rf[mask_case, 0, 2] + Rf[mask_case, 2, 0]) / S
                            q[rows, 2] = (Rf[mask_case, 1, 2] + Rf[mask_case, 2, 1]) / S
                            q[rows, 3] = 0.25 * S
            
            # Normalize for numerical safety
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            return q
        
        def quat_to_rotm_batch(q):
            # q: (N,4) [w,x,y,z] -> R: (N,3,3)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            x2, y2, z2 = x * x, y * y, z * z
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z
            
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
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
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
            tl = t[lin_mask][:, None]
            q[lin_mask] = (1.0 - tl) * q1[lin_mask] + tl * q2[lin_mask]
        
        # Normalize and convert back to rotation matrices
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return quat_to_rotm_batch(q)
    
    def generate_fibers(self, params):
        """Generate fiber directions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate_fibers()")


class FibGenBayer(FibGen):
    """Fiber generator using the Bayer et al. (2012) method.
    
    Suitable for truncated biventricular geometries with and without outflow tracts.
    """
    
    # Field names in Laplace solution
    FIELD_NAMES = ['Phi_BiV_EPI', 'Phi_BiV_LV', 'Phi_BiV_RV', 'Phi_BiV_AB']
    
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
        self.lap = {
            'epi': np.asarray(mesh_cells.cell_data['Phi_BiV_EPI']),
            'lv': np.asarray(mesh_cells.cell_data['Phi_BiV_LV']),
            'rv': np.asarray(mesh_cells.cell_data['Phi_BiV_RV']),
            'ab': np.asarray(mesh_cells.cell_data['Phi_BiV_AB']),
        }
        
        self.grad = {
            'epi': np.asarray(mesh_cells.cell_data['Phi_BiV_EPI_grad']),
            'lv': np.asarray(mesh_cells.cell_data['Phi_BiV_LV_grad']),
            'rv': np.asarray(mesh_cells.cell_data['Phi_BiV_RV_grad']),
            'ab': np.asarray(mesh_cells.cell_data['Phi_BiV_AB_grad']),
        }
        
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
        d = self.lap['rv'] / (self.lap['lv'] + self.lap['rv'])
        
        # Septum angles (interpolated between LV and RV)
        alfaS = self.calculate_angle(d, params['ALFA_END'], params['ALFA_END'])
        betaS = self.calculate_angle(d, params['BETA_END'], params['BETA_END'])
        
        # Wall angles (interpolated from endo to epi)
        alfaW = self.calculate_angle(self.lap['epi'], params['ALFA_END'], params['ALFA_EPI'])
        betaW = self.calculate_angle(self.lap['epi'], params['BETA_END'], params['BETA_EPI'])
        
        # Build LV and RV basis
        Q_LV0 = self.calculate_basis(self.grad['ab'], -self.grad['lv'])
        Q_LV = self.rotate_basis_matrix(Q_LV0, alfaS, betaS)
        
        Q_RV0 = self.calculate_basis(self.grad['ab'], self.grad['rv'])
        Q_RV = self.rotate_basis_matrix(Q_RV0, alfaS, -betaS)
        
        # Interpolate between LV and RV (endocardial layer)
        Q_END = self.interpolate_basis(Q_LV, Q_RV, d)
        # Flip for consistency
        Q_END[d > 0.5, :, 0] = -Q_END[d > 0.5, :, 0]
        Q_END[d > 0.5, :, 2] = -Q_END[d > 0.5, :, 2]
        
        # Build epicardial basis
        Q_EPI0 = self.calculate_basis(self.grad['ab'], self.grad['epi'])
        Q_EPI = self.rotate_basis_matrix(Q_EPI0, alfaW, betaW)
        
        # Interpolate from endo to epi
        FST = self.interpolate_basis(Q_END, Q_EPI, self.lap['epi'])
        
        F = FST[:, :, 0]  # Fiber direction
        S = FST[:, :, 1]  # Sheet normal
        T = FST[:, :, 2]  # Sheet direction
        
        return F, S, T


class FibGenDoste(FibGen):
    """Fiber generator using the Doste et al. (2019) method.
    
    Suitable for biventricular geometries with outflow tracts.
    """
    
    # Field names in Laplace solution
    FIELD_NAMES = ['Trans_BiV', 'Long_AV', 'Long_MV', 'Long_PV', 'Long_TV',
                   'Weight_LV', 'Weight_RV', 'Trans_EPI', 'Trans_LV', 'Trans_RV']
    
    # Mapping from VTU field names to internal keys
    NAME_MAP = {
        'Trans_BiV': 'ven_trans',
        'Long_AV': 'lv_av_long',
        'Long_MV': 'lv_mv_long',
        'Long_PV': 'rv_pv_long',
        'Long_TV': 'rv_tv_long',
        'Weight_LV': 'lv_weight',
        'Weight_RV': 'rv_weight',
        'Trans_EPI': 'epi_trans',
        'Trans_LV': 'lv_trans',
        'Trans_RV': 'rv_trans',
    }
    
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
        
        for vtu_name, key in self.NAME_MAP.items():
            self.lap[key] = np.asarray(mesh_cells.cell_data[vtu_name])
            self.grad[key] = np.asarray(mesh_cells.cell_data[vtu_name + "_grad"])
        
        return self.lap, self.grad
    
    @staticmethod
    def _redistribute_weight(weight, up, low):
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
        lv_glong = (grad['lv_mv_long'] * lap['lv_weight'][:, None] + 
                   grad['lv_av_long'] * (1 - lap['lv_weight'][:, None]))

        # Calculate LV basis
        Q_lv = self.calculate_basis(lv_glong, grad['lv_trans'])
        eC_lv = Q_lv[:, :, 0]  # Circumferential
        eL_lv = Q_lv[:, :, 1]  # Longitudinal
        eT_lv = Q_lv[:, :, 2]  # Transmural
        
        # Calculate combined RV longitudinal
        rv_glong = (grad['rv_tv_long'] * lap['rv_weight'][:, None] + 
                   grad['rv_pv_long'] * (1 - lap['rv_weight'][:, None]))
        Q_rv = self.calculate_basis(rv_glong, grad['rv_trans'])
        eC_rv = Q_rv[:, :, 0]  # Circumferential
        eL_rv = Q_rv[:, :, 1]  # Longitudinal
        eT_rv = Q_rv[:, :, 2]  # Transmural
        
        # Global circumferential (blended)
        eC = eC_rv * (1 - lap['ven_trans'][:, None]) + eC_lv * lap['ven_trans'][:, None]
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
        lv_weight = self._redistribute_weight(lap['lv_weight'], 0.7, 0.01)
        rv_weight = self._redistribute_weight(lap['rv_weight'], 0.1, 0.001)
        
        # LV angles
        alpha_lv_endo = params['AENDOLV'] * lv_weight + params['AOTENDOLV'] * (1 - lv_weight)
        alpha_lv_epi = params['AEPILV'] * lv_weight + params['AOTEPILV'] * (1 - lv_weight)
        alpha_wall_lv = self.calculate_angle(lap['epi_trans'], alpha_lv_endo, alpha_lv_epi)
        beta_wall_lv = self.calculate_angle(lap['epi_trans'], params['BENDOLV'], params['BEPILV']) * lv_weight
        
        # RV angles
        alpha_rv_endo = params['AENDORV'] * rv_weight + params['AOTENDORV'] * (1 - rv_weight)
        alpha_rv_epi = params['AEPIRV'] * rv_weight + params['AOTEPIRV'] * (1 - rv_weight)
        alpha_wall_rv = self.calculate_angle(lap['epi_trans'], alpha_rv_endo, alpha_rv_epi)
        beta_wall_rv = self.calculate_angle(lap['epi_trans'], params['BENDORV'], params['BEPIRV']) * rv_weight
        
        # Septum angles
        sep = np.abs(lap['ven_trans'] - 0.5)
        sep = (sep - np.min(sep)) / (np.max(sep) - np.min(sep))
        alpha_septum = alpha_lv_endo * sep * lap['lv_trans'] + alpha_rv_endo * sep * lap['rv_trans']
        beta_septum = params['BENDOLV'] * lap['lv_trans'] * lv_weight + params['BENDORV'] * lap['rv_trans'] * rv_weight
        
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
        
        # Interpolate across ventricles
        Qepi = self.interpolate_basis(Qrv_wall, Qlv_wall, self.lap['ven_trans'])
        Qendo = self.interpolate_basis(Qrv_sep, Qlv_sep, self.lap['ven_trans'])
        
        # Interpolate from endo to epi
        Q = self.interpolate_basis(Qendo, Qepi, self.lap['epi_trans'])
        
        print("   Done!")
        F = Q[:, :, 0]  # Fiber direction
        S = Q[:, :, 1]  # Sheet normal
        T = Q[:, :, 2]  # Sheet direction
        
        return F, S, T
