#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Surface mesh processing utilities for biventricular heart models.

This module provides classes and utilities for processing surface meshes,
including generating epicardial apex surfaces.
"""

import os
import numpy as np
import pyvista as pv
from .SurfaceNames import SurfaceName


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
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


class SurfaceProcessor:
    """Processor for surface mesh operations.
    
    This class handles operations on surface meshes, such as generating
    epicardial apex surfaces from existing surface meshes.
    
    Attributes:
        surfaces_dir: Directory containing surface mesh files.
        surface_paths: Dictionary mapping SurfaceName enum to file paths.
    """
    
    def __init__(self, surfaces_dir, surface_paths):
        """Initialize the SurfaceProcessor.
        
        Args:
            surfaces_dir: Directory containing surface mesh files.
            surface_paths: Dictionary mapping SurfaceName enum values to filenames
                (not full paths, just filenames).
        """
        self.surfaces_dir = surfaces_dir
        self.surface_paths = surface_paths
    
    def get_surface_path(self, surface_name):
        """Get full path to a surface file.
        
        Args:
            surface_name: SurfaceName enum value.
            
        Returns:
            str: Full path to the surface file.
        """
        if surface_name not in self.surface_paths:
            raise KeyError(f"Surface {surface_name.value} not found in surface_paths")
        return os.path.join(self.surfaces_dir, self.surface_paths[surface_name])
    
    def generate_epi_apex(self):
        """Generate the epicardial apex surface from the epicardial surface of the BiV.
        
        This method identifies the apex point of the epicardium and creates a surface
        mesh containing elements that include the apex point. The surface is saved with
        global node and element IDs.
        
        The method requires:
            - SurfaceName.EPI: Epicardial surface
            - SurfaceName.BASE: Base surface (for finding apex)
            - SurfaceName.EPI_APEX: Output surface name (will be created)
        """
        # Load the epi surface
        epi_path = self.get_surface_path(SurfaceName.EPI)
        epi_mesh = pv.read(epi_path)
        epi_points = epi_mesh.points
        epi_cells = epi_mesh.faces
        epi_eNoN = epi_cells[0]
        epi_cells = epi_cells.reshape((-1, epi_eNoN + 1))
        epi_cells = epi_cells[:, 1:]
        epi_global_node_id = epi_mesh.point_data['GlobalNodeID']
        epi_global_cell_id = epi_mesh.cell_data['GlobalElementID']

        # Load the base surface
        base_path = self.get_surface_path(SurfaceName.BASE)
        base_mesh = pv.read(base_path)
        base_global_node_id = base_mesh.point_data['GlobalNodeID']

        # Extract the boundary of the epi surface (at the top) to find the apex point
        epi_base_global_node_id = np.intersect1d(epi_global_node_id, base_global_node_id)
        epi_base_nodes = np.where(np.isin(epi_global_node_id, epi_base_global_node_id))[0]

        # Get normal
        base_normal, base_centroid = get_normal_plane_svd(epi_points[epi_base_nodes, :])

        # Find the index of the apex point of the epi surface
        distance = np.abs(base_normal @ (epi_points - base_centroid).T)
        epi_apex_point_index = np.argmax(distance)

        # Find elements containing the apex point
        epi_apex_cell_index = np.where(epi_cells == epi_apex_point_index)[0]

        # Create epi_apex mesh
        submesh_cells = epi_cells[epi_apex_cell_index]
        submesh_xyz = np.zeros([len(np.unique(submesh_cells)), epi_points.shape[1]])
        map_mesh_submesh = np.ones(epi_points.shape[0], dtype=int) * -1
        map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
        child_elems_new = np.zeros(submesh_cells.shape, dtype=int)

        cont = 0
        for e in range(submesh_cells.shape[0]):
            for i in range(submesh_cells.shape[1]):
                if map_mesh_submesh[submesh_cells[e, i]] == -1:
                    child_elems_new[e, i] = cont
                    submesh_xyz[cont] = epi_points[submesh_cells[e, i]]
                    map_mesh_submesh[submesh_cells[e, i]] = cont
                    map_submesh_mesh[cont] = submesh_cells[e, i]
                    cont += 1
                else:
                    child_elems_new[e, i] = map_mesh_submesh[submesh_cells[e, i]]

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

        epi_apex_path = self.get_surface_path(SurfaceName.EPI_APEX)
        epi_apex_mesh.save(epi_apex_path)
