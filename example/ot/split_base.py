import pyvista as pv
import numpy as np
import os


def _get_cells_array(mesh):
    """
    Extract the cell connectivity from a pyvista.PolyData as either:
      - a (ncells, nodes_per_cell) ndarray if all faces are uniform, or
      - a list of 1D ndarrays (one per cell) for mixed-size faces.
    """
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.size == 0:
        return np.empty((0, 0), dtype=int)
    # Fast path: assume uniform face size (common for triangulated surfaces)
    first = int(faces[0])
    step = first + 1
    if faces.size % step == 0:
        cells = faces.reshape((-1, step))[:, 1:]
        # Sanity check uniformity
        if np.all(cells.shape[1] == first):
            return cells
    # Fallback: mixed polygons; build a ragged list
    cells_list = []
    i = 0
    n = faces.size
    while i < n:
        eNoN = int(faces[i])
        start = i + 1
        end = start + eNoN
        if end > n:
            raise ValueError("Invalid PolyData faces array encoding.")
        cells_list.append(faces[start:end].copy())
        i = end
    return cells_list


def split_base_into_four_regions(cells, base_elem_ids):
    """
    Split the set of base elements into four disconnected regions using cell adjacency
    defined by shared nodes.
    
    ARGS:
        cells : ndarray (ncells, eNoN) or list of 1D ndarrays
            Cell connectivity (indices into mesh points) for the same set that
            'base_elem_ids' indices refer to.
        base_elem_ids : 1D array-like of ints
            Indices (within 'cells') of elements belonging to the base.
    
    RETURNS:
        regions : list of 1D numpy arrays
            Four arrays, each containing the element indices (within 'cells')
            for one disconnected region of the base.
    """
    base_elem_ids = np.asarray(base_elem_ids, dtype=int)
    # Map node -> set of base element ids that touch the node
    node_to_base_elems = {}
    for eid in base_elem_ids:
        nodes_iter = cells[eid] if isinstance(cells, list) else cells[eid]
        for node in nodes_iter:
            s = node_to_base_elems.get(node)
            if s is None:
                node_to_base_elems[node] = {int(eid)}
            else:
                s.add(int(eid))
    # BFS over base elements using shared-node adjacency
    visited = set()
    regions = []
    for start in base_elem_ids:
        if int(start) in visited:
            continue
        queue = [int(start)]
        visited.add(int(start))
        comp = []
        while queue:
            eid = queue.pop()
            comp.append(eid)
            nodes_iter = cells[eid] if isinstance(cells, list) else cells[eid]
            for node in nodes_iter:
                neighs = node_to_base_elems.get(node)
                if not neighs:
                    continue
                for nb in neighs:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        regions.append(np.array(sorted(comp), dtype=int))
    # Sort regions by size (descending) for determinism
    regions.sort(key=lambda a: a.size, reverse=True)
    if len(regions) != 4:
        raise ValueError(f"Expected 4 disconnected regions, found {len(regions)}.")
    return regions


def split_base_from_mesh(mesh, base_label=1):
    """
    Convenience wrapper that reads labels from mesh.cell_data['ModelFaceID'],
    extracts base elements, and returns the four disconnected regions.
    """
    labels_block = np.asarray(mesh.cell_data['ModelFaceID'])
    base_ids = np.where(labels_block == base_label)[0]
    cells = _get_cells_array(mesh)
    return split_base_into_four_regions(cells, base_ids)

surf_fldr = './mesh-surfaces'
surf_mesh = pv.read("./mesh-complete.exterior.vtp")
labels = surf_mesh.cell_data['ModelFaceID']
max_label = np.max(labels)

regions = split_base_from_mesh(surf_mesh, base_label=1)

new_labels = np.copy(labels)
for i, region in enumerate(regions):
    new_labels[region] = i + max_label + 1  # Assign new labels starting from max_label + 1

surf_mesh.cell_data['ModelFaceID'] = new_labels
surf_mesh.save("./mesh-complete.exterior.split.vtp")

# TODO figure out which one is which automatically
new_names = {5: 'mv',
             6: 'tv',
             7: 'av',
             8: 'pv'}

# Save each region as a separate surface with proper fields
for label_value, surf_name in new_names.items():
    cell_ids = np.where(new_labels == label_value)[0]
    if cell_ids.size == 0:
        continue
    submesh = surf_mesh.extract_cells(cell_ids)
    # Ensure PolyData so we can write .vtp
    try:
        submesh = submesh.cast_to_polydata()
    except Exception:
        submesh = submesh.extract_surface().clean()
    # Ensure fields are present and consistent
    submesh.cell_data['ModelFaceID'] = np.full(submesh.n_cells, label_value, dtype=new_labels.dtype)
    # GlobalElementID: copy from parent if available
    if 'GlobalElementID' in surf_mesh.cell_data:
        submesh.cell_data['GlobalElementID'] = np.asarray(surf_mesh.cell_data['GlobalElementID'])[cell_ids]
    # GlobalNodeID: prefer preserved data; else reconstruct via vtkOriginalPointIds if present
    if 'GlobalNodeID' not in submesh.point_data and 'GlobalNodeID' in surf_mesh.point_data:
        if 'vtkOriginalPointIds' in submesh.point_data:
            orig_pids = np.asarray(submesh.point_data['vtkOriginalPointIds']).astype(int)
            submesh.point_data['GlobalNodeID'] = np.asarray(surf_mesh.point_data['GlobalNodeID'])[orig_pids]
    out_path = os.path.join(surf_fldr, f"{surf_name}.vtp")
    submesh.save(out_path)
