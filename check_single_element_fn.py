#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2026/01/15 21:25:52

@author: Javiera Jilberto Vallejos 
'''


import numpy as np
import pyvista as pv
from pyvista import CellType
import sys
sys.path.insert(0, 'src')
from quat_utils import rotm_to_quat_batch, quat_to_rotm_batch, find_best_quaternions

# Load mesh
mesh_path = "check.vtu"
result_mesh = pv.read(mesh_path)
element = 166502

###########################################################
# Bayer method parameters
###########################################################
ALFA_END = np.deg2rad(60.0)
ALFA_EPI = np.deg2rad(-60.0)
BETA_END = np.deg2rad(0.0)
BETA_EPI = np.deg2rad(0.0)

###########################################################
# Step 1: Load and normalize Laplace fields
###########################################################
print("Loading Laplace solution for element", element)

# Get Laplace field values at the cell
Phi_EPI = np.asarray(result_mesh.cell_data['Trans_EPI'])[element]
Phi_LV = np.asarray(result_mesh.cell_data['Trans_LV'])[element]
Phi_RV = np.asarray(result_mesh.cell_data['Trans_RV'])[element]
Phi_AB = np.asarray(result_mesh.cell_data['Long_AB'])[element]

# Get gradients at the cell
gPhi_EPI = np.asarray(result_mesh.cell_data['Trans_EPI_grad'])[element]
gPhi_LV = np.asarray(result_mesh.cell_data['Trans_LV_grad'])[element]
gPhi_RV = np.asarray(result_mesh.cell_data['Trans_RV_grad'])[element]
gPhi_AB = np.asarray(result_mesh.cell_data['Long_AB_grad'])[element]

###########################################################
# Step 2: Compute interpolation factor d
###########################################################
d = Phi_RV / (Phi_LV + Phi_RV)

###########################################################
# Step 3: Compute alpha and beta angles
###########################################################
alfaS = ALFA_END * (1 - d) - ALFA_END * d
betaS = BETA_END * (1 - d) - BETA_END * d
alfaW = ALFA_END * (1 - Phi_EPI) + ALFA_EPI * Phi_EPI
betaW = BETA_END * (1 - Phi_EPI) + BETA_EPI * Phi_EPI

###########################################################
# Step 4: Construct Q_LV0 basis (axis function for LV)
###########################################################
# axis(gPhi_AB, -gPhi_LV)
# u = gPhi_AB (longitudinal)
# v = -gPhi_LV (secondary direction)
u_LV = gPhi_AB
v_LV = -gPhi_LV

# e1 = normalize(u)
norm_u_LV = np.linalg.norm(u_LV)
e1_LV = u_LV / norm_u_LV if norm_u_LV > 0 else np.zeros(3)

# e2 = v - proj_{e1}(v), then normalize
proj_LV = np.dot(e1_LV, v_LV) * e1_LV
e2_LV = v_LV - proj_LV
norm_e2_LV = np.linalg.norm(e2_LV)
e2_LV = e2_LV / norm_e2_LV if norm_e2_LV > 0 else np.zeros(3)

# e0 = cross(e1, e2), then normalize
e0_LV = np.cross(e1_LV, e2_LV)
norm_e0_LV = np.linalg.norm(e0_LV)
e0_LV = e0_LV / norm_e0_LV if norm_e0_LV > 0 else np.zeros(3)

# Q_LV0 = [e0, e1, e2] as columns
Q_LV0 = np.column_stack([e0_LV, e1_LV, e2_LV])

###########################################################
# Step 5: Apply rotations to get Q_LV (orient function)
###########################################################
# orient(Q_LV0, alfaS, betaS)
# Rotation about z-axis (Ra) by alfaS
ca_LV = np.cos(alfaS)
sa_LV = np.sin(alfaS)
Ra_LV = np.array([
    [ca_LV, -sa_LV, 0],
    [sa_LV, ca_LV, 0],
    [0, 0, 1]
])

# Rotation about y-axis (Rb) by betaS
cb_LV = np.cos(betaS)
sb_LV = np.sin(betaS)
Rb_LV = np.array([
    [cb_LV, 0, sb_LV],
    [0, 1, 0],
    [-sb_LV, 0, cb_LV]
])

# Compose rotations
RaRb_LV = Ra_LV @ Rb_LV
Q_LV = Q_LV0 @ RaRb_LV
print(Q_LV)

###########################################################
# Step 6: Construct Q_RV0 basis (axis function for RV)
###########################################################
# axis(gPhi_AB, gPhi_RV) - note positive gPhi_RV
u_RV = gPhi_AB
v_RV = gPhi_RV

# e1 = normalize(u)
norm_u_RV = np.linalg.norm(u_RV)
e1_RV = u_RV / norm_u_RV if norm_u_RV > 0 else np.zeros(3)

# e2 = v - proj_{e1}(v), then normalize
proj_RV = np.dot(e1_RV, v_RV) * e1_RV
e2_RV = v_RV - proj_RV
norm_e2_RV = np.linalg.norm(e2_RV)
e2_RV = e2_RV / norm_e2_RV if norm_e2_RV > 0 else np.zeros(3)

# e0 = cross(e1, e2), then normalize
e0_RV = np.cross(e1_RV, e2_RV)
norm_e0_RV = np.linalg.norm(e0_RV)
e0_RV = e0_RV / norm_e0_RV if norm_e0_RV > 0 else np.zeros(3)

# Q_RV0 = [e0, e1, e2] as columns
Q_RV0 = np.column_stack([e0_RV, e1_RV, e2_RV])

###########################################################
# Step 7: Apply rotations to get Q_RV (orient function)
###########################################################
# orient(Q_RV0, alfaS, -betaS) - note the minus in betaS
ca_RV = np.cos(alfaS)
sa_RV = np.sin(alfaS)
Ra_RV = np.array([
    [ca_RV, -sa_RV, 0],
    [sa_RV, ca_RV, 0],
    [0, 0, 1]
])

cb_RV = np.cos(-betaS)
sb_RV = np.sin(-betaS)
Rb_RV = np.array([
    [cb_RV, 0, sb_RV],
    [0, 1, 0],
    [-sb_RV, 0, cb_RV]
])

RaRb_RV = Ra_RV @ Rb_RV
Q_RV = Q_RV0 @ RaRb_RV
print(Q_RV)

###########################################################
# Step 8: Interpolate between Q_LV and Q_RV (bislerp)
###########################################################

# Convert rotation matrices to quaternions [w, x, y, z]
q1_END = rotm_to_quat_batch(Q_LV.reshape(1, 3, 3))[0]
q2_END = rotm_to_quat_batch(Q_RV.reshape(1, 3, 3))[0]

# Ensure shortest path
dot_END = np.dot(q1_END, q2_END)
if dot_END < 0.0:
    q2_END = -q2_END
    dot_END = -dot_END

# SLERP
dot_END = np.clip(dot_END, -1.0, 1.0)
theta0_END = np.arccos(dot_END)
sin_theta0_END = np.sin(theta0_END)

if sin_theta0_END < 1e-6:
    # Linear interpolation
    q_END = (1.0 - d) * q1_END + d * q2_END
else:
    # Spherical interpolation
    theta_END = theta0_END * d
    s0_END = np.sin(theta0_END - theta_END) / sin_theta0_END
    s1_END = np.sin(theta_END) / sin_theta0_END
    q_END = s0_END * q1_END + s1_END * q2_END

# Normalize
q_END = q_END / np.linalg.norm(q_END)

# Convert quaternion back to rotation matrix
Q_END = quat_to_rotm_batch(q_END.reshape(1, 4))[0]


###########################################################
# Step 9: Construct Q_EPI0 basis (axis function for EPI)
###########################################################
# axis(gPhi_AB, gPhi_EPI)
u_EPI = gPhi_AB
v_EPI = gPhi_EPI

# e1 = normalize(u)
norm_u_EPI = np.linalg.norm(u_EPI)
e1_EPI = u_EPI / norm_u_EPI if norm_u_EPI > 0 else np.zeros(3)

# e2 = v - proj_{e1}(v), then normalize
proj_EPI = np.dot(e1_EPI, v_EPI) * e1_EPI
e2_EPI = v_EPI - proj_EPI
norm_e2_EPI = np.linalg.norm(e2_EPI)
e2_EPI = e2_EPI / norm_e2_EPI if norm_e2_EPI > 0 else np.zeros(3)

# e0 = cross(e1, e2), then normalize
e0_EPI = np.cross(e1_EPI, e2_EPI)
norm_e0_EPI = np.linalg.norm(e0_EPI)
e0_EPI = e0_EPI / norm_e0_EPI if norm_e0_EPI > 0 else np.zeros(3)

# Q_EPI0 = [e0, e1, e2] as columns
Q_EPI0 = np.column_stack([e0_EPI, e1_EPI, e2_EPI])

###########################################################
# Step 10: Apply rotations to get Q_EPI (orient function)
###########################################################
# orient(Q_EPI0, alfaW, betaW)
ca_EPI = np.cos(alfaW)
sa_EPI = np.sin(alfaW)
Ra_EPI = np.array([
    [ca_EPI, -sa_EPI, 0],
    [sa_EPI, ca_EPI, 0],
    [0, 0, 1]
])

cb_EPI = np.cos(betaW)
sb_EPI = np.sin(betaW)
Rb_EPI = np.array([
    [cb_EPI, 0, sb_EPI],
    [0, 1, 0],
    [-sb_EPI, 0, cb_EPI]
])

RaRb_EPI = Ra_EPI @ Rb_EPI
Q_EPI = Q_EPI0 @ RaRb_EPI

###########################################################
# Step 11: Final interpolation (bislerp between Q_END and Q_EPI)
###########################################################
# bislerp(Q_END, Q_EPI, Phi_EPI)

# Use find_best_quaternions to find the best representation
# q1_final, q2_final = find_best_quaternions(Q_END.reshape(1, 3, 3), Q_EPI.reshape(1, 3, 3))
# q1_final = q1_final[0]
# q2_final = q2_final[0]

q1_final = rotm_to_quat_batch(Q_END.reshape(1, 3, 3))[0]
q2_final = rotm_to_quat_batch(Q_EPI.reshape(1, 3, 3))[0]

dot_final = np.dot(q1_final, q2_final)

# SLERP
t_final = Phi_EPI
dot_final = np.clip(dot_final, -1.0, 1.0)
theta0_final = np.arccos(dot_final)
sin_theta0_final = np.sin(theta0_final)

if sin_theta0_final < 1e-6:
    # Linear interpolation
    q_final = (1.0 - t_final) * q1_final + t_final * q2_final
else:
    # Spherical interpolation
    theta_final = theta0_final * t_final
    s0_final = np.sin(theta0_final - theta_final) / sin_theta0_final
    s1_final = np.sin(theta_final) / sin_theta0_final
    q_final = s0_final * q1_final + s1_final * q2_final

# Normalize
q_final = q_final / np.linalg.norm(q_final)

# Convert quaternion back to rotation matrix
FST = quat_to_rotm_batch(q_final.reshape(1, 4))[0]
print("\nFinal FST rotation matrix:")
print(FST)

###########################################################
# Step 12: Extract fiber, sheet, and normal directions
###########################################################
F = FST[:, 0]  # Fiber direction (first column)
S = FST[:, 1]  # Sheet normal direction (second column)
T = FST[:, 2]  # Sheet direction (third column)

###########################################################
# Step 13: Create a mesh with the single element and fiber directions
###########################################################

# Extract the element's connectivity and points
cell = result_mesh.get_cell(element)
point_ids = np.array(cell.point_ids)

# Get the coordinates of the element's points
points = result_mesh.points[point_ids]

# Create a new mesh with just this element
# Use sequential indices [0, 1, 2, ...] for the new mesh
num_points = len(point_ids)
connectivity = np.arange(num_points).reshape(1, -1)
single_element_mesh = pv.UnstructuredGrid(
    {cell.type: connectivity},
    points
)

# Add eC_END and eC_EPI as point data for reference (optional)
single_element_mesh.cell_data['eC_END'] = Q_END[:, 0].reshape([1,3])
single_element_mesh.cell_data['eC_EPI'] = Q_EPI[:, 0].reshape([1,3])

# Add the fiber directions as cell data
single_element_mesh.cell_data['Fiber'] = F.reshape(1, 3)
single_element_mesh.cell_data['Sheet'] = S.reshape(1, 3)
single_element_mesh.cell_data['Normal'] = T.reshape(1, 3)

# Also add the original Laplace fields for reference
single_element_mesh.cell_data['Trans_EPI'] = np.array([Phi_EPI])
single_element_mesh.cell_data['Trans_LV'] = np.array([Phi_LV])
single_element_mesh.cell_data['Trans_RV'] = np.array([Phi_RV])
single_element_mesh.cell_data['Long_AB'] = np.array([Phi_AB])

# Save the single element mesh
output_path = "single_element_fibers.vtu"
single_element_mesh.save(output_path)
print(f"\nSingle element mesh saved to: {output_path}")
