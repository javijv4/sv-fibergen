#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2026/01/15 21:25:52

@author: Javiera Jilberto Vallejos 
'''


import numpy as np
import pyvista as pv
from pyvista import CellType

# Load mesh
mesh_path = "check.vtu"
result_mesh = pv.read(mesh_path)
element = 145215

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

print(f"Phi_EPI = {Phi_EPI}")
print(f"Phi_LV = {Phi_LV}")
print(f"Phi_RV = {Phi_RV}")
print(f"Phi_AB = {Phi_AB}")
print(f"gPhi_EPI = {gPhi_EPI}")
print(f"gPhi_LV = {gPhi_LV}")
print(f"gPhi_RV = {gPhi_RV}")
print(f"gPhi_AB = {gPhi_AB}")

###########################################################
# Step 2: Compute interpolation factor d
###########################################################
d = Phi_RV / (Phi_LV + Phi_RV)
print(f"\nd = {d}")

###########################################################
# Step 3: Compute alpha and beta angles
###########################################################
alfaS = ALFA_END * (1 - d) - ALFA_END * d
betaS = BETA_END * (1 - d) - BETA_END * d
alfaW = ALFA_END * (1 - Phi_EPI) + ALFA_EPI * Phi_EPI
betaW = BETA_END * (1 - Phi_EPI) + BETA_EPI * Phi_EPI

print(f"\nalfaS = {alfaS} rad = {np.rad2deg(alfaS)} deg")
print(f"betaS = {betaS} rad = {np.rad2deg(betaS)} deg")
print(f"alfaW = {alfaW} rad = {np.rad2deg(alfaW)} deg")
print(f"betaW = {betaW} rad = {np.rad2deg(betaW)} deg")

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

print(f"\nQ_LV0:")
print(f"  e0 (circumferential) = {e0_LV}")
print(f"  e1 (longitudinal) = {e1_LV}")
print(f"  e2 (transmural) = {e2_LV}")

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

print(f"\nQ_LV (after rotations):")
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

print(f"\nQ_RV0:")
print(f"  e0 (circumferential) = {e0_RV}")
print(f"  e1 (longitudinal) = {e1_RV}")
print(f"  e2 (transmural) = {e2_RV}")

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

print(f"\nQ_RV (after rotations):")
print(Q_RV)

###########################################################
# Step 8: Interpolate between Q_LV and Q_RV (bislerp)
###########################################################

# Convert rotation matrices to quaternions [w, x, y, z]
def rotm_to_quat(R):
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # Choose major diagonal
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return q

q1_END = rotm_to_quat(Q_LV)
q2_END = rotm_to_quat(Q_RV)

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
def quat_to_rotm(q):
    w, x, y, z = q
    x2 = x * x
    y2 = y * y
    z2 = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    R = np.array([
        [1.0 - 2.0*y2 - 2.0*z2, 2.0*xy - 2.0*wz, 2.0*xz + 2.0*wy],
        [2.0*xy + 2.0*wz, 1.0 - 2.0*x2 - 2.0*z2, 2.0*yz - 2.0*wx],
        [2.0*xz - 2.0*wy, 2.0*yz + 2.0*wx, 1.0 - 2.0*x2 - 2.0*y2]
    ])
    return R

Q_END = quat_to_rotm(q_END)

print(f"\nQ_END (after SLERP interpolation):")
print(Q_END)


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

print(f"\nQ_EPI0:")
print(f"  e0 (circumferential) = {e0_EPI}")
print(f"  e1 (longitudinal) = {e1_EPI}")
print(f"  e2 (transmural) = {e2_EPI}")

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

print(f"\nQ_EPI (after rotations):")
print(Q_EPI)

###########################################################
# Step 11: Final interpolation (bislerp between Q_END and Q_EPI)
###########################################################
# bislerp(Q_END, Q_EPI, Phi_EPI)

q1_final = rotm_to_quat(Q_END)
q2_final = rotm_to_quat(Q_EPI)

# Generate all possible quaternion options by rotating 180 degrees
# about each basis vector of Q_END (and combinations)
print(f"\n" + "="*60)
print("EXPLORING QUATERNION OPTIONS")
print("="*60)

# Function to multiply two quaternions
def quat_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

# Get basis vectors from Q_END
eC = Q_END[:, 0]  # circumferential
eL = Q_END[:, 1]  # longitudinal
eT = Q_END[:, 2]  # transmural

# Create quaternions for 180-degree rotations about each axis
# For 180 deg rotation about unit vector v: q = [0, vx, vy, vz]
q_180_eC = np.array([0.0, eC[0], eC[1], eC[2]])
q_180_eC = q_180_eC / np.linalg.norm(q_180_eC)

q_180_eL = np.array([0.0, eL[0], eL[1], eL[2]])
q_180_eL = q_180_eL / np.linalg.norm(q_180_eL)

q_180_eT = np.array([0.0, eT[0], eT[1], eT[2]])
q_180_eT = q_180_eT / np.linalg.norm(q_180_eT)

# Generate all 8 combinations (2^3)
quaternion_options = []
option_names = []

# Option 0: No rotation (identity)
quaternion_options.append(q1_final)
option_names.append("Original (no rotation)")

# Option 1: Rotate 180° about eC
q_opt1 = quat_multiply(q1_final, q_180_eC)
quaternion_options.append(q_opt1)
option_names.append("180° about eC (circumferential)")

# Option 2: Rotate 180° about eL
q_opt2 = quat_multiply(q1_final, q_180_eL)
quaternion_options.append(q_opt2)
option_names.append("180° about eL (longitudinal)")

# Option 3: Rotate 180° about eT
q_opt3 = quat_multiply(q1_final, q_180_eT)
quaternion_options.append(q_opt3)
option_names.append("180° about eT (transmural)")

# Option 4: Rotate 180° about eC, then eL
q_opt4 = quat_multiply(quat_multiply(q1_final, q_180_eC), q_180_eL)
quaternion_options.append(q_opt4)
option_names.append("180° about eC + eL")

# Option 5: Rotate 180° about eC, then eT
q_opt5 = quat_multiply(quat_multiply(q1_final, q_180_eC), q_180_eT)
quaternion_options.append(q_opt5)
option_names.append("180° about eC + eT")

# Option 6: Rotate 180° about eL, then eT
q_opt6 = quat_multiply(quat_multiply(q1_final, q_180_eL), q_180_eT)
quaternion_options.append(q_opt6)
option_names.append("180° about eL + eT")

# Option 7: Rotate 180° about all three axes
q_opt7 = quat_multiply(quat_multiply(quat_multiply(q1_final, q_180_eC), q_180_eL), q_180_eT)
quaternion_options.append(q_opt7)
option_names.append("180° about eC + eL + eT")

# Print all options and compute dot product with q2_final
print(f"\nq1_final (original) = {q1_final}")
print(f"q2_final (target)   = {q2_final}")
print(f"\nAll 8 possible quaternion options:\n")

for i, (q_opt, name) in enumerate(zip(quaternion_options, option_names)):
    dot_prod = np.dot(q_opt, q2_final)
    angle = 2 * np.arccos(np.clip(np.abs(dot_prod), 0, 1))
    print(f"Option {i}: {name}")
    print(f"  q = {q_opt}")
    print(f"  dot(q, q2_final) = {dot_prod:8.5f}")
    print(f"  rotation angle   = {np.rad2deg(angle):7.2f}°")
    print()

# Find the option with maximum absolute dot product (closest to target)
abs_dots = [np.abs(np.dot(q, q2_final)) for q in quaternion_options]
print(abs_dots)
best_idx = np.argmax(abs_dots)

# Use the best quaternion option for final SLERP
q1_final_to_use = quaternion_options[best_idx]
dot_final = np.dot(q1_final_to_use, q2_final)

# SLERP
t_final = Phi_EPI
dot_final = np.clip(dot_final, -1.0, 1.0)
theta0_final = np.arccos(dot_final)
sin_theta0_final = np.sin(theta0_final)

if sin_theta0_final < 1e-6:
    # Linear interpolation
    q_final = (1.0 - t_final) * q1_final_to_use + t_final * q2_final
else:
    # Spherical interpolation
    theta_final = theta0_final * t_final
    s0_final = np.sin(theta0_final - theta_final) / sin_theta0_final
    s1_final = np.sin(theta_final) / sin_theta0_final
    q_final = s0_final * q1_final_to_use + s1_final * q2_final

# Normalize
q_final = q_final / np.linalg.norm(q_final)

# Convert quaternion back to rotation matrix
FST = quat_to_rotm(q_final)

print(f"\nFST (final rotation matrix):")
print(FST)

###########################################################
# Step 12: Extract fiber, sheet, and normal directions
###########################################################
F = FST[:, 0]  # Fiber direction (first column)
S = FST[:, 1]  # Sheet normal direction (second column)
T = FST[:, 2]  # Sheet direction (third column)

print(f"\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)
print(f"Fiber direction (F) = {F}")
print(f"Sheet normal direction (S) = {S}")
print(f"Sheet direction (T) = {T}")
print(f"\nNorm of F = {np.linalg.norm(F)}")
print(f"Norm of S = {np.linalg.norm(S)}")
print(f"Norm of T = {np.linalg.norm(T)}")
print(f"\nF · S = {np.dot(F, S)} (should be ~0)")
print(f"F · T = {np.dot(F, T)} (should be ~0)")
print(f"S · T = {np.dot(S, T)} (should be ~0)")
print("="*60)


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
