#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Plot the 8 rotation options for quaternion representation

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a random orthonormal basis
def generate_random_basis():
    # Start with a random vector for e0
    e0 = np.random.randn(3)
    e0 = e0 / np.linalg.norm(e0)
    
    # Generate e1 orthogonal to e0
    v1 = np.random.randn(3)
    e1 = v1 - np.dot(v1, e0) * e0
    e1 = e1 / np.linalg.norm(e1)
    
    # Generate e2 as cross product
    e2 = np.cross(e0, e1)
    e2 = e2 / np.linalg.norm(e2)
    
    return e0, e1, e2

# Quaternion multiplication
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

# Convert rotation matrix to quaternion
def rotm_to_quat(R):
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
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
    return q / np.linalg.norm(q)

# Convert quaternion to rotation matrix
def quat_to_rotm(q):
    w, x, y, z = q
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    R = np.array([
        [1.0 - 2.0*y2 - 2.0*z2, 2.0*xy - 2.0*wz, 2.0*xz + 2.0*wy],
        [2.0*xy + 2.0*wz, 1.0 - 2.0*x2 - 2.0*z2, 2.0*yz - 2.0*wx],
        [2.0*xz - 2.0*wy, 2.0*yz + 2.0*wx, 1.0 - 2.0*x2 - 2.0*y2]
    ])
    return R

# SLERP (Spherical Linear Interpolation)
def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions.
    
    Args:
        q1: Starting quaternion (4,) [w, x, y, z]
        q2: Ending quaternion (4,) [w, x, y, z]
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion (4,) [w, x, y, z]
    """
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product to avoid numerical issues with arccos
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle between quaternions
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    # Compute interpolation weights
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    # Compute interpolated quaternion
    result = w1 * q1 + w2 * q2
    return result / np.linalg.norm(result)

# Plot basis vectors
def plot_basis_multi(ax, Q_rotated, Q_target, Q_slerp, title):
    """Plot multiple bases: rotated option, target, and SLERP result
    
    Args:
        Q_rotated: Rotated option basis (3, 3)
        Q_target: Target basis (3, 3)
        Q_slerp: SLERP interpolated basis (3, 3)
        title: Plot title
    """
    origin = np.array([0, 0, 0])
    
    colors = ['red', 'green', 'blue']
    
    # Plot rotated option as faint dashed (starting point)
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                 Q_rotated[0, i], Q_rotated[1, i], Q_rotated[2, i],
                 color=colors[i], alpha=0.25, arrow_length_ratio=0.15, 
                 linewidth=1.5, linestyle='--')
    
    # Plot target as faint dotted (ending point)
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                 Q_target[0, i], Q_target[1, i], Q_target[2, i],
                 color=colors[i], alpha=0.25, arrow_length_ratio=0.15, 
                 linewidth=1.5, linestyle=':')
    
    # Plot SLERP result as bold solid (interpolated)
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                 Q_slerp[0, i], Q_slerp[1, i], Q_slerp[2, i],
                 color=colors[i], arrow_length_ratio=0.2, linewidth=2.5,
                 alpha=1.0)
    
    # Set equal aspect ratio and limits
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)

# Main script
np.random.seed(41)  # For reproducibility

# Generate random orthonormal basis Q1
e0, e1, e2 = generate_random_basis()
Q = np.column_stack([e0, e1, e2])

print("Original basis Q:")
print(Q)
print(f"\nOrthogonality check:")
print(f"e0 · e1 = {np.dot(e0, e1):.10f}")
print(f"e0 · e2 = {np.dot(e0, e2):.10f}")
print(f"e1 · e2 = {np.dot(e1, e2):.10f}")

# Convert to quaternion
q_original = rotm_to_quat(Q)
print(f"\nOriginal quaternion: {q_original}")

# Generate a second random orthonormal basis Q2
e0_2, e1_2, e2_2 = generate_random_basis()
Q2 = np.column_stack([e0_2, e1_2, e2_2])

print("\n" + "="*60)
print("Second random basis Q2:")
print(Q2)
print(f"\nOrthogonality check:")
print(f"e0_2 · e1_2 = {np.dot(e0_2, e1_2):.10f}")
print(f"e0_2 · e2_2 = {np.dot(e0_2, e2_2):.10f}")
print(f"e1_2 · e2_2 = {np.dot(e1_2, e2_2):.10f}")

# Convert to quaternion
q_target = rotm_to_quat(Q2)
print(f"\nTarget quaternion: {q_target}")
print(f"dot(q_original, q_target) = {np.dot(q_original, q_target):.4f}")

# Create quaternions for 180° rotations about each basis vector
# For 180° rotation about unit vector v: q = [0, vx, vy, vz]
q_180_e0 = np.array([0.0, e0[0], e0[1], e0[2]])
q_180_e0 = q_180_e0 / np.linalg.norm(q_180_e0)

q_180_e1 = np.array([0.0, e1[0], e1[1], e1[2]])
q_180_e1 = q_180_e1 / np.linalg.norm(q_180_e1)

q_180_e2 = np.array([0.0, e2[0], e2[1], e2[2]])
q_180_e2 = q_180_e2 / np.linalg.norm(q_180_e2)

# Generate 8 rotation options using quaternion multiplication
# q_result = q_rot * q_original represents composing rotations
quaternions = []
option_names = []

# Option 0: Original (no rotation)
quaternions.append(q_original)
option_names.append("0: Original")

# Option 1: Rotate 180° about e0
q1 = quat_multiply(q_180_e0, q_original)
quaternions.append(q1)
option_names.append("1: 180° about e0")

# Option 2: Rotate 180° about e1
q2 = quat_multiply(q_180_e1, q_original)
quaternions.append(q2)
option_names.append("2: 180° about e1")

# Option 3: Rotate 180° about e2
q3 = quat_multiply(q_180_e2, q_original)
quaternions.append(q3)
option_names.append("3: 180° about e2")

# Option 4: Rotate 180° about e0, then e1
# Apply rotations right to left: first e0, then e1
q4 = quat_multiply(q_180_e1, quat_multiply(q_180_e0, q_original))
quaternions.append(q4)
option_names.append("4: 180° about e0+e1")

# Option 5: Rotate 180° about e2, then e0
q5 = quat_multiply(q_180_e0, quat_multiply(q_180_e2, q_original))
quaternions.append(q5)
option_names.append("5: 180° about e2+e0")

# Option 6: Rotate 180° about e1, then e2
q6 = quat_multiply(q_180_e2, quat_multiply(q_180_e1, q_original))
quaternions.append(q6)
option_names.append("6: 180° about e1+e2")

# Option 7: Rotate 180° about all three axes
q7 = quat_multiply(q_180_e2, quat_multiply(q_180_e1, quat_multiply(q_180_e0, q_original)))
quaternions.append(q7)
option_names.append("7: 180° about e0+e1+e2")

# Convert quaternions to rotation matrices for visualization
rotation_matrices = [quat_to_rotm(q) for q in quaternions]

# Print comparison
print("\n" + "="*60)
print("Quaternion comparisons with target Q2:")
print("="*60)
for i, (q, name, R) in enumerate(zip(quaternions, option_names, rotation_matrices)):
    print(f"\nOption {name}")
    print(f"  q = [{q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f}]")
    dot = np.dot(q, q_target)
    abs_dot = np.abs(dot)
    angle = 2 * np.arccos(np.clip(abs_dot, 0, 1))
    print(f"  dot(q, q_target) = {dot:7.4f} (abs: {abs_dot:.4f})")
    print(f"  rotation angle from target = {np.rad2deg(angle):7.2f}°")


# Create figure with 8 subplots (2 rows x 4 columns)
fig = plt.figure(figsize=(18, 9))
fig.suptitle('SLERP Interpolation (t=0.5) for 8 Rotation Options\n(Dashed=Option, Dotted=Target, Solid=SLERP)', 
             fontsize=14, fontweight='bold')

# Compute SLERP for each option
for i, (R, name, q) in enumerate(zip(rotation_matrices, option_names, quaternions)):
    ax = fig.add_subplot(2, 4, i+1, projection='3d')
    
    # Compute SLERP at t=0.5
    q_slerp = slerp(q, q_target, 0.5)
    R_slerp = quat_to_rotm(q_slerp)
    
    # Compute dot product with target quaternion
    dot = np.dot(q, q_target)
    
    # Create title with option name and dot product
    title_with_info = f"{name}\ndot={dot:.4f}"
    
    # Plot all three bases
    plot_basis_multi(ax, R, Q2, R_slerp, title_with_info)

plt.tight_layout()
plt.savefig('rotation_options_slerp.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as 'rotation_options_slerp.png'")
plt.show()
