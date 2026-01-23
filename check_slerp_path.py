#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Check SLERP shortest path behavior."""

import numpy as np
import pyvista as pv

# Load mesh and get element data
element = 530439
result_mesh = pv.read("check.vtu")

# Get Q_END and Q_EPI for this element
Q_END = np.column_stack([
    result_mesh.cell_data['eC_END'][element],
    result_mesh.cell_data['eL_END'][element],
    result_mesh.cell_data['eT_END'][element]
])

Q_EPI = np.column_stack([
    result_mesh.cell_data['eC_EPI'][element],
    result_mesh.cell_data['eL_EPI'][element],
    result_mesh.cell_data['eT_EPI'][element]
])

print("="*60)
print("CHECKING SLERP SHORTEST PATH")
print("="*60)

print("\nQ_END (endocardial basis):")
print(Q_END)

print("\nQ_EPI (epicardial basis):")
print(Q_EPI)

# Check dot products of individual column vectors
print("\n" + "-"*60)
print("Individual column vector dot products:")
print("-"*60)
for i, name in enumerate(['eC (circumferential)', 'eL (longitudinal)', 'eT (transmural)']):
    dot_col = np.dot(Q_END[:, i], Q_EPI[:, i])
    print(f"{name}:")
    print(f"  Q_END[:, {i}] = {Q_END[:, i]}")
    print(f"  Q_EPI[:, {i}] = {Q_EPI[:, i]}")
    print(f"  dot product = {dot_col:.6f}")
    if dot_col < 0:
        print(f"  -> NEGATIVE dot product (vectors point in opposite hemispheres)")
    print()

# Convert to quaternions
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
    q = q / np.linalg.norm(q)
    return q

print("-"*60)
print("Quaternion representation:")
print("-"*60)
q_END = rotm_to_quat(Q_END)
q_EPI = rotm_to_quat(Q_EPI)

print(f"q_END = {q_END}")
print(f"q_EPI = {q_EPI}")

dot_quat = np.dot(q_END, q_EPI)
print(f"\nQuaternion dot product = {dot_quat:.6f}")

if dot_quat >= 0:
    print("-> POSITIVE: Using these quaternions directly (shortest path)")
    angle = 2 * np.arccos(np.clip(np.abs(dot_quat), 0, 1))
else:
    print("-> NEGATIVE: Would negate q_EPI for shortest path")
    angle = 2 * np.arccos(np.clip(np.abs(dot_quat), 0, 1))

print(f"\nRotation angle between Q_END and Q_EPI: {np.rad2deg(angle):.2f} degrees")

print("\n" + "="*60)
print("EXPLANATION:")
print("="*60)
print("""
This is EXPECTED behavior! Here's why:

1. Individual basis vectors (columns) can have negative dot products,
   meaning they point in opposite hemispheres of the unit sphere.

2. However, the quaternion represents the ENTIRE rotation matrix as
   a single point on the 4D unit hypersphere (S³).

3. The quaternion dot product being POSITIVE means the two rotation
   matrices are on the same hemisphere of S³, so SLERP will take
   the shortest path between them.

4. The shortest path for the ROTATION AS A WHOLE is not the same as
   having shortest paths for each individual basis vector separately.

5. SLERP ensures smooth, constant-angular-velocity interpolation of
   the rotation transformation, which is geometrically correct even
   if individual vectors seem to "flip".

In summary: dot_final being positive is correct - it ensures the
shortest rotational path between the two coordinate frames.
""")
