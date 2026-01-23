# Cardiac Fiber Generation Documentation

This document describes the mathematical framework for generating myocardial fiber orientations in cardiac geometries using Laplace-Dirichlet rule-based methods.

# General Overview of the Fiber Generation Methods
To explain the main steps and functions of the code, in the following we explain the fiber generation steps for a **single ventricle**. In the next section these concepts are expanded to the biventricular geometries using the Bayer or Doste method. 

## 1. Boundaries

The fiber generation algorithm requires specific boundary surfaces to be defined on the cardiac mesh. For a single ventricle geometry, the following surfaces must be identified:

- **Epicardium (epi)**: The outer surface of the ventricle
- **Endocardium (endo)**: The inner surface of the ventricle
- **Base**: The basal (top) boundary of the ventricle
- **Apex (epi_apex)**: The epicardial apex region

The apex surface is automatically generated from the epicardium by identifying the point furthest from the base. Specifically, the apex point $\mathbf{p}_{\text{apex}}$ is found as:

$$
\mathbf{p}_{\text{apex}} = \arg\min_{\mathbf{p} \in S_{\text{epi}} \setminus S_{\text{base}}} \|\mathbf{p} - \mathbf{c}_{\text{base}}\|
$$

where $S_{\text{epi}}$ is the epicardial surface, $S_{\text{base}}$ is the base surface, and $\mathbf{c}_{\text{base}}$ is the centroid of the base. The apex surface consists of all elements in the epicardium that contain this apex point.

## 2. Laplace Problem

The foundation of the rule-based fiber generation is the solution of Laplace-Dirichlet boundary value problems. These provide scalar fields $\phi$ (surface1 → surface2) that satisfy:

$$
\nabla^2 \phi = 0 \quad \text{in } \Omega
\phi_t = 0 & \text{on } S_{\text{surface1}} \\
\phi_t = 1 & \text{on } S_{\text{surface2}}
$$

### 2.1 Transmural Direction

The transmural field $\phi_t$ (endo → epi) characterizes the wall thickness direction, varying from endocardium to epicardium:

$$
\begin{cases}
\nabla^2 \phi_t = 0 & \text{in } \Omega \\
\phi_t = 0 & \text{on } S_{\text{endo}} \\
\phi_t = 1 & \text{on } S_{\text{epi}}
\end{cases}
$$

This field is normalized to $[0, 1]$ range where $\phi_t = 0$ at the endocardium and $\phi_t = 1$ at the epicardium.

### 2.2 Longitudinal Direction

The longitudinal field $\phi_\ell$ (apex → base) characterizes the apex-to-base direction:

$$
\begin{cases}
\nabla^2 \phi_\ell = 0 & \text{in } \Omega \\
\phi_\ell = 0 & \text{on } S_{\text{apex}} \\
\phi_\ell = 1 & \text{on } S_{\text{base}}
\end{cases}
$$

This field is also normalized to $[0, 1]$ where $\phi_\ell = 0$ at the apex and $\phi_\ell = 1$ at the base.

**Implementation Note**: The Laplace equations are solved using the SVMultiphysics solver configured to solve a steady-state heat equation (which is equivalent to the Laplace equation). The solver is configured with:
- `Conductivity = 1.0`, `Source_term = 0.0`, `Density = 0.0`
- `Spectral_radius_of_infinite_time_step = 0.0`
- Single time step to obtain the steady-state solution directly

## 3. Definition of Basis

A local orthonormal basis $\{\mathbf{e}_c, \mathbf{e}_\ell, \mathbf{e}_t\}$ is constructed at each point in the myocardium, where:
- $\mathbf{e}_c$: circumferential direction
- $\mathbf{e}_\ell$: longitudinal direction  
- $\mathbf{e}_t$: transmural direction

### 3.1 Obtain Gradients from Laplace Solutions

The gradients of the Laplace fields provide natural directional vectors. The gradients are computed at mesh nodes and then averaged to cell centers for smoother results:

$$
\mathbf{g}_t = \nabla \phi_t, \quad \mathbf{g}_\ell = \nabla \phi_\ell
$$

These gradients are normalized to unit vectors:

$$
\hat{\mathbf{g}}_t = \frac{\mathbf{g}_t}{\|\mathbf{g}_t\|}, \quad \hat{\mathbf{g}}_\ell = \frac{\mathbf{g}_\ell}{\|\mathbf{g}_\ell\|}
$$

### 3.2 Calculate Circumferential Direction

The local orthonormal basis is constructed using the `axis` function. Given the longitudinal direction $\hat{\mathbf{g}}_\ell$ and the transmural direction $\hat{\mathbf{g}}_t$:

1. **Longitudinal basis vector**: 
    $$\mathbf{e}_\ell = \frac{\hat{\mathbf{g}}_\ell}{\|\hat{\mathbf{g}}_\ell\|}$$

2. **Transmural basis vector** (orthogonalized to $\mathbf{e}_\ell$):
    $$\mathbf{e}_t' = \hat{\mathbf{g}}_t - (\hat{\mathbf{g}}_t \cdot \mathbf{e}_\ell)\mathbf{e}_\ell$$
    $$\mathbf{e}_t = \frac{\mathbf{e}_t'}{\|\mathbf{e}_t'\|}$$

3. **Circumferential basis vector** (orthogonal to both):
    $$\mathbf{e}_c = \mathbf{e}_\ell \times \mathbf{e}_t$$

This ensures a right-handed orthonormal coordinate system at each element.

## 4. Definition of Angles over the Geometry

Two angles define the fiber orientation relative to the local basis $\mathbf Q=(\mathbf e_c, \mathbf e_\ell, \mathbf e_t)$:

- **$\alpha$ (helix angle)**: Rotation angle using as axis of rotation $\mathbf{e}_t$. Returns a rotated basis $\mathbf Q^{(\alpha)}=(\mathbf e_c', \mathbf e_\ell', \mathbf e_t)$
- **$\beta$ (transverse angle)**: Rotation angle using as axis of rotation $\mathbf{e}_\ell'$. Returns the fiber ($\mathbf f$), sheetnormal ($\mathbf n$), and sheet ($\mathbf s$) vectors $\mathbf Q^{(\alpha, \beta)} = (\mathbf f, \mathbf n, \mathbf s)$.

The angles vary linearly across the wall thickness based on the transmural coordinate $\phi_t$. For the single ventricle this is:

$$
\alpha(\phi_t) = \alpha_{\text{endo}}(1 - \phi_t) + \alpha_{\text{epi}}\phi_t
$$

$$
\beta(\phi_t) = \beta_{\text{endo}}(1 - \phi_t) + \beta_{\text{epi}}\phi_t
$$

where:
- $\alpha_{\text{endo}}, \alpha_{\text{epi}}$: helix angles at endocardium and epicardium (typically $60°$ and $-60°$)
- $\beta_{\text{endo}}, \beta_{\text{epi}}$: transverse angles at endocardium and epicardium (typically $20°$ and $-20°$)

## 5. Rotation of the Basis

The fiber direction is obtained by applying two successive rotations to the local basis.

### 5.1 Rotation Using Matrices

The two-step rotation can be expressed with standard rotation matrices applied to the local basis $\mathbf{Q} = [\mathbf{e}_c, \mathbf{e}_\ell, \mathbf{e}_t]$:

- Helix rotation by $\alpha$ about the transmural axis $\mathbf{e}_t$:

$$
\mathbf{R}_\alpha = \begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 \\
\sin\alpha & \cos\alpha & 0 \\
0 & 0 & 1
\end{bmatrix}, \quad
\mathbf{Q}^{(\alpha)} = \mathbf{Q}\,\mathbf{R}_\alpha.
$$

- Transverse rotation by $\beta$ about the rotated longitudinal axis $\mathbf{e}_\ell^{(\alpha)}$ (the second column of $\mathbf{Q}^{(\alpha)}$):

$$
\mathbf{R}_\beta = \begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}, \quad
\mathbf{Q}^{(\alpha,\beta)} = \mathbf{Q}^{(\alpha)}\,\mathbf{R}_\beta.
$$

This formulation rotates the circumferential direction toward the longitudinal direction (helix), then applies the transverse tilt about the updated longitudinal axis to obtain the final fiber, sheet, and sheet-normal directions.

### 5.2 Rotation Using Rodrigues' Formula

An equivalent, axis–angle formulation uses the Rodrigues rotation formula. For a unit axis $\mathbf{n}$ and angle $\theta$:

$$
\mathbf{R}(\theta, \mathbf{n}) = \mathbf{I}\cos\theta + [\mathbf{n}]_\times \sin\theta + \mathbf{n}\,\mathbf{n}^T (1 - \cos\theta),
$$

where the skew-symmetric matrix $[\mathbf{n}]_\times$ is

$$
[\mathbf{n}]_\times = \begin{bmatrix}
0 & -n_z & n_y \\
n_z & 0 & -n_x \\
-n_y & n_x & 0
\end{bmatrix}, \quad \mathbf{n} = (n_x, n_y, n_z)^T.
$$

Applying the two-step rotation to the local basis:

- Helix rotation by $\alpha$ about $\mathbf{e}_t$:

$$
\mathbf{Q}^{(\alpha)} = \mathbf{Q}\,\mathbf{R}(\alpha, \mathbf{e}_t).
$$

- Transverse rotation by $\beta$ about the rotated longitudinal axis $\mathbf{e}_\ell^{(\alpha)}$:

$$
\mathbf{Q}^{(\alpha,\beta)} = \mathbf{Q}^{(\alpha)}\,\mathbf{R}(\beta, \mathbf{e}_\ell^{(\alpha)}), \quad \mathbf{e}_\ell^{(\alpha)} = \big(\mathbf{Q}^{(\alpha)}\big)[:, 1].
$$

This axis–angle approach is numerically robust and generalizes to arbitrary rotation axes, aligning with the implementation used for outflow tract geometries.

## 6. Basis Interpolation (Bislerp)

When working with biventricular geometries, different basis vectors are computed for the left ventricle (LV), right ventricle (RV), and epicardium. These must be smoothly interpolated to avoid discontinuities.

### Spherical Linear Interpolation (SLERP)

Simple linear interpolation of rotation matrices can produce non-orthogonal results. Instead, **bislerp** (bilinear spherical interpolation) is used, which operates via quaternion representation:

Given two rotation matrices $\mathbf{Q}_1$ and $\mathbf{Q}_2$, and interpolation parameter $t \in [0, 1]$:

1. **Convert to quaternions**: 
   $$\mathbf{q}_1 = \text{rotm2quat}(\mathbf{Q}_1), \quad \mathbf{q}_2 = \text{rotm2quat}(\mathbf{Q}_2)$$

2. **Ensure shortest path** (quaternion double cover):
   $$\text{if } \mathbf{q}_1 \cdot \mathbf{q}_2 < 0: \quad \mathbf{q}_2 \leftarrow -\mathbf{q}_2$$

3. **SLERP formula**:
   $$\theta_0 = \arccos(\mathbf{q}_1 \cdot \mathbf{q}_2)$$
   $$\mathbf{q}(t) = \frac{\sin((1-t)\theta_0)}{\sin\theta_0}\mathbf{q}_1 + \frac{\sin(t\theta_0)}{\sin\theta_0}\mathbf{q}_2$$

4. **Convert back to rotation matrix**:
   $$\mathbf{Q}(t) = \text{quat2rotm}(\mathbf{q}(t))$$

For nearly parallel quaternions ($\sin\theta_0 < 10^{-6}$), linear interpolation is used instead to avoid numerical issues.

# Bayer Method

The Bayer et al. (2012) method is designed for truncated biventricular geometries without outflow tracts. 

### Required Laplace Fields

Four Laplace problems are solved:

1. **Transmural**: $\phi_{\text{epi}}$ (endo → epi)
2. **LV chamber**: $\phi_{\text{LV}}$ (RV free wall → LV free wall)
3. **RV chamber**: $\phi_{\text{RV}}$ (LV free wall → RV free wall)
4. **Apex-to-base**: $\phi_{\text{AB}}$ (apex → base)

### Angle Definition

The angles vary based on the interventricular coordinate $d$ and transmural coordinate $\phi_{\text{epi}}$:

$$
d = \frac{\phi_{\text{RV}}}{\phi_{\text{LV}} + \phi_{\text{RV}}}
$$

**Septum angles** (interpolated between LV and RV):
$$
\alpha_s = \alpha_{\text{endo}}(1 - d) - \alpha_{\text{endo}} d
$$
$$
\beta_s = \beta_{\text{endo}}(1 - d) - \beta_{\text{endo}} d
$$

**Wall angles** (transmural variation):
$$
\alpha_w = \alpha_{\text{endo}}(1 - \phi_{\text{epi}}) + \alpha_{\text{epi}}\phi_{\text{epi}}
$$
$$
\beta_w = \beta_{\text{endo}}(1 - \phi_{\text{epi}}) + \beta_{\text{epi}}\phi_{\text{epi}}
$$

### Algorithm Steps

1. Construct LV basis: $\mathbf{Q}_{\text{LV}}^0 = \text{axis}(\nabla\phi_{\text{AB}}, -\nabla\phi_{\text{LV}})$
2. Rotate by septum angles: $\mathbf{Q}_{\text{LV}} = \text{orient}(\mathbf{Q}_{\text{LV}}^0, \alpha_s, \beta_s)$
3. Construct RV basis: $\mathbf{Q}_{\text{RV}}^0 = \text{axis}(\nabla\phi_{\text{AB}}, \nabla\phi_{\text{RV}})$
4. Rotate by septum angles (note sign flip for $\beta$): $\mathbf{Q}_{\text{RV}} = \text{orient}(\mathbf{Q}_{\text{RV}}^0, \alpha_s, -\beta_s)$
5. Interpolate endocardial basis: $\mathbf{Q}_{\text{endo}} = \text{bislerp}(\mathbf{Q}_{\text{LV}}, \mathbf{Q}_{\text{RV}}, d)$
6. Flip vectors for $d > 0.5$ (to maintain coherence)
7. Construct epicardial basis: $\mathbf{Q}_{\text{epi}}^0 = \text{axis}(\nabla\phi_{\text{AB}}, \nabla\phi_{\text{epi}})$
8. Rotate by wall angles: $\mathbf{Q}_{\text{epi}} = \text{orient}(\mathbf{Q}_{\text{epi}}^0, \alpha_w, \beta_w)$
9. Interpolate final basis: $\mathbf{Q} = \text{bislerp}(\mathbf{Q}_{\text{endo}}, \mathbf{Q}_{\text{epi}}, \phi_{\text{epi}})$
10. Extract fiber directions: $\mathbf{f} = \mathbf{Q}[:, 0]$, $\mathbf{s} = \mathbf{Q}[:, 1]$, $\mathbf{n} = \mathbf{Q}[:, 2]$

### Implementation ([main_bayer.py](main_bayer.py))

The implementation follows this workflow:

```python
params = {
    "ALFA_END": 60.0,   # Endocardial helix angle (degrees)
    "ALFA_EPI": -60.0,  # Epicardial helix angle (degrees)
    "BETA_END": 20.0,   # Endocardial transverse angle (degrees)
    "BETA_EPI": -20.0,  # Epicardial transverse angle (degrees)
}
```

The script:
1. Generates the apex surface from the epicardium
2. Runs the Laplace solver with appropriate boundary conditions
3. Loads the Laplace solutions and computes gradients at cell centers
4. Applies the Bayer algorithm using vectorized operations
5. Saves fiber, sheet, and sheet-normal directions to separate VTU files

**Key differences from original Bayer paper**:
- Vector flipping strategy instead of the correction term (eliminates discontinuity)
- Beta rotation applied correctly about the longitudinal axis (not circumferential)
- All operations fully vectorized for performance

## 8. Doste Method

The Doste et al. (2019) method extends the fiber generation to biventricular geometries with **outflow tracts**, requiring valve surfaces to be explicitly defined.

### Required Surfaces

In addition to standard boundaries:
- **Mitral valve (mv)**: LV inflow
- **Aortic valve (av)**: LV outflow
- **Tricuspid valve (tv)**: RV inflow
- **Pulmonary valve (pv)**: RV outflow

### Required Laplace Fields

Eight Laplace problems are solved (4 per ventricle):

**Left Ventricle**:
1. $\phi_{\text{LV,trans}}$: LV endocardium → epicardium (transmural)
2. $\phi_{\text{LV,mv}}$: Mitral valve → other boundaries (MV longitudinal)
3. $\phi_{\text{LV,av}}$: Aortic valve → other boundaries (AV longitudinal)
4. $\phi_{\text{LV,ven}}$: RV → LV (ventricular)

**Right Ventricle**:
1. $\phi_{\text{RV,trans}}$: RV endocardium → epicardium (transmural)
2. $\phi_{\text{RV,tv}}$: Tricuspid valve → other boundaries (TV longitudinal)
3. $\phi_{\text{RV,pv}}$: Pulmonary valve → other boundaries (PV longitudinal)
4. $\phi_{\text{RV,ven}}$: LV → RV (ventricular)

Additionally, $\phi_{\text{epi,trans}}$ provides global transmural coordinate.

### Valve Weight Functions

To localize the influence of each valve, weight functions are computed and redistributed:

$$
w_{\text{LV}} = \text{redistribute}(\phi_{\text{LV,mv}}, q_{\text{up}} = 0.7, q_{\text{low}} = 0.01)
$$
$$
w_{\text{RV}} = \text{redistribute}(\phi_{\text{RV,tv}}, q_{\text{up}} = 0.1, q_{\text{low}} = 0.001)
$$

The redistribution clips values at quantiles $q_{\text{low}}$ and $q_{\text{up}}$, then renormalizes to $[0, 1]$.

### Angle Definition

The angles are region-specific with outflow tract blending:

**LV helix angles**:
$$
\alpha_{\text{LV,endo}} = \alpha_{\text{endo,LV}} w_{\text{LV}} + \alpha_{\text{OT,endo,LV}}(1 - w_{\text{LV}})
$$
$$
\alpha_{\text{LV,epi}} = \alpha_{\text{epi,LV}} w_{\text{LV}} + \alpha_{\text{OT,epi,LV}}(1 - w_{\text{LV}})
$$
$$
\alpha_{\text{LV}} = \alpha_{\text{LV,endo}}(1 - \phi_{\text{epi,trans}}) + \alpha_{\text{LV,epi}}\phi_{\text{epi,trans}}
$$

Similar formulations apply for $\alpha_{\text{RV}}$ and $\beta$ angles.

**Septum angles** are computed by blending LV and RV contributions using septal position.

### Basis Construction

For each ventricle, the longitudinal direction blends the two valve gradients:

$$
\mathbf{g}_{\ell,\text{LV}} = w_{\text{LV}} \nabla\phi_{\text{LV,mv}} + (1 - w_{\text{LV}})\nabla\phi_{\text{LV,av}}
$$

The transmural direction is orthogonalized:

$$
\mathbf{e}_{t,\text{LV}} = \text{normalize}\left(\nabla\phi_{\text{LV,trans}} - (\mathbf{e}_{\ell,\text{LV}} \cdot \nabla\phi_{\text{LV,trans}})\mathbf{e}_{\ell,\text{LV}}\right)
$$

Circumferential direction completes the orthonormal system:

$$
\mathbf{e}_{c,\text{LV}} = \mathbf{e}_{\ell,\text{LV}} \times \mathbf{e}_{t,\text{LV}}
$$

### Rotation and Interpolation

The rotation uses Rodrigues' formula (see Section 5.2) for arbitrary rotation axes. For rotation by $\alpha$ about axis $\mathbf{n}$:

$$
\mathbf{R}(\alpha, \mathbf{n}) = \mathbf{I}\cos\alpha + [\mathbf{n}]_\times \sin\alpha + \mathbf{n}\mathbf{n}^T(1 - \cos\alpha)
$$

where $[\mathbf{n}]_\times$ is the skew-symmetric matrix of $\mathbf{n}$.

The rotations are applied:
1. First rotation by $\alpha$ about $\mathbf{e}_t$
2. Second rotation by $\beta$ about rotated $\mathbf{e}_\ell$

Finally, bislerp interpolation is performed in three stages:
$$
\mathbf{Q}_{\text{epi}} = \text{bislerp}(\mathbf{Q}_{\text{RV,epi}}, \mathbf{Q}_{\text{LV,epi}}, \phi_{\text{ven,trans}})
$$
$$
\mathbf{Q}_{\text{endo}} = \text{bislerp}(\mathbf{Q}_{\text{RV,septum}}, \mathbf{Q}_{\text{LV,septum}}, \phi_{\text{ven,trans}})
$$
$$
\mathbf{Q} = \text{bislerp}(\mathbf{Q}_{\text{endo}}, \mathbf{Q}_{\text{epi}}, \phi_{\text{epi,trans}})
$$

### Implementation ([main_doste.py](main_doste.py))

The implementation uses different angle parameters for each region:

```python
params = {
    'AENDORV': 90,    'AEPIRV': -25,     # RV helix angles
    'AENDOLV': 60,    'AEPILV': -60,     # LV helix angles
    'AOTENDOLV': 90,  'AOTEPILV': 0,     # LV outflow tract helix
    'AOTENDORV': 90,  'AOTEPIRV': 0,     # RV outflow tract helix
    'BENDORV': 20,    'BEPIRV': -20,     # RV transverse angles
    'BENDOLV': 20,    'BEPILV': -20,     # LV transverse angles
}
```

The workflow:
1. Generates apex surface
2. Runs Laplace solver for all 8+ fields
3. Computes basis vectors separately for LV and RV
4. Applies region-specific angle formulations with valve weighting
5. Performs three-stage bislerp interpolation
6. Saves fiber, sheet, and sheet-normal directions

**Key features**:
- Handles complex geometries with outflow tracts
- Valve-specific angle definitions
- Smooth transitions between ventricular and outflow tract regions
- More anatomically accurate for complete heart models


# Notes
## On the BISLERP function