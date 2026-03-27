"""
Example: XPBD Deformable Mesh Drop with Gaussian Splat Deformation

Demonstrates dropping a doll mesh as a deformable body onto a ground plane
using Newton's XPBD solver. RKPM skinning weights propagate the mesh
deformation to Gaussian splat points via least-squares DOF fitting.

Step 1: XPBD deformable tet mesh simulation (Newton + solve_tetrahedra2).
Step 2: At each frame, fit Simplicits DOFs to the deformed mesh vertices,
        then apply those DOFs to splat positions via linear blend skinning.

The tet mesh is pre-generated (e.g. via TetWild) and loaded from a .msh file.

Newton note: We monkey-patch ``solve_tetrahedra`` → ``solve_tetrahedra2`` so
the XPBD solver uses material-aware NeoHookean tet constraints (k_mu, k_lambda).

Requirements:
    pip install meshio polyscope plyfile

Usage:
    python newton_xpbd_splat_deformation.py
"""

from __future__ import annotations

import os
from collections import Counter

import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
import warp as wp
from plyfile import PlyData

import newton
import newton._src.solvers.xpbd.solver_xpbd as _xpbd_mod
from newton._src.solvers.xpbd.kernels import solve_tetrahedra2
from newton.solvers import SolverXPBD

from kaolin.physics.simplicits import SimplicitsObject
from kaolin.physics.simplicits.easy_api import SimulatedObject
from gaussian_utils import transform_gaussians_lbs, pad_transforms

# Swap solve_tetrahedra → solve_tetrahedra2 so XPBD uses material-aware
# NeoHookean tet constraints (k_mu, k_lambda) instead of relaxation-only.
_xpbd_mod.solve_tetrahedra = solve_tetrahedra2

# SH DC to RGB conversion constant
SH_C0 = 0.28209479177387814

# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOLL_DIR = os.path.join(SCRIPT_DIR, "inria", "gaussian-splatting", "output", "doll")
TET_MESH_PATH = os.path.join(DOLL_DIR, "mesh.BluehairRagdoll_.msh")
SPLAT_PATH = os.path.join(DOLL_DIR, "point_cloud", "iteration_30000", "point_cloud.ply")

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

# Material (stored on the model; used by other solvers — XPBD tet pass uses SOFT_BODY_RELAXATION)
YOUNGS_MODULUS = 5e7
POISSON_RATIO = 0.3
K_MU = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
K_LAMBDA = YOUNGS_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
K_DAMP = 0.01
DENSITY = 500.0  # kg/m^3
PARTICLE_RADIUS = 0.005

# XPBD: lower compliance → stiffer tets (see Newton solver_xpbd.solve_tetrahedra)
SOFT_BODY_RELAXATION = 0.02
SOFT_CONTACT_RELAXATION = 0.65

# Contact — keep below extreme values or ground correction overwhelms soft-body constraints
SOFT_CONTACT_KE = 2.0e5
SOFT_CONTACT_KD = 40.0
SOFT_CONTACT_MU = 0.5

# Simulation
FRAME_DT = 1.0 / 30.0
SIM_SUBSTEPS = 10
SIM_DT = FRAME_DT / SIM_SUBSTEPS
XPBD_ITERATIONS = 100

# Scene
DROP_HEIGHT = 0.5  # Z offset above mesh's lowest point

# RKPM
NUM_RKPM_HANDLES = 30
NUM_RKPM_NODES = 1024
RKPM_APPX_VOL = 0.5


# ============================================================================
# HELPERS
# ============================================================================

def load_gaussian_splat(path: str):
    """Load Gaussian splat data from a PLY file.

    Returns:
        xyz: (N, 3) float32 numpy array of positions.
        rotation: (N, 4) float32 numpy array of quaternions (wxyz).
        scaling: (N, 3) float32 numpy array of log-scales.
        rgb: (N, 3) float32 numpy array in [0, 1].
    """
    plydata = PlyData.read(path)
    elem = plydata.elements[0]
    xyz = np.stack([
        np.asarray(elem["x"], dtype=np.float32),
        np.asarray(elem["y"], dtype=np.float32),
        np.asarray(elem["z"], dtype=np.float32),
    ], axis=1)
    rotation = np.stack([
        np.asarray(elem["rot_0"], dtype=np.float32),
        np.asarray(elem["rot_1"], dtype=np.float32),
        np.asarray(elem["rot_2"], dtype=np.float32),
        np.asarray(elem["rot_3"], dtype=np.float32),
    ], axis=1)
    scaling = np.stack([
        np.asarray(elem["scale_0"], dtype=np.float32),
        np.asarray(elem["scale_1"], dtype=np.float32),
        np.asarray(elem["scale_2"], dtype=np.float32),
    ], axis=1)
    # Convert SH DC coefficients to RGB
    sh_dc = np.stack([
        np.asarray(elem["f_dc_0"], dtype=np.float32),
        np.asarray(elem["f_dc_1"], dtype=np.float32),
        np.asarray(elem["f_dc_2"], dtype=np.float32),
    ], axis=1)
    rgb = np.clip(sh_dc * SH_C0 + 0.5, 0.0, 1.0)
    return xyz, rotation, scaling, rgb


def load_tet_mesh(msh_path: str):
    """Load a tetrahedral mesh from a .msh file.

    Extracts vertices, tet indices, and boundary surface triangles.
    Filters out tiny-volume tets that can cause solver instability.

    Returns:
        nodes: (V, 3) float64 vertices.
        tets: (T, 4) int32 tet indices.
        surface_faces: (F, 3) int32 boundary triangle indices.
    """
    mesh = meshio.read(msh_path)
    nodes = mesh.points.astype(np.float64)

    tets = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            break
    if tets is None:
        raise RuntimeError(f"No tetrahedra found in {msh_path}")

    # Filter tiny-volume tets
    v0, v1, v2, v3 = nodes[tets[:, 0]], nodes[tets[:, 1]], nodes[tets[:, 2]], nodes[tets[:, 3]]
    d = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)
    vols = np.abs(np.linalg.det(d)) / 6.0
    good = vols > 1e-8
    n_removed = (~good).sum()
    tets = tets[good]

    # Extract boundary surface triangles (faces belonging to exactly one tet)
    face_count = Counter()
    for tet in tets:
        for tri in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_count[tuple(sorted(tri))] += 1
    surface_faces = np.array([f for f, cnt in face_count.items() if cnt == 1], dtype=np.int32)

    print(f"Tet mesh: {nodes.shape[0]} verts, {tets.shape[0]} tets, "
          f"{surface_faces.shape[0]} surface tris"
          + (f" ({n_removed} tiny tets removed)" if n_removed else ""))

    return nodes, tets, surface_faces


# ============================================================================
# EXAMPLE CLASS
# ============================================================================

class Example:
    """XPBD deformable doll mesh drop with RKPM splat deformation."""

    def __init__(self):
        # ---- Load tet mesh ----
        tet_nodes, tet_elems, self.surface_faces = load_tet_mesh(TET_MESH_PATH)

        # Shift mesh so its lowest point starts at DROP_HEIGHT above ground
        z_min = tet_nodes[:, 2].min()
        self.z_offset = DROP_HEIGHT - z_min
        tet_nodes_shifted = tet_nodes.copy()
        tet_nodes_shifted[:, 2] += self.z_offset

        # ---- Load splats ----
        splat_xyz, splat_rot, splat_scale, self.splat_rgb = load_gaussian_splat(SPLAT_PATH)
        splat_xyz_shifted = splat_xyz.copy()
        splat_xyz_shifted[:, 2] += self.z_offset

        # ---- Build Newton model ----
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=tet_nodes_shifted.tolist(),
            indices=tet_elems.flatten().tolist(),
            density=DENSITY,
            k_mu=K_MU,
            k_lambda=K_LAMBDA,
            k_damp=K_DAMP,
            particle_radius=PARTICLE_RADIUS,
        )

        self.model = builder.finalize()
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_MU

        # ---- Solver ----
        self.solver = SolverXPBD(
            self.model,
            iterations=XPBD_ITERATIONS,
            soft_body_relaxation=SOFT_BODY_RELAXATION,
            soft_contact_relaxation=SOFT_CONTACT_RELAXATION,
        )

        # ---- States ----
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # ---- RKPM skinning weights for splat deformation ----
        print("Building RKPM skinning weights...")
        rest_verts_torch = torch.tensor(tet_nodes_shifted, dtype=torch.float32, device="cuda")
        splat_xyz_torch = torch.tensor(splat_xyz_shifted, dtype=torch.float32, device="cuda")

        # Create RKPM object from mesh vertices
        sim_obj = SimplicitsObject.create_rkpm(
            rest_verts_torch,
            yms=float(YOUNGS_MODULUS),
            prs=float(POISSON_RATIO),
            rhos=float(DENSITY),
            appx_vol=RKPM_APPX_VOL,
            num_handles=NUM_RKPM_HANDLES,
            num_nodes=NUM_RKPM_NODES,
        )

        # Bake simulation weights at mesh vertices → get B matrix for DOF fitting
        sim_pts_baked = sim_obj.bake_for_simulation()
        sim_object = SimulatedObject.from_skinned_physics_points(sim_pts_baked, init_transform=None)
        self.B_dense = sim_object.B_dense  # (3*N_mesh, 12*(H+1))
        self.rest_verts = rest_verts_torch  # (N_mesh, 3)

        # Bake rendering weights at splat positions
        splat_skinned = sim_obj.bake_for_rendering(splat_xyz_torch)
        self.splat_weights = splat_skinned.skinning_weights  # (N_splat, H+1)
        self.splat_rest_pts = splat_skinned.pts  # (N_splat, 3)

        # Store rest-pose splat attributes for transform_gaussians_lbs
        self.splat_rest_rot = torch.tensor(splat_rot, dtype=torch.float32, device="cuda")
        self.splat_rest_scales = torch.tensor(splat_scale, dtype=torch.float32, device="cuda")

        num_handles = self.splat_weights.shape[1]
        print(f"RKPM: {num_handles} handles (including constant), "
              f"B_dense shape: {list(self.B_dense.shape)}, "
              f"splat weights shape: {list(self.splat_weights.shape)}")

        self.sim_time = 0.0
        self.playing = False
        print(f"Loaded {splat_xyz.shape[0]} splats")

    def step(self):
        """Advance the simulation by one substep."""
        self.state_0.clear_forces()
        contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, contacts, SIM_DT)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def restart(self):
        """Reset simulation to initial state."""
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_time = 0.0
        self.playing = False

    def get_particle_positions(self) -> np.ndarray:
        """Get current particle positions from state."""
        return self.state_0.particle_q.numpy().astype(np.float64)

    def get_deformed_splats(self) -> np.ndarray:
        """Fit Simplicits DOFs to current mesh deformation, apply to splats.

        Uses transform_gaussians_lbs (same pipeline as the Simplicits notebook)
        to compute per-point 4x4 transforms from skinning weights and handle
        transforms, then applies them to splat positions.

        Returns:
            (N_splat, 3) float64 deformed splat positions.
        """
        # Current deformed mesh verts from XPBD
        deformed_np = self.state_0.particle_q.numpy().astype(np.float32)
        deformed = torch.tensor(deformed_np, device="cuda")

        # Displacement from rest
        dx = (deformed - self.rest_verts).reshape(-1, 1)  # (3*N, 1)

        # Least-squares: B @ z = dx → z = B^+ @ dx
        z = torch.linalg.lstsq(self.B_dense, dx).solution  # (12*(H+1), 1)

        # Reshape to per-handle relative transforms: (H+1, 3, 4)
        num_handles = self.splat_weights.shape[1]
        tfms = z.squeeze(1).reshape(num_handles, 3, 4)

        # Pad to (1, H+1, 4, 4) and apply via transform_gaussians_lbs
        tfms_padded = pad_transforms(tfms).unsqueeze(0)
        new_xyz, _, _, _ = transform_gaussians_lbs(
            self.splat_rest_pts, self.splat_rest_rot, self.splat_rest_scales,
            self.splat_weights, tfms_padded,
        )

        return new_xyz.cpu().numpy().astype(np.float64)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    example = Example()

    # ---- Polyscope setup ----
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.0, is_relative=False)

    # Deformable mesh (boundary surface of tet mesh)
    ps_mesh = ps.register_surface_mesh(
        "doll_mesh",
        example.get_particle_positions(),
        example.surface_faces,
        smooth_shade=True,
    )

    # Splat point cloud with RGB colors
    ps_splats = ps.register_point_cloud(
        "splats",
        example.get_deformed_splats(),
        point_render_mode="quad",
        radius=0.001,
    )
    ps_splats.add_color_quantity("rgb", example.splat_rgb, enabled=True)

    # Ground plane visual
    g = 2.0
    ground_verts = np.array([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]])
    ground_faces = np.array([[0, 1, 2], [0, 2, 3]])
    ps.register_surface_mesh(
        "ground", ground_verts, ground_faces,
        color=(0.85, 0.85, 0.85), transparency=0.3,
    )

    def callback():
        if example.playing:
            if psim.Button("Pause"):
                example.playing = False
        else:
            if psim.Button("Play"):
                example.playing = True

        psim.SameLine()
        if psim.Button("Restart"):
            example.restart()
            ps_mesh.update_vertex_positions(example.get_particle_positions())
            ps_splats.update_point_positions(example.get_deformed_splats())

        psim.Text(f"Time: {example.sim_time:.2f}s")

        if example.playing:
            for _ in range(SIM_SUBSTEPS):
                example.step()
            ps_mesh.update_vertex_positions(example.get_particle_positions())
            ps_splats.update_point_positions(example.get_deformed_splats())
            example.sim_time += FRAME_DT

    ps.set_user_callback(callback)
    ps.show()
