# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RKPM Fracture Demo — 4D Lifted Coordinates
===========================================

Demonstrates progressive fracture using RKPM skinning weights lifted to 4D
coordinates (x, y, z, H) where H is the disc winding number.  No pre-training
is required — RKPM is fully analytical.

Controls (Polyscope GUI):
  - Play / Pause     : toggle simulation
  - Reset            : restore rest state
  - Cut progress α   : animate fracture (calls set_cut_progress_rkpm)
  - Handle index     : visualize which RKPM eigenmode is active
"""

import torch
import numpy as np

from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene, DiscCut

# ── Scene configuration ────────────────────────────────────────────────────

DEVICE = 'cuda'
DTYPE = torch.float

# Object geometry: random points in a [-0.5, 0.5]^3 box
N_PTS = 50000
torch.manual_seed(0)
pts = (torch.rand(N_PTS, 3, device=DEVICE) - 0.5).to(DTYPE)

# Material properties
YMS = 1e4    # Young's modulus
PRS = 0.45   # Poisson's ratio
RHOS = 500.  # Density
APPX_VOL = 1.0

# RKPM parameters
NUM_HANDLES = 10   # number of RKPM eigenmodes (total handles = NUM_HANDLES + 1)
NUM_NODES = 128   # RKPM kernel nodes
CUTOFF_FACTOR = 1.5  # kernel truncation radius in units of node radius (default 3.0); lower = more local

# Disc cut: YZ-plane, centred at origin, radius covers the box
CUT_NORMAL = torch.tensor([1., 0., 0.])
CUT_CENTER = torch.tensor([0., 0.7, 0.5])
CUT_RADIUS = 2.0

cut = DiscCut(
    normal=CUT_NORMAL.to(DEVICE),
    center=CUT_CENTER.to(DEVICE),
    radius=CUT_RADIUS,
    alpha=0.0,
    tanh_sharpness=1000.0,
    h_scale=10.0,
)

# Colours for each side of the cut
COLOR_POS = np.array([0.25, 0.55, 0.95])  # blue — positive side (x > 0)
COLOR_NEG = np.array([0.95, 0.35, 0.25])  # red  — negative side (x < 0)

FLOOR_HEIGHT = -1.5


# ── Helpers ─────────────────────────────────────────────────────────────────

def side_colors(pts_np, cut_normal_np, cut_center_np):
    """Per-point RGB colours: blue if on positive side of cut, red if negative."""
    H = (pts_np - cut_center_np) @ cut_normal_np  # (N,)
    return np.where(H[:, None] >= 0,
                    COLOR_POS[None, :],
                    COLOR_NEG[None, :]).astype(np.float32)


def make_disc_verts(alpha, cut_center_np, t1, t2, num_segs=64):
    """Vertices for a growing disc of radius CUT_RADIUS * alpha."""
    r = CUT_RADIUS * alpha
    angles = np.linspace(0, 2 * np.pi, num_segs, endpoint=False)
    rim = cut_center_np + r * (np.outer(np.cos(angles), t1) + np.outer(np.sin(angles), t2))
    return np.vstack([cut_center_np[None], rim]).astype(np.float32)

# ── Build object ────────────────────────────────────────────────────────────

print("Initialising 4D RKPM skinning weights … ", end='', flush=True)
obj = SimplicitsObject.create_rkpm_with_cut(
    pts=pts,
    yms=YMS, prs=PRS, rhos=RHOS, appx_vol=APPX_VOL,
    cut=cut,
    num_handles=NUM_HANDLES,
    num_nodes=NUM_NODES,
    use_double=True,
    initial_cut_alpha=0.0,
    cutoff_factor=CUTOFF_FACTOR,
)
print(f"done. num_handles={obj.num_handles}")

# ── Build scene ─────────────────────────────────────────────────────────────

scene = SimplicitsScene(
    device=DEVICE,
    dtype=DTYPE,
    direct_solve=True,
    use_cuda_graphs=False,
    timestep=0.03,
    max_newton_steps=5,
)

obj_id = scene.add_object(obj, num_qp=2000)
scene.set_scene_gravity(acc_gravity=torch.tensor([0., 9.8, 0.], device=DEVICE))
scene.set_scene_floor(floor_height=FLOOR_HEIGHT, floor_axis=1, floor_penalty=1000.)

# Pin the max-x edge so the object hangs and the fracture is visible
x_vals = obj.pts[:, 0]
x_thresh = x_vals.max() - 0.05 * (x_vals.max() - x_vals.min())
scene.set_object_boundary_condition(
    obj_id, "pin_edge_x",
    lambda x: x[:, 0] >= x_thresh,
    bdry_penalty=10000.)

print(f"Scene ready. obj_id={obj_id}")

# ── Polyscope visualisation ─────────────────────────────────────────────────

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ImportError:
    print("polyscope not installed — running headless for 10 steps.")
    for step in range(10):
        scene.run_sim_step()
        deformed = scene.get_object_deformed_pts(obj_id)
        print(f"  step {step}: mean y = {deformed[:, 1].mean():.4f}")
    raise SystemExit(0)

ps.init()
ps.set_up_dir("y_up")
ps.set_ground_plane_height(FLOOR_HEIGHT)

# Build disc tangent frame from cut normal
cut_normal_np = CUT_NORMAL.numpy()
cut_center_np = CUT_CENTER.numpy()
n = cut_normal_np
ref = np.array([0., 1., 0.]) if abs(n[1]) < 0.9 else np.array([1., 0., 0.])
t1 = ref - np.dot(ref, n) * n; t1 /= np.linalg.norm(t1)
t2 = np.cross(n, t1)
num_segs = 64
disc_faces = np.array([[0, i + 1, (i + 1) % num_segs + 1] for i in range(num_segs)])

pts_np = pts.cpu().numpy()
rest_colors = side_colors(pts_np, cut_normal_np, cut_center_np)

pc = ps.register_point_cloud("object", pts_np, radius=0.005)
pc.add_color_quantity("side", rest_colors, enabled=True)

# Growing disc mesh (starts collapsed at alpha=0)
disc_mesh = ps.register_surface_mesh(
    "cut disc", make_disc_verts(0.0, cut_center_np, t1, t2, num_segs),
    disc_faces, transparency=0.4)
disc_mesh.set_color((0.8, 0.8, 0.2))

# State
state = {'playing': False, 'alpha': 0.0, 'handle': 0}


def update_pts():
    deformed = scene.get_object_deformed_pts(obj_id, points=pts).cpu().numpy()
    pc.update_point_positions(deformed)


def update_weights(handle_idx):
    with torch.no_grad():
        w = obj.skinning_weight_function(pts, obj.cut).cpu().numpy()
    pc.add_scalar_quantity("handle_weight", w[:, handle_idx], enabled=True, cmap='coolwarm')


def play_callback():
    s = state

    # Play / Pause
    _, s['playing'] = psim.Checkbox("Play", s['playing'])
    psim.SameLine()

    # Reset
    if psim.Button("Reset"):
        scene.reset_scene()
        s['playing'] = False

    # Cut progress slider
    psim.Separator()
    psim.TextUnformatted("Cut progress α  (0 = uncut  →  1 = fully cut)")
    changed_alpha, new_alpha = psim.SliderFloat("##alpha", s['alpha'], v_min=0.0, v_max=1.0)
    if changed_alpha:
        s['alpha'] = new_alpha
        scene.set_cut_progress_rkpm(obj_id, alpha=new_alpha)
        disc_mesh.update_vertex_positions(
            make_disc_verts(new_alpha, cut_center_np, t1, t2, num_segs))
        update_pts()
        update_weights(s['handle'])

    # Handle weight visualiser
    psim.Separator()
    psim.TextUnformatted("Handle weight visualizer")
    changed_h, new_h = psim.SliderInt("##handle", s['handle'],
                                      v_min=0, v_max=obj.num_handles - 1)
    if changed_h:
        s['handle'] = new_h
        update_weights(new_h)

    if s['playing']:
        scene.run_sim_step()
        update_pts()


update_pts()
ps.set_user_callback(play_callback)
ps.show()
