"""
Using Simplicit's Easy API To Simulate Example Mesh

Simplicits is a mesh-free, representation-agnostic way to simulate elastic deformations.
This script demonstrates a simple way to use the simplicit's code base to create a simple object,
train it, simulate it and visualize it using polyscope.
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import kaolin as kal
import warp as wp
import polyscope as ps

# Local logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Logger used in the api code
logging.getLogger('kaolin.physics').setLevel(logging.INFO)

sys.path.append(str(Path(__file__).parent.parent))
from tutorial_common import COMMON_DATA_DIR

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Simulate elastic deformations using Simplicits"
)
parser.add_argument(
    "--load-skinning",
    type=str,
    default=".default_cache.pth",
    help="Path to load pre-trained skinning function from disk"
)
parser.add_argument(
    "--save-skinning",
    type=str,
    default=".default_cache.pth",
    help="Path to save trained skinning function to disk"
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=100,
    help="Number of simulation steps to run (default: 200)"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Run polyscope visualization (default: False)"
)
parser.add_argument(
    "--headless-steps",
    type=int,
    default=None,
    help="Run simulation headless for N steps (no interactive window)"
)
parser.add_argument(
    "--method",
    type=str,
    choices=["trained", "rkpm"],
    default="trained",
    help="Method to create SimplicitsObject: 'trained' (neural network) or 'rkpm' (RKPM) (default: trained)"
)
parser.add_argument(
    "--visualize-weights",
    action="store_true",
    help="Visualize RKPM skinning weights on sample points"
)

args = parser.parse_args()

# Load and prepare mesh
logger.info("Loading geometry...")
mesh = kal.io.import_mesh(os.path.join(COMMON_DATA_DIR, 'meshes', 'fox.obj'), triangulate=True).cuda()
mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
orig_vertices = mesh.vertices.clone()
logger.info(f"Mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

# Sample geometry
logger.info("Sampling geometry...")

# Physics material parameters
# soft_youngs_modulus = 1e5
soft_youngs_modulus = 1e4
poisson_ratio = 0.45
rho = 500.0  # kg/m^3
approx_volume = 0.5  # m^3

# Points sampled over the object's bounding box
num_samples = 1000000
uniform_pts = torch.rand(num_samples, 3, device='cuda') * (
    orig_vertices.max(dim=0).values - orig_vertices.min(dim=0).values
) + orig_vertices.min(dim=0).values
boolean_signs = kal.ops.mesh.check_sign(
    mesh.vertices.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512
)

# Use pts within the object
pts = uniform_pts[boolean_signs.squeeze()]
yms = torch.full((pts.shape[0],), soft_youngs_modulus, device="cuda")
prs = torch.full((pts.shape[0],), poisson_ratio, device="cuda")
rhos = torch.full((pts.shape[0],), rho, device="cuda")

logger.info(f"Sampled {pts.shape[0]} points within mesh volume")

# Create or load trained object
skinning_path = args.load_skinning or args.save_skinning

if args.load_skinning and os.path.exists(args.load_skinning):
    logger.info(f"Loading pre-trained skinning function from {args.load_skinning}...")
    skinning_fcn = torch.load(args.load_skinning)
    sim_obj = kal.physics.simplicits.SimplicitsObject.create_from_function(
        pts, yms, prs, rhos, approx_volume, skinning_fcn
    )
else:
    if args.method == "trained":
        logger.info("Training SimplicitsObject using neural network (this will take a couple of minutes)...")
        sim_obj = kal.physics.simplicits.SimplicitsObject.create_trained(
            pts,  # sampled points
            yms,  # material stiffness
            prs,  # material compressibility ratio
            rhos,  # material density
            approx_volume,  # volume
            num_handles=5,  # skinning handles (DOFs)
            training_num_steps=10000,
            training_lr_start=1e-3,
            training_lr_end=1e-3,
            training_le_coeff=1e-1,
            training_lo_coeff=1e6,
            training_log_every=1000,
            normalize_for_training=True
        )
        logger.info("Training completed!")
    elif args.method == "rkpm":
        logger.info("Creating SimplicitsObject using RKPM method...")
        sim_obj = kal.physics.simplicits.SimplicitsObject.create_rkpm(
            pts,  # sampled points
            yms,  # material stiffness
            prs,  # material compressibility ratio
            rhos,  # material density
            approx_volume,  # volume
            num_handles=10,  # skinning handles (DOFs)
            num_points=16384,  # number of sample points for RKPM
            num_nodes=1024,  # number of nodes for RKPM
            use_double=True,
        )
        logger.info("RKPM object created!")

    # Optionally save the trained function
    if args.save_skinning:
        save_dir = os.path.dirname(args.save_skinning)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(sim_obj.skinning_weight_function, args.save_skinning)
        logger.info(f"Saved skinning function to {args.save_skinning}")

# Optionally visualize skinning weights
if args.visualize_weights:
    logger.info("Visualizing skinning weights...")
    ps.init()

    # Compute skinning weights
    with torch.no_grad():
        weights = sim_obj.skinning_weight_function(pts)  # Shape: (num_pts, num_handles)

    logger.info(f"Skinning weights shape: {weights.shape}")

    # Register point cloud
    ps_pts = ps.register_point_cloud("Sample Points", pts.cpu().detach().numpy(), radius=0.003)

    # Visualize each handle's weights as a scalar quantity
    num_handles = weights.shape[1]
    for i in range(num_handles):
        weight_values = weights[:, i].cpu().detach().numpy()
        ps_pts.add_scalar_quantity(f"Handle_{i}_weight", weight_values, enabled=(i==0))

    # Also show the dominant handle for each point
    dominant_handle = torch.argmax(weights, dim=1).cpu().detach().numpy()
    ps_pts.add_scalar_quantity("Dominant_Handle", dominant_handle, enabled=True, cmap="rainbow")

    logger.info(f"Visualizing {num_handles} skinning weight handles")
    ps.show()

# Create scene
logger.info("Creating scene...")
scene = kal.physics.simplicits.SimplicitsScene()
scene.max_newton_steps = 5
# scene.timestep = 0.03
scene.timestep = 0.01
scene.direct_solve = True

# Add object to scene
obj_idx = scene.add_object(sim_obj)

# Set gravity and floor forces
scene.set_scene_gravity(acc_gravity=torch.tensor([0, 9.8, 0]))
scene.set_scene_floor(floor_height=-0.8, floor_axis=1, floor_penalty=1000)

logger.info("Scene created with gravity and floor")

if args.visualize:
    # Run simulation
    logger.info("Starting simulation...")

    # Initialize polyscope
    if args.headless_steps is not None:
        ps.init("openGL3_egl")
    else:
        ps.init()
    ps.set_ground_plane_mode("tile_reflection")
    ps.set_ground_plane_height(-0.8)
    ps.set_up_dir("y_up")

    # Set frame rate to match simulation timestep
    target_fps = 1.0 / scene.timestep
    ps.set_max_fps(int(target_fps))
    logger.info(f"Set max FPS to {int(target_fps)} (dt={scene.timestep})")

    # Reset scene
    scene.reset_scene()
    mesh.vertices = scene.get_object_deformed_pts(obj_idx, orig_vertices)

    # Register the mesh
    ps_mesh = ps.register_surface_mesh(
        "Fox",
        mesh.vertices.cpu().detach().numpy(),
        mesh.faces.cpu().detach().numpy(),
        smooth_shade=True
    )

    # Run simulation headless or interactive
    if args.headless_steps is not None:
        # Headless mode: run simulation directly without pre-computing
        logger.info(f"Running simulation headless for {args.headless_steps} steps...")
        scene.reset_scene()
        initial_vertices = scene.get_object_deformed_pts(obj_idx, orig_vertices).clone()

        for step in range(args.headless_steps):
            scene.run_sim_step()
            mesh.vertices = scene.get_object_deformed_pts(obj_idx, orig_vertices)
            ps_mesh.update_vertex_positions(mesh.vertices.cpu().detach().numpy())
            if step % 10 == 0:
                vertex_diff = (mesh.vertices - initial_vertices).norm(dim=1).mean().item()
                logger.info(f"Simulation step {step}/{args.headless_steps}, avg vertex displacement: {vertex_diff:.6f}")

        final_vertex_diff = (mesh.vertices - initial_vertices).norm(dim=1).mean().item()
        logger.info(f"Headless simulation completed! Final avg displacement: {final_vertex_diff:.6f}")
    else:
        # Interactive mode: pre-compute all frames for scrubbing
        logger.info(f"Pre-computing {args.num_steps} simulation steps...")
        simulation_frames = []
        scene.reset_scene()

        # Store initial state
        simulation_frames.append(scene.get_object_deformed_pts(obj_idx, orig_vertices).cpu().clone())

        # Run simulation and store each frame
        for step in range(args.num_steps):
            scene.run_sim_step()
            deformed_verts = scene.get_object_deformed_pts(obj_idx, orig_vertices)
            simulation_frames.append(deformed_verts.cpu().clone())
            if step % 10 == 0:
                logger.info(f"  Computing step {step}/{args.num_steps}")

        logger.info(f"Pre-computation complete! {len(simulation_frames)} frames stored.")

        # Simulation state
        sim_state = {"current_frame": 0, "max_frames": len(simulation_frames) - 1, "playing": False}

        # Set initial mesh to first frame
        mesh.vertices = simulation_frames[0].cuda()
        ps_mesh.update_vertex_positions(mesh.vertices.cpu().detach().numpy())

        # Combined callback for both simulation and UI
        def combined_callback():
            # Play/Pause button
            button_label = "Pause" if sim_state["playing"] else "Play"
            if ps.imgui.Button(button_label):
                sim_state["playing"] = not sim_state["playing"]
                logger.info(f"Animation {'playing' if sim_state['playing'] else 'paused'}")

            ps.imgui.SameLine()
            if ps.imgui.Button("Reset to Start"):
                sim_state["current_frame"] = 0
                sim_state["playing"] = False
                mesh.vertices = simulation_frames[0].cuda()
                ps_mesh.update_vertex_positions(mesh.vertices.cpu().detach().numpy())
                logger.info("Reset to frame 0")

            # UI controls - slider to scrub through simulation
            changed, new_frame = ps.imgui.SliderInt("Simulation Step", sim_state["current_frame"], 0, sim_state["max_frames"])

            if changed:
                sim_state["current_frame"] = new_frame
                sim_state["playing"] = False  # Pause when user scrubs manually
                mesh.vertices = simulation_frames[new_frame].cuda()
                ps_mesh.update_vertex_positions(mesh.vertices.cpu().detach().numpy())

            # Auto-advance frame when playing
            if sim_state["playing"]:
                sim_state["current_frame"] += 1
                if sim_state["current_frame"] > sim_state["max_frames"]:
                    sim_state["current_frame"] = 0  # Loop back to start
                mesh.vertices = simulation_frames[sim_state["current_frame"]].cuda()
                ps_mesh.update_vertex_positions(mesh.vertices.cpu().detach().numpy())

        ps.set_user_callback(combined_callback)
        ps.show()
else:
    logger.info("Skipping visualization. Scene is ready for simulation.")
    logger.info("Use --visualize flag to run polyscope visualization.")
