#!/usr/bin/env python
from __future__ import print_function

import pybullet as p
import pybullet_planning as pp
import pybullet_data
import numpy as np
import time
import json
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from PIL import Image

from pybullet_planning import (
    connect,
    disconnect,
    load_pybullet,
    wait_if_gui,
    set_joint_positions,
    dump_world,
    convex_hull,
    create_mesh,
    remove_body,
)
from pybullet_tools.ur5e_robotiq_utils import (
    INITIAL_CONF,
    VERTICAL_CONF,
    get_arm_joints,
    setup_mimic_joints,
    set_gripper_open,
    set_gripper_close,
    move_gripper,
    close_until_collision,
    set_gripper,
    move_arm,
    get_arm_conf,
    pairwise_collision_with_allowed,
    get_top_grasps,
    get_side_grasps,
)
from pybullet_tools.ur5e_robotiq_primitives import (
    UR5E_ROBOTIQ_URDF,
    GRASP_INFO,
    BodyPose,
    BodyConf,
    BodyGrasp,
    get_grasp_gen,
    get_ik_fn,
    get_free_motion_gen,
    get_holding_motion_gen,
)
from pybullet_tools.utils import (
    WorldSaver,
    get_box_geometry,
    get_cylinder_geometry,
    create_shape,
    create_body,
    draw_pose,
    update_state,
    set_default_camera,
    set_camera,
    get_link_pose,
    draw_pose,
    end_effector_from_body,
    link_from_name,
    set_dynamics,
    get_dynamics_info,
    control_joints,
    get_links,
    add_fixed_constraint,
    Attachment,
    create_attachment,
    get_pose,
    set_pose,
    multiply,
    invert,
    matrix_from_quat,
    approximate_as_prism,
    unit_pose,
    Pose,
    Euler,
    set_quat,
    multiply,
    plan_joint_motion,
    remove_body,
    get_distance,
    get_aabb,
    draw_aabb,
    buffer_aabb,
    get_aabb_center,
    get_aabb_extent,
    get_self_link_pairs,
    pairwise_link_collision,
    plan_direct_joint_motion,
    get_aabb_vertices,
    get_extend_fn,
    get_collision_fn,
    get_sample_fn,
    get_distance_fn,
    pairwise_collision,
    set_all_color,
    create_sphere,
    draw_mesh,
    add_line,
)
from pybullet_tools.ikfast.ur5e_robotiq.ik import (
    ur5e_robotiq_inverse_kinematics,
    ikfast_compute_inverse_kinematics,
    sample_tool_ik,
)
from pybullet_tools.ikfast.utils import select_solution
from ikfast_ur5e_robotiq import get_ik
from pybullet_tools.motion import compute_motion
# from pybullet_tools.ompl_plugin import compute_motion

import pyb_utils
from pyb_utils import (
    GhostObject,
    CollisionDetector,
)

connect(use_gui=True, shadows=False, color=[1, 1, 1, 0.8], width=1920, height=1440)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Start video recording
output_dir = "column_assembly_rigid_elastic_0.2_steps_screenshots"
os.makedirs(output_dir, exist_ok=True)
video_filename = os.path.join(output_dir, "column_assembly_rigid_elastic_0.2_viz.mp4")
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
print(f"Started recording video to {video_filename}")

# Create simulation environment
plane_id = p.loadURDF("plane_transparent.urdf", [0, 0, -0.0])
set_all_color(plane_id, color=[1, 1, 1, 0.001])

supports_geometry = get_cylinder_geometry(0.005, 0.188)  # Updated height to match functional code
supports_ori = p.getQuaternionFromEuler([0, 0, 0])
supports_1_pose = ([0.31999999880790714, 0.20000000000000004, 0.094], supports_ori)
supports_2_pose = ([0.4400000005960465, 0.13071796894073487, 0.094], supports_ori)
supports_3_pose = ([0.4400000005960465, 0.2692820310592652, 0.094], supports_ori)

supports_1_collision_id, supports_1_visual_id = create_shape(
    supports_geometry, supports_1_pose, color=[1.0, 1.0, 1.0, 0.6]
)
supports_1_id = create_body(supports_1_collision_id, supports_1_visual_id)

supports_2_collision_id, supports_2_visual_id = create_shape(
    supports_geometry, supports_2_pose, color=[1.0, 1.0, 1.0, 0.6]
)
supports_2_id = create_body(supports_2_collision_id, supports_2_visual_id)

supports_3_collision_id, supports_3_visual_id = create_shape(
    supports_geometry, supports_3_pose, color=[1.0, 1.0, 1.0, 0.6]
)
supports_3_id = create_body(supports_3_collision_id, supports_3_visual_id)

# Load robot
robot_id = load_pybullet(UR5E_ROBOTIQ_URDF, fixed_base=True)
arm_joints = get_arm_joints(robot_id)
set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
set_gripper_open(robot_id)
setup_mimic_joints(robot_id)
set_gripper(robot_id, 0.72)

# set_all_color(robot_id, [178/255, 187/255, 190/255, 1.0])

gripper_center_pose = get_link_pose(robot_id, 15)

# Set camera
set_camera(yaw=150, pitch=-15, distance=1.0, target_position=[0.15, 0.15, 0.55])

# Load structure data
with open("column_infos.json", "r", encoding="utf-8") as file:
    bar_structures = json.load(file)

# Create all bars as target
bar_objects = {}
for bar_id, bar_info in bar_structures.items():
    base_geometry = get_box_geometry(
        bar_info["geometry"][0], bar_info["geometry"][1], bar_info["geometry"][2]
    )
    bar_initial_ori = p.getQuaternionFromEuler([0, 0, 0])
    bar_initial_pos = (0.0, 0.0, 0.0)
    bar_initial_pose = (bar_initial_pos, bar_initial_ori)
    base_collision_id, base_visual_id = create_shape(
        base_geometry, bar_initial_pose, color=[46/255, 89/255, 167/255, 0.1]
    )
    bar_body_id = create_body(base_collision_id, base_visual_id)
    bar_objects[bar_id] = bar_body_id
    bar_target_pos = (
        bar_info["position"][0],
        bar_info["position"][1],
        bar_info["position"][2],
    )
    bar_target_ori = R.from_matrix(bar_info["rotation_matrix"]).as_quat()
    bar_target_pose = (bar_target_pos, bar_target_ori)
    set_pose(bar_body_id, bar_target_pose)
    p.setCollisionFilterGroupMask(
        bar_body_id, -1, collisionFilterGroup=0, collisionFilterMask=0
    )

# Load paths from JSON
paths_file = "column_assembly_paths_rigid_elastic_0.2.json"
try:
    with open(paths_file, "r", encoding="utf-8") as f:
        all_paths_data = json.load(f)
    print(f"Loaded paths from {paths_file}")
except Exception as e:
    print(f"Error loading paths from {paths_file}: {e}")
    exit(1)
    
# Load deformation data
deformation_file = "column_deformation_data_rigid_elastic_0.2.json"
try:
    with open(deformation_file, "r", encoding="utf-8") as f:
        deformation_data = json.load(f)
    print(f"Loaded deformation data from {deformation_file}")
except Exception as e:
    print(f"Error loading deformation data from {deformation_file}: {e}")
    exit(1)

# Assembly simulation
obstacles = [plane_id, supports_1_id, supports_2_id, supports_3_id]
assembled_bars = []
joint_bodies = []
joints_pos = set()
# Dictionary to map bar_id to body ID
bar_id_to_body = {}


# Function to create ribbon mesh (retained from original)
def create_ribbon_mesh(start_points, end_points, center_points):
    vertices = []
    indices = []
    for i in range(len(start_points)):
        vertices.append(start_points[i])
        vertices.append(end_points[i])
        if len(center_points) > i:
            vertices.append(center_points[i])
    for i in range(len(start_points)-1):
        idx = i * 3
        indices.extend([idx, idx+3, idx+1])
        indices.extend([idx+1, idx+3, idx])
        indices.extend([idx+1, idx+3, idx+4])
        indices.extend([idx+4, idx+3, idx+1])
    return vertices, indices

# Replay assembly using loaded paths
for idx, path_data in enumerate(all_paths_data):
    
    ribbon_body = None
    
    bar_id = path_data["bar_id"]
    seq_index = path_data["sequence_index"]
    approach_path = path_data["approach_path"]
    grasp_path = path_data["grasp_path"]
    target_pose = (
        tuple(path_data["target_pose"]["position"]),
        tuple(path_data["target_pose"]["orientation"])
    )
    rgba_color = path_data["color"]

    # Get bar information
    obj_bar_info = bar_structures.get(bar_id)
    if obj_bar_info is None:
        print(f"Warning: Bar ID {bar_id} not found in bar_structures. Skipping.")
        continue
    
    # # Determine if paths are empty and set color
    # is_path_empty = not approach_path and not grasp_path
    # bar_color = [1.0, 0.0, 0.0, 1.0] if is_path_empty else rgba_color
    # Determine bar color based on path availability
    has_approach = bool(approach_path)
    has_grasp = bool(grasp_path)
    if not has_approach and not has_grasp:
        bar_color = [226/255, 0/255, 26/255, 1.0]  # Red for no approach and no grasp
    elif has_approach and not has_grasp:
        bar_color = [250/250, 187/250, 0/250, 1.0]  # Yellow for approach but no grasp
    else:
        bar_color = rgba_color  # Use original color if both paths exist

    # Create bar geometry
    obj_bar_base_geometry = get_box_geometry(
        obj_bar_info["geometry"][0],
        obj_bar_info["geometry"][1],
        obj_bar_info["geometry"][2] - 0.02,
    )
    obj_bar_initial_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_bar_initial_pos = (0.0, 0.0, 0.0)
    obj_bar_initial_pose = (obj_bar_initial_pos, obj_bar_initial_ori)
    obj_bar_base_collision_id, obj_bar_base_visual_id = create_shape(
        obj_bar_base_geometry,
        obj_bar_initial_pose,
        color=bar_color,
    )
    obj_bar_body_id = create_body(
        obj_bar_base_collision_id, obj_bar_base_visual_id, mass=0.001
    )
    
    # Store bar_id to body_id mapping
    bar_id_to_body[bar_id] = obj_bar_body_id
    
    if not has_approach and not has_grasp:
        # Directly place bar at target pose
        obj_bar_grasp_pose = multiply(gripper_center_pose, Pose(euler=[-np.pi / 2, 0, 0]))
        set_pose(obj_bar_body_id, obj_bar_grasp_pose)
        
        # Update target bar color
        bar_target_id = bar_objects.get(bar_id)
        set_all_color(bar_target_id, color=[27/255, 167/255, 132/255, 0.05])
        
        # set_pose(obj_bar_body_id, target_pose)
        print(f"Placed bar {bar_id} directly at target pose with red color due to empty paths")
        
        obstacles.append(obj_bar_body_id)
        assembled_bars.append(obj_bar_body_id)
    else:
        # Set initial grasp pose and attach bar
        obj_bar_grasp_pose = multiply(gripper_center_pose, Pose(euler=[-np.pi / 2, 0, 0]))
        set_pose(obj_bar_body_id, obj_bar_grasp_pose)
        attachments = create_attachment(robot_id, 15, obj_bar_body_id)
        attachments.assign()
        time.sleep(0.5)

        # Update target bar color
        bar_target_id = bar_objects.get(bar_id)
        set_all_color(bar_target_id, color=[27/255, 167/255, 132/255, 0.05])

        # Execute approach path with visualization
        if approach_path:
            bar_length = obj_bar_info["geometry"][2]
            start_points = []
            end_points = []
            center_points = []
            center_line_ids = []
            for conf in approach_path:
                set_joint_positions(robot_id, arm_joints, conf)
                attachments.assign()
                gripper_pos, gripper_quat = get_link_pose(robot_id, 15)
                rotation_matrix = matrix_from_quat(gripper_quat)
                y_axis = rotation_matrix[:, 1]
                line_start = np.array(gripper_pos) - (bar_length / 2) * y_axis
                line_end = line_start + bar_length * y_axis
                center_point = (line_start + line_end) / 2
                start_points.append(line_start)
                end_points.append(line_end)
                center_points.append(center_point)
                if len(center_points) > 1:
                    line_id = p.addUserDebugLine(
                        lineFromXYZ=center_points[-2],
                        lineToXYZ=center_points[-1],
                        lineColorRGB=[rgba_color[0], rgba_color[1], rgba_color[2]],
                        lineWidth=2,
                        lifeTime=0
                    )
                    center_line_ids.append(line_id)
                time.sleep(1.0 / 240)
            vertices, indices = create_ribbon_mesh(start_points, end_points, center_points)
            col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
            vis_shape = p.createVisualShape(
                p.GEOM_MESH, vertices=vertices, indices=indices,
                rgbaColor=[rgba_color[0], rgba_color[1], rgba_color[2], 0.4]
            )
            ribbon_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[0, 0, 0]
            )
        # wait_if_gui()
        time.sleep(0.25)

        # Execute grasp path
        if grasp_path:
            for conf in grasp_path:
                set_joint_positions(robot_id, arm_joints, conf)
                attachments.assign()
                time.sleep(1.0 / 240)
            # wait_if_gui()
            time.sleep(0.5)

    # Post-grasp operations
    # p.changeDynamics(obj_bar_body_id, -1, mass=0.0)
    set_gripper_open(robot_id)
    set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)

    # Set bar to target pose
    set_pose(obj_bar_body_id, target_pose)
    
    # Apply deformation data for all assembled bars
    deformation_entry = next((entry for entry in deformation_data if entry["sequence_index"] == seq_index), None)
    
    # Clear previous joint spheres
    for joint_body in joint_bodies:
        remove_body(joint_body)
    joint_bodies.clear()
    joints_pos.clear()  # Clear joint positions to avoid duplicates

    if deformation_entry:
        for bar_deform in deformation_entry["bar_deformations"]:
            deform_bar_id = bar_deform["bar_id"]
            if deform_bar_id in bar_id_to_body:
                deformed_pos = bar_deform["deformed_position"]
                deformed_quat = bar_deform["deformed_quaternion"]
                deformed_pose = (deformed_pos, deformed_quat)
                set_pose(bar_id_to_body[deform_bar_id], deformed_pose)
                print(f"Applied deformation to bar {deform_bar_id}: position={deformed_pos}, quaternion={deformed_quat}")
                
        # Create joint spheres using deformed positions
        joint_radius = 0.012
        def is_position_close(new_pos, existing_positions, threshold):
            for existing_pos in existing_positions:
                distance = np.linalg.norm(np.array(new_pos) - np.array(existing_pos))
                if distance < threshold:
                    return True
            return False

        for joint_deform in deformation_entry["joint_deformations"]:
            node_id = joint_deform["node_id"]
            deformed_pos = tuple(joint_deform["deformed_position"])
            if not is_position_close(deformed_pos, joints_pos, 1e-3):
                joint_id = create_sphere(
                    joint_radius, color=[249/255, 241/255, 219/255, 1.0]
                )
                set_pose(joint_id, (deformed_pos, [0, 0, 0, 1]))  # Default orientation
                joints_pos.add(deformed_pos)
                joint_bodies.append(joint_id)
                print(f"Created joint {node_id} at deformed position: {deformed_pos}")
    else:
        print(f"Warning: No deformation data found for sequence index {seq_index}")

        # Create joint spheres
        bar_length = obj_bar_info["geometry"][2]
        joint_radius = 0.012
        joint1_offset = bar_length / 2
        joint2_offset = -joint1_offset
        joint1_pose = multiply(target_pose, Pose(point=[0, 0, joint1_offset]))
        joint1_pos = tuple(joint1_pose[0])
        joint2_pose = multiply(target_pose, Pose(point=[0, 0, joint2_offset]))
        joint2_pos = tuple(joint2_pose[0])

        def is_position_close(new_pos, existing_positions, threshold):
            for existing_pos in existing_positions:
                distance = np.linalg.norm(np.array(new_pos) - np.array(existing_pos))
                if distance < threshold:
                    return True
            return False

        if not is_position_close(joint1_pos, joints_pos, 1e-3):
            joint1_id = create_sphere(
                joint_radius, color=[249/255, 241/255, 219/255, 1.0]
            )
            set_pose(joint1_id, joint1_pose)
            joints_pos.add(joint1_pos)
            joint_bodies.append(joint1_id)
        if not is_position_close(joint2_pos, joints_pos, 1e-3):
            joint2_id = create_sphere(
                joint_radius, color=[249/255, 241/255, 219/255, 1.0]
            )
            set_pose(joint2_id, joint2_pose)
            joints_pos.add(joint2_pos)
            joint_bodies.append(joint2_id)

    set_gripper(robot_id, 0.72)
    obstacles.append(obj_bar_body_id)
    assembled_bars.append(obj_bar_body_id)
    
    # obj_bar_body_illustrated_id = create_body(
    #     obj_bar_base_collision_id, obj_bar_base_visual_id, mass=0.001
    # )

    # # Set initial grasp pose and attach bar
    # obj_bar_grasp_pose = multiply(gripper_center_pose, Pose(euler=[-np.pi / 2, 0, 0]))
    # set_pose(obj_bar_body_illustrated_id, obj_bar_grasp_pose)
        
    # # wait_if_gui()
    
    # cam_info = p.getDebugVisualizerCamera()
    # width, height = cam_info[0] * 2, cam_info[1] * 2

    # _, _, rgb_img, _, _ = p.getCameraImage(
    #     width, height,
    #     viewMatrix=cam_info[2],
    #     projectionMatrix=cam_info[3],
    #     shadow=False,
    #     renderer=p.ER_BULLET_HARDWARE_OPENGL
    # )
    
    # rgb_array = np.array(rgb_img)
    # rgb_array = rgb_array[:, :, :3]
    # image = Image.fromarray(rgb_array)
    
    # seq_num = f"{idx:02d}"
    # filename = f"{seq_num}_{int(bar_id):02d}.png"
    # filepath = os.path.join(output_dir, filename)
    
    # image.save(filepath)
    # print(f"Saved: {filepath}")
    
    # remove_body(obj_bar_body_illustrated_id)
    if ribbon_body is not None:
        remove_body(ribbon_body)
    p.removeAllUserDebugItems()
    
    # wait_if_gui()
    time.sleep(0.25)

# Stop video recording
p.stopStateLogging(log_id)
print(f"Video recording saved to {video_filename}")

set_gripper_open(robot_id)
wait_if_gui()
disconnect()