#!/usr/bin/env python
from __future__ import print_function

import pybullet as p
import pybullet_planning as pp
import pybullet_data

import numpy as np
import time
import json
import csv
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from pybullet_planning import (
    connect,
    disconnect,
    load_pybullet,
    wait_if_gui,
    set_joint_positions,
    dump_world,
    convex_hull,
    create_mesh,
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
    get_side_grasps_multi_link,
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
    LinkInfo,
    create_multi_body,
    get_joints,
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
from pybullet_tools.motion import compute_motion, compute_motion_with_deformation
# from pybullet_tools.ompl_plugin import compute_motion

import pyb_utils
from pyb_utils import (
    GhostObject,
    CollisionDetector,
)

# from FEA_plugin import fea_analysis
from FEA_single_bar_plugin import single_bar_analysis
from FEA_plugin_decompose import fea_analysis

def create_segmented_bar(geometry, num_segments, base_pose=([0, 0, 0], [0, 0, 0, 1]), color=[1.0, 0.0, 0.0, 1.0]):
    """
    Create a multi-body bar with odd number of segments, centered at base link, using spherical joints.
    Segments are split into negative and positive chains for clarity.
    
    Args:
        geometry (list): [width, depth, total_length] of the bar
        num_segments (int): Odd number of segments
        base_pose (tuple): ((x, y, z), (qx, qy, qz, qw)) for base link
        color (list): RGBA color for segments
    
    Returns:
        int: Body ID of the created multi-body
    """
    assert num_segments % 2 == 1, "Number of segments must be odd"
    
    width, depth, total_length = geometry
    segment_length = total_length / num_segments
    segment_geometry = get_box_geometry(width, depth, segment_length)
    
    # Create base link
    base_collision_id, base_visual_id = create_shape(segment_geometry, pose=base_pose, color=color)
    base_link = LinkInfo(collision_id=base_collision_id, visual_id=base_visual_id)
    
    # Create links for segments
    links = []
    half_segments = num_segments // 2
    
    # Negative chain (links extending downward)
    for i in range(half_segments):
        z_offset = -segment_length / 2
        link_pose = ([0, 0, z_offset], [0, 0, 0, 1])
        link_collision_id, link_visual_id = create_shape(segment_geometry, pose=link_pose, color=color)
        
        parent = 0 if i == 0 else i  # Base for first link, previous link otherwise
        joint_point = [0, 0, -segment_length / 2 if i == 0 else -segment_length]
        
        link_info = LinkInfo(
            collision_id=link_collision_id,
            visual_id=link_visual_id,
            point=joint_point,
            parent=parent,
            joint_type=p.JOINT_SPHERICAL
        )
        links.append(link_info)
    
    # Positive chain (links extending upward)
    for i in range(half_segments):
        z_offset = segment_length / 2
        link_pose = ([0, 0, z_offset], [0, 0, 0, 1])
        link_collision_id, link_visual_id = create_shape(segment_geometry, pose=link_pose, color=color)
        
        parent = 0 if i == 0 else half_segments + i  # Base for first link, previous link otherwise
        joint_point = [0, 0, segment_length / 2 if i == 0 else segment_length]
        
        link_info = LinkInfo(
            collision_id=link_collision_id,
            visual_id=link_visual_id,
            point=joint_point,
            parent=parent,
            joint_type=p.JOINT_SPHERICAL
        )
        links.append(link_info)
    
    return create_multi_body(base_link, links)

def get_segmented_bar_endpoints(body_id, bar_data):
    """
    Calculate the coordinates of the two endpoints of a segmented bar in the world frame using bar_data.
    
    Args:
        body_id (int): The PyBullet body ID of the segmented bar.
        bar_data (dict): Dictionary containing geometry (width, depth, total_length).
    
    Returns:
        dict: Dictionary containing the coordinates of the two endpoints.
              Format: {"nodes": [{"id": 1, "x": x1, "y": y1, "z": z1},
                                {"id": 2, "x": x2, "y": y2, "z": z2}]}
    """
    # Extract total length from bar_data geometry
    total_length = bar_data["geometry"][2]
    
    # Base link (link 0) pose
    base_pose = get_link_pose(body_id, -1)
    
    # Calculate endpoints relative to base link
    # Negative endpoint (bottom end, along local -z axis)
    neg_endpoint = multiply(base_pose, Pose(point=[0, 0, -total_length / 2]))[0]
    
    # Positive endpoint (top end, along local +z axis)
    pos_endpoint = multiply(base_pose, Pose(point=[0, 0, total_length / 2]))[0]
    
    return {
        "nodes": [
            {"id": 1, "x": neg_endpoint[0], "y": neg_endpoint[1], "z": neg_endpoint[2]},
            {"id": 2, "x": pos_endpoint[0], "y": pos_endpoint[1], "z": pos_endpoint[2]}
        ]
    }

def set_spherical_joint_pose(body, joint, quaternion):
    """
    Set the pose of a spherical joint using a quaternion.
    
    Args:
        body: The body ID.
        joint: The joint index.
        quaternion: List [qx, qy, qz, qw] representing the target orientation.
    """
    # if get_joint_type(body, joint) != p.JOINT_SPHERICAL:
    #     raise ValueError(f"Joint {joint} is not a spherical joint")
    p.resetJointStateMultiDof(
        bodyUniqueId=body,
        jointIndex=joint,
        targetValue=quaternion,
    )

def set_spherical_joint_poses(body, joints, quaternions):
    """
    Set the poses of multiple spherical joints.
    
    Args:
        body: The body ID.
        joints: List of joint indices.
        quaternions: List of quaternions [[qx, qy, qz, qw], ...].
    """
    assert len(joints) == len(quaternions), f"joints {joints} | quaternions {quaternions}"
    for joint, quat in zip(joints, quaternions):
        set_spherical_joint_pose(body, joint, quat)
        
def reset_spherical_joint_poses(body, joints):
    """
    Reset all specified spherical joints to their default orientation (identity quaternion).
    
    Args:
        body: The body ID.
        joints: List of joint indices to reset.
    """
    identity_quat = [0, 0, 0, 1]  # Identity quaternion (no rotation)
    for joint in joints:
        p.resetJointStateMultiDof(
            bodyUniqueId=body,
            jointIndex=joint,
            targetValue=identity_quat,
        )
        

connect(use_gui=False, shadows=False, color=[1, 1, 1])
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create simulation environment
plane_id = p.loadURDF("plane_transparent.urdf", [0, 0, -0.0])
set_all_color(plane_id, color=[1, 1, 1, 0.001])

# plane_geometry = get_box_geometry(100, 100, 0.001)
# plane_ori = p.getQuaternionFromEuler([0, 0, 0])
# plane_pose = ([0, 0, 0], plane_ori)
# plane_collision_id, plane_visual_id = create_shape(
#     plane_geometry, plane_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# plane_id = create_body(plane_collision_id, plane_visual_id)


# Create supports
supports_geometry = get_cylinder_geometry(0.005, 0.188)
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

# base_geometry = get_box_geometry(0.15, 0.15, 0.4)
# base_ori = p.getQuaternionFromEuler([0, 0, 0])
# base_pose = ([0.0, 0.0, -0.2], base_ori)
# base_collision_id, base_visual_id = create_shape(
#     base_geometry, base_pose, color=[1.0, 1.0, 1.0, 0.8]
# )
# base_id = create_body(base_collision_id, base_collision_id)

# robot_id = p.loadURDF(UR5E_ROBOTIQ_URDF, basePosition=[0, -0.5, 0.4], useFixedBase=True)
robot_id = load_pybullet(UR5E_ROBOTIQ_URDF, fixed_base=True)
arm_joints = get_arm_joints(robot_id)
set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
set_gripper_open(robot_id)
setup_mimic_joints(robot_id)
set_gripper(robot_id, 0.72)

gripper_center_pose = get_link_pose(robot_id, 15)

# set_default_camera(pitch=-20, distance=2)
set_camera(yaw=150, pitch=-15, distance=1.0, target_position=[0.15, 0.15, 0.55])

# Load structure data
with open("column_infos.json", "r", encoding="utf-8") as file:
    bar_structures = json.load(file)
    
# Load complete structure for FEA
with open("column_info_structure.json", "r", encoding="utf-8") as file:
    full_structure = json.load(file)

# Create all bars as target
bar_objects = {}
for bar_id, bar_info in bar_structures.items():

    # Create geometry for each bar
    base_geometry = get_box_geometry(
        bar_info["geometry"][0], bar_info["geometry"][1], bar_info["geometry"][2]
    )

    # Initial orientation and position
    bar_initial_ori = p.getQuaternionFromEuler([0, 0, 0])
    bar_initial_pos = (0.0, 0.0, 0.0)
    bar_initial_pose = (bar_initial_pos, bar_initial_ori)

    # Create shape with green color
    base_collision_id, base_visual_id = create_shape(
        base_geometry, bar_initial_pose, color=[46 / 255, 89 / 255, 167 / 255, 0.2]
    )

    # Create body and store it
    bar_body_id = create_body(base_collision_id, base_visual_id)
    bar_objects[bar_id] = bar_body_id

    # Set target position and orientation
    bar_target_pos = (
        bar_info["position"][0],
        bar_info["position"][1],
        bar_info["position"][2],
    )
    bar_target_ori = R.from_matrix(bar_info["rotation_matrix"]).as_quat()
    bar_target_pose = (bar_target_pos, bar_target_ori)

    # Apply the target pose
    set_pose(bar_body_id, bar_target_pose)

    p.setCollisionFilterGroupMask(
        bar_body_id, -1, collisionFilterGroup=0, collisionFilterMask=0
    )
    

cmap = plt.get_cmap("PuBu")
num_bars = len(bar_structures.items())
    
# Assembly simulation
obstacles = [plane_id, supports_1_id, supports_2_id, supports_3_id]
assembled_bars = []
joint_bodies = []
bars_convex_mesh = None

assembly_sequence = [15, 9, 14, 18, 1, 10, 13, 22, 12, 0, 19, 20, 11, 21, 4, 5, 6, 7, 8, 17, 16, 3, 2, 23]

all_paths_data = []
approach_stats = []
deformation_data = []

obj_bar_objects = {}
# for i, (obj_bar_id, obj_bar_info) in enumerate(bar_structures.items()):
for seq_index, obj_bar_id in enumerate(assembly_sequence):

    # Get bar information
    obj_bar_info = bar_structures.get(str(obj_bar_id))  # Ensure bar_id is string
    if obj_bar_info is None:
        print(f"Warning: Bar ID {obj_bar_id} not found in bar_structures. Skipping.")
        continue
    
    color_value = seq_index / max(1, num_bars - 1)
    rgba_color = cmap(color_value)
    
    obj_bar_body_geometry = [
        obj_bar_info["geometry"][0],
        obj_bar_info["geometry"][1],
        obj_bar_info["geometry"][2] - 0.02,
    ]
    
    obj_bar_body_id = create_segmented_bar(obj_bar_body_geometry, 7, color=rgba_color)
    obj_bar_objects[obj_bar_id] = obj_bar_body_id

    obj_bar_grasp_pose = multiply(gripper_center_pose, Pose(euler=[-np.pi / 2, 0, 0]))
    set_pose(obj_bar_body_id, obj_bar_grasp_pose)
    attachements = create_attachment(robot_id, 15, obj_bar_body_id)
    attachements.assign()
    time.sleep(0.5)

    # Set target position and orientation
    bar_target_pos = (
        obj_bar_info["position"][0],
        obj_bar_info["position"][1],
        obj_bar_info["position"][2],
    )
    bar_target_ori = R.from_matrix(obj_bar_info["rotation_matrix"]).as_quat()
    bar_target_pose = (bar_target_pos, bar_target_ori)
    # draw_pose(bar_target_pose)
    bar_target_id = bar_objects.get(str(obj_bar_id))
    set_all_color(bar_target_id, color=[27 / 255, 167 / 255, 132 / 255, 0.1])

    bar_length = obj_bar_info["geometry"][2]
    joint_radius = 0.012
    joint1_offset = bar_length / 2
    joint2_offset = -joint1_offset

    joint1_pose = multiply(bar_target_pose, Pose(point=[0, 0, joint1_offset]))
    joint1_pos = tuple(joint1_pose[0])
    joint2_pose = multiply(bar_target_pose, Pose(point=[0, 0, joint2_offset]))
    joint2_pos = tuple(joint2_pose[0])

    # Define initial tool pose
    TOOL_POSE = Pose()
    grasp_confs = []
    approach_confs = []

    # Generate side grasps for the object
    grasps = get_side_grasps_multi_link(
        obj_bar_body_id,
        under=True,
        max_width=0.085,
        grasp_length=0.0025,
        body_pose=bar_target_pose,
        top_offset=0.0,
    )
    grasp_conf_attempts = 0

    # Iterate through each grasp to compute configurations
    for grasp in grasps:
        approach = multiply(
            grasp, Pose(point=[0, 0, -0.01])
        )  # Define approach pose 20mm above grasp
        print(f"Attempts for grasp conf: {grasp_conf_attempts + 1}")

        # Compute grasp configuration using inverse kinematics
        grasp_conf = ur5e_robotiq_inverse_kinematics(
            robot_id, 15, grasp, use_pybullet=False, obstacles=obstacles
        )
        if grasp_conf is None:
            print("IKfast failed for grasp, use pybullet instead.")
            grasp_conf = ur5e_robotiq_inverse_kinematics(
                robot_id, 15, grasp, use_pybullet=True, obstacles=obstacles
            )

        if grasp_conf is not None:
            # Compute approach configuration using inverse kinematics
            approach_conf = ur5e_robotiq_inverse_kinematics(
                robot_id, 15, approach, use_pybullet=False, obstacles=obstacles
            )
            if approach_conf is None:
                print("IKfast failed for approach, use pybullet instead.")
                approach_conf = ur5e_robotiq_inverse_kinematics(
                    robot_id, 15, approach, use_pybullet=True, obstacles=obstacles
                )

            if approach_conf is not None:
                # Check for self-collision in approach configuration
                set_joint_positions(robot_id, arm_joints, approach_conf)
                link_pairs = get_self_link_pairs(robot_id, arm_joints)
                approach_self_collision = False
                for link1, link2 in link_pairs:
                    if pairwise_link_collision(robot_id, link1, robot_id, link2):
                        approach_self_collision = True
                        print(
                            f"Approach conf {grasp_conf_attempts + 1} discarded due to self-collision between links {link1} and {link2}"
                        )
                        break

                # Check for self-collision in grasp configuration
                set_joint_positions(robot_id, arm_joints, grasp_conf)
                link_pairs = get_self_link_pairs(robot_id, arm_joints)
                grasp_self_collision = False
                for link1, link2 in link_pairs:
                    if pairwise_link_collision(robot_id, link1, robot_id, link2):
                        grasp_self_collision = True
                        print(
                            f"Grasp conf {grasp_conf_attempts + 1} discarded due to self-collision between links {link1} and {link2}"
                        )
                        break

                # Store valid configurations if no self-collisions occur
                if not grasp_self_collision and not approach_self_collision:
                    approach_confs.append(approach_conf)
                    grasp_confs.append(grasp_conf)

        grasp_conf_attempts += 1

    # Filter valid configuration pairs
    valid_pairs = [
        (g, a)
        for g, a in zip(grasp_confs, approach_confs)
        if g is not None and a is not None
    ]
    grasp_confs, approach_confs = zip(*valid_pairs) if valid_pairs else ([], [])
    grasp_confs = list(grasp_confs)
    approach_confs = list(approach_confs)

    # Select final configurations
    set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
    selected_approach_conf = select_solution(
        robot_id, arm_joints, approach_confs, nearby_conf=None
    )
    selected_grasp_conf = select_solution(
        robot_id, arm_joints, grasp_confs, nearby_conf=selected_approach_conf
    )

    # if selected_approach_conf is not None:
    #     set_joint_positions(robot_id, arm_joints, selected_grasp_conf)
    #     set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
    # else:
    # if selected_approach_conf is None:
    #     print("No feasible conf found.")
    #     wait_if_gui()
    if selected_approach_conf is None or selected_grasp_conf is None:
        print(f"No feasible grasp configuration found for bar {obj_bar_id}. Placing directly at target pose.")
        # wait_if_gui()
        # Set bar to target pose
        set_pose(obj_bar_body_id, bar_target_pose)
        p.changeDynamics(obj_bar_body_id, -1, mass=0.0)  # Make object massless
        set_gripper_open(robot_id)  # Open the gripper
        set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)  # Return to vertical configuration
        set_gripper(robot_id, 0.72)  # Reset gripper
        obstacles.append(obj_bar_body_id)
        assembled_bars.append(obj_bar_body_id)

        # Record zero planning times and costs
        approach_stats.append({
            "bar_id": str(obj_bar_id),
            "sequence_index": seq_index,
            "planning_times": [float("inf")] * 10,  # Zero for all attempts
            "planning_costs": [float("inf")] * 10,   # Zero for all attempts
            "grasp_planning_time": float("inf")
        })

        # Store empty path data
        path_data = {
            "bar_id": str(obj_bar_id),
            "sequence_index": seq_index,
            "approach_path": [],  # Empty list for no path
            "approach_deformations": [],
            "grasp_path": [],    # Empty list for no path
            "target_pose": {
                "position": list(bar_target_pose[0]),
                "orientation": list(bar_target_pose[1])
            },
            "color": list(rgba_color)
        }
        all_paths_data.append(path_data)
    
    else:
        # set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)

        # Define path planning parameters
        resolutions = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        # weights = np.reciprocal(resolutions)
        weights = np.array([100.0, 50.0, 20.0, 1.0, 1.0, 1.0])
        dynamic_resolution_ratio = 0.5
        fine_resolutions = dynamic_resolution_ratio * resolutions

        # Function to calculate path cost based on weighted joint differences
        def calculate_path_cost(paths, weights=weights):
            if not paths or len(paths) <= 5:  # Invalid path if too short
                return float("inf")

            total_cost = 0.0
            num_joints = len(paths[0])
            if weights is None:
                weights = [1.0] * num_joints

            for i in range(len(paths) - 1):
                config1 = paths[i]
                config2 = paths[i + 1]
                step_cost = sum(
                    weights[j] * abs(config2[j] - config1[j]) for j in range(num_joints)
                )
                total_cost += step_cost

            return total_cost
        
        # # Calculate path length in Cartesian space
        # def calculate_path_length(robot_id, arm_joints, paths):
        #     if not paths or len(paths) <= 5:
        #         return float('inf')
        #     total_length = 0.0
        #     ee_link = link_from_name(robot_id, 'gripper_center_link')  # End effector link name for UR5e
            
        #     for i in range(len(paths) - 1):
        #         # Set joint positions for config1 and get end effector position
        #         set_joint_positions(robot_id, arm_joints, paths[i])
        #         pos1, _ = get_link_pose(robot_id, ee_link)
        #         pos1 = np.array(pos1)
                
        #         # Set joint positions for config2 and get end effector position
        #         set_joint_positions(robot_id, arm_joints, paths[i + 1])
        #         pos2, _ = get_link_pose(robot_id, ee_link)
        #         pos2 = np.array(pos2)
                
        #         # Calculate Euclidean distance between consecutive positions
        #         distance = np.linalg.norm(pos2 - pos1)
        #         total_length += distance
            
        #     return total_length

        # Collision detection between robot and hull
        def convex_mesh_detection(q, convex_mesh):
            set_joint_positions(robot_id, arm_joints, q)
            attachements.assign()
            collision = (convex_mesh is not None) and (
                (pairwise_collision(robot_id, convex_mesh))
                or (pairwise_collision(attachements.child, convex_mesh))
            )

            return q, collision

        # Plan the approach path
        print("Approach path planning...")
        shortest_approach_path = None
        bar_deformation_data = None
        min_approach_cost = float("inf")
        planning_times = []
        planning_costs = []

        for i in range(10):
            print(f"Attempts for approach paths: {i + 1}")
            print(f"Obstacles: {obstacles}")
            start_time = time.time()
            # approach_paths = plan_joint_motion(robot_id, arm_joints, selected_approach_conf, start_conf=VERTICAL_CONF,
            #                                 obstacles=obstacles, attachments=[attachements],
            #                                 weights=weights, resolution=resolutions, algorithm='rrt_connect', num_samples=400,)
            approach_paths, bar_deformation_results = compute_motion_with_deformation(
                robot_id,
                arm_joints,
                start_conf=VERTICAL_CONF,
                end_conf=selected_approach_conf,
                obstacles=obstacles,
                attachments=[attachements],
                attachments_data=obj_bar_info,
                assembled_elements=assembled_bars,
                self_collisions=True,
                algorithm="rrt_connect",
                # resolutions=resolutions,
                # weights=weights,
            )

            if approach_paths is not None:
                planning_time = time.time() - start_time
                planning_times.append(planning_time)
                
                approach_cost = calculate_path_cost(approach_paths, weights)
                planning_costs.append(approach_cost)
                # approach_cost = calculate_path_length(robot_id, arm_joints, approach_paths)
                print(
                    f"Attempts for approach paths: {i + 1} - Path found with cost: {approach_cost}"
                )
                if approach_cost < min_approach_cost:
                    min_approach_cost = approach_cost
                    shortest_approach_path = approach_paths
                    bar_deformation_data = bar_deformation_results
            else:
                print("No approach paths found.")
                planning_times.append(float("inf"))
                planning_costs.append(float("inf"))

        if shortest_approach_path is not None:
            print(f"Shortest approach path found with cost: {min_approach_cost}")
            print(f"Shortest approach path: {shortest_approach_path}")
            print(f"Corresponding deformation data: {bar_deformation_data}")
        else:
            print("No valid approach paths found after 10 attempts.")

        # Plan the grasp path
        print("Grasp path planning...")

        for i in range(1):
            grasp_planning_start_time = time.time()
            print(f"Attempts for grasp paths: {i + 1}")
            print(f"Obstacles: {obstacles}")
            grasp_paths = plan_direct_joint_motion(
                robot_id,
                arm_joints,
                selected_grasp_conf,
                start_conf=selected_approach_conf,
                obstacles=obstacles,
                attachments=[attachements],
            )
            if grasp_paths is not None:
                grasp_planning_time = time.time() - grasp_planning_start_time
                print(f"Attempts for find paths: {i + 1}")
                break
            else:
                grasp_planning_time = float("inf")
                print("No valid grasp paths found after 1 attempts.")

        # if grasp_paths is None:
        #     print("No valid grasp paths found after 10 attempts.")
                
        approach_stats.append({
            "bar_id": str(obj_bar_id),
            "sequence_index": seq_index,
            "planning_times": planning_times,
            "planning_costs": planning_costs,
            "grasp_planning_time": grasp_planning_time
        })

        set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
        attachements.assign()
        
        path_data = {
            "bar_id": str(obj_bar_id),
            "sequence_index": seq_index,
            "approach_path": [list(conf) for conf in shortest_approach_path] if shortest_approach_path else [],
            # "approach_deformations": [list(bar_conf) for bar_conf in bar_deformation_data] if shortest_approach_path else [],
            "approach_deformations": [
                {
                    "joint_quaternions": [quat.tolist() if isinstance(quat, np.ndarray) else list(quat) for quat in bar_conf["joint_quaternions"]],
                    "deformed_coords": [coord.tolist() if isinstance(coord, np.ndarray) else list(coord) for coord in bar_conf["deformed_coords"]]
                }
                for bar_conf in bar_deformation_data
            ] if shortest_approach_path else [],
            "grasp_path": [list(conf) for conf in grasp_paths] if grasp_paths else [],
            "target_pose": {
                "position": list(bar_target_pose[0]),
                "orientation": list(bar_target_pose[1])
            },
            "color": list(rgba_color)
        }
        all_paths_data.append(path_data)
        
        # end_points = get_segmented_bar_endpoints(obj_bar_body_id, obj_bar_info)
        # deformation_results = single_bar_analysis(end_points, n_segments=7)
        # set_spherical_joint_poses(obj_bar_body_id, range(6), deformation_results["joint_quaternions"])
        # wait_if_gui()
        # reset_spherical_joint_poses(obj_bar_body_id, range(6))
        
        print("Execute the paths...")
        # wait_if_gui()

        # Execute the approach path
        # if shortest_approach_path:
        #     for i in range(len(shortest_approach_path)):
        #         set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
        #         # reset_spherical_joint_poses(obj_bar_body_id, range(6))
        #         attachements.assign()
        #         # # print(attachements.child)
        #         # # wait_if_gui()
        #         # end_points = get_segmented_bar_endpoints(obj_bar_body_id, obj_bar_info)
        #         # orientation_quat = get_link_pose(obj_bar_body_id, -1)[1]
        #         # print(get_link_pose(obj_bar_body_id, -1)[0])
        #         # print(orientation_quat)
        #         # deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=500)
        #         # set_spherical_joint_poses(obj_bar_body_id, range(6), deformation_results["joint_quaternions"])
        #         # # q, is_collision = convex_mesh_detection(shortest_approach_path[i], bars_convex_mesh)
        #         # # if is_collision:
        #         # # wait_if_gui()
        #         set_spherical_joint_poses(obj_bar_body_id, range(6), bar_deformation_data[i]["joint_quaternions"])
                
        #         deformed_points = bar_deformation_data[i]["deformed_coords"]
        #         for j in range(len(deformed_points)-1):
        #             add_line(
        #                 start=deformed_points[j],
        #                 end=deformed_points[j+1],
        #                 color=[159/255, 163/255, 154/255, 0.4],
        #                 width=10.0,
        #                 lifetime=0
        #             )
                
        #         time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
        #     wait_if_gui()
        
        if shortest_approach_path:
            all_left_edges = []
            all_right_edges = []
            center_points = []
            center_line_ids = []
            
            for i in range(len(shortest_approach_path)):
                set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
                attachements.assign()
                gripper_pos, gripper_quat = get_link_pose(robot_id, 15)
                set_spherical_joint_poses(obj_bar_body_id, range(6), bar_deformation_data[i]["joint_quaternions"])
                
                deformed_points = bar_deformation_data[i]["deformed_coords"]
                current_left = deformed_points[::2]
                current_right = deformed_points[1::2]
                
                all_left_edges.append(current_left)
                all_right_edges.append(current_right)
                center_points.append(gripper_pos)
                
                if i > 0:
                    line_id = p.addUserDebugLine(
                        lineFromXYZ=center_points[i-1],
                        lineToXYZ=center_points[i],
                        lineColorRGB=[rgba_color[0], rgba_color[1], rgba_color[2]],
                        lineWidth=2,
                        lifeTime=0
                    )
                    center_line_ids.append(line_id)
                
                time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
                
            vertices = []
            indices = []
            
            for i in range(len(all_left_edges)):
                vertices.extend(all_left_edges[i] + all_right_edges[i][::-1])
                
            n_segments = len(all_left_edges[0])
            for i in range(len(all_left_edges)-1):
                for j in range(n_segments):
                    idx0 = i * 2 * n_segments + j
                    idx1 = idx0 + 1
                    idx2 = (i+1) * 2 * n_segments + j
                    idx3 = idx2 + 1
                    
                    indices.extend([idx0, idx2, idx1])
                    indices.extend([idx1, idx2, idx3])
                    
                    indices.extend([idx1, idx2, idx0])
                    indices.extend([idx3, idx2, idx1])
            
            vis_shape = p.createVisualShape(
                p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                rgbaColor=[rgba_color[0], rgba_color[1], rgba_color[2], 0.4],
                # specularColor=[0.4, 0.4, 0.4]
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis_shape,
                basePosition=[0, 0, 0]
            )
                
            # wait_if_gui()

        # Execute the grasp path
        if grasp_paths:
            for i in range(len(grasp_paths)):
                set_joint_positions(robot_id, arm_joints, grasp_paths[i])
                reset_spherical_joint_poses(obj_bar_body_id, range(6))
                attachements.assign()
                time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
            # wait_if_gui()

        # Post-grasp operations
        p.changeDynamics(obj_bar_body_id, -1, mass=0.0)  # Make object massless

        set_gripper_open(robot_id)  # Open the gripper
        # set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)  # Return to vertical configuration

        # Reverse the grasp path to move back from the grasp position
        # if grasp_paths:
        #     print("Reversing grasp path to retreat...")
        #     for i in range(len(grasp_paths) - 1, -1, -1):  # Iterate backwards
        #         set_joint_positions(robot_id, arm_joints, grasp_paths[i])
        #         time.sleep(1.0 / 960)  # Match original grasp execution speed
        # else:
        #     print("No grasp path available to reverse.")

        # # Reverse the approach path to return to VERTICAL_CONF
        # if shortest_approach_path:
        #     print("Reversing approach path to return to VERTICAL_CONF...")
        #     for i in range(len(shortest_approach_path) - 1, -1, -1):  # Iterate backwards
        #         set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
        #         time.sleep(1.0 / 960)  # Match original approach execution speed
        # else:
        #     print(
        #         "No approach path available to reverse, setting to VERTICAL_CONF directly."
        #     )
        #     set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
        #     wait_if_gui()
        set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)

        # def is_position_close(new_pos, existing_positions, threshold):
        #     for existing_pos in existing_positions:
        #         distance = np.linalg.norm(np.array(new_pos) - np.array(existing_pos))
        #         if distance < threshold:
        #             return True
        #     return False

        # if not is_position_close(joint1_pos, joints_pos, 1e-3):
        #     joint1_id = create_sphere(
        #         joint_radius, color=[249 / 255, 241 / 255, 219 / 255, 1.0]
        #     )
        #     set_pose(joint1_id, joint1_pose)
        #     joints_pos.add(joint1_pos)
        # if not is_position_close(joint2_pos, joints_pos, 1e-3):
        #     joint2_id = create_sphere(
        #         joint_radius, color=[249 / 255, 241 / 255, 219 / 255, 1.0]
        #     )
        #     set_pose(joint2_id, joint2_pose)
        #     joints_pos.add(joint2_pos)

        set_gripper(robot_id, 0.72)  # Set gripper to a specific position

        # # Update obstacles list with the grasped object
        # if bars_convex_mesh is not None:
        #     remove_body(bars_convex_mesh)

        obstacles.append(obj_bar_body_id)
        assembled_bars.append(obj_bar_body_id)

    # Extract current structure for FEA
    active_bar_ids = [int(i) for i, bar in obj_bar_objects.items() if bar in assembled_bars]
    active_bars = [b for b in full_structure["bars"] if b["id"] in active_bar_ids]

    # Find all nodes connected to active bars
    active_node_ids = set()
    for bar in active_bars:
        active_node_ids.add(bar["start"])
        active_node_ids.add(bar["end"])

    active_nodes = [n for n in full_structure["nodes"] if n["id"] in active_node_ids]

    # Construct current structure in FEA format
    current_structure = {
        "nodes": active_nodes,
        "bars": active_bars
    }

    # Get FEA results with relative displacement, rotation matrix, and joint quaternions
    fea_results = fea_analysis(current_structure, n_segments=7, k_rot_base=1.6, k_rot_bar_sim=4.2e9, scale=1.0)
    deformed_bars = fea_results["deformed_bars"]
    deformed_nodes = fea_results["deformed_nodes"]

    # Store initial poses for target bars (middle segment)
    initial_poses = {}
    for bar_id, bar_info in bar_structures.items():
        bar_target_pos = np.array(bar_info["position"])
        bar_target_rot = np.array(bar_info["rotation_matrix"])  # Store as rotation matrix
        initial_poses[int(bar_id)] = (bar_target_pos, bar_target_rot)

    # Create a mapping from node ID to deformed position
    node_deformations = {node["id"]: np.array(node["deformed_position"]) for node in deformed_nodes}
        
    # Store deformation data for this assembly step
    step_deformation = {
        "sequence_index": seq_index,
        "bar_deformations": [],
        "joint_deformations": []
    }

    # Update the pose of all assembled bars (middle segment) based on FEA results
    for bar_id, obj_bar_body_id in obj_bar_objects.items():
        if obj_bar_body_id in assembled_bars:
            bar_id_int = int(bar_id)

            # Get the bar deformation info from FEA results
            bar_deform = next((b for b in deformed_bars if b["id"] == bar_id_int), None)
            if bar_deform is None:
                print(f"Warning: No deformation info found for bar {bar_id}")
                continue

            # Get initial pose from initial_poses
            if bar_id not in initial_poses:
                print(f"Warning: No initial pose found for bar {bar_id}")
                continue
            initial_pos, initial_rot = initial_poses[bar_id]

            # Get relative displacement and apply it to initial position
            relative_displacement = np.array(bar_deform["relative_displacement"])
            deformed_pos = initial_pos + relative_displacement
            
            # Set the local coordinate system
            local_coordinate_system = bar_deform["local_coordinate_system"]

            # Get relative rotation matrix and combine with initial rotation
            relative_rot = np.array(bar_deform["rotation_matrix"])
            deformed_rot = relative_rot @ local_coordinate_system # Apply relative rotation to initial

            # Convert deformed rotation matrix to quaternion
            deformed_quat = R.from_matrix(deformed_rot).as_quat()

            # Set the deformed pose for the bar (obj_bar_body_id)
            deformed_pose = (deformed_pos.tolist(), deformed_quat.tolist())
            set_pose(obj_bar_body_id, deformed_pose)
            print(f"Bar {bar_id} initial position: {initial_pos}, quaternion: {R.from_matrix(initial_rot).as_quat()}")
            print(f"Bar {bar_id} deformed to position: {deformed_pos}, quaternion: {deformed_quat}")
            
            # Set the deformed quat for the links of the bar
            deformed_link_quat = bar_deform["joint_quaternions"]
            set_spherical_joint_poses(obj_bar_body_id, range(6), deformed_link_quat)
            
            # Store deformation data for this bar
            bar_deformation = {
                "bar_id": str(bar_id),
                "deformed_position": deformed_pos.tolist(),
                "deformed_quaternion": deformed_quat.tolist(),
                "deformed_link_quaternion": [quat for quat in deformed_link_quat],
                "relative_displacement": relative_displacement.tolist(),
                "rotation_matrix": relative_rot.tolist(),
                "local_coordinate_system": local_coordinate_system
            }
            step_deformation["bar_deformations"].append(bar_deformation)
            
    # Store joint deformation data
    for node_id, deformed_pos in node_deformations.items():
        joint_deformation = {
            "node_id": str(node_id),
            "deformed_position": deformed_pos.tolist()
        }
        step_deformation["joint_deformations"].append(joint_deformation)
            
    # Append deformation data for this step
    deformation_data.append(step_deformation)
    
    # Function to clear all existing joints
    def clear_joints():
        for joint_body in joint_bodies:
            remove_body(joint_body)
        joint_bodies.clear()

    # Function to create new joints at given positions
    def create_joints(joint_positions, joint_radius=0.012):
        clear_joints()  # Clear existing joints first
        for pos in joint_positions:
            joint_id = create_sphere(joint_radius, color=[249/255, 241/255, 219/255, 1.0])
            set_pose(joint_id, (pos, [0, 0, 0, 1]))  # Default orientation
            joint_bodies.append(joint_id)
            
    # # Collect all unique joint positions from deformed nodes
    joint_positions = bar_deform["deformed_joint_positions"]

    # Create/update joints at deformed positions
    create_joints(joint_positions)

    # wait_if_gui()
    
wait_if_gui()

csv_output_file = "column_approach_path_stats_flexible_4.2_elastic_1.6.csv"
try:
    # Prepare CSV headers: bar_id, 10 approach times, 10 approach costs, 1 grasp time
    headers = ["bar_id"]
    for i in range(10):
        headers.append(f"approach_time_{i+1}")
    for i in range(10):
        headers.append(f"approach_cost_{i+1}")
    headers.append("grasp_time")
    
    # Prepare data rows: one row per bar
    rows = []
    for stat in approach_stats:
        row = {"bar_id": stat["bar_id"]}
        # Add 10 approach times (or None if attempt doesn't exist)
        for i in range(10):
            time_val = stat["planning_times"][i] if i < len(stat["planning_times"]) else None
            row[f"approach_time_{i+1}"] = time_val
        # Add 10 approach costs (or None if attempt doesn't exist)
        for i in range(10):
            cost_val = stat["planning_costs"][i] if i < len(stat["planning_costs"]) else None
            row[f"approach_cost_{i+1}"] = cost_val
        # Add grasp time (single value, defaults to inf if not present)
        row["grasp_time"] = stat.get("grasp_planning_time")
        rows.append(row)
    
    # Write to CSV
    with open(csv_output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved approach path statistics to {csv_output_file}")
except Exception as e:
    print(f"Error saving approach path statistics to {csv_output_file}: {e}")

output_file = "column_assembly_paths_flexible_4.2_elastic_1.6.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_paths_data, f, indent=4)
    print(f"Saved all paths to {output_file}")
except Exception as e:
    print(f"Error saving paths to {output_file}: {e}")
    
# Save deformation data to JSON file
deformation_output_file = "column_deformation_data_flexible_4.2_elastic_1.6.json"
try:
    with open(deformation_output_file, "w", encoding="utf-8") as f:
        json.dump(deformation_data, f, indent=4)
    print(f"Saved deformation data to {deformation_output_file}")
except Exception as e:
    print(f"Error saving deformation data to {deformation_output_file}: {e}")