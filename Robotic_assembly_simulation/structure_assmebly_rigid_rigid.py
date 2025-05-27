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

connect(use_gui=True, shadows=False, color=[1, 1, 1, 0.8])
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create simulation environment
plane_id = p.loadURDF("plane_transparent.urdf", [0, 0, -0.0])
set_all_color(plane_id, color=[1, 1, 1, 0.001])

# Create supports
# supports_geometry = get_cylinder_geometry(0.005, 0.188)
# supports_ori = p.getQuaternionFromEuler([0, 0, 0])
# supports_1_pose = ([0.31999999880790714, 0.20000000000000004, 0.094], supports_ori)
# supports_2_pose = ([0.4400000005960465, 0.13071796894073487, 0.094], supports_ori)
# supports_3_pose = ([0.4400000005960465, 0.2692820310592652, 0.094], supports_ori)

# supports_1_collision_id, supports_1_visual_id = create_shape(
#     supports_geometry, supports_1_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_1_id = create_body(supports_1_collision_id, supports_1_visual_id)

# supports_2_collision_id, supports_2_visual_id = create_shape(
#     supports_geometry, supports_2_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_2_id = create_body(supports_2_collision_id, supports_2_visual_id)

# supports_3_collision_id, supports_3_visual_id = create_shape(
#     supports_geometry, supports_3_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_3_id = create_body(supports_3_collision_id, supports_3_visual_id)

# supports_geometry = get_cylinder_geometry(0.005, 0.24)
# supports_ori = p.getQuaternionFromEuler([0, 0, 0])
# supports_1_pose = ([0.48660254089588534, 0.235355339586163, 0.134], supports_ori)
# supports_2_pose = ([0.13304914108298843, 0.5102521280478652, 0.134], supports_ori)
# supports_3_pose = ([0.3133974591041147, 0.235355339586163, 0.134], supports_ori)
# supports_4_pose = ([-0.04015592358662501, 0.5102521082915299, 0.134], supports_ori)

# supports_1_collision_id, supports_1_visual_id = create_shape(
#     supports_geometry, supports_1_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_1_id = create_body(supports_1_collision_id, supports_1_visual_id)

# supports_2_collision_id, supports_2_visual_id = create_shape(
#     supports_geometry, supports_2_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_2_id = create_body(supports_2_collision_id, supports_2_visual_id)

# supports_3_collision_id, supports_3_visual_id = create_shape(
#     supports_geometry, supports_3_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_3_id = create_body(supports_3_collision_id, supports_3_visual_id)

# supports_4_collision_id, supports_4_visual_id = create_shape(
#     supports_geometry, supports_4_pose, color=[1.0, 1.0, 1.0, 0.6]
# )
# supports_4_id = create_body(supports_4_collision_id, supports_4_visual_id)


# base_geometry = get_box_geometry(0.15, 0.15, 0.2)
# base_ori = p.getQuaternionFromEuler([0, 0, 0])
# base_pose = ([0.0, 0.0, -0.1], base_ori)
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
with open("arc_infos.json", "r", encoding="utf-8") as file:
    bar_structures = json.load(file)

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
obstacles = [plane_id]
# obstacles = [plane_id, supports_1_id, supports_2_id, supports_3_id, supports_4_id]
assembled_bars = []
joints_pos = set()
bars_convex_mesh = None

# assembly_sequence = [15, 9, 14, 18, 1, 10, 13, 22, 12, 0, 19, 20, 11, 21, 4, 5, 6, 7, 8, 17, 16, 3, 2, 23]
assembly_sequence = [10, 9, 8, 23, 20, 6, 11, 15, 13, 1, 5, 18, 2, 12, 16, 7, 0, 22, 4, 14, 19, 17, 21, 3]

all_paths_data = []
approach_stats = []

obj_bar_objects = {}
# for i, (obj_bar_id, obj_bar_info) in enumerate(bar_structures.items()):
for seq_index, bar_id in enumerate(assembly_sequence):

    # Get bar information
    obj_bar_info = bar_structures.get(str(bar_id))  # Ensure bar_id is string
    if obj_bar_info is None:
        print(f"Warning: Bar ID {bar_id} not found in bar_structures. Skipping.")
        continue
    
    color_value = seq_index / max(1, num_bars - 1)
    rgba_color = cmap(color_value)
    
    # Create geometry for each bar
    obj_bar_base_geometry = get_box_geometry(
        obj_bar_info["geometry"][0],
        obj_bar_info["geometry"][1],
        obj_bar_info["geometry"][2] - 0.02,
    )

    # Initial orientation and position
    obj_bar_initial_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_bar_initial_pos = (0.0, 0.0, 0.0)
    obj_bar_initial_pose = (obj_bar_initial_pos, obj_bar_initial_ori)

    # Create shape with green color
    obj_bar_base_collision_id, obj_bar_base_visual_id = create_shape(
        obj_bar_base_geometry,
        obj_bar_initial_pose,
        color=rgba_color,
    )

    # Create body and store it
    obj_bar_body_id = create_body(
        obj_bar_base_collision_id, obj_bar_base_visual_id, mass=0.001
    )
    obj_bar_objects[bar_id] = obj_bar_body_id

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
    bar_target_id = bar_objects.get(str(bar_id))
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
    grasps = get_side_grasps(
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
    if selected_approach_conf is None:
        print("No feasible conf found.")
        wait_if_gui()

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
        approach_paths = compute_motion(
            robot_id,
            arm_joints,
            start_conf=VERTICAL_CONF,
            end_conf=selected_approach_conf,
            obstacles=obstacles,
            attachments=[attachements],
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
        else:
            print("No approach paths found.")
            planning_costs.append(float("inf"))

    if shortest_approach_path is not None:
        print(f"Shortest approach path found with cost: {min_approach_cost}")
        print(f"Shortest approach path: {shortest_approach_path}")
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
        "bar_id": str(bar_id),
        "sequence_index": seq_index,
        "planning_times": planning_times,
        "planning_costs": planning_costs,
        "grasp_planning_time": grasp_planning_time
    })

    set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)
    attachements.assign()
    
    path_data = {
        "bar_id": str(bar_id),
        "sequence_index": seq_index,
        "approach_path": [list(conf) for conf in shortest_approach_path] if shortest_approach_path else [],
        "grasp_path": [list(conf) for conf in grasp_paths] if grasp_paths else [],
        "target_pose": {
            "position": list(bar_target_pose[0]),
            "orientation": list(bar_target_pose[1])
        },
        "color": list(rgba_color)
    }
    all_paths_data.append(path_data)    
    
    print("Execute the paths...")
    # wait_if_gui()

    # Execute the approach path
    # if shortest_approach_path:
    #     for i in range(len(shortest_approach_path)):
    #         set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
    #         attachements.assign()
    #         time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
    #     wait_if_gui()
    
    # if shortest_approach_path:
    #     bar_length = obj_bar_info["geometry"][2]  # Get the length of the current bar
    #     start_points = []  # To store all start points of the lines
    #     end_points = []    # To store all end points of the lines
    #     center_points = [] # To store all center points
        
    #     for i in range(len(shortest_approach_path)):
    #         set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
    #         attachements.assign()
            
    #         # Get the current gripper pose
    #         gripper_pos, gripper_quat = get_link_pose(robot_id, 15)
            
    #         # Convert quaternion to rotation matrix to extract y-axis
    #         rotation_matrix = matrix_from_quat(gripper_quat)
    #         y_axis = rotation_matrix[:, 1]  # Second column is the y-axis
            
    #         # Compute the end points of the line (start at gripper position, extend along y-axis)
    #         line_start = np.array(gripper_pos) - (bar_length / 2) * y_axis
    #         line_end = line_start + bar_length * y_axis
    #         center_point = (line_start + line_end) / 2  # Calculate center point
            
    #         # Store the points
    #         start_points.append(line_start)
    #         end_points.append(line_end)
    #         center_points.append(center_point)
            
    #         # Draw the current line in the simulation (lighter color)
    #         if i == 0 or i == len(shortest_approach_path) - 1:
    #             add_line(
    #                 start=line_start,
    #                 end=line_end,
    #                 color=[228/255, 223/255, 215/255, 0.6],  # Light gray for individual lines
    #                 width=15.0,
    #                 lifetime=0
    #             )
            
    #         # Draw connecting lines as soon as we have at least 2 points
    #         if i > 0:
    #             # Connect start points (left side)
    #             add_line(
    #                 start=start_points[i-1],
    #                 end=start_points[i],
    #                 color=[228/255, 223/255, 215/255, 0.5],  # Darker color for ribbon edges
    #                 width=1.5,
    #                 lifetime=0
    #             )
    #             # Connect end points (right side)
    #             add_line(
    #                 start=end_points[i-1],
    #                 end=end_points[i],
    #                 color=[228/255, 223/255, 215/255, 0.5],  # Darker color for ribbon edges
    #                 width=1.5,
    #                 lifetime=0
    #             )
    #             # Connect center points
    #             add_line(
    #                 start=center_points[i-1],
    #                 end=center_points[i],
    #                 color=[249/255, 232/255, 208/255, 1.0],  # Solid color for center line
    #                 width=2.0,
    #                 lifetime=0
    #             )
            
    #         time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
        
    #     wait_if_gui()
    
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
            
            # indices.extend([idx+1, idx+4, idx+2])
            # indices.extend([idx+2, idx+4, idx+1])

            # indices.extend([idx+2, idx+4, idx+5])
            # indices.extend([idx+5, idx+4, idx+2])
        
        return vertices, indices

    if shortest_approach_path:
        bar_length = obj_bar_info["geometry"][2]
        start_points = []
        end_points = []
        center_points = []
        center_line_ids = []
        
        for i in range(len(shortest_approach_path)):
            set_joint_positions(robot_id, arm_joints, shortest_approach_path[i])
            attachements.assign()
            
            gripper_pos, gripper_quat = get_link_pose(robot_id, 15)
            rotation_matrix = matrix_from_quat(gripper_quat)
            y_axis = rotation_matrix[:, 1]
            
            line_start = np.array(gripper_pos) - (bar_length / 2) * y_axis
            line_end = line_start + bar_length * y_axis
            center_point = (line_start + line_end) / 2
            
            start_points.append(line_start)
            end_points.append(line_end)
            center_points.append(center_point)
            
            # if i > 0:
            #     line_id = p.addUserDebugLine(
            #         lineFromXYZ=center_points[i-1],
            #         lineToXYZ=center_points[i],
            #         # lineColorRGB=[159/255, 163/255, 154/255],
            #         lineColorRGB=[rgba_color[0], rgba_color[1], rgba_color[2]],
            #         lineWidth=2,
            #         lifeTime=0
            #     )
            #     center_line_ids.append(line_id)
            
            time.sleep(1.0 / 240)
        
        vertices, indices = create_ribbon_mesh(start_points, end_points, center_points)
        col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
        vis_shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[rgba_color[0], rgba_color[1], rgba_color[2], 0.4])
                                   # rgbaColor=[159/255, 163/255, 154/255, 0.4])
        # ribbon_body = p.createMultiBody(baseMass=0, 
        #                             baseCollisionShapeIndex=col_shape,
        #                             baseVisualShapeIndex=vis_shape,
        #                             basePosition=[0, 0, 0])
        
        # wait_if_gui()
        
        # p.removeAllUserDebugItems()
        # remove_body(ribbon_body)
        
        # wait_if_gui()

    # Execute the grasp path
    if grasp_paths:
        for i in range(len(grasp_paths)):
            set_joint_positions(robot_id, arm_joints, grasp_paths[i])
            attachements.assign()
            time.sleep(1.0 / 240)  # Simulate motion at 60 Hz
        # wait_if_gui()

    # Post-grasp operations
    p.changeDynamics(obj_bar_body_id, -1, mass=0.0)  # Make object massless

    set_gripper_open(robot_id)  # Open the gripper
    # set_joint_positions(robot_id, arm_joints, VERTICAL_CONF)  # Return to vertical configuration

    # # Reverse the grasp path to move back from the grasp position
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
        
    # Path to return to VERTICAL_CONF
    # return_paths = compute_motion(
    #         robot_id,
    #         arm_joints,
    #         start_conf=selected_grasp_conf,
    #         end_conf=VERTICAL_CONF,
    #         obstacles=obstacles,
    #         # attachments=[attachements],
    #         assembled_elements=assembled_bars,
    #         self_collisions=True,
    #         algorithm="rrt_connect",
    #         # resolutions=resolutions,
    #         # weights=weights,
    #     )  
    # if return_paths:
    #     for i in range(len(return_paths)):
    #         set_joint_positions(robot_id, arm_joints, return_paths[i])
    #         time.sleep(1.0 / 480)  # Simulate motion at 60 Hz
    #     # wait_if_gui()  
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
    # bars_nodes = []

    # for bar in assembled_bars:
    #     aabb = get_aabb(bar)
    #     aabb = buffer_aabb(aabb, 0.1)
    #     # center = get_aabb_center(aabb)
    #     # extent = get_aabb_extent(aabb)
    #     bar_nodes = get_aabb_vertices(aabb)
    #     bars_nodes.extend(bar_nodes)

    # bars_convex_hull = convex_hull(bars_nodes)
    # bars_convex_mesh = create_mesh(bars_convex_hull, under=True, color=[1, 0, 0, 0.2])
    
    # wait_if_gui()

    # def draw_filled_aabb_visual_only(aabb, color=[1, 0, 0, 0.2], physicsClientId=0):
    #     center = get_aabb_center(aabb)
    #     extent = get_aabb_extent(aabb)

    #     visual_shape_id = p.createVisualShape(
    #         shapeType=p.GEOM_BOX,
    #         halfExtents=extent / 2.0,
    #         rgbaColor=color,
    #         physicsClientId=physicsClientId
    #     )

    #     body_id = p.createMultiBody(
    #         baseMass=0,
    #         baseVisualShapeIndex=visual_shape_id,
    #         basePosition=center,
    #         physicsClientId=physicsClientId
    #     )

    #     return body_id

    # draw_filled_aabb_visual_only(aabb)

wait_if_gui()

csv_output_file = "arc_approach_grasp_path_stats.csv"
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

output_file = "arc_assembly_paths.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_paths_data, f, indent=4)
    print(f"Saved all paths to {output_file}")
except Exception as e:
    print(f"Error saving paths to {output_file}: {e}")