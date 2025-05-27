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

from FEA_plugin import fea_analysis

connect(use_gui=True, shadows=False, color=[1, 1, 1, 0.8])
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create simulation environment
plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
set_all_color(plane_id, color=[1, 1, 1, 0.001])

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

# # Load structure data
# with open("tetrahedron_infos_deformation.json", "r", encoding="utf-8") as file:
#     bar_structures = json.load(file)
    
# # Load complete structure for FEA
# with open("tetrahedron_info_deformation_update.json", "r", encoding="utf-8") as file:
#     full_structure = json.load(file)

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
# obstacles = [plane_id, base_id]
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
            "grasp_path": [],    # Empty list for no path
            "target_pose": {
                "position": list(bar_target_pose[0]),
                "orientation": list(bar_target_pose[1])
            },
            "color": list(rgba_color)
        }
        all_paths_data.append(path_data)

        # Proceed to FEA and joint creation
        # (The FEA and joint creation code remains unchanged)
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
                planning_times.append(float("inf"))
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
        #         # q, is_collision = convex_mesh_detection(shortest_approach_path[i], bars_convex_mesh)
        #         # if is_collision:
        #         #     wait_if_gui()
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
                #         lineColorRGB=[rgba_color[0], rgba_color[1], rgba_color[2]],
                #         lineWidth=2,
                #         lifeTime=0
                #     )
                #     center_line_ids.append(line_id)
                
                time.sleep(1.0 / 240)
            
            vertices, indices = create_ribbon_mesh(start_points, end_points, center_points)
            col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
            vis_shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, 
                                        rgbaColor=[rgba_color[0], rgba_color[1], rgba_color[2], 0.4])
            # ribbon_body = p.createMultiBody(baseMass=0, 
            #                             baseCollisionShapeIndex=col_shape,
            #                             baseVisualShapeIndex=vis_shape,
            #                             basePosition=[0, 0, 0])
            
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

    # Get FEA results with relative displacement and rotation matrix
    fea_results = fea_analysis(current_structure, k_rot=3.2, scale=1)
    deformed_bars = fea_results["deformed_bars"]
    deformed_nodes = fea_results["deformed_nodes"]

    # Store initial poses for target bars (full length) - now storing both position and rotation matrix
    initial_poses = {}
    for bar_id, bar_info in bar_structures.items():
        bar_target_pos = np.array(bar_info["position"])
        bar_target_rot = np.array(bar_info["rotation_matrix"])  # Store as rotation matrix
        initial_poses[int(bar_id)] = (bar_target_pos, bar_target_rot)

    # Create a mapping from node ID to deformed position
    node_deformations = {node["id"]: np.array(node["deformed_position"]) for node in deformed_nodes}

    # # Update the pose of all assembled bars (short bars) based on FEA results
    # for bar_id, obj_bar_body_id in obj_bar_objects.items():
    #     if obj_bar_body_id in assembled_bars:
    #         bar_id_int = int(bar_id)

    #         # Get the bar information from the full structure
    #         bar_info = next((b for b in full_structure["bars"] if b["id"] == bar_id_int), None)
    #         if bar_info is None:
    #             print(f"Warning: No bar info found for bar {bar_id}")
    #             continue

    #         # Get start and end node IDs
    #         start_node_id = bar_info["start"]
    #         end_node_id = bar_info["end"]

    #         # Get deformed positions for start and end nodes
    #         start_pos = node_deformations.get(start_node_id)
    #         end_pos = node_deformations.get(end_node_id)

    #         if start_pos is None or end_pos is None:
    #             print(f"Warning: Missing node deformation data for bar {bar_id}")
    #             continue

    #         # Calculate new bar position (midpoint between deformed nodes)
    #         deformed_pos = (start_pos + end_pos) / 2

    #         # Calculate direction vector and rotation
    #         direction = end_pos - start_pos
    #         length = np.linalg.norm(direction)
    #         if length > 1e-6:
    #             direction = direction / length

    #         # Calculate rotation matrix to align with new direction
    #         initial_dir = np.array([0, 0, 1])  # Assuming bars are initially along Z-axis
    #         deformed_dir = direction

    #         # Calculate rotation between initial and deformed direction
    #         dot = np.dot(initial_dir, deformed_dir)
    #         dot = np.clip(dot, -1.0, 1.0)
            
    #         if abs(1.0 - dot) < 1e-6:
    #             # No significant rotation - identity matrix
    #             rotation_matrix = np.eye(3)
    #         else:
    #             # Calculate rotation matrix
    #             axis = np.cross(initial_dir, deformed_dir)
    #             axis = axis / np.linalg.norm(axis)
    #             angle = np.arccos(dot)
    #             rotation = R.from_rotvec(axis * angle)
    #             rotation_matrix = rotation.as_matrix()

    #         # Convert rotation matrix to quaternion
    #         deformed_quat = R.from_matrix(rotation_matrix).as_quat()

    #         # Set the deformed pose for the short bar (obj_bar_body_id)
    #         deformed_pose = (deformed_pos.tolist(), deformed_quat.tolist())
    #         set_pose(obj_bar_body_id, deformed_pose)
    #         print(f"Short bar {bar_id} deformed to position: {deformed_pos}, quaternion: {deformed_quat}")
    
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
            
            # Store deformation data for this bar
            bar_deformation = {
                "bar_id": str(bar_id),
                "deformed_position": deformed_pos.tolist(),
                "deformed_quaternion": deformed_quat.tolist(),
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
            
    # Collect all unique joint positions from deformed nodes
    joint_positions = list(node_deformations.values())

    # Create/update joints at deformed positions
    create_joints(joint_positions)

    # wait_if_gui()
    
wait_if_gui()

csv_output_file = "column_approach_path_stats_rigid_elastic_3.2.csv"
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

output_file = "column_assembly_paths_rigid_elastic_3.2.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_paths_data, f, indent=4)
    print(f"Saved all paths to {output_file}")
except Exception as e:
    print(f"Error saving paths to {output_file}: {e}")
    
# Save deformation data to JSON file
deformation_output_file = "column_deformation_data_rigid_elastic_3.2.json"
try:
    with open(deformation_output_file, "w", encoding="utf-8") as f:
        json.dump(deformation_data, f, indent=4)
    print(f"Saved deformation data to {deformation_output_file}")
except Exception as e:
    print(f"Error saving deformation data to {deformation_output_file}: {e}")