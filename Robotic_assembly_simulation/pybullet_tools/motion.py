import numpy as np
from termcolor import cprint

from pybullet_planning import (
    get_sample_fn,
    get_distance_fn,
    get_extend_fn,
    set_joint_positions,
    get_collision_fn,
    pairwise_collision,
    remove_body,
    birrt,
    check_initial_end,
    get_aabb,
    get_aabb_vertices,
    convex_hull,
    create_mesh,
    wait_if_gui,
    )
from motion_planners.meta import solve
from pybullet_tools.utils import (
    buffer_aabb,
    get_self_link_pairs,
    pairwise_link_collision,
    get_link_pose,
    multiply,
    Pose,
    Euler,
)
from pybullet_tools.ur5e_robotiq_utils import (set_spherical_joint_pose, set_spherical_joint_poses, reset_spherical_joint_poses)

from FEA_single_bar_plugin import single_bar_analysis

MAX_DISTANCE = 0.0

DYNMAIC_RES_RATIO = 0.5
CONVEX_BUFFER = 0.1

RESOLUTIONS = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
WEIGHTS = np.reciprocal(RESOLUTIONS)

##################################################


def get_pairs(iterator):
    try:
        last = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield last, current
        last = current

##################################################


def create_convex_bounding(bodies=None, buffer=CONVEX_BUFFER, color=[27 / 255, 167 / 255, 132 / 255, 0.2]):
    bodies_nodes = []

    if bodies is None:
        return None
    else:
        for body in bodies:
            aabb = get_aabb(body)
            aabb = buffer_aabb(aabb, buffer)
            body_nodes = get_aabb_vertices(aabb)
            bodies_nodes.extend(body_nodes)

        bodies_convex_hull = convex_hull(bodies_nodes)
        bodies_convex_mesh = create_mesh(
            bodies_convex_hull, under=True, color=color)
        
        return bodies_convex_mesh

###############################################


def compute_motion(robot, joints, start_conf, end_conf, obstacles=[], attachments=[],
                   assembled_elements=None, self_collisions=True, weights=WEIGHTS, resolutions=RESOLUTIONS,
                   disabled_collisions={}, custom_limits={}, algorithm=None,
                   buffer=CONVEX_BUFFER, max_distance=MAX_DISTANCE, **kwargs):

    assert len(joints) == len(end_conf)

    set_joint_positions(robot, joints, start_conf)
    for attachment in attachments:
        attachment.assign()

    # construct a bounding box around the built elements
    convex_bounding = None
    if assembled_elements:
        convex_bounding = create_convex_bounding(assembled_elements)

    try:
        if (weights is None) and (resolutions is not None):
            weights = np.reciprocal(resolutions)

        sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
        distance_fn = get_distance_fn(robot, joints, weights=weights)
        extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
        fine_extend_fn = get_extend_fn(
            robot, joints, resolutions=DYNMAIC_RES_RATIO*resolutions)
        collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits=custom_limits, max_distance=max_distance)

        # Collision detection between robot and mesh
        def convex_mesh_detection(q, convex_mesh=convex_bounding):
            set_joint_positions(robot, joints, q)
            for attachment in attachments:
                attachment.assign()
            collision = (convex_mesh is not None) and ((pairwise_collision(
                robot, convex_mesh)) or (any(pairwise_collision(attachment.child, convex_mesh) for attachment in attachments)))

            return q, collision

        def dynamic_extend_fn(q_start, q_end):
            for (q1, c1), (q2, c2) in get_pairs(map(convex_mesh_detection, extend_fn(q_start, q_end))):
                if c1 and c2:
                    for q in fine_extend_fn(q1, q2):
                        yield q
                else:
                    yield q2

        def get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                     disabled_collisions={}, custom_limits={}, max_distance=max_distance):
            check_link_pairs = (
                get_self_link_pairs(robot, joints, disabled_collisions)
                if self_collisions
                else []
            )
            
            if convex_bounding is None:
                def element_collision_fn(q, diagnosis=False):
                    return collision_fn(q, diagnosis=diagnosis)
            else:
                def element_collision_fn(q, diagnosis=False):
                    set_joint_positions(robot, joints, q)
                    for attachment in attachments:
                        attachment.assign()

                    if self_collisions:
                        for link1, link2 in check_link_pairs:
                            if pairwise_link_collision(robot, link1, robot, link2):
                                return True
                    
                    if pairwise_collision(robot, convex_bounding):
                        return collision_fn(q, diagnosis=diagnosis)
                    return False
        
            return element_collision_fn
    
        element_collision_fn = get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions)
    
        paths = None

        if not check_initial_end(start_conf, end_conf, element_collision_fn):
            return None

        if algorithm is None:
            paths = birrt(start_conf, end_conf, distance_fn, sample_fn,
                          dynamic_extend_fn, element_collision_fn, **kwargs)
        else:
            paths = solve(start_conf, end_conf, distance_fn, sample_fn,
                          dynamic_extend_fn, element_collision_fn, algorithm=algorithm, **kwargs)

        if paths is None:
            cprint('Failed to find a motion plan!', 'red')
            return None
        else:
            return paths

    finally:
        # Always remove convex_bounding if it exists
        if convex_bounding is not None:
            remove_body(convex_bounding)
    

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
    
# def compute_motion_with_deformation(robot, joints, start_conf, end_conf, obstacles=[], attachments=[], attachments_data=None,
#                    assembled_elements=None, self_collisions=True, weights=WEIGHTS, resolutions=RESOLUTIONS,
#                    disabled_collisions={}, custom_limits={}, algorithm=None,
#                    buffer=CONVEX_BUFFER, max_distance=MAX_DISTANCE, **kwargs):

#     assert len(joints) == len(end_conf)

#     set_joint_positions(robot, joints, start_conf)
#     for attachment in attachments:
#         attachment.assign()

#     # construct a bounding box around the built elements
#     convex_bounding = None
#     if assembled_elements:
#         convex_bounding = create_convex_bounding(assembled_elements)

#     if (weights is None) and (resolutions is not None):
#         weights = np.reciprocal(resolutions)

#     sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
#     distance_fn = get_distance_fn(robot, joints, weights=weights)
#     extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
#     fine_extend_fn = get_extend_fn(
#         robot, joints, resolutions=DYNMAIC_RES_RATIO*resolutions)
#     collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
#                                     disabled_collisions=disabled_collisions,
#                                     custom_limits=custom_limits, max_distance=max_distance)

#     # Collision detection between robot and mesh
#     def convex_mesh_detection(q, convex_mesh=convex_bounding):
#         set_joint_positions(robot, joints, q)
#         for attachment in attachments:
#             attachment.assign()
#         collision = (convex_mesh is not None) and ((pairwise_collision(
#             robot, convex_mesh)) or (any(pairwise_collision(attachment.child, convex_mesh) for attachment in attachments)))

#         return q, collision

#     def dynamic_extend_fn(q_start, q_end):
#         for (q1, c1), (q2, c2) in get_pairs(map(convex_mesh_detection, extend_fn(q_start, q_end))):
#             if c1 and c2:
#                 for q in fine_extend_fn(q1, q2):
#                     yield q
#             else:
#                 yield q2
    
#     def get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
#                                  disabled_collisions={}, custom_limits={}, max_distance=max_distance):
#         check_link_pairs = (
#             get_self_link_pairs(robot, joints, disabled_collisions)
#             if self_collisions
#             else []
#         )
        
#         if convex_bounding is None:
#             def element_collision_fn(q):
#                 set_joint_positions(robot, joints, q)
#                 for attachment in attachments:
#                     attachment.assign()
#                     end_points = get_segmented_bar_endpoints(attachment.child, attachments_data)
#                     orientation_quat = get_link_pose(attachment.child, -1)[1]
#                     deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=500)
#                     set_spherical_joint_poses(attachment.child, range(6), deformation_results["joint_quaternions"])
                    
#                 return collision_fn(q)
#         else:
#             def element_collision_fn(q):
#                 set_joint_positions(robot, joints, q)
#                 for attachment in attachments:
#                     attachment.assign()
#                     end_points = get_segmented_bar_endpoints(attachment.child, attachments_data)
#                     orientation_quat = get_link_pose(attachment.child, -1)[1]
#                     deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=500)
#                     set_spherical_joint_poses(attachment.child, range(6), deformation_results["joint_quaternions"])

#                 if self_collisions:
#                     for link1, link2 in check_link_pairs:
#                         if pairwise_link_collision(robot, link1, robot, link2):
#                             return True
                
#                 if pairwise_collision(robot, convex_bounding):
#                     return collision_fn(q)
#                 return False
        
#         return element_collision_fn
    
#     element_collision_fn = get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions)
    
#     paths = None

#     if not check_initial_end(start_conf, end_conf, collision_fn):
#         return None

#     if algorithm is None:
#         paths = birrt(start_conf, end_conf, distance_fn, sample_fn,
#                       dynamic_extend_fn, element_collision_fn, **kwargs)
#     else:
#         paths = solve(start_conf, end_conf, distance_fn, sample_fn,
#                       dynamic_extend_fn, element_collision_fn, algorithm=algorithm, **kwargs)
        
#     for attachment in attachments:
#         reset_spherical_joint_poses(attachment.child, range(6))

#     if convex_bounding is not None:
#         remove_body(convex_bounding)

#     if paths is None:
#         cprint('Failed to find a motion plan!', 'red')
#         return None
#     else:
#         return paths

def compute_motion_with_deformation(robot, joints, start_conf, end_conf, obstacles=[], attachments=[], attachments_data=None,
                                   assembled_elements=None, self_collisions=True, weights=WEIGHTS, resolutions=RESOLUTIONS,
                                   disabled_collisions={}, custom_limits={}, algorithm=None,
                                   buffer=CONVEX_BUFFER, max_distance=MAX_DISTANCE, **kwargs):
    """
    Compute motion plan with deformation information for attachments.
    Returns:
        tuple: (paths, deformation_data) where deformation_data is a list of deformation results for each path point
    """
    assert len(joints) == len(end_conf)

    # Initialize
    convex_bounding = None
    deformation_data = []
    paths = None

    try:
        # Set initial configuration and assign attachments
        set_joint_positions(robot, joints, start_conf)
        for attachment in attachments:
            attachment.assign()

        # Construct a bounding box around the built elements
        if assembled_elements:
            convex_bounding = create_convex_bounding(assembled_elements)

        # Set default resolutions if None
        if resolutions is None:
            resolutions = [0.01] * len(joints)  # Default resolution of 0.01 per joint
            cprint(f"Warning: resolutions not provided, using default: {resolutions}", "yellow")

        # Set weights based on resolutions
        if (weights is None) and (resolutions is not None):
            weights = np.reciprocal(resolutions)

        # Initialize motion planning functions
        sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
        distance_fn = get_distance_fn(robot, joints, weights=weights)
        extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
        fine_extend_fn = get_extend_fn(robot, joints, resolutions=DYNMAIC_RES_RATIO * np.array(resolutions))
        collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                       self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                                       custom_limits=custom_limits, max_distance=max_distance)

        # Collision detection between robot and mesh
        def convex_mesh_detection(q, convex_mesh=convex_bounding):
            set_joint_positions(robot, joints, q)
            for attachment in attachments:
                attachment.assign()
            collision = (convex_mesh is not None) and ((pairwise_collision(robot, convex_mesh)) or
                        (any(pairwise_collision(attachment.child, convex_mesh) for attachment in attachments)))
            return q, collision

        def dynamic_extend_fn(q_start, q_end):
            for (q1, c1), (q2, c2) in get_pairs(map(convex_mesh_detection, extend_fn(q_start, q_end))):
                if c1 and c2:
                    for q in fine_extend_fn(q1, q2):
                        yield q
                else:
                    yield q2

        def get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                    self_collisions=self_collisions, disabled_collisions={},
                                    custom_limits={}, max_distance=max_distance):
            check_link_pairs = get_self_link_pairs(robot, joints, disabled_collisions) if self_collisions else []

            if convex_bounding is None:
                def element_collision_fn(q):
                    set_joint_positions(robot, joints, q)
                    for attachment in attachments:
                        attachment.assign()
                        end_points = get_segmented_bar_endpoints(attachment.child, attachments_data)
                        orientation_quat = get_link_pose(attachment.child, -1)[1]
                        deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=1.0)
                        set_spherical_joint_poses(attachment.child, range(6), deformation_results["joint_quaternions"])
                    return collision_fn(q)
            else:
                def element_collision_fn(q):
                    set_joint_positions(robot, joints, q)
                    for attachment in attachments:
                        attachment.assign()
                        end_points = get_segmented_bar_endpoints(attachment.child, attachments_data)
                        orientation_quat = get_link_pose(attachment.child, -1)[1]
                        deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=1.0)
                        set_spherical_joint_poses(attachment.child, range(6), deformation_results["joint_quaternions"])

                    if self_collisions:
                        for link1, link2 in check_link_pairs:
                            if pairwise_link_collision(robot, link1, robot, link2):
                                return True
                    if convex_bounding and pairwise_collision(robot, convex_bounding):
                        return collision_fn(q)
                    return False
            return element_collision_fn

        element_collision_fn = get_element_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                                        self_collisions=self_collisions)

        # Check initial and endRegenerate Response configurations
        if not check_initial_end(start_conf, end_conf, collision_fn):
            cprint('Initial or end configuration in collision!', 'red')
            return None, None

        # Compute motion plan
        if algorithm is None:
            paths = birrt(start_conf, end_conf, distance_fn, sample_fn,
                          dynamic_extend_fn, element_collision_fn, **kwargs)
        else:
            paths = solve(start_conf, end_conf, distance_fn, sample_fn,
                          dynamic_extend_fn, element_collision_fn, algorithm=algorithm, **kwargs)
        
        # # Log the result for debugging
        # print(f"Path planning returned type: {type(paths)}, value: {paths}")

        # Check paths validity
        if paths is None:
            cprint('Failed to find a motion plan! Got paths=None', 'red')
            return None, None
        if not isinstance(paths, (list, tuple)):
            cprint(f'Error: Expected paths to be a list/tuple, got {type(paths)}, value: {paths}', 'red')
            return None, None

        # Compute deformation information for each path point
        for q in paths:
            set_joint_positions(robot, joints, q)
            for attachment in attachments:
                attachment.assign()
                end_points = get_segmented_bar_endpoints(attachment.child, attachments_data)
                orientation_quat = get_link_pose(attachment.child, -1)[1]
                deformation_results = single_bar_analysis(end_points, orientation_quat=orientation_quat, n_segments=7, scale=1.0)
                joint_quats = deformation_results.get("joint_quaternions", [])
                if len(joint_quats) != 6:
                    cprint(f"Error: Expected 6 joint quaternions, got {len(joint_quats)}", "red")
                    return None, None
                deformation_data.append({
                    "joint_quaternions": joint_quats,
                    "deformed_coords": deformation_results.get("deformed_coords", [])
                })

        return paths, deformation_data

    finally:
        # Reset attachments to original state
        for attachment in attachments:
            reset_spherical_joint_poses(attachment.child, range(6))

        # Clean up convex bounding
        if convex_bounding is not None:
            remove_body(convex_bounding)