import math
import random
import numpy as np
from collections import namedtuple
from itertools import combinations
import pybullet as p
from itertools import product

from .utils import (
    multiply,
    get_link_pose,
    set_joint_position,
    set_joint_positions,
    get_joint_positions,
    get_min_limit,
    get_max_limit,
    quat_from_euler,
    read_pickle,
    set_pose,
    get_pose,
    euler_from_quat,
    link_from_name,
    point_from_pose,
    invert,
    Pose,
    unit_pose,
    joints_from_names,
    joint_from_name,
    PoseSaver,
    get_aabb,
    get_joint_limits,
    ConfSaver,
    get_bodies,
    create_mesh,
    remove_body,
    unit_from_theta,
    violates_limit,
    violates_limits,
    add_line,
    get_body_name,
    get_num_joints,
    approximate_as_cylinder,
    approximate_as_prism,
    unit_quat,
    unit_point,
    angle_between,
    quat_from_pose,
    compute_jacobian,
    movable_from_joints,
    quat_from_axis_angle,
    LockRenderer,
    Euler,
    get_links,
    get_link_name,
    get_extend_fn,
    get_moving_links,
    link_pairs_collision,
    get_link_subtree,
    clone_body,
    get_all_links,
    pairwise_collision,
    tform_point,
    get_camera_matrix,
    ray_from_pixel,
    pixel_from_ray,
    dimensions_from_camera_matrix,
    wrap_angle,
    TRANSPARENT,
    PI,
    OOBB,
    pixel_from_point,
    set_all_color,
    wait_if_gui,
    expand_links,
    pairwise_link_collision,
    approximate_as_prism_multi_link
)

UR5E_ROBOTIQ_URDF = "urdf/ur5e_robotiq.urdf"

# UR5e joint names
UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Robotiq 85 joint names (only movable joints)
ROBOTIQ_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_right_finger_tip_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
]

# Robotiq 85 link names
ROBOTIQ_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_right_finger_tip_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
]

# UR5e tool frame
UR5E_TOOL_FRAME = "gripper_center_link"

INITIAL_CONF = [
    0,
    -np.pi / 2,
    np.pi / 2,
    0,
    -np.pi / 2,
    0,
]

VERTICAL_CONF = [
    0,
    -np.pi / 2,
    0,
    0,
    np.pi / 2,
    -np.pi / 2,
]

# Grasp parameters
GRASP_LENGTH = 0.02  # 35~40mm for robotiq 85 2f gripper
MAX_GRASP_WIDTH = 0.085  # 85mm for robotiq 85 2f gripper
SIDE_HEIGHT_OFFSET = 0.02
OPEN_POSITION = 0.0
CLOSE_POSITION = 0.8

# Side grasp tool pose
# TOOL_POSE = Pose(euler=Euler(pitch=0))
TOOL_POSE = Pose()

#####################################
# Arm control
#####################################


def get_arm_joints(robot):
    """Get the UR5e arm joints."""
    return joints_from_names(robot, UR5E_JOINTS)


def set_arm_conf(robot, conf):
    """Set the UR5e arm joint positions."""
    set_joint_positions(robot, get_arm_joints(robot), conf)


def move_arm(robot, conf, max_velocity=1.0, force=100):
    """
    Move the UR5e arm to the specified target positions.
    :param robot: The ID of the robot in PyBullet.
    :param conf: A list of target positions for each joint in the arm.
    :param max_velocity: The maximum velocity for the joint movement.
    :param force: The maximum force to apply to the joints.
    """
    arm_joints = get_arm_joints(robot)

    # Ensure the target positions match the number of arm joints
    if len(conf) != len(arm_joints):
        raise ValueError(
            "The number of target positions must match the number of arm joints."
        )

    # Set the target position for each joint
    for i, joint_id in enumerate(arm_joints):
        p.setJointMotorControl2(
            robot,
            joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=conf[i],
            force=force,
            maxVelocity=max_velocity,
        )


def get_arm_conf(robot):
    """Get the current joint positions of the UR5e arm."""
    return get_joint_positions(robot, get_arm_joints(robot))


#####################################
# Gripper control
#####################################

# Mimic joints setting
mimic_parent = "robotiq_85_left_knuckle_joint"
mimic_children = {
    "robotiq_85_left_knuckle_joint": 1,  # include the parent joint in order to achieve the same velocity
    "robotiq_85_left_finger_tip_joint": -1,
    "robotiq_85_right_knuckle_joint": -1,
    "robotiq_85_right_finger_tip_joint": 1,
    "robotiq_85_left_inner_knuckle_joint": 1,
    "robotiq_85_right_inner_knuckle_joint": -1,
}


def setup_mimic_joints(robot):
    """
    Set up mimic joints for the Robotiq gripper using PyBullet constraints.
    :param robot: The ID of the robot in PyBullet.
    """
    # Get the parent joint ID
    mimic_parent_id = joint_from_name(robot, mimic_parent)

    # Get the child joint IDs and their multipliers
    mimic_children_multiplier = {
        joint_from_name(robot, joint_name): multiplier
        for joint_name, multiplier in mimic_children.items()
    }

    # Create constraints for mimic joints
    for joint_id, multiplier in mimic_children_multiplier.items():
        mimic_constraints = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=mimic_parent_id,
            childBodyUniqueId=robot,
            childLinkIndex=joint_id,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, -1, 0],  # Axis for the gear joint
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(
            mimic_constraints, gearRatio=-multiplier, maxForce=10000, erp=1.0
        )


def get_gripper_joints(robot):
    """Get the Robotiq gripper joints."""
    return joints_from_names(robot, ROBOTIQ_JOINTS)


def move_gripper(robot, position):
    """
    Set the Robotiq gripper joint positions, including mimic joints.
    :param robot: The ID of the robot in PyBullet.
    :param position: The target position for the parent joint.
    """
    # Set the parent joint position
    parent_joint_id = joint_from_name(robot, mimic_parent)
    p.setJointMotorControl2(
        robot,
        parent_joint_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=position,
        force=50,
        maxVelocity=2.0,
    )


# def move_gripper(robot, velocity):
#     """
#     Set the Robotiq gripper joint positions, including mimic joints.
#     :param robot: The ID of the robot in PyBullet.
#     :param position: The target position for the parent joint.
#     """
#     # Set the parent joint position
#     parent_joint_id = joint_from_name(robot, mimic_parent)
#     p.setJointMotorControl2(
#         robot,
#         parent_joint_id,
#         controlMode=p.VELOCITY_CONTROL,
#         # targetPosition=position,
#         targetVelocity=velocity,
#         force=5,
#         maxVelocity=1.0,
#     )


def open_gripper(robot):
    """Open the gripper by setting the parent joint to its maximum limit."""
    move_gripper(robot, OPEN_POSITION)


def close_gripper(robot):
    """Close the gripper."""
    move_gripper(robot, CLOSE_POSITION)


def set_gripper(robot, position):
    """
    Set the Robotiq gripper to a specific position.
    :param robot: The ID of the robot in PyBullet.
    :param position: The target position for the parent joint.
    """
    # Set the parent joint position
    parent_joint_id = joint_from_name(robot, mimic_parent)
    set_joint_position(robot, parent_joint_id, position)

    # Set the mimic children joints based on the parent joint position
    for joint_name, multiplier in mimic_children.items():
        child_joint_id = joint_from_name(robot, joint_name)
        child_position = position * multiplier
        set_joint_position(robot, child_joint_id, child_position)


def set_gripper_open(robot):
    """
    Open the Robotiq gripper by setting the parent joint to its maximum limit.
    :param robot: The ID of the robot in PyBullet.
    """
    set_gripper(robot, OPEN_POSITION)


def set_gripper_close(robot):
    """
    Close the Robotiq gripper by setting the parent joint to its minimum limit.
    :param robot: The ID of the robot in PyBullet.
    """
    set_gripper(robot, CLOSE_POSITION)
    

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


#####################################
# Grasp planning
#####################################


# Box grasp


def get_top_grasps(
    body,
    under=False,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
):
    """
    Compute top grasps for a given body based on its dimensions and pose.

    Args:
        body: The object for which grasps are being computed.
        under (bool): If True, adds additional grasps for underhand manipulation.
        tool_pose: The pose of the tool relative to the body.
        body_pose: The pose of the body in the world frame.
        max_width: The maximum width allowed for a grasp.
        grasp_length: The length of the grasp along the z-axis.

    Returns:
        list: A list of computed grasp poses.
    """
    # Approximate the body as a prism and get its center and dimensions (width, length, height)
    # Use unit_pose() to ignore the body's current pose for approximation
    center, (w, l, h) = approximate_as_prism(body, body_pose=unit_pose())
    # Transform the center point to the body's current pose in the world frame
    center = point_from_pose(body_pose)

    # Define a translation along the z-axis to position the grasp
    # The grasp is positioned at a height of (grasp_length - h / 2) above the body's base
    translate_z = Pose(point=[0, 0, grasp_length - h / 2])

    # Define a translation to move the grasp to the center of the body
    translate_center = Pose(point=center)

    # Extract the orientation component of the body's pose
    # This ensures that only the rotation is applied, ignoring the translation
    body_pose_ori = Pose(unit_point(), euler_from_quat(quat_from_pose(body_pose)))

    # Initialize an empty list to store the computed grasps
    grasps = []

    # Check if the body's width is within the allowed grasp width
    if w <= max_width:
        # Outer loop for rotate_z: handles rotation around the z-axis
        # Assuming rotate_z has two possible values (e.g., 0 and π/2)
        for j in range(2):  
            # Define a rotation around the z-axis (0° or 90°)
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + j * math.pi])
            
            # Inner loop for reflect_z and under: handles reflection and underhand grasps
            # reflect_z is dependent on the `under` parameter
            for i in range(1 + under):
                # Define a reflection around the z-axis (0° or 180°)
                reflect_z = Pose(euler=[0, math.pi + i * math.pi, 0])
                # Combine all transformations to compute the final grasp pose
                grasps += [
                    multiply(
                        tool_pose,          # Tool pose relative to the body
                        translate_center,   # Move to the body's center
                        body_pose_ori,      # Apply the body's orientation
                        reflect_z,          # Apply reflection (if underhand)
                        translate_z,        # Position the grasp along the z-axis
                        rotate_z,           # Apply rotation around the z-axis
                    )
                ]

    # Check if the body's length is within the allowed grasp width
    if l <= max_width:
        # Outer loop for rotate_z: handles rotation around the z-axis
        # Assuming rotate_z has two possible values (e.g., 0 and π)
        for j in range(2):  
            # Define a rotation around the z-axis (0° or 180°)
            rotate_z = Pose(euler=[0, 0, j * math.pi])
            
            # Inner loop for reflect_z and under: handles reflection and underhand grasps
            # reflect_z is dependent on the `under` parameter
            for i in range(1 + under):
                # Define a reflection around the z-axis (0° or 180°)
                reflect_z = Pose(euler=[0, math.pi + i * math.pi, 0])
                # Combine all transformations to compute the final grasp pose
                grasps += [
                    multiply(
                        tool_pose,          # Tool pose relative to the body
                        translate_center,   # Move to the body's center
                        body_pose_ori,      # Apply the body's orientation
                        reflect_z,          # Apply reflection (if underhand)
                        translate_z,        # Position the grasp along the z-axis
                        rotate_z,           # Apply rotation around the z-axis
                    )
                ]

    # Return the list of computed grasp poses
    return grasps


def get_side_grasps(
    body,
    under=False,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
    top_offset=SIDE_HEIGHT_OFFSET,
):
    # Approximate the body as a prism and get its center and dimensions (width, length, height)
    # Use unit_pose() to ignore the body's current pose for approximation
    center, (w, l, h) = approximate_as_prism(body, body_pose=unit_pose())
    # Transform the center point to the body's current pose in the world frame
    center = point_from_pose(body_pose)

    # Compute the translation to align with the object's center in world coordinates
    translate_center = Pose(point=center)
    
    # Extract the orientation component of the body's pose
    # This ensures that only the rotation is applied, ignoring the translation
    body_pose_ori = Pose(unit_point(), euler_from_quat(quat_from_pose(body_pose)))

    # Initialize list to store grasp poses
    grasps = []

    # Define the vertical offset from the top of the object
    # z_offset = h / 2 - top_offset
    z_offset = top_offset

    # Check if grasping is impossible (both dimensions exceed max_width)
    if w > max_width and l > max_width:
        print(
            "Cannot grasp: Both width (w) and length (l) exceed max grasp width (max_width)"
        )
        return None

    # Iterate over top (j=0) and optionally bottom (j=1) if under=True
    for j in range(1 + under):
        # Define rotation to switch between top and bottom grasps (0 or π around Z-axis)
        swap_xz = Pose(
            euler=[
                math.pi / 2 + j * math.pi,
                0,
                0,
            ]
        )

        # Grasp along width if within max_width
        if w <= max_width:
            # Position the tool at the side along the length dimension
            translate_z = Pose(point=[0, z_offset, 0])
            grasp_depth = Pose(point=[0, 0, w / 2 - grasp_length])
            for i in range(2):
                # Rotate around Z-axis to grasp from both sides (π/2 or 3π/2)
                rotate_z = Pose(euler=[0, 0, i * math.pi])
                for j in range(2):
                    rotate_x = Pose(euler=[j * math.pi, 0, 0])
                    grasps += [
                        multiply(
                            tool_pose,  # Initial tool pose
                            translate_center,  # Center alignment
                            body_pose_ori,  # Body orientation
                            swap_xz,  # Top/bottom rotation
                            translate_z,  # Position offset
                            rotate_x,
                            grasp_depth,
                            rotate_z,  # Side-to-side rotation
                        )
                    ]

        # Grasp along length if within max_width
        if l <= max_width:
            # Position the tool at the side along the length dimension
            translate_z = Pose(point=[0, z_offset, 0])
            grasp_depth = Pose(point=[0, 0, l / 2 - grasp_length])
            for i in range(2):
                # Rotate around Z-axis to grasp from both sides (π/2 or 3π/2)
                rotate_z = Pose(euler=[0, 0, i * math.pi])
                for j in range(2):
                    rotate_x = Pose(euler=[j * math.pi, math.pi / 2, 0])
                    grasps += [
                        multiply(
                            tool_pose,  # Initial tool pose
                            translate_center,  # Center alignment
                            body_pose_ori,  # Body orientation
                            swap_xz,  # Top/bottom rotation
                            translate_z,  # Position offset
                            rotate_x,
                            grasp_depth,
                            rotate_z,  # Side-to-side rotation
                        )
                    ]

    # Return the list of grasp poses
    return grasps

def get_side_grasps_multi_link(
    body,
    under=False,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
    top_offset=SIDE_HEIGHT_OFFSET,
):
    # Approximate the body as a prism and get its center and dimensions (width, length, height)
    # Use unit_pose() to ignore the body's current pose for approximation
    center, (w, h, l) = approximate_as_prism_multi_link(body, body_pose=unit_pose())
    # Transform the center point to the body's current pose in the world frame
    center = point_from_pose(body_pose)

    # Compute the translation to align with the object's center in world coordinates
    translate_center = Pose(point=center)
    
    # Extract the orientation component of the body's pose
    # This ensures that only the rotation is applied, ignoring the translation
    body_pose_ori = Pose(unit_point(), euler_from_quat(quat_from_pose(body_pose)))

    # Initialize list to store grasp poses
    grasps = []

    # Define the vertical offset from the top of the object
    # z_offset = h / 2 - top_offset
    z_offset = top_offset

    # Check if grasping is impossible (both dimensions exceed max_width)
    if w > max_width and l > max_width:
        print(
            "Cannot grasp: Both width (w) and length (l) exceed max grasp width (max_width)"
        )
        return None

    # Iterate over top (j=0) and optionally bottom (j=1) if under=True
    for j in range(1 + under):
        # Define rotation to switch between top and bottom grasps (0 or π around Z-axis)
        swap_xz = Pose(
            euler=[
                math.pi / 2 + j * math.pi,
                0,
                0,
            ]
        )

        # Grasp along width if within max_width
        if w <= max_width:
            # Position the tool at the side along the length dimension
            translate_z = Pose(point=[0, z_offset, 0])
            grasp_depth = Pose(point=[0, 0, w / 2 - grasp_length])
            for i in range(2):
                # Rotate around Z-axis to grasp from both sides (π/2 or 3π/2)
                rotate_z = Pose(euler=[0, 0, i * math.pi])
                for j in range(2):
                    rotate_x = Pose(euler=[j * math.pi, 0, 0])
                    grasps += [
                        multiply(
                            tool_pose,  # Initial tool pose
                            translate_center,  # Center alignment
                            body_pose_ori,  # Body orientation
                            swap_xz,  # Top/bottom rotation
                            translate_z,  # Position offset
                            rotate_x,
                            grasp_depth,
                            rotate_z,  # Side-to-side rotation
                        )
                    ]

        # Grasp along length if within max_width
        if l <= max_width:
            # Position the tool at the side along the length dimension
            translate_z = Pose(point=[0, z_offset, 0])
            grasp_depth = Pose(point=[0, 0, l / 2 - grasp_length])
            for i in range(2):
                # Rotate around Z-axis to grasp from both sides (π/2 or 3π/2)
                rotate_z = Pose(euler=[0, 0, i * math.pi])
                for j in range(2):
                    rotate_x = Pose(euler=[j * math.pi, math.pi / 2, 0])
                    grasps += [
                        multiply(
                            tool_pose,  # Initial tool pose
                            translate_center,  # Center alignment
                            body_pose_ori,  # Body orientation
                            swap_xz,  # Top/bottom rotation
                            translate_z,  # Position offset
                            rotate_x,
                            grasp_depth,
                            rotate_z,  # Side-to-side rotation
                        )
                    ]

    # Return the list of grasp poses
    return grasps



# Cylinder grasps


def get_top_cylinder_grasps(
    body,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
):
    # Apply transformations right to left on object pose
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, height / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose) - center)
    if max_width < diameter:
        return
    while True:
        theta = random.uniform(0, 2 * np.pi)
        rotate_z = Pose(euler=[0, 0, theta])
        yield multiply(
            tool_pose, translate_z, rotate_z, reflect_z, translate_center, body_pose
        )


def get_side_cylinder_grasps(
    body,
    under=False,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
    top_offset=SIDE_HEIGHT_OFFSET,
):
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    translate_center = Pose(point_from_pose(body_pose) - center)
    # x_offset = 0
    x_offset = height / 2 - top_offset
    if max_width < diameter:
        return
    while True:
        theta = random.uniform(0, 2 * np.pi)
        translate_rotate = (
            [x_offset, 0, diameter / 2 - grasp_length],
            quat_from_euler([theta, 0, 0]),
        )
        for j in range(1 + under):
            swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
            yield multiply(
                tool_pose, translate_rotate, swap_xz, translate_center, body_pose
            )


def get_edge_cylinder_grasps(
    body,
    under=False,
    tool_pose=TOOL_POSE,
    body_pose=unit_pose(),
    grasp_length=GRASP_LENGTH,
):
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    translate_yz = Pose(point=[0, diameter / 2, height / 2 - grasp_length])
    reflect_y = Pose(euler=[0, math.pi, 0])
    translate_center = Pose(point=point_from_pose(body_pose) - center)
    while True:
        theta = random.uniform(0, 2 * np.pi)
        rotate_z = Pose(euler=[0, 0, theta])
        for i in range(1 + under):
            rotate_under = Pose(euler=[0, 0, i * math.pi])
            yield multiply(
                tool_pose,
                rotate_under,
                translate_yz,
                rotate_z,
                reflect_y,
                translate_center,
                body_pose,
            )


#####################################
# Collision for grasps detection
#####################################


def close_until_collision(
    robot,
    bodies=[],
    num_steps=25,
):
    """
    Close the gripper until a collision is detected.
    :param robot: The ID of the robot in PyBullet.
    :param bodies: List of bodies to check for collision.
    :param num_steps: Number of steps to interpolate between open and closed configurations.
    :return: The last safe position before collision, or None if collision occurs at the start.
    """
    # Get the gripper joints
    gripper_joints = get_gripper_joints(robot)
    if not gripper_joints:
        return None

    # Get the moving links of the gripper
    collision_links = frozenset(get_moving_links(robot, gripper_joints))

    # Interpolate between open and closed positions
    for i in range(num_steps + 1):
        position = OPEN_POSITION + (CLOSE_POSITION - OPEN_POSITION) * (i / num_steps)
        set_gripper(robot, position)

        # Check for collisions
        if any(pairwise_collision((robot, collision_links), body) for body in bodies):
            if i == 0:
                return None  # Collision at the start
            return OPEN_POSITION + (CLOSE_POSITION - OPEN_POSITION) * (
                (i - 1) / num_steps
            )  # Last safe position

    return CLOSE_POSITION  # Return the fully closed position if no collision


def compute_grasp_width(robot, body, grasp_pose, **kwargs):
    """
    Compute the grasp width for a given object and grasp pose.
    :param robot: The ID of the robot in PyBullet.
    :param body: The ID of the object to grasp.
    :param grasp_pose: The grasp pose relative to the tool frame.
    :return: The grasp width (position), or None if no valid grasp is found.
    """
    # Get the tool pose and set the object pose
    tool_link = link_from_name(robot, UR5E_TOOL_FRAME)
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)

    # Compute the grasp width using close_until_collision
    return close_until_collision(robot, bodies=[body], **kwargs)


# def create_gripper(robot, visual=True):
#     """
#     Create a clone of the gripper for visualization or collision detection.
#     :param robot: The ID of the robot in PyBullet.
#     :param visual: Whether to create a visual copy (default: True).
#     :return: The ID of the cloned gripper.
#     """
#     # Get all link IDs for the gripper
#     link_ids = [link_from_name(robot, link_name) for link_name in ROBOTIQ_LINKS]

#     # Clone the gripper using the link IDs
#     gripper = clone_body(robot, links=link_ids, visual=visual, collision=True)

#     # Set the gripper to transparent if not visual
#     if not visual:
#         set_all_color(gripper, TRANSPARENT)

#     return gripper


def create_gripper(robot, visual=True):
    """
    Create a clone of the gripper for visualization or collision detection.
    :param robot: The ID of the robot in PyBullet.
    :param visual: Whether to create a visual copy (default: True).
    :return: The ID of the cloned gripper.
    """
    # Define the root link of the gripper
    root_link_name = ROBOTIQ_LINKS[0]  # Root link of the gripper
    root_link_id = link_from_name(robot, root_link_name)

    # Get the subtree of links starting from the root link
    link_ids = get_link_subtree(robot, root_link_id)

    # Clone the gripper using the subtree links
    gripper = clone_body(robot, links=link_ids, visual=visual, collision=True)

    # Set the gripper to transparent if not visual
    if not visual:
        set_all_color(gripper, TRANSPARENT)
    else:
        set_all_color(gripper, [0.5, 0.5, 0.5, 0.5])

    return gripper


#####################################
# Collision check considering allowed collisions
#####################################

#####################################
# Collision for grasps detection
#####################################


# def pairwise_collision_with_allowed(body1, body2, allowed_collisions=None, **kwargs):
#     """
#     Check for collisions between two bodies, considering a set of allowed collisions.

#     Args:
#         body1: The first body.
#         body2: The second body.
#         allowed_collisions: A set of tuples, where each tuple contains two (body_id, link_id) pairs.
#                             These pairs represent collisions that are allowed and should be ignored.
#         **kwargs: Additional arguments to pass to the collision detection functions.

#     Returns:
#         bool: True if there is a collision between the bodies that is not allowed, False otherwise.
#     """
#     if allowed_collisions is None:
#         allowed_collisions = set()

#     if isinstance(body1, tuple) or isinstance(body2, tuple):
#         body1, links1 = expand_links(body1)
#         body2, links2 = expand_links(body2)
#     else:
#         links1 = get_all_links(body1)
#         links2 = get_all_links(body2)

#     for link1, link2 in product(links1, links2):
#         if (body1 == body2) and (link1 == link2):
#             continue

#         # Check if the collision is allowed
#         collision_pair = ((body1, link1), (body2, link2))
#         reverse_collision_pair = ((body2, link2), (body1, link1))

#         if (
#             collision_pair in allowed_collisions
#             or reverse_collision_pair in allowed_collisions
#         ):
#             continue

#         if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
#             return True

#     return False


def pairwise_collision_with_allowed(body1, body2, allowed_collisions=None, **kwargs):
    """
    Check for collisions between two bodies, considering a set of allowed collisions.

    Args:
        body1: The first body.
        body2: The second body.
        allowed_collisions: A set of tuples, where each tuple contains two (body_id, link_id) pairs.
                            These pairs represent collisions that are allowed and should be ignored.
        **kwargs: Additional arguments to pass to the collision detection functions.

    Returns:
        bool: True if there is a collision between the bodies that is not allowed, False otherwise.
    """
    if allowed_collisions is None:
        allowed_collisions = set()

    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
    else:
        links1 = get_all_links(body1)
        links2 = get_all_links(body2)

    if len(links1) >= 2:
        links1 = links1[1:]

    if len(links2) >= 2:
        links2 = links2[1:]

    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue

        collision_pair = ((body1, link1), (body2, link2))
        reverse_collision_pair = ((body2, link2), (body1, link1))

        if (
            collision_pair in allowed_collisions
            or reverse_collision_pair in allowed_collisions
        ):
            continue

        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True

    return False
