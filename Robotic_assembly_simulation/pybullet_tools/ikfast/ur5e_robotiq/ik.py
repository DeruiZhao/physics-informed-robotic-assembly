from ..ikfast import *  # For legacy purposes

import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from ..utils import (
    get_ik_limits,
    compute_forward_kinematics,
    select_solution,
    USE_ALL,
    USE_CURRENT,
)
from ...ur5e_robotiq_utils import (
    UR5E_TOOL_FRAME,
    UR5E_JOINTS,
    get_arm_joints,
    pairwise_collision_with_allowed,
)
from ...utils import (
    matrix_from_quat,
    multiply,
    get_link_pose,
    link_from_name,
    get_joint_positions,
    joint_from_name,
    invert,
    get_custom_limits,
    all_between,
    inverse_kinematics,
    set_joint_positions,
    get_joint_positions,
    pairwise_collision,
    get_all_links,
    get_joints,
)
from ...ikfast.utils import IKFastInfo
from ikfast_ur5e_robotiq import get_ik, get_fk

BASE_FRAME = "base_link"
TOOL_FRAME = "gripper_center_link"

UR5E_ROBOTIQ_URDF = "urdf/ur5e_robotiq.urdf"

UR5E_ROBOTIQ_IKFAST_INFO = IKFastInfo(
    module_name="ur5e_robotiq.ikfast_ur5e_robotiq",
    base_link="base_link",
    ee_link="gripper_center_link",
    free_joints=[],
)

#####################################


#####################################


def is_ik_compiled():
    """Check if the IKFast module for UR5e with Robotiq gripper is compiled and available."""
    try:
        import ikfast_ur5e_robotiq

        return True
    except ImportError:
        return False


def ikfast_compute_inverse_kinematics(
    ik_fn, pose, position_noise=0.001, quaternion_noise=0.0005
):
    """
    Compute inverse kinematics solutions using IKFast. If no solutions are found, add small perturbations
    to the target pose and retry once. If still no solutions, return an empty list.

    Args:
        ik_fn: The IKFast inverse kinematics function.
        pose: A tuple of (position, quaternion) representing the target pose in the base frame.
        position_noise: Magnitude of noise to add to the position (in meters).
        quaternion_noise: Magnitude of noise to add to the quaternion.

    Returns:
        A list of joint configurations (solutions) or an empty list if no solutions are found after retrying.
    """
    pos = np.array(pose[0])  # Target position
    quat = np.array(pose[1])  # Target quaternion

    # First attempt: Try solving without any noise
    rot = matrix_from_quat(tuple(quat))  # Convert quaternion to rotation matrix
    solutions = ik_fn(pos, rot, [])  # Compute IK solutions
    if solutions is not None:
        return solutions  # Return solutions if found

    # Second attempt: Add noise to the pose and try again
    noisy_quat = quat + np.random.randn(4) * quaternion_noise
    noisy_quat /= np.linalg.norm(noisy_quat)  # Normalize the quaternion
    noisy_pos = pos + np.random.randn(3) * position_noise

    rot = matrix_from_quat(
        tuple(noisy_quat)
    )  # Convert noisy quaternion to rotation matrix
    solutions = ik_fn(noisy_pos, rot, [])  # Compute IK solutions with noisy pose
    if solutions is not None:
        return solutions  # Return solutions if found

    # If no solutions are found after retrying, return an empty list
    return []


def get_ik_generator(robot, ik_pose, custom_limits={}):
    """
    Create a generator yielding valid inverse kinematics solutions for the UR5e arm.

    Args:
        robot: The PyBullet robot ID.
        ik_pose: The target pose for the tool (position, quaternion) in the base frame.
        custom_limits: Dictionary of custom joint limits (joint_name: (min, max)).

    Yields:
        Lists of joint configurations that satisfy the IK and joint limits.
    """
    # Get the joints involved in IK for the specified tool link
    arm_joints = get_arm_joints(robot)
    # Retrieve joint limits, applying custom limits if provided
    min_limits, max_limits = get_custom_limits(robot, arm_joints, custom_limits)

    while True:
        # Compute IK solutions using IKFast
        confs = ikfast_compute_inverse_kinematics(get_ik, ik_pose)
        # Filter solutions to ensure they are within joint limits
        solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
        yield solutions
        if all(min_limits[i] == max_limits[i] for i in range(len(arm_joints))):
            break


def sample_tool_ik(robot, tool_pose, nearby_conf=USE_CURRENT, max_attempts=25, **kwargs):
    """
    Sample a valid IK solution for a given tool pose.

    Args:
        robot: The PyBullet robot ID.
        tool_pose: The target pose for the tool (position, quaternion) in the base frame.
        nearby_conf: Strategy for selecting solutions (USE_ALL or USE_CURRENT for current pose).
        max_attempts: Maximum number of IK attempts before giving up.
        **kwargs: Additional arguments passed to get_ik_generator.

    Returns:
        A joint configuration (list of floats) or None if no valid solution is found.
    """
    # Create an IK solution generator
    generator = get_ik_generator(robot, tool_pose, **kwargs)

    # Get the joints involved in IK for the tool link
    arm_joints = get_arm_joints(robot)

    # Attempt to find a valid solution up to max_attempts times
    for _ in range(max_attempts):
        try:
            solutions = next(generator)
            if solutions:
                # Select a solution based on the nearby_conf strategy
                return select_solution(
                    robot, arm_joints, solutions, nearby_conf=nearby_conf
                )
        except StopIteration:
            break
    return None


def ur5e_robotiq_inverse_kinematics(
    robot,
    link,
    target_pose,
    obstacles=[],
    custom_limits={},
    use_pybullet=False,
    allowed_collisions=[],
    **kwargs,
):
    """
    Compute inverse kinematics for the UR5e arm with a Robotiq gripper.

    Args:
        robot: The PyBullet robot ID.
        link: The target link (e.g., gripper_center_link) for the IK computation.
        target_pose: The target pose (position, quaternion) for the specified link in the base frame.
        obstacles: List of PyBullet body IDs to check for collisions.
        custom_limits: Dictionary of custom joint limits (joint_name: (min, max)).
        use_pybullet: Boolean to use PyBullet's built-in IK solver instead of IKFast.
        **kwargs: Additional arguments passed to sample_tool_ik.

    Returns:
        A joint configuration (list of floats) for the arm joints, or None if no valid solution exists.
    """
    # Define the arm link and joints for IK
    arm_link = link  # The target link for IK
    arm_joints = get_arm_joints(robot)

    # Use IKFast if compiled and not overridden by use_pybullet
    if not use_pybullet and is_ik_compiled():
        # Sample an IK solution using IKFast
        conf = sample_tool_ik(
            robot,
            target_pose,
            custom_limits=custom_limits,
            **kwargs,
        )
        if conf is None:
            return None
        # Apply the configuration to the robot
        set_joint_positions(robot, arm_joints, conf)
    else:
        # Fall back to PyBullet's IK solver
        conf = inverse_kinematics(
            robot, arm_link, target_pose, custom_limits=custom_limits
        )
        if conf is None:
            return None
        # Apply the configuration to the robot
        set_joint_positions(robot, arm_joints, conf[: len(arm_joints)])

    # Check for collisions with obstacles, skipping allowed collisions
    for obstacle in obstacles:
        if pairwise_collision_with_allowed(
            robot,
            obstacle,
            allowed_collisions=allowed_collisions,
            **kwargs,
        ):
            return None

    # Return the current joint positions of the arm
    return get_joint_positions(robot, arm_joints)
