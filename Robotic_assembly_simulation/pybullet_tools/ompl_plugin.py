import numpy as np
from termcolor import cprint
from ompl import base as ob
from ompl import geometric as og
import pybullet as p

from pybullet_planning import (
    get_sample_fn,
    get_distance_fn,
    get_extend_fn,
    set_joint_positions,
    get_collision_fn,
    pairwise_collision,
    remove_body,
    check_initial_end,
    get_aabb,
    get_aabb_vertices,
    convex_hull,
    create_mesh,
)
from pybullet_tools.utils import (
    buffer_aabb,
    get_self_link_pairs,
    pairwise_link_collision,
)

# Constants
MAX_DISTANCE = 0.0
DYNAMIC_RES_RATIO = 0.5
CONVEX_BUFFER = 0.1
RESOLUTIONS = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
WEIGHTS = np.reciprocal(RESOLUTIONS)
DEFAULT_PLANNING_TIME = 1e6  # Default planning time of 1 million seconds

class PbStateSpace(ob.RealVectorStateSpace):
    """Custom state space for PyBullet integration with OMPL"""
    def __init__(self, num_dim):
        super().__init__(num_dim)
        self.num_dim = num_dim

    def allocStateSampler(self):
        return self.allocDefaultStateSampler()

def get_pairs(iterator):
    """Generate consecutive pairs from an iterator.
    Args:
        iterator: Any iterable object
    Yields:
        Pairs of consecutive elements (current, next)
    """
    iterator = iter(iterator)
    try:
        last = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield last, current
        last = current
        
def create_convex_bounding(bodies=None, buffer=CONVEX_BUFFER, color=(0.106, 0.655, 0.518, 0.2)):
    """Create a convex hull around given bodies for simplified collision checking.
    Args:
        bodies: List of PyBullet body IDs to enclose
        buffer: Padding around the bodies
        color: RGBA color for visualization
    Returns:
        PyBullet body ID of the convex mesh
    """
    bodies_nodes = []
    if bodies is None:
        return None
    
    for body in bodies:
        aabb = get_aabb(body)
        aabb = buffer_aabb(aabb, buffer)
        body_nodes = get_aabb_vertices(aabb)
        bodies_nodes.extend(body_nodes)
    
    bodies_convex_hull = convex_hull(bodies_nodes)
    bodies_convex_mesh = create_mesh(bodies_convex_hull, under=True, color=color)
    return bodies_convex_mesh

def dynamic_interpolate_path(robot, joints, path, collision_fn, convex_bounding, attachments, 
                           resolutions, dynamic_res_ratio=DYNAMIC_RES_RATIO, self_collisions=True, 
                           disabled_collisions=None):
    """Refine path with adaptive resolution based on proximity to obstacles.
    Args:
        robot: PyBullet robot ID
        joints: List of joint indices
        path: Initial path to refine
        collision_fn: Collision checking function
        convex_bounding: Convex hull body ID for simplified collision checking
        attachments: List of attached objects
        resolutions: Base resolution values for path interpolation
        dynamic_res_ratio: Ratio for finer resolution near obstacles
        self_collisions: Flag to check self-collisions
        disabled_collisions: Pairs of links to ignore in collision checking
    Returns:
        Refined path as list of configurations
    """
    if disabled_collisions is None:
        disabled_collisions = {}
        
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    fine_extend_fn = get_extend_fn(robot, joints, resolutions=dynamic_res_ratio * resolutions)
    refined_path = []

    def convex_mesh_detection(q):
        """Check collision with convex bounding and update attachments."""
        set_joint_positions(robot, joints, q)
        for attachment in attachments:
            attachment.assign()
        collision = (convex_bounding is not None) and (
            pairwise_collision(robot, convex_bounding) or 
            any(pairwise_collision(attachment.child, convex_bounding) for attachment in attachments))
        return q, collision

    for q1, q2 in get_pairs(iter(path)):
        refined_path.append(q1)
        extended_segment = list(extend_fn(q1, q2))
        if not extended_segment:
            continue
        
        # Check collisions along the segment
        collision_checks = list(map(convex_mesh_detection, extended_segment))
        for (q_prev, c_prev), (q_curr, c_curr) in get_pairs(collision_checks):
            if c_prev and c_curr:  # If both points are near obstacles
                fine_segment = list(fine_extend_fn(q_prev, q_curr))
                refined_path.extend(fine_segment[:-1])
            elif not collision_fn(q_curr):  # If current point is collision-free
                refined_path.append(q_curr)

    # Ensure goal configuration is included
    if path and refined_path[-1] != path[-1]:
        refined_path.append(path[-1])

    return refined_path

def compute_motion(robot, joints, start_conf, end_conf, obstacles=None, attachments=None,
                  assembled_elements=None, self_collisions=True, weights=WEIGHTS, 
                  resolutions=RESOLUTIONS, disabled_collisions=None, custom_limits=None, 
                  algorithm="RRTConnect", buffer=CONVEX_BUFFER, max_distance=MAX_DISTANCE, 
                  planning_time=DEFAULT_PLANNING_TIME, **kwargs):
    """Compute motion plan between start and end configurations.
    Args:
        robot: PyBullet robot ID
        joints: List of joint indices
        start_conf: Start configuration
        end_conf: Goal configuration
        obstacles: List of obstacle body IDs
        attachments: List of attached objects
        assembled_elements: List of bodies to create convex hull around
        self_collisions: Flag to check self-collisions
        weights: Distance metric weights
        resolutions: Interpolation resolutions
        disabled_collisions: Pairs of links to ignore in collision checking
        custom_limits: Custom joint limits
        algorithm: OMPL planner algorithm name
        buffer: Padding for convex hull creation
        max_distance: Maximum allowed distance between collision checks
        planning_time: Maximum planning time in seconds
    Returns:
        List of configurations representing the path, or None if planning fails
    """
    if obstacles is None:
        obstacles = []
    if attachments is None:
        attachments = []
    if disabled_collisions is None:
        disabled_collisions = {}
    if custom_limits is None:
        custom_limits = {}

    assert len(joints) == len(end_conf)

    # Set initial state and update attachments
    set_joint_positions(robot, joints, start_conf)
    for attachment in attachments:
        attachment.assign()

    # Create convex hull for simplified collision checking if needed
    convex_bounding = None
    if assembled_elements:
        convex_bounding = create_convex_bounding(assembled_elements, buffer=buffer)

    # Set default weights if not provided
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)

    # Setup planning components
    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    collision_fn = get_collision_fn(
        robot, joints, obstacles=obstacles, attachments=attachments, 
        self_collisions=self_collisions, disabled_collisions=disabled_collisions,
        custom_limits=custom_limits, max_distance=max_distance)

    # Configure OMPL state space
    num_dim = len(joints)
    space = PbStateSpace(num_dim)
    bounds = ob.RealVectorBounds(num_dim)
    for i, joint in enumerate(joints):
        joint_info = p.getJointInfo(robot, joint)
        bounds.setLow(i, joint_info[8])  # Joint lower limit
        bounds.setHigh(i, joint_info[9])  # Joint upper limit
    space.setBounds(bounds)

    # Setup SimpleSetup for planning
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    def is_state_valid(state):
        """Check if state is valid considering collisions and limits."""
        q = [state[i] for i in range(num_dim)]
        set_joint_positions(robot, joints, q)
        for attachment in attachments:
            attachment.assign()

        # Check self-collisions if enabled
        if self_collisions:
            check_link_pairs = get_self_link_pairs(robot, joints, disabled_collisions)
            for link1, link2 in check_link_pairs:
                if pairwise_link_collision(robot, link1, robot, link2):
                    return False

        # Check collisions with convex bounding if it exists
        if convex_bounding is not None:
            if (pairwise_collision(robot, convex_bounding) or 
                any(pairwise_collision(attachment.child, convex_bounding) for attachment in attachments)):
                return not collision_fn(q)
            return True
        return not collision_fn(q)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))

    # Select planning algorithm
    if algorithm == "PRM":
        planner = og.PRM(si)
    elif algorithm == "LazyPRM":
        planner = og.LazyPRM(si)
    elif algorithm == "LazyPRMstar":
        planner = og.LazyPRMstar(si)
    elif algorithm == "SPARS":
        planner = og.SPARS(si)
    elif algorithm == "SPARStwo":
        planner = og.SPARStwo(si)
    elif algorithm == "EST":
        planner = og.EST(si)
    elif algorithm == "BiEST":
        planner = og.BiEST(si)
    elif algorithm == "ProjEST":
        planner = og.ProjEST(si)
    elif algorithm == "KPIECE1":
        planner = og.KPIECE1(si)
    elif algorithm == "BKPIECE1":
        planner = og.BKPIECE1(si)
    elif algorithm == "LBKPIECE1":
        planner = og.LBKPIECE1(si)
    elif algorithm == "PDST":
        planner = og.PDST(si)
    elif algorithm == "RRT":
        planner = og.RRT(si)
    elif algorithm == "RRTConnect":
        planner = og.RRTConnect(si)
        planner.setRange(0.05)  # Set step size
    elif algorithm == "LazyRRT":
        planner = og.LazyRRT(si)
    elif algorithm == "TRRT":
        planner = og.TRRT(si)
    elif algorithm == "RRTstar":
        planner = og.RRTstar(si)
    elif algorithm == "RRTXstatic":
        planner = og.RRTXstatic(si)
    elif algorithm == "RRTsharp":
        planner = og.RRTsharp(si)
    elif algorithm == "LBTRRT":
        planner = og.LBTRRT(si)
    elif algorithm == "LazyLBTRRT":
        planner = og.LazyLBTRRT(si)
    elif algorithm == "InformedRRTstar":
        planner = og.InformedRRTstar(si)
    elif algorithm == "SORRTstar":
        planner = og.SORRTstar(si)
    elif algorithm == "STRRTstar":
        planner = og.STRRTstar(si)
    elif algorithm == "BITstar":
        planner = og.BITstar(si)
    elif algorithm == "ABITstar":
        planner = og.ABITstar(si)
    elif algorithm == "AITstar":
        planner = og.AITstar(si)
    elif algorithm == "EITstar":
        planner = og.EITstar(si)
    elif algorithm == "SBL":
        planner = og.SBL(si)
    elif algorithm == "STRIDE":
        planner = og.STRIDE(si)
    elif algorithm == "FMT":
        planner = og.FMT(si)
    elif algorithm == "BFMT":
        planner = og.BFMT(si)
    elif algorithm == "SST":
        planner = og.SST(si)
    else:
        cprint(f"Algorithm {algorithm} not supported, defaulting to RRTConnect", "yellow")
        planner = og.RRTConnect(si)
        planner.setRange(0.05)
    ss.setPlanner(planner)

    # Set start and goal states
    start_state = ob.State(space)
    goal_state = ob.State(space)
    for i in range(num_dim):
        start_state[i] = start_conf[i]
        goal_state[i] = end_conf[i]

    ss.setStartAndGoalStates(start_state, goal_state, 1e-3)  # Set goal tolerance

    # Validate start and goal states
    if not (check_initial_end(start_conf, end_conf, collision_fn) and 
            is_state_valid(start_state) and is_state_valid(goal_state)):
        if convex_bounding is not None:
            remove_body(convex_bounding)
        cprint("No motion plan found due to invalid start or goal state", "red")
        return None
    
    # Attempt to solve the planning problem
    simplifier = og.PathSimplifier(si)
    solved = ss.solve(planning_time)
    path = None
    
    if solved:
        sol_path_geometric = ss.getSolutionPath()
        simplifier.smoothBSpline(sol_path_geometric, maxSteps=5, minChange=1e-3)
        sol_path_states = sol_path_geometric.getStates()
        raw_path = [[state[i] for i in range(num_dim)] for state in sol_path_states]
        
        # Refine path with dynamic interpolation
        path = dynamic_interpolate_path(
            robot, joints, raw_path, collision_fn, convex_bounding, 
            attachments, resolutions, dynamic_res_ratio=DYNAMIC_RES_RATIO,
            self_collisions=self_collisions, disabled_collisions=disabled_collisions)
        
        # Verify goal achievement
        tolerance = 1e-3
        final_conf = path[-1]
        if not np.allclose(final_conf, end_conf, atol=tolerance):
            path = None
            cprint("No valid motion plan found (path does not reach goal)", "red")
        else:
            cprint(f"Found solution with {len(path)} waypoints", "green")
    else:
        cprint("No motion plan found", "red")

    # Clean up convex bounding if created
    if convex_bounding is not None:
        remove_body(convex_bounding)

    return path