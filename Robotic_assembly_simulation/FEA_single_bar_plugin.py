import openseespy.opensees as ops
import numpy as np
import opsvis as opsv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

def auto_geom_transf(transf_tag, i_node, j_node):
    """
    Automatically determine the geometric transformation vector based on element direction.
    """
    xi, yi, zi = ops.nodeCoord(i_node)
    xj, yj, zj = ops.nodeCoord(j_node)
    
    x_axis = np.array([xj - xi, yj - yi, zj - zi])
    x_axis /= np.linalg.norm(x_axis)
    
    global_axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    selected_axis = global_axes[np.argmin([abs(np.dot(x_axis, ax)) for ax in global_axes])]
    
    vecxz = selected_axis - np.dot(selected_axis, x_axis) * x_axis
    vecxz /= np.linalg.norm(vecxz)
    
    ops.geomTransf('Corotational', transf_tag, *vecxz)
    return vecxz

def calculate_rotational_stiffness(E, I, L, n_segments):
    """Calculate rotational stiffness for multi-segment model."""
    return n_segments * (E * I) / L

def compute_relative_rotation(parent_start, parent_end, child_start, child_end):
    """
    Compute relative rotation between two segments.
    
    Args:
        parent_start: Starting coordinates of parent segment
        parent_end: Ending coordinates of parent segment
        child_start: Starting coordinates of child segment
        child_end: Ending coordinates of child segment
    """
    parent_vec = (parent_end - parent_start) / np.linalg.norm(parent_end - parent_start)
    child_vec = (child_end - child_start) / np.linalg.norm(child_end - child_start)
    
    cross_prod = np.cross(parent_vec, child_vec)
    cross_norm = np.linalg.norm(cross_prod)
    dot_product = np.clip(np.dot(parent_vec, child_vec), -1.0, 1.0)
    
    if cross_norm < 1e-6:
        return [0.0, 0.0, 0.0, 1.0] if dot_product > 0 else [0.0, 0.0, 1.0, 0.0]
    
    axis = cross_prod / cross_norm
    angle = np.arccos(dot_product)
    quat = Rotation.from_rotvec(axis * angle).as_quat()
    
    if quat[-1] < 0:
        quat = -np.array(quat)
    
    return list(quat / np.linalg.norm(quat))

def single_bar_analysis(structure_data, orientation_quat=[0, 0, 0, 1], n_segments=7, scale=100.0, plot=False):
    """Main analysis function with node translation to move midpoint to origin and rotation to xz-plane."""
    if n_segments % 2 == 0:
        raise ValueError("n_segments must be odd")

    # Extract node coordinates
    nodes = structure_data["nodes"]
    if len(nodes) != 2:
        raise ValueError("Exactly two nodes must be provided")
    
    start_coord = np.array([nodes[0]["x"], nodes[0]["y"], nodes[0]["z"]])
    end_coord = np.array([nodes[1]["x"], nodes[1]["y"], nodes[1]["z"]])
    
    # Calculate midpoint and translate structure so that midpoint is at (0, 0, 0)
    midpoint = (start_coord + end_coord) / 2
    translation_vector = -midpoint
    start_coord += translation_vector
    end_coord += translation_vector
    
    # Calculate midpoint and vector between nodes (after translation)
    midpoint = (start_coord + end_coord) / 2  # Should be close to (0, 0, 0)
    bar_vector = end_coord - start_coord
    
    # Project bar vector onto xy-plane (z=0) to determine rotation needed
    xy_projection = np.array([bar_vector[0], bar_vector[1], 0])
    if np.linalg.norm(xy_projection) > 1e-6:
        # Normalize projection
        xy_projection /= np.linalg.norm(xy_projection)
        # Target direction is along x-axis ([1, 0, 0])
        target = np.array([1, 0, 0])
        # Compute angle to rotate xy_projection to x-axis around z-axis
        cos_theta = np.dot(xy_projection, target)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        # Determine rotation direction using cross product
        cross = np.cross(xy_projection, target)
        if cross[2] < 0:
            angle = -angle
        # Create rotation around z-axis
        rotation = Rotation.from_euler('z', angle)
    else:
        # If no xy-projection, no rotation needed
        rotation = Rotation.from_quat([0, 0, 0, 1])
    
    # Apply rotation to both nodes around midpoint
    start_rel = start_coord - midpoint
    end_rel = end_coord - midpoint
    rotated_start_rel = rotation.apply([start_rel])[0]
    rotated_end_rel = rotation.apply([end_rel])[0]
    rotated_start = rotated_start_rel + midpoint
    rotated_end = rotated_end_rel + midpoint
    
    # Update structure_data with translated and rotated coordinates
    structure_data["nodes"] = [
        {"id": 1, "x": rotated_start[0], "y": rotated_start[1], "z": rotated_start[2]},
        {"id": 2, "x": rotated_end[0], "y": rotated_end[1], "z": rotated_end[2]}
    ]
    
    # Original analysis code continues
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    E, G = 2.1e11, 8.0e10
    b = 0.004
    A, I, J = b**2, b**4/12, 2.25*(b**4)/6
    rho, g = 7850, 9.81
    
    BASE_NODE_TAG = 1
    BASE_BEAM_ELEMENT_TAG = 1
    BASE_ZERO_LENGTH_TAG = 1000
    BASE_TRANSF_TAG = 1
    BASE_MATERIAL_TAG = 1
    
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG, E)
    
    start_coord = np.array([structure_data["nodes"][0]["x"], structure_data["nodes"][0]["y"], structure_data["nodes"][0]["z"]])
    end_coord = np.array([structure_data["nodes"][1]["x"], structure_data["nodes"][1]["y"], structure_data["nodes"][1]["z"]])
    length = np.linalg.norm(end_coord - start_coord)
    
    segment_nodes = []
    reference_nodes = []
    current_node_tag = BASE_NODE_TAG
    
    for i in range(n_segments):
        xi_start = start_coord + (end_coord - start_coord) * i / n_segments
        ops.node(current_node_tag, *xi_start)
        start_node_tag = current_node_tag
        current_node_tag += 1
        
        xi_end = start_coord + (end_coord - start_coord) * (i + 1) / n_segments
        ops.node(current_node_tag, *xi_end)
        end_node_tag = current_node_tag
        current_node_tag += 1
        
        segment_nodes.append([start_node_tag, end_node_tag])
        
        if i < n_segments - 1:
            ops.node(current_node_tag, *xi_end)
            reference_nodes.append(current_node_tag)
            current_node_tag += 1
    
    k_rot = calculate_rotational_stiffness(4.2e8, I, length, n_segments)
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG+1, k_rot)
    
    for i in range(n_segments - 1):
        ref_node = reference_nodes[i]
        node_a = segment_nodes[i][1]
        node_b = segment_nodes[i + 1][0]
        
        ops.element('zeroLength', BASE_ZERO_LENGTH_TAG,
                   node_a, ref_node,
                   '-mat', *[BASE_MATERIAL_TAG]*3 + [BASE_MATERIAL_TAG+1]*3,
                   '-dir', *range(1,7))
        BASE_ZERO_LENGTH_TAG += 1
        
        ops.element('zeroLength', BASE_ZERO_LENGTH_TAG,
                   node_b, ref_node,
                   '-mat', *[BASE_MATERIAL_TAG]*3 + [BASE_MATERIAL_TAG+1]*3,
                   '-dir', *range(1,7))
        BASE_ZERO_LENGTH_TAG += 1
    
    middle_idx = n_segments // 2
    ops.fix(reference_nodes[middle_idx-1], 1, 1, 1, 1, 1, 1)
    ops.fix(reference_nodes[middle_idx], 1, 1, 1, 1, 1, 1)
    
    auto_geom_transf(BASE_TRANSF_TAG, segment_nodes[0][0], segment_nodes[0][1])
    for i in range(n_segments):
        ops.element('elasticBeamColumn', BASE_BEAM_ELEMENT_TAG, 
                   segment_nodes[i][0], segment_nodes[i][1], 
                   A, 1e3*E, G, J, I, I, BASE_TRANSF_TAG)
        BASE_BEAM_ELEMENT_TAG += 1
    
    weight_per_node = (rho * A * length * g) / (2 * n_segments)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for seg in segment_nodes:
        for node in seg:
            ops.load(node, 0, 0, -weight_per_node, 0, 0, 0)
    
    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1)
    ops.analysis('Static')
    ops.test('NormDispIncr', 1.0e-6, 100)
    
    if ops.analyze(10) != 0:
        ops.wipe()
        raise RuntimeError("Analysis failed to converge")
    
    deformed_coords = []
    for start, end in segment_nodes:
        coords_start = np.array(ops.nodeCoord(start)) + np.array([ops.nodeDisp(start, dof) for dof in range(1,4)]) * scale
        coords_end = np.array(ops.nodeCoord(end)) + np.array([ops.nodeDisp(end, dof) for dof in range(1,4)]) * scale
        deformed_coords.extend([coords_start, coords_end])
    deformed_coords = np.array(deformed_coords)
    
    # Quaternion computation for middle joints
    quaternions = []
    middle_idx = n_segments // 2

    # Compute rotation to align local y-axis with global xy-plane and ensure local x-axis points downward
    local_rot = Rotation.from_quat(orientation_quat)
    local_y = np.array([0, 1, 0])  # Local y-axis
    local_x = np.array([1, 0, 0])  # Local x-axis
    local_z = np.array([0, 0, 1])  # Local z-axis

    # Transform local axes to global coordinates
    global_y = local_rot.apply([local_y])[0]
    global_x = local_rot.apply([local_x])[0]
    global_z = local_rot.apply([local_z])[0]

    # Step 1: Compute rotation to make local y-axis perpendicular to global z-axis
    z_global = np.array([0, 0, 1])
    # Project global_y onto the plane perpendicular to global_z
    y_proj = global_y - np.dot(global_y, global_z) * global_z
    # Compute global_z × y_proj
    z_cross_y = np.cross(global_z, y_proj)

    # Solve for θ such that (cosθ * y_proj + sinθ * (global_z × y_proj)) · z_global = 0
    a = np.dot(y_proj, z_global)
    b = np.dot(z_cross_y, z_global)

    if abs(b) < 1e-10:
        if abs(a) < 1e-10:
            # y_proj is already perpendicular to z_global, no rotation needed
            rotation_y = Rotation.from_quat([0, 0, 0, 1])  # Identity rotation
        else:
            # No solution exists, use identity as fallback
            rotation_y = Rotation.from_quat([0, 0, 0, 1])
    else:
        # Compute θ = arctan(-a / b)
        theta = np.arctan2(-a, b)
        # Create rotation around local z-axis
        rotation_y = Rotation.from_rotvec(theta * local_z)

    # Apply rotation_y to local x-axis to check its orientation
    # global_x_after_y = rotation_y.apply([global_x])[0]
    global_x_after_y = rotation_y.apply([local_x])[0]

    # Step 2: Compute rotation to make local x-axis point downward
    if global_x_after_y[2] > 0:
        # If x-axis z-component is positive, rotate 180 degrees around local z-axis
        rotation_x = Rotation.from_euler('z', np.pi)
    else:
        # No additional rotation needed
        rotation_x = Rotation.from_quat([0, 0, 0, 1])

    # Combine rotations
    additional_rotation = rotation_y * rotation_x

    # Compute quaternions for all joints (only adjust middle two)
    for i in range(middle_idx, 0, -1):
        q = compute_relative_rotation(
            deformed_coords[2*i], deformed_coords[2*i+1],
            deformed_coords[2*(i-1)], deformed_coords[2*(i-1)+1]
        )
        if i == middle_idx:
            # Apply additional rotation to align y-axis and ensure x-axis points downward
            q_rot = Rotation.from_quat(q)
            adjusted_q = (additional_rotation * q_rot).as_quat()
            quaternions.append(adjusted_q)
        else:
            quaternions.append(q)

    for i in range(middle_idx, n_segments - 1):
        q = compute_relative_rotation(
            deformed_coords[2*i], deformed_coords[2*i+1],
            deformed_coords[2*(i+1)], deformed_coords[2*(i+1)+1]
        )
        if i == middle_idx:
            # Apply additional rotation to align y-axis and ensure x-axis points downward
            q_rot = Rotation.from_quat(q)
            adjusted_q = (additional_rotation * q_rot).as_quat()
            quaternions.append(adjusted_q)
        else:
            quaternions.append(q)
            
    inverse_rotation = rotation.inv()
    real_deformed_coords = []

    unique_coords = []  

    for coord in deformed_coords:
        untranslated = coord - midpoint
        unrotated = inverse_rotation.apply(untranslated)
        real_coord = unrotated + midpoint - translation_vector

        is_unique = True
        if unique_coords:
            distances = cdist([real_coord], unique_coords)
            min_dist = np.min(distances)
            if min_dist < 1e-6:
                is_unique = False
        
        if is_unique:
            unique_coords.append(real_coord)
            real_deformed_coords.append(real_coord)

    result = {
        "id": 1,
        "joint_quaternions": quaternions,
        "deformed_coords": np.array(real_deformed_coords).tolist()
    }
    
    if plot:
        opsv.plot_model()
        opsv.plot_load()
        opsv.plot_defo(sfac=scale)
        plt.show()
    
    ops.wipe()
    return result


if __name__ == "__main__":
    # structure_data = {
    #     'nodes': [
    #         {'id': 1, 'x': 0.1, 'y': 0.1, 'z': 0.1},
    #         {'id': 2, 'x': 0.4, 'y': 0.4, 'z': 0.4}
    #     ]
    # }
    
    structure_data = {'nodes': [{'id': 1, 'x': 0.6777076125144958, 'y': 0.2037447690963745, 'z': 0.650910496711731}, {'id': 2, 'x': 0.6367971301078796, 'y': 0.35022640228271484, 'z': 0.7807929515838623}]}
    
    try:
        result = single_bar_analysis(
            structure_data,
            n_segments=7,
            scale=1,
            plot=True
        )
        
        print("\nResults:")
        for i, q in enumerate(result['joint_quaternions']):
            print(f"Joint {i} relative quat: {np.round(q, 4)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Analysis complete")