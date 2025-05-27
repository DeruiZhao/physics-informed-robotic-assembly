import json
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import opsvis as opsv
from scipy.spatial.transform import Rotation as R

def auto_geom_transf(transfTag, iNode, jNode):
    """Automatically determine geometric transformation vector with local z-axis along element direction"""
    xi, yi, zi = ops.nodeCoord(iNode)
    xj, yj, zj = ops.nodeCoord(jNode)
    z_axis = np.array([xj-xi, yj-yi, zj-zi])  # Local z-axis is the element direction
    z_axis /= np.linalg.norm(z_axis)
    
    # Choose a global axis to define x-axis (perpendicular to z_axis)
    global_axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    selected_axis = global_axes[np.argmin([abs(np.dot(z_axis, ax)) for ax in global_axes])]
    
    # Compute local x-axis (perpendicular to z_axis)
    x_axis = selected_axis - np.dot(selected_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    
    # Compute local y-axis (perpendicular to z_axis and x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # In OpenSees, geomTransf expects a vector in the local x-z plane
    ops.geomTransf('Corotational', transfTag, *x_axis)
    return x_axis, y_axis, z_axis  # Return local axes for use in quaternion computation

def calculate_rotational_stiffness(E, I, L, n_segments):
    """Calculate rotational stiffness for multi-segment rigid body model"""
    return n_segments * (E * I) / L

def create_semi_rigid_joint(node_tag, ref_node_tag, trans_mat_tag, rot_mat_tag, zero_length_tag):
    """Create semi-rigid joint connection with unified stiffness"""
    ops.element('zeroLength', zero_length_tag, node_tag, ref_node_tag,
               '-mat', trans_mat_tag, trans_mat_tag, trans_mat_tag,  # Translation stiffness
               rot_mat_tag, rot_mat_tag, rot_mat_tag,  # Rotation stiffness
               '-dir', 1, 2, 3, 4, 5, 6)  # All DOFs

def compute_relative_rotation(parent_start, parent_end, child_start, child_end, local_x_axis, local_y_axis, local_z_axis):
    """
    Compute relative rotation between two segments in the local coordinate system with z-axis along element.
    
    Args:
        parent_start: Starting coordinates of parent segment
        parent_end: Ending coordinates of parent segment
        child_start: Starting coordinates of child segment
        child_end: Ending coordinates of child segment
        local_x_axis: Local x-axis of the middle segment
        local_y_axis: Local y-axis of the middle segment
        local_z_axis: Local z-axis of the middle segment (along element direction)
    """
    # Compute segment direction vectors
    parent_vec = (parent_end - parent_start) / np.linalg.norm(parent_end - parent_start)
    child_vec = (child_end - child_start) / np.linalg.norm(child_end - child_start)
    
    # Define local coordinate system
    x_local = local_x_axis / np.linalg.norm(local_x_axis)
    y_local = local_y_axis / np.linalg.norm(local_y_axis)
    z_local = local_z_axis / np.linalg.norm(local_z_axis)
    
    # Transformation matrix from global to local coordinates
    global_to_local = np.vstack([x_local, y_local, z_local]).T  # Columns are basis vectors
    
    # Project parent and child vectors onto local coordinate system
    parent_vec_local = global_to_local.T @ parent_vec
    child_vec_local = global_to_local.T @ child_vec
    
    # Normalize vectors in local coordinate system
    parent_vec_local /= np.linalg.norm(parent_vec_local)
    child_vec_local /= np.linalg.norm(child_vec_local)
    
    # Compute relative rotation in local coordinate system
    cross_prod = np.cross(parent_vec_local, child_vec_local)
    cross_norm = np.linalg.norm(cross_prod)
    # dot_product = np.clip(np.dot(parent_vec_local, child_vec_local), -1.0, 1.0)
    dot_product = np.dot(parent_vec_local, child_vec_local)
    
    if cross_norm < 1e-6:
        return [0.0, 0.0, 0.0, 1.0]  # No rotation or 180-degree rotation
    
    axis_local = cross_prod / cross_norm
    angle = np.arccos(dot_product)
    
    # Compute quaternion directly in local coordinate system
    quat = R.from_rotvec(axis_local * angle).as_quat()
    
    if quat[-1] < 0:
        quat = -np.array(quat)
    
    return list(quat / np.linalg.norm(quat))

def fea_analysis(structure_data, n_segments=3, k_rot_base=1e1, k_rot_bar_sim=2e11, scale=1.0, plot=False):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Material parameters
    E = 2.1e11  # Elastic modulus (Pa)
    G = 8.0e10  # Shear modulus (Pa)
    b = 0.004    # Section width (m)
    A = b**2    # Cross-sectional area (m^2)
    I = b**4/12 # Moment of inertia (m^4)
    J = 2.25*(b**4)/6  # Torsional constant (m^4)
    rho = 7850  # Density (kg/m^3)
    g = 9.81    # Gravity (m/s^2)
    k_trans = 2.1e11  # High translation stiffness
    
    nodes, bars = structure_data["nodes"], structure_data["bars"]
    
    # Base tags
    BASE_NODE_TAG = 1
    BASE_BEAM_ELEMENT_TAG = 1
    BASE_ZERO_LENGTH_TAG = 1000
    BASE_TRANSF_TAG = 1
    BASE_MATERIAL_TAG = 1
    BASE_GROUND_NODE_TAG = 10000
    
    # Create base materials
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG, k_trans)
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG+1, k_rot_base)
    
    current_material_tag = BASE_MATERIAL_TAG + 2
    node_id_map = {}
    joint_ref_nodes = {}  # Reference nodes for joints (connecting bars or supports)
    segment_ref_nodes = {}  # Reference nodes for segment connections within bars
    ground_nodes = {}
    current_node_tag = BASE_NODE_TAG
    current_ground_tag = BASE_GROUND_NODE_TAG

    # Create joint reference nodes
    for node in nodes:
        pos_key = f"{node['x']}_{node['y']}_{node['z']}"
        if pos_key not in joint_ref_nodes:
            ref_node_tag = current_node_tag
            ops.node(ref_node_tag, node["x"], node["y"], node["z"])
            joint_ref_nodes[pos_key] = ref_node_tag
            current_node_tag += 1
            
            if node["is_support"]:
                ground_tag = current_ground_tag
                ops.node(ground_tag, node["x"], node["y"], node["z"])
                ops.fix(ground_tag, 1, 1, 1, 1, 1, 1)
                ground_nodes[pos_key] = ground_tag
                current_ground_tag += 1
                
                create_semi_rigid_joint(ref_node_tag, ground_tag, 
                                      BASE_MATERIAL_TAG, BASE_MATERIAL_TAG+1, 
                                      BASE_ZERO_LENGTH_TAG)
                BASE_ZERO_LENGTH_TAG += 1
    
    bar_segment_nodes = {}
    bar_local_axes = {}  # Store local axes for each bar
    
    # Create multi-segment beams
    for bar in bars:
        start_node = next(n for n in nodes if n["id"] == bar["start"])
        end_node = next(n for n in nodes if n["id"] == bar["end"])
        start_pos_key = f"{start_node['x']}_{start_node['y']}_{start_node['z']}"
        end_pos_key = f"{end_node['x']}_{end_node['y']}_{end_node['z']}"
        
        start_ref_node = joint_ref_nodes[start_pos_key]
        end_ref_node = joint_ref_nodes[end_pos_key]
        
        start_coord = np.array([start_node["x"], start_node["y"], start_node["z"]])
        end_coord = np.array([end_node["x"], end_node["y"], end_node["z"]])
        length = np.linalg.norm(end_coord - start_coord)
        
        k_rot_bar = calculate_rotational_stiffness(k_rot_bar_sim, I, length, n_segments)
        bar_material_tag = current_material_tag
        ops.uniaxialMaterial('Elastic', bar_material_tag, k_rot_bar)
        current_material_tag += 1
        
        # Create segment reference nodes and segment end nodes
        segment_end_nodes = []  # Store (start_node, end_node) for each segment
        segment_ref_nodes_list = []  # Store reference nodes for segment connections
        
        # Create reference nodes for segment connections (n_segments - 1)
        for i in range(1, n_segments):
            xi = start_coord + (end_coord - start_coord) * i / n_segments
            pos_key = f"bar_{bar['id']}_{i}"  # Unique key for segment reference node
            ref_node_tag = current_node_tag
            ops.node(ref_node_tag, *xi)
            segment_ref_nodes_list.append(ref_node_tag)
            segment_ref_nodes[pos_key] = ref_node_tag
            current_node_tag += 1
        
        # Create segment end nodes
        for i in range(n_segments):
            # Start node of segment
            start_node_tag = current_node_tag
            xi_start = start_coord + (end_coord - start_coord) * i / n_segments
            ops.node(start_node_tag, *xi_start)
            current_node_tag += 1
            
            # End node of segment
            end_node_tag = current_node_tag
            xi_end = start_coord + (end_coord - start_coord) * (i + 1) / n_segments
            ops.node(end_node_tag, *xi_end)
            current_node_tag += 1
            
            segment_end_nodes.append((start_node_tag, end_node_tag))
            
            # Connect start node to joint reference node for first segment
            if i == 0:
                create_semi_rigid_joint(start_node_tag, start_ref_node, 
                                      BASE_MATERIAL_TAG, BASE_MATERIAL_TAG+1, 
                                      BASE_ZERO_LENGTH_TAG)
                BASE_ZERO_LENGTH_TAG += 1
            # Connect end node to joint reference node for last segment
            if i == n_segments - 1:
                create_semi_rigid_joint(end_node_tag, end_ref_node, 
                                      BASE_MATERIAL_TAG, BASE_MATERIAL_TAG+1, 
                                      BASE_ZERO_LENGTH_TAG)
                BASE_ZERO_LENGTH_TAG += 1
        
        # Connect segments via reference nodes
        for i in range(n_segments - 1):
            # Connect end of segment i to segment reference node
            segment_ref_node = segment_ref_nodes_list[i]
            prev_segment_end = segment_end_nodes[i][1]  # End node of segment i
            ops.element('zeroLength', BASE_ZERO_LENGTH_TAG, 
                        prev_segment_end, segment_ref_node,
                        '-mat', BASE_MATERIAL_TAG, BASE_MATERIAL_TAG, BASE_MATERIAL_TAG,
                        bar_material_tag, bar_material_tag, bar_material_tag,
                        '-dir', 1, 2, 3, 4, 5, 6)
            BASE_ZERO_LENGTH_TAG += 1
            
            # Connect start of segment i+1 to segment reference node
            next_segment_start = segment_end_nodes[i + 1][0]  # Start node of segment i+1
            ops.element('zeroLength', BASE_ZERO_LENGTH_TAG, 
                        next_segment_start, segment_ref_node,
                        '-mat', BASE_MATERIAL_TAG, BASE_MATERIAL_TAG, BASE_MATERIAL_TAG,
                        bar_material_tag, bar_material_tag, bar_material_tag,
                        '-dir', 1, 2, 3, 4, 5, 6)
            BASE_ZERO_LENGTH_TAG += 1
        
        # Create beam elements for segments and get local coordinate system
        transf_tag = BASE_TRANSF_TAG
        # Get local coordinate system for the middle segment
        middle_segment_idx = n_segments // 2
        middle_start_node, middle_end_node = segment_end_nodes[middle_segment_idx]
        x_axis, y_axis, z_axis = auto_geom_transf(transf_tag, middle_start_node, middle_end_node)
        bar_local_axes[bar["id"]] = (x_axis, y_axis, z_axis)  # Store local axes for this bar
        
        for i in range(n_segments):
            start_node_tag, end_node_tag = segment_end_nodes[i]
            ops.element('elasticBeamColumn', BASE_BEAM_ELEMENT_TAG, 
                        start_node_tag, end_node_tag, 
                        A, 1e3*E, G, J, I, I, transf_tag)
            BASE_BEAM_ELEMENT_TAG += 1
        
        BASE_TRANSF_TAG += 1
        
        # Store segment nodes for load application
        segment_nodes = []
        for start_tag, end_tag in segment_end_nodes:
            segment_nodes.append(start_tag)
            segment_nodes.append(end_tag)
        segment_nodes = list(dict.fromkeys(segment_nodes))  # Remove duplicates
        bar_segment_nodes[bar["id"]] = segment_nodes
        node_id_map[f"{bar['id']}_start"] = segment_end_nodes[0][0]
        node_id_map[f"{bar['id']}_end"] = segment_end_nodes[-1][1]
    
    # Apply gravity loads
    weight_per_length = rho * A * g
    nodal_forces = {}
    
    for node_tag in range(BASE_NODE_TAG, current_node_tag):
        nodal_forces[node_tag] = [0, 0, 0]
    
    for bar in bars:
        start_node = next(n for n in nodes if n["id"] == bar["start"])
        end_node = next(n for n in nodes if n["id"] == bar["end"])
        start_coord = np.array([start_node["x"], start_node["y"], start_node["z"]])
        end_coord = np.array([end_node["x"], end_node["y"], end_node["z"]])
        length = np.linalg.norm(end_coord - start_coord)
        
        total_weight = weight_per_length * length
        bar_nodes = bar_segment_nodes[bar["id"]]
        node_weight = total_weight / len(bar_nodes)
        for node_tag in bar_nodes:
            nodal_forces[node_tag][2] -= node_weight
    
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for node_id, forces in nodal_forces.items():
        ops.load(node_id, *forces, 0, 0, 0)
    
    # Analysis settings
    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.algorithm('Newton')
    # ops.integrator('ArcLength', 0.01, 1.0)
    ops.integrator('LoadControl', 0.01)
    ops.analysis('Static')
    ops.test('NormDispIncr', 1.0e-6, 100)
    
    if ops.analyze(100) != 0:
        ops.wipe()
        raise RuntimeError("Analysis failed to converge")
    
    # Collect results with relative displacement, rotation matrix, joint quaternions, local coordinate system, and deformed joint positions
    deformed_bars = []
    deformed_nodes = []

    # Collect deformed joint positions for all joint reference nodes
    all_joint_positions = []
    for pos_key, ref_node_tag in joint_ref_nodes.items():
        # Extract initial coordinates from pos_key
        x, y, z = map(float, pos_key.split('_'))
        initial_pos = np.array([x, y, z])
        # Get displacements
        disp = np.array([ops.nodeDisp(ref_node_tag, dof) for dof in range(1, 4)]) * scale
        # Compute deformed position
        deformed_pos = initial_pos + disp
        # Store deformed position with node ID
        node_id = next(node["id"] for node in nodes if f"{node['x']}_{node['y']}_{node['z']}" == pos_key)
        all_joint_positions.append(deformed_pos.tolist())

    # Collect node information
    for node in nodes:
        pos_key = f"{node['x']}_{node['y']}_{node['z']}"
        ref_node_tag = joint_ref_nodes[pos_key]
        
        # Get initial and deformed positions
        initial_pos = np.array([node["x"], node["y"], node["z"]])
        disp = np.array([ops.nodeDisp(ref_node_tag, dof) for dof in range(1, 4)]) * scale
        deformed_pos = initial_pos + disp
        
        # Store node information (no quaternion)
        deformed_nodes.append({
            "id": node["id"],
            "initial_position": initial_pos.tolist(),
            "deformed_position": deformed_pos.tolist(),
            "displacement": disp.tolist()
        })

    # Collect bar information and compute joint quaternions
    for bar in bars:
        # Get segment nodes
        segment_nodes = bar_segment_nodes[bar["id"]]
        
        # Compute middle segment index
        middle_segment_idx = n_segments // 2
        
        # Middle segment's start and end nodes
        start_tag = segment_nodes[2 * middle_segment_idx]
        end_tag = segment_nodes[2 * middle_segment_idx + 1]
        
        # Middle segment's initial coordinates
        initial_i = np.array(ops.nodeCoord(start_tag))
        initial_j = np.array(ops.nodeCoord(end_tag))
        initial_center = (initial_i + initial_j) / 2
        
        # Middle segment's displacements
        disp_i = np.array([ops.nodeDisp(start_tag, dof) for dof in range(1, 4)]) * scale
        disp_j = np.array([ops.nodeDisp(end_tag, dof) for dof in range(1, 4)]) * scale
        
        # Deformed coordinates
        deformed_i = initial_i + disp_i
        deformed_j = initial_j + disp_j
        deformed_center = (deformed_i + deformed_j) / 2
        
        # Relative displacement of middle segment center
        relative_displacement = deformed_center - initial_center
        
        # Compute rotation matrix for middle segment
        initial_dir = initial_j - initial_i
        initial_length = np.linalg.norm(initial_dir)
        initial_dir = initial_dir / initial_length if initial_length > 1e-10 else initial_dir
        
        deformed_dir = deformed_j - deformed_i
        deformed_length = np.linalg.norm(deformed_dir)
        deformed_dir = deformed_dir / deformed_length if deformed_length > 1e-10 else deformed_dir
        
        direction_threshold = 1e-10
        dot = np.dot(initial_dir, deformed_dir)
        dot = np.clip(dot, -1.0, 1.0)
        
        if abs(1.0 - dot) < direction_threshold:
            rotation_matrix = np.eye(3)
            middle_quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = np.cross(initial_dir, deformed_dir)
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-10 else np.array([1.0, 0.0, 0.0])
            angle = np.arccos(dot)
            rotation = R.from_rotvec(axis * angle)
            rotation_matrix = rotation.as_matrix()
            middle_quat = rotation.as_quat()
        
        # Compute deformed coordinates for all segments
        deformed_coords = []
        for i in range(n_segments):
            start = segment_nodes[2 * i]
            end = segment_nodes[2 * i + 1]
            coords_start = np.array(ops.nodeCoord(start)) + np.array([ops.nodeDisp(start, dof) for dof in range(1, 4)]) * scale
            coords_end = np.array(ops.nodeCoord(end)) + np.array([ops.nodeDisp(end, dof) for dof in range(1, 4)]) * scale
            deformed_coords.extend([coords_start, coords_end])
        deformed_coords = np.array(deformed_coords)
        
        # Get local coordinate system for this bar (from middle segment)
        local_x_axis, local_y_axis, local_z_axis = bar_local_axes[bar["id"]]
        
        # Compute quaternions for intermediate joints (n_segments - 1 joints)
        joint_quaternions = []
        
        # Backward: from middle_idx to 1
        for i in range(middle_segment_idx, 0, -1):
            q = compute_relative_rotation(
                deformed_coords[2*i], deformed_coords[2*i+1],
                deformed_coords[2*(i-1)], deformed_coords[2*(i-1)+1],
                local_x_axis, local_y_axis, local_z_axis
            )
            joint_quaternions.append(q)
        
        # Forward: from middle_idx to n_segments - 2
        for i in range(middle_segment_idx, n_segments - 1):
            q = compute_relative_rotation(
                deformed_coords[2*i], deformed_coords[2*i+1],
                deformed_coords[2*(i+1)], deformed_coords[2*(i+1)+1],
                local_x_axis, local_y_axis, local_z_axis
            )
            joint_quaternions.append(q)
        
        # Store local coordinate system
        local_coordinate_system = [
            local_x_axis.tolist(),
            local_y_axis.tolist(),
            local_z_axis.tolist(),
        ]
        local_coordinate_system = list(map(list, zip(*local_coordinate_system)))
                
        deformed_bars.append({
            "id": bar["id"],
            "relative_displacement": relative_displacement.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "joint_quaternions": joint_quaternions,
            "local_coordinate_system": local_coordinate_system,
            "deformed_joint_positions": all_joint_positions
        })
    
    if plot:
        opsv.plot_model(node_labels=0, element_labels=0)
        opsv.plot_load()
        opsv.plot_defo(sfac=scale)
        plt.show()
    
    ops.wipe()
    
    return {
        "deformed_bars": deformed_bars,
        "deformed_nodes": deformed_nodes
    }

if __name__ == "__main__":
    try:
        with open("column_info_structure_test.json", "r") as f:
            data = json.load(f)
        
        deformed_elements1 = fea_analysis(data, n_segments=7, k_rot_base=1.6, k_rot_bar_sim=0.5025e9, scale=1, plot=True)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Execution completed")