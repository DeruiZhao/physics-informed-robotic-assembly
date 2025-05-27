import json
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import opsvis as opsv

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
    
    # In OpenSees, geomTransf expects a vector in the local x-z plane (x_axis in this case)
    ops.geomTransf('Corotational', transfTag, *x_axis)
    return x_axis, y_axis, z_axis  # Return local axes for use in result storage

def create_semi_rigid_joint(node_tag, ref_node_tag, k_trans, k_rot, zero_length_tag):
    """Create semi-rigid joint connection with unified stiffness"""
    ops.element('zeroLength', zero_length_tag, node_tag, ref_node_tag,
               '-mat', k_trans, k_trans, k_trans,  # Translation stiffness
               k_rot, k_rot, k_rot,  # Rotation stiffness
               '-dir', 1, 2, 3, 4, 5, 6)  # All DOFs

def fea_analysis(structure_data, k_rot=1e1, scale=1.0):
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
    k_trans = 2e11  # Translation stiffness
    
    nodes, bars = structure_data["nodes"], structure_data["bars"]
    
    # Define base tags
    BASE_NODE_TAG = 1
    BASE_BEAM_ELEMENT_TAG = 1
    BASE_ZERO_LENGTH_TAG = 1000
    BASE_TRANSF_TAG = 1
    BASE_MATERIAL_TAG = 1
    BASE_GROUND_NODE_TAG = 10000
    
    # Create materials
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG, k_trans)
    ops.uniaxialMaterial('Elastic', BASE_MATERIAL_TAG+1, k_rot)
    
    node_id_map = {}
    joint_ref_nodes = {}
    ground_nodes = {}
    bar_local_axes = {}  # Store local axes for each bar
    current_node_tag = BASE_NODE_TAG
    current_ground_tag = BASE_GROUND_NODE_TAG

    # Create reference nodes for all joint positions
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
    
    # Create beam nodes and connect to reference nodes
    for bar in bars:
        start_node = next(n for n in nodes if n["id"] == bar["start"])
        end_node = next(n for n in nodes if n["id"] == bar["end"])
        start_pos_key = f"{start_node['x']}_{start_node['y']}_{start_node['z']}"
        end_pos_key = f"{end_node['x']}_{end_node['y']}_{end_node['z']}"
        
        start_ref_node = joint_ref_nodes[start_pos_key]
        end_ref_node = joint_ref_nodes[end_pos_key]
        
        start_tag = current_node_tag
        ops.node(start_tag, start_node["x"], start_node["y"], start_node["z"])
        node_id_map[f"{bar['id']}_start"] = start_tag
        current_node_tag += 1
        
        end_tag = current_node_tag
        ops.node(end_tag, end_node["x"], end_node["y"], end_node["z"])
        node_id_map[f"{bar['id']}_end"] = end_tag
        current_node_tag += 1
        
        create_semi_rigid_joint(start_tag, start_ref_node, BASE_MATERIAL_TAG, BASE_MATERIAL_TAG+1, BASE_ZERO_LENGTH_TAG)
        BASE_ZERO_LENGTH_TAG += 1
        
        create_semi_rigid_joint(end_tag, end_ref_node, BASE_MATERIAL_TAG, BASE_MATERIAL_TAG+1, BASE_ZERO_LENGTH_TAG)
        BASE_ZERO_LENGTH_TAG += 1
        
        # Compute and store local coordinate system
        x_axis, y_axis, z_axis = auto_geom_transf(BASE_TRANSF_TAG, start_tag, end_tag)
        bar_local_axes[bar["id"]] = (x_axis, y_axis, z_axis)
        
        ops.element('elasticBeamColumn', BASE_BEAM_ELEMENT_TAG, 
                   start_tag, end_tag, A, E, G, J, I, I, BASE_TRANSF_TAG)
        BASE_BEAM_ELEMENT_TAG += 1
        BASE_TRANSF_TAG += 1
    
    # Apply gravity loads
    weight_per_length = rho * A * g
    nodal_forces = {node_id: [0, 0, 0] for node_id in node_id_map.values()}
    for bar in bars:
        start_tag = node_id_map[f"{bar['id']}_start"]
        end_tag = node_id_map[f"{bar['id']}_end"]
        length = np.linalg.norm(np.array(ops.nodeCoord(end_tag)) - np.array(ops.nodeCoord(start_tag)))
        half_weight = weight_per_length * length / 2
        nodal_forces[start_tag][2] -= half_weight
        nodal_forces[end_tag][2] -= half_weight
    
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for node_id, forces in nodal_forces.items():
        ops.load(node_id, *forces, 0, 0, 0)
    
    # Analysis settings
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
    
    # Collect results with relative displacement, rotation matrix, and local coordinate system
    deformed_bars = []
    deformed_nodes = []
    
    # Collect all reference node displacements (joint positions)
    for node in nodes:
        pos_key = f"{node['x']}_{node['y']}_{node['z']}"
        ref_node_tag = joint_ref_nodes[pos_key]
        
        # Get initial and deformed positions
        initial_pos = np.array([node["x"], node["y"], node["z"]])
        disp = np.array([ops.nodeDisp(ref_node_tag, dof) for dof in range(1, 4)]) * scale
        deformed_pos = initial_pos + disp
        
        deformed_nodes.append({
            "id": node["id"],
            "initial_position": initial_pos.tolist(),
            "deformed_position": deformed_pos.tolist(),
            "displacement": disp.tolist()
        })
    
    # Collect bar information
    for bar in bars:
        start_tag = node_id_map[f"{bar['id']}_start"]
        end_tag = node_id_map[f"{bar['id']}_end"]
        
        # Initial coordinates
        initial_i = np.array(ops.nodeCoord(start_tag))
        initial_j = np.array(ops.nodeCoord(end_tag))
        initial_center = (initial_i + initial_j) / 2
        
        # Displacements (translation only)
        disp_i = np.array([ops.nodeDisp(start_tag, dof) for dof in range(1, 4)]) * scale
        disp_j = np.array([ops.nodeDisp(end_tag, dof) for dof in range(1, 4)]) * scale
        
        deformed_i = initial_i + disp_i
        deformed_j = initial_j + disp_j
        deformed_center = (deformed_i + deformed_j) / 2
        
        # Relative displacement at center
        relative_displacement = deformed_center - initial_center
        
        # Initial and deformed directions (normalized)
        initial_dir = (initial_j - initial_i)
        initial_length = np.linalg.norm(initial_dir)
        initial_dir = initial_dir / initial_length
        
        deformed_dir = (deformed_j - deformed_i)
        deformed_dir = deformed_dir / np.linalg.norm(deformed_dir)
        
        # Calculate rotation matrix
        direction_threshold = 1e-6
        dot = np.dot(initial_dir, deformed_dir)
        dot = np.clip(dot, -1.0, 1.0)
        
        if abs(1.0 - dot) < direction_threshold:
            rotation_matrix = np.eye(3).tolist()
        else:
            axis = np.cross(initial_dir, deformed_dir)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            rotation = R.from_rotvec(axis * angle)
            rotation_matrix = rotation.as_matrix().tolist()
        
        # Get local coordinate system
        local_x_axis, local_y_axis, local_z_axis = bar_local_axes[bar["id"]]
        local_coordinate_system = [
            local_x_axis.tolist(),
            local_y_axis.tolist(),
            local_z_axis.tolist(),
        ]
        local_coordinate_system = list(map(list, zip(*local_coordinate_system)))  # Transpose to get columns as axes
        
        deformed_bars.append({
            "id": bar["id"],
            "relative_displacement": relative_displacement.tolist(),
            "rotation_matrix": rotation_matrix,
            "local_coordinate_system": local_coordinate_system
        })
        
    # opsv.plot_model()
    # opsv.plot_load()
    # opsv.plot_defo(sfac=scale)
    # plt.show()
    
    ops.wipe()
    return {
        "deformed_bars": deformed_bars,
        "deformed_nodes": deformed_nodes
    }

if __name__ == "__main__":
    try:
        # with open("tetrahedron_info_deformation_update.json", "r") as f:
        #     data = json.load(f)
        with open("column_info_structure.json", "r") as f:
            data = json.load(f)
        
        result1 = fea_analysis(data, k_rot=4e1, scale=50)
        result2 = fea_analysis(data, k_rot=2e1, scale=50)
        
        # Compare bar results
        for elem1, elem2 in zip(result1["deformed_bars"], result2["deformed_bars"]):
            print(f"\nElement {elem1['id']}:")
            print(f"k_rot * 2: Displacement = {elem1['relative_displacement']}")
            print(f"k_rot: Displacement = {elem2['relative_displacement']}")
            
            print("\nRotation matrix (k_rot * 2):")
            for row in elem1['rotation_matrix']:
                print(row)
            
            print("\nRotation matrix (k_rot):")
            for row in elem2['rotation_matrix']:
                print(row)
            
            # Print local coordinate system
            print("\nLocal coordinate system:")
            for row in elem1['local_coordinate_system']:
                print(row)
            
            # Calculate displacement magnitude ratio
            disp1 = np.linalg.norm(elem1['relative_displacement'])
            disp2 = np.linalg.norm(elem2['relative_displacement'])
            disp_ratio = disp2 / disp1 if disp1 != 0 else 0
            print(f"\nDisplacement magnitude ratio: {disp_ratio:.2f}")
        
        # Print node displacements
        print("\nNode displacements (k_rot * 2):")
        for node in result1["deformed_nodes"]:
            print(f"Node {node['id']}:")
            print(f"Initial position: {node['initial_position']}")
            print(f"Deformed position: {node['deformed_position']}")
            print(f"Displacement: {node['displacement']}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Execution completed")