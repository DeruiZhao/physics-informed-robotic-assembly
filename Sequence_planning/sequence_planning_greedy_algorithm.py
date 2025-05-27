import math
import random
from collections import defaultdict
import numpy as np

# Karamba imports
import Karamba.Models
import Karamba.Elements
import Karamba.Results
from System.Collections.Generic import List
from System import String
from Karamba.Algorithms import Analyze
from Karamba.Results import BeamResultantForces, NodeDisplacements

# Global variables
cached_results = defaultdict(float)
cache_hits = 0
cache_misses = 0
current_best_fitness = float("-inf")


def check_sequence_feasibility(model, sequence):
    """Check if a construction sequence is feasible."""
    try:
        element_from_id = {e.ind: tuple(e.node_inds) for e in model.elems}
        ground_nodes = {s.node_ind for s in model.supports}
        active_nodes, active_elements = set(ground_nodes), set()

        for elem in sequence:
            node1, node2 = element_from_id[elem]
            if node1 not in active_nodes and node2 not in active_nodes:
                return False
            active_elements.add(elem)
            active_nodes.add(node1)
            active_nodes.add(node2)
        return True
    except Exception as e:
        print(f"Feasibility check error: {e}")
        return False


def evaluate_sequence(
    model,
    sequence,
    objective="displacement",
    CroSecs_in=None,
    enable_early_stopping=True,
):
    """Evaluate a construction sequence with caching and optional early termination."""
    global cache_hits, cache_misses, current_best_fitness

    if not check_sequence_feasibility(model, sequence):
        return float("inf")

    try:
        fea_model = model.Clone()
        fea_model.cloneElements()
        load_cases = ["LC0"]
        max_value = 0.0
        stress_scale = 1e3  # Store scaling factor for stress

        if objective == "displacement":
            disps = []
            active_nodes = set()
            element_node_map = {e.ind: set(e.node_inds) for e in fea_model.elems}

            for i, elem_ind in enumerate(sequence):
                active_nodes.update(element_node_map[elem_ind])
                sub_seq = tuple(sequence[: i + 1])
                sub_key = (sub_seq, objective)

                if sub_key in cached_results:
                    cache_hits += 1
                    current_disp = cached_results[sub_key]
                else:
                    cache_misses += 1
                    for ele in fea_model.elems:
                        ele.set_is_active(fea_model, ele.ind in sub_seq)

                    _, _, _, out_model, _ = Analyze.solve(fea_model, load_cases)
                    node_indices = List[int](active_nodes)
                    trans, _, _, _, _ = NodeDisplacements.solve(
                        out_model, load_cases[0], node_indices
                    )

                    max_disp = max(
                        math.sqrt(t.X**2 + t.Y**2 + t.Z**2) for nt in trans for t in nt
                    )
                    current_disp = max_disp * 1000  # Convert to mm
                    cached_results[sub_key] = current_disp

                disps.append(current_disp)
                max_value = max(disps)

                if enable_early_stopping and -max_value < current_best_fitness:
                    return max_value

            return max_value

        elif objective == "stress":
            if (
                CroSecs_in is None
                or not hasattr(CroSecs_in, "A")
                or not hasattr(CroSecs_in, "Wely_z_pos")
            ):
                raise ValueError(
                    "Valid Cross-section (CroSecs_in) must be provided for stress objective!"
                )

            stresses = []
            for i, elem_ind in enumerate(sequence):
                sub_seq = tuple(sequence[: i + 1])
                sub_key = (
                    sub_seq,
                    objective,
                    stress_scale,
                )  # Include scale in cache key

                if sub_key in cached_results:
                    cache_hits += 1
                    current_stress = cached_results[sub_key]
                else:
                    cache_misses += 1
                    for ele in fea_model.elems:
                        ele.set_is_active(fea_model, ele.ind in sub_seq)

                    _, _, _, out_model, _ = Analyze.solve(fea_model, load_cases)
                    max_stress = 0.0

                    for elem_ind in sub_seq:
                        beam = out_model.elems[elem_ind]
                        if not isinstance(beam, Karamba.Elements.ModelBeam):
                            continue

                        elem_id = List[String]([str(elem_ind)])
                        N, V, M = BeamResultantForces.solve(
                            out_model, elem_id, load_cases[0], 1, 1
                        )
                        axial_stress = abs(N[0][0]) / CroSecs_in.A
                        bending_stress = abs(M[0][0]) / CroSecs_in.Wely_z_pos
                        total_stress = (
                            axial_stress + bending_stress
                        ) / stress_scale  # Convert to MPa
                        max_stress = max(max_stress, total_stress)

                    current_stress = max_stress
                    cached_results[sub_key] = current_stress

                stresses.append(current_stress)
                max_value = max(stresses)

                if enable_early_stopping and -max_value < current_best_fitness:
                    return max_value

            return max_value

        raise ValueError(f"Unsupported objective: {objective}")
    except Exception as e:
        print(f"Evaluation failed for sequence {sequence}: {e}")
        return float("inf")


def compute_element_distances(model):
    """Compute minimum distance from each element to any support node."""
    element_from_id = {e.ind: tuple(e.node_inds) for e in model.elems}
    ground_nodes = {s.node_ind for s in model.supports}
    distances = {}

    for elem_id, (node1, node2) in element_from_id.items():
        pos1, pos2 = model.nodes[node1].pos, model.nodes[node2].pos
        min_dist = float("inf")
        for ground_node in ground_nodes:
            ground_pos = model.nodes[ground_node].pos
            dist1 = math.sqrt(
                sum(
                    (getattr(pos1, dim) - getattr(ground_pos, dim)) ** 2
                    for dim in ("X", "Y", "Z")
                )
            )
            dist2 = math.sqrt(
                sum(
                    (getattr(pos2, dim) - getattr(ground_pos, dim)) ** 2
                    for dim in ("X", "Y", "Z")
                )
            )
            min_dist = min(min_dist, dist1, dist2)
        distances[elem_id] = min_dist

    return distances


def greedy_construction_sequence(model, objective="displacement", CroSecs_in=None):
    """Greedily construct a sequence minimizing maximum objective during assembly."""
    try:
        element_from_id = {e.ind: tuple(e.node_inds) for e in model.elems}
        ground_nodes = {s.node_ind for s in model.supports}
        all_elements = [e.ind for e in model.elems]
        active_nodes = set(ground_nodes)
        sequence = []
        remaining_elements = set(all_elements)
        max_value_global = 0.0
        step_history = []

        element_distances = compute_element_distances(model)

        while remaining_elements:
            best_elem = None
            best_value = float("inf")

            feasible_elements = [
                e
                for e in remaining_elements
                if any(n in active_nodes for n in element_from_id[e])
            ]
            if not feasible_elements:
                print("Warning: No feasible element found to continue the sequence!")
                return None, float("inf"), []

            for elem in feasible_elements:
                temp_seq = sequence + [elem]
                current_value = evaluate_sequence(
                    model, temp_seq, objective, CroSecs_in, enable_early_stopping=False
                )
                if current_value < best_value:
                    best_value = current_value
                    best_elem = elem
                elif current_value == best_value and best_elem is not None:
                    if element_distances[elem] < element_distances[best_elem]:
                        best_elem = elem

            if best_elem is None:
                print("Warning: No feasible element selected!")
                return None, float("inf"), []

            sequence.append(best_elem)
            active_nodes.update(element_from_id[best_elem])
            remaining_elements.remove(best_elem)
            step_history.append(best_value)
            max_value_global = max(max_value_global, best_value)

            if len(sequence) % max(1, len(all_elements) // 10) == 0:
                print(
                    f"Step {len(sequence)}: Added element {best_elem}, {objective}: {best_value:.6f}, Global max: {max_value_global:.6f}"
                )

        return sequence, max_value_global, step_history
    except Exception as e:
        print(f"Greedy construction failed: {e}")
        return None, float("inf"), []


# Main optimization
if "activate" in globals() and activate and "Model_in" in globals() and Model_in:
    all_elements = [elem.ind for elem in Model_in.elems]

    # Initialize global variables
    cached_results.clear()
    cache_hits = 0
    cache_misses = 0
    current_best_fitness = float("-inf")

    try:
        print(
            f"Starting greedy optimization for {objective} minimization with stress scaling factor: 1e3..."
        )
        opt_seq, global_max, history = greedy_construction_sequence(
            Model_in, objective=objective, CroSecs_in=CroSecs_in
        )

        if opt_seq is None:
            print("Failed to construct a feasible sequence.")
        else:
            print("\nOptimization Results:")
            print(f"Optimized Sequence: {opt_seq}")
            print(f"Number of elements: {len(opt_seq)}")
            print(f"Global max {objective} during assembly: {global_max:.6f}")
            print(f"Final {objective}: {history[-1]:.6f}")

            print("\nAssembly History:")
            for step, (elem, val) in enumerate(zip(opt_seq, history)):
                print(f"Step {step+1}: Element {elem}, {objective}: {val:.6f}")

            total_evaluations = cache_hits + cache_misses
            cache_hit_rate = (
                cache_hits / total_evaluations * 100 if total_evaluations > 0 else 0
            )
            print(f"\nCache Statistics:")
            print(f"Sequence Evaluations: {total_evaluations}")
            print(f"Cache Hits: {cache_hits}")
            print(f"Cache Misses: {cache_misses}")
            print(f"Cache Hit Rate: {cache_hit_rate:.2f}%")

    except Exception as e:
        print(f"Greedy optimization failed: {e}")
        raise
    finally:
        cached_results.clear()
        print("Cache cleared. Current cache size:", len(cached_results))
else:
    print("Greedy optimization not activated or Model_in not defined.")
