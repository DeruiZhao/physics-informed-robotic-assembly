import math
import random
import pygad
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
cached_results = defaultdict(lambda: (float("inf"), -1))  # (value, early_stop_idx)
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


def evaluate_sequence(model, sequence, objective="displacement", CroSecs_in=None):
    """Evaluate a construction sequence with caching, early termination, and early stop index recording."""
    global cache_hits, cache_misses, current_best_fitness

    if not check_sequence_feasibility(model, sequence):
        return float("inf")

    fea_model = model.Clone()
    fea_model.cloneElements()
    load_cases = ["LC0"]
    max_value = 0.0

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
                current_disp, _ = cached_results[sub_key]
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
                cached_results[sub_key] = (
                    current_disp,
                    i,
                )  # Cache value and early stop index

            disps.append(current_disp)
            max_value = max(disps)

            if -max_value < current_best_fitness:
                cached_results[sub_key] = (
                    max_value,
                    i,
                )  # Update cache with early stop index
                return max_value

        cached_results[tuple(sequence), objective] = (
            max_value,
            len(sequence),
        )  # No early stop
        return max_value

    elif objective == "stress":
        if CroSecs_in is None:
            raise ValueError(
                "Cross-section (CroSecs_in) must be provided for stress objective!"
            )

        stresses = []
        for i, elem_ind in enumerate(sequence):
            sub_seq = tuple(sequence[: i + 1])
            sub_key = (sub_seq, objective)

            if sub_key in cached_results:
                cache_hits += 1
                current_stress, _ = cached_results[sub_key]
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
                    ) / 1e3  # Convert to MPa
                    max_stress = max(max_stress, total_stress)

                current_stress = max_stress
                cached_results[sub_key] = (
                    current_stress,
                    i,
                )  # Cache value and early stop index

            stresses.append(current_stress)
            max_value = max(stresses)

            if -max_value < current_best_fitness:
                cached_results[sub_key] = (
                    max_value,
                    i,
                )  # Update cache with early stop index
                return max_value

        cached_results[tuple(sequence), objective] = (
            max_value,
            len(sequence),
        )  # No early stop
        return max_value

    raise ValueError(f"Unsupported objective: {objective}")


def fitness_func_max_disp(ga_instance, solution, solution_idx):
    """Fitness function for displacement minimization."""
    global current_best_fitness
    sequence = [int(all_elements[i]) for i in solution]
    fitness = -evaluate_sequence(Model_in, sequence, objective="displacement")
    current_best_fitness = max(current_best_fitness, fitness)
    return fitness


def fitness_func_max_stress(ga_instance, solution, solution_idx, CroSecs_in):
    """Fitness function for stress minimization."""
    global current_best_fitness
    sequence = [int(all_elements[i]) for i in solution]
    fitness = -evaluate_sequence(
        Model_in, sequence, objective="stress", CroSecs_in=CroSecs_in
    )
    current_best_fitness = max(current_best_fitness, fitness)
    return fitness


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


def generate_feasible_initial_solution(model, element_distances, randomness=0.4):
    """Generate a single feasible construction sequence prioritizing proximity to supports."""
    try:
        element_from_id = {e.ind: tuple(e.node_inds) for e in model.elems}
        ground_nodes = {s.node_ind for s in model.supports}
        active_nodes, remaining_elements = set(ground_nodes), set(
            element_from_id.keys()
        )
        sequence = []

        while remaining_elements:
            feasible_elements = [
                e
                for e in remaining_elements
                if any(n in active_nodes for n in element_from_id[e])
            ]
            if not feasible_elements:
                print("Warning: Construction sequence generation failed!")
                return None

            feasible_elements.sort(key=lambda e: element_distances[e])
            elem = (
                random.choice(feasible_elements[: max(3, len(feasible_elements) // 2)])
                if random.random() < randomness and len(feasible_elements) > 1
                else feasible_elements[0]
            )

            sequence.append(elem)
            active_nodes.update(element_from_id[elem])
            remaining_elements.remove(elem)

        return sequence if check_sequence_feasibility(model, sequence) else None
    except Exception as e:
        print(f"Initial solution error: {e}")
        return None


def generate_feasible_population(model, pop_size, all_elements):
    """Generate a population of feasible construction sequences."""
    population = []
    seen_sequences = set()
    element_distances = compute_element_distances(model)

    while len(population) < pop_size:
        feasible_seq = generate_feasible_initial_solution(
            model, element_distances, randomness=0.2
        )
        if feasible_seq:
            seq_tuple = tuple(feasible_seq)
            if seq_tuple not in seen_sequences:
                population.append([all_elements.index(e) for e in feasible_seq])
                seen_sequences.add(seq_tuple)

    return population


def initialize_with_feasible_population(ga_instance, model):
    """Initialize GA with feasible population."""
    pop = generate_feasible_population(model, ga_instance.sol_per_pop, all_elements)
    ga_instance.population = np.array(pop)
    print(f"Initialized population with {len(pop)} feasible solutions")


def feasible_crossover(parents, offspring_size, ga_instance):
    """Crossover operator maintaining feasibility, with crossover point before early stopping."""
    offspring = []
    element_from_id = {e.ind: tuple(e.node_inds) for e in Model_in.elems}
    ground_nodes = {s.node_ind for s in Model_in.supports}
    objective = (
        "displacement"
        if ga_instance.fitness_func.__name__.endswith("disp")
        else "stress"
    )
    CroSecs_in = (
        getattr(ga_instance, "CroSecs_in", None) if objective == "stress" else None
    )

    for _ in range(offspring_size[0]):
        parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
        seq1 = [all_elements[int(i)] for i in parent1]
        seq2 = [all_elements[int(i)] for i in parent2]

        # Get early stopping index from cache for parent1
        seq1_key = (tuple(seq1), objective)
        if seq1_key in cached_results:
            _, early_stop_idx = cached_results[seq1_key]
        else:
            # Evaluate sequence to ensure cache is populated
            evaluate_sequence(Model_in, seq1, objective, CroSecs_in)
            _, early_stop_idx = cached_results[seq1_key]

        # Choose crossover point as the position before early stopping, or random if no early stop
        if early_stop_idx < len(seq1):
            point = early_stop_idx
        else:
            point = random.randint(1, len(seq1) - 1)

        child = seq1[:point]
        active_nodes = set(ground_nodes).union(
            *(element_from_id[elem] for elem in child)
        )

        # Add remaining elements from parent2 that are feasible
        remaining = [e for e in seq2 if e not in child]
        for elem in remaining:
            if any(n in active_nodes for n in element_from_id[elem]):
                child.append(elem)
                active_nodes.update(element_from_id[elem])

        # If child is incomplete, add remaining feasible elements from parent1
        if len(child) < len(seq1):
            remaining = [e for e in seq1 if e not in child]
            for elem in remaining:
                if any(n in active_nodes for n in element_from_id[elem]):
                    child.append(elem)
                    active_nodes.update(element_from_id[elem])

        # Ensure child is feasible; if not, revert to parent1
        if check_sequence_feasibility(Model_in, child):
            offspring.append([all_elements.index(e) for e in child])
        else:
            offspring.append(parent1)

    return np.array(offspring)


def feasible_mutation(offspring, ga_instance, initial_temp=1000, cooling_rate=0.95):
    """Mutation operator with simulated annealing maintaining feasibility."""
    element_from_id = {e.ind: tuple(e.node_inds) for e in Model_in.elems}
    ground_nodes = {s.node_ind for s in Model_in.supports}
    current_temp = initial_temp * (cooling_rate**ga_instance.generations_completed)

    for solution_idx in range(offspring.shape[0]):
        seq = [all_elements[int(i)] for i in offspring[solution_idx]]
        if random.random() < ga_instance.mutation_probability:
            pos1, pos2 = random.sample(range(len(seq)), 2)
            new_seq = seq.copy()
            new_seq[pos1], new_seq[pos2] = new_seq[pos2], new_seq[pos1]
            if check_sequence_feasibility(Model_in, new_seq):
                old_fitness = -evaluate_sequence(Model_in, seq, "displacement")
                new_fitness = -evaluate_sequence(Model_in, new_seq, "displacement")
                if new_fitness > old_fitness or random.random() < math.exp(
                    (new_fitness - old_fitness) / current_temp
                ):
                    seq = new_seq
        offspring[solution_idx] = [all_elements.index(e) for e in seq]

    return offspring


def get_ga_config(objective, CroSecs_in=None):
    """Get GA configuration based on optimization objective."""
    base_params = {
        "num_parents_mating": 10,
        "sol_per_pop": 20,
        "num_genes": num_elements,
        "gene_type": int,
        "gene_space": list(range(num_elements)),
        "mutation_probability": 0.6,
        "crossover_probability": 0.8,
        "parent_selection_type": "tournament",
        "keep_elitism": 1,
        "keep_parents": 0,
        "allow_duplicate_genes": False,
        "on_generation": lambda ga: on_generation(ga),
        "num_generations": 3000,
        "stop_criteria": ["reach_0", "saturate_200"],
        "crossover_type": feasible_crossover,
        "mutation_type": lambda offspring, ga: feasible_mutation(
            offspring, ga, initial_temp=10000, cooling_rate=0.995
        ),
    }

    if objective == "displacement":
        return {**base_params, "fitness_func": fitness_func_max_disp}
    elif objective == "stress":
        if CroSecs_in is None:
            raise ValueError(
                "Cross-section (CroSecs_in) must be provided for stress objective!"
            )
        return {
            **base_params,
            "fitness_func": lambda ga, sol, idx: fitness_func_max_stress(
                ga, sol, idx, CroSecs_in
            ),
            "CroSecs_in": CroSecs_in,
        }

    raise ValueError(f"Unsupported objective: {objective}")


# Main optimization
if "activate" in globals() and activate and "Model_in" in globals() and Model_in:
    all_elements = [elem.ind for elem in Model_in.elems]
    num_elements = len(all_elements)
    fitness_history = []
    current_best_fitness = float("-inf")

    def on_generation(ga_instance):
        """Callback for each generation."""
        global current_best_fitness
        best_fitness = ga_instance.best_solution()[1]
        current_best_fitness = best_fitness
        fitness_history.append(-best_fitness)
        if ga_instance.generations_completed % 100 == 0:
            print(
                f"Generation {ga_instance.generations_completed}, Best Fitness: {-best_fitness:.8f}"
            )

    try:
        ga_params = get_ga_config(objective, CroSecs_in)
        ga_instance = pygad.GA(**ga_params)
        initialize_with_feasible_population(ga_instance, Model_in)
        print(f"Starting GA run for {objective} optimization...")
        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()
        opt_seq = [all_elements[int(i)] for i in solution]
        result_value = evaluate_sequence(
            Model_in, opt_seq, objective=objective, CroSecs_in=CroSecs_in
        )

        print("\nOptimization Results:")
        print(f"Optimized Sequence: {opt_seq}")
        print(
            f"Max {'Displacement' if objective == 'displacement' else 'Stress'}: {result_value}"
        )
        print(f"Generations Completed: {ga_instance.generations_completed}")
        print(f"Cached Sequences Size: {len(cached_results)}")

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
        print(f"GA execution failed: {e}")
        raise
    finally:
        cached_results.clear()
        print("Cache cleared. Current cache size:", len(cached_results))
else:
    print("GA not activated or Model_in not defined.")
