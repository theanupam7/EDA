import heapq

def dijkstra(graph, start_node):
    """
    Calculates the shortest paths from a start_node to all other nodes
    in a graph using Dijkstra's algorithm.

    Args:
        graph (dict): A dictionary representing the graph as an adjacency list.
                      Keys are node names (str or int).
                      Values are lists of tuples, where each tuple contains
                      (neighbor_node, weight).
                      Example: {'A': [('B', 1), ('C', 4)], 'B': [('A', 1), ('C', 2)], ...}
        start_node: The node from which to calculate shortest paths.

    Returns:
        tuple: A tuple containing two dictionaries:
               - distances (dict): Shortest distances from start_node to all other nodes.
                                   Nodes not reachable will have a distance of float('inf').
               - predecessors (dict): A dictionary mapping each node to its predecessor
                                      in the shortest path from the start_node.
                                      The start_node itself will have None as a predecessor.

    Raises:
        ValueError: If the start_node is not in the graph.
        TypeError: If the graph is not a dictionary or if node keys/values are malformed.
                   (Basic type checking, not exhaustive)

    Edge Cases Handled:
        - Empty graph: Returns empty distances and predecessors.
        - Start node not in graph: Raises ValueError.
        - Disconnected components: Unreachable nodes will have float('inf') distance.
        - Single node graph: Distance to itself is 0.
        - Graph with no edges from start_node: Other nodes remain at float('inf').
    """
    if not isinstance(graph, dict):
        raise TypeError("Graph must be represented as a dictionary (adjacency list).")

    if not graph:
        return {}, {}

    if start_node not in graph:
        # Check if any node exists. If graph has nodes but start_node is not one of them.
        if any(graph): # True if graph has at least one node defined
             raise ValueError(f"Start node '{start_node}' not found in the graph.")
        else: # Graph is effectively empty or malformed if start_node isn't there AND graph has no keys
            return {}, {}


    # Initialize distances: infinity for all, 0 for start_node
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    # Predecessor dictionary to reconstruct paths
    predecessors = {node: None for node in graph}

    # Priority queue: (distance, node)
    # We use a min-heap, so nodes with smaller distances have higher priority.
    priority_queue = [(0, start_node)]
    heapq.heapify(priority_queue)

    # Set of visited nodes to prevent reprocessing
    # While Dijkstra's doesn't strictly need a 'visited' set if edge weights are non-negative
    # and priority queue updates handle finding shorter paths, it can be an optimization
    # in some implementations or make logic clearer. Here, the check `if new_distance < distances[neighbor]`
    # effectively serves a similar purpose for path relaxation.

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we've found a shorter path already, skip
        if current_distance > distances[current_node]:
            continue

        # Process neighbors
        if current_node in graph: # Ensure current_node is a valid key
            # Validate neighbors format
            if not isinstance(graph.get(current_node), list):
                raise TypeError(f"Neighbors for node '{current_node}' should be a list of (neighbor, weight) tuples.")

            for neighbor_entry in graph.get(current_node, []):
                if not (isinstance(neighbor_entry, (list, tuple)) and len(neighbor_entry) == 2):
                    raise TypeError(
                        f"Neighbor entry for node '{current_node}' is malformed. "
                        f"Expected (neighbor, weight), got: {neighbor_entry}"
                    )
                neighbor, weight = neighbor_entry

                if not isinstance(weight, (int, float)):
                    raise TypeError(
                        f"Edge weight between '{current_node}' and '{neighbor}' must be numeric. Got: {weight}"
                    )
                if weight < 0:
                    raise ValueError(
                        "Dijkstra's algorithm does not support negative edge weights. "
                        f"Negative weight {weight} found between '{current_node}' and '{neighbor}'."
                    )


                # If neighbor is not in distances, it means it's a new node not in the initial graph keys.
                # This could be an error in graph definition, or the graph is dynamic.
                # For a static graph, all nodes should be keys in the `graph` dict.
                # We'll assume valid graph structure where all reachable nodes are graph keys.
                if neighbor not in distances:
                    # This case implies an inconsistency in the graph definition
                    # (e.g. 'A': [('X', 1)] but 'X' is not a key in the graph).
                    # Depending on requirements, this could raise an error or initialize 'X'.
                    # For this implementation, we assume all nodes are predefined as keys in `graph`.
                    # If you want to dynamically add nodes, you'd initialize distances[neighbor] = float('inf') here.
                    # However, it's safer to require all nodes to be defined upfront.
                    print(f"Warning: Neighbor '{neighbor}' of node '{current_node}' is not a key in the graph. "
                          "It will be ignored unless it was meant to be implicitly added.")
                    # To handle implicitly added nodes:
                    # if neighbor not in distances:
                    # distances[neighbor] = float('inf')
                    # predecessors[neighbor] = None
                    # And ensure graph[neighbor] = [] if it doesn't exist to avoid KeyErrors later
                    # For now, we assume this is a malformed graph or an edge to an undefined node.
                    continue # Skip this neighbor if it's not a defined node

                distance_through_current = current_distance + weight

                if distance_through_current < distances[neighbor]:
                    distances[neighbor] = distance_through_current
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance_through_current, neighbor))
        else:
            # This should not happen if graph is well-formed and start_node is in graph.
            # It might indicate a node was in priority_queue but not in the graph structure.
            print(f"Warning: Node '{current_node}' from priority queue not found as a key in the graph.")


    return distances, predecessors

if __name__ == "__main__":
    print("Dijkstra's Algorithm Examples:")

    # Helper to print results nicely
    def print_paths(start_node, distances, predecessors):
        print(f"\nShortest paths from '{start_node}':")
        if not distances:
            print("  No distances calculated (e.g., empty graph or start node not found).")
            return

        for node, dist in sorted(distances.items()):
            path = []
            curr = node
            while curr is not None:
                path.append(curr)
                if curr not in predecessors: # Should not happen if logic is correct
                    print(f"Error: Node {curr} not in predecessors during path reconstruction.")
                    path.append("!PREDECESSOR_ERROR!")
                    break
                curr = predecessors[curr]
                if curr is not None and curr not in distances: # Cycle or error
                     path.append(f"!UNKNOWN_NODE_IN_PATH:{curr}!")
                     break

            path.reverse()
            path_str = " -> ".join(map(str, path))
            print(f"  To '{node}': Distance = {dist if dist != float('inf') else 'infinity'}, Path = {path_str}")

    # 1. Standard Graph
    print("\n--- 1. Standard Graph ---")
    graph1 = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    start_node1 = 'A'
    distances1, predecessors1 = dijkstra(graph1, start_node1)
    print_paths(start_node1, distances1, predecessors1)
    # Expected: A:0 (A), B:1 (A->B), C:3 (A->B->C), D:4 (A->B->C->D)

    # 2. Graph with Disconnected Components
    print("\n--- 2. Graph with Disconnected Components ---")
    graph2 = {
        'A': [('B', 1)],
        'B': [('A', 1)],
        'C': [('D', 3)],
        'D': [('C', 3)],
        'E': [] # Isolated node
    }
    start_node2 = 'A'
    distances2, predecessors2 = dijkstra(graph2, start_node2)
    print_paths(start_node2, distances2, predecessors2)
    # Expected: A:0, B:1, C:inf, D:inf, E:inf

    start_node2b = 'E'
    distances2b, predecessors2b = dijkstra(graph2, start_node2b)
    print_paths(start_node2b, distances2b, predecessors2b)
    # Expected: A:inf, B:inf, C:inf, D:inf, E:0


    # 3. Empty Graph
    print("\n--- 3. Empty Graph ---")
    graph3 = {}
    start_node3 = 'A' # Node doesn't exist here
    try:
        # An empty graph means start_node cannot be in it.
        # The current implementation will have `start_node not in graph` be true.
        # If the graph dict is empty, `any(graph)` is False, so it returns {}, {}
        distances3, predecessors3 = dijkstra(graph3, start_node3)
        print_paths(start_node3, distances3, predecessors3)
    except ValueError as e:
        print(f"  Caught expected error for empty graph: {e}")
    # Expected: If graph is truly empty, distances/predecessors should be {}

    # 4. Graph with a Single Node
    print("\n--- 4. Graph with a Single Node ---")
    graph4 = {'A': []}
    start_node4 = 'A'
    distances4, predecessors4 = dijkstra(graph4, start_node4)
    print_paths(start_node4, distances4, predecessors4)
    # Expected: A:0

    # 5. Start Node Not in Graph (but graph is not empty)
    print("\n--- 5. Start Node Not in Graph ---")
    graph5 = {'A': [('B', 1)], 'B': [('A', 1)]}
    start_node5 = 'C'
    try:
        distances5, predecessors5 = dijkstra(graph5, start_node5)
        print_paths(start_node5, distances5, predecessors5)
    except ValueError as e:
        print(f"  Caught expected error: {e}")
    # Expected: ValueError

    # 6. Graph with no edges from start_node (isolated start_node in a larger graph)
    print("\n--- 6. Isolated Start Node in Larger Graph ---")
    graph6 = {
        'A': [],
        'B': [('C', 1)],
        'C': [('B', 1)]
    }
    start_node6 = 'A'
    distances6, predecessors6 = dijkstra(graph6, start_node6)
    print_paths(start_node6, distances6, predecessors6)
    # Expected: A:0, B:inf, C:inf

    # 7. More complex graph
    print("\n--- 7. More Complex Graph ---")
    graph7 = {
        'S': [('A', 7), ('B', 2), ('C', 3)],
        'A': [('S', 7), ('D', 4), ('B', 3)],
        'B': [('S', 2), ('A', 3), ('D', 4), ('H', 1)],
        'C': [('S', 3), ('L', 2)],
        'D': [('A', 4), ('B', 4), ('F', 5)],
        'H': [('B', 1), ('F', 3), ('G', 2)],
        'G': [('H', 2), ('E', 2)],
        'L': [('C', 2), ('I', 4), ('J', 4)],
        'F': [('D', 5), ('H', 3)],
        'I': [('L', 4), ('J', 6), ('K', 4)],
        'J': [('L', 4), ('I', 6), ('K', 4)],
        'K': [('I', 4), ('J', 4), ('E', 5)],
        'E': [('G', 2), ('K', 5)]
    }
    start_node7 = 'S'
    distances7, predecessors7 = dijkstra(graph7, start_node7)
    print_paths(start_node7, distances7, predecessors7)
    # Expected: S:0, A:5 (S->B->A), B:2 (S->B), C:3 (S->C), D:6 (S->B->A->D or S->B->D), H:3 (S->B->H), etc.

    # 8. Graph with potential for errors
    print("\n--- 8. Graph with type errors / negative weights (expect errors) ---")
    graph_bad_neighbor = {'A': ['B']} # Malformed neighbor
    try:
        print("Testing malformed neighbor:")
        dijkstra(graph_bad_neighbor, 'A')
    except TypeError as e:
        print(f"  Caught expected error: {e}")

    graph_bad_weight_type = {'A': [('B', 'heavy')]} # Non-numeric weight
    try:
        print("Testing non-numeric weight:")
        dijkstra(graph_bad_weight_type, 'A')
    except TypeError as e:
        print(f"  Caught expected error: {e}")

    graph_neg_weight = {'A': [('B', -1)]} # Negative weight
    try:
        print("Testing negative weight:")
        dijkstra(graph_neg_weight, 'A')
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    graph_node_as_value_not_key = {'A': [('X',1)]} # X is not a key in graph
    print("Testing edge to undefined node (expect warning, X ignored):")
    # This will print a warning and X won't be in distances
    dist_undef, pred_undef = dijkstra(graph_node_as_value_not_key, 'A')
    print_paths('A', dist_undef, pred_undef)

    print("\n--- Test with a graph that is not a dict ---")
    try:
        dijkstra(["Not", "a", "dict"], "A")
    except TypeError as e:
        print(f"  Caught expected error: {e}")

    print("\n--- Test with a graph where a node's neighbors is not a list ---")
    graph_malformed_neighbors = {'A': ('B', 1)} # Should be {'A': [('B', 1)]}
    try:
        dijkstra(graph_malformed_neighbors, "A")
    except TypeError as e:
        print(f"  Caught expected error: {e}")

    print("\nAll tests finished.")
