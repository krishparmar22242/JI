import heapq
import itertools
import matplotlib.pyplot as plt
import networkx as nx

# --- Visualization Functions (Modified Titles for Clarity) ---

def visualize_step(graph_data, path, g_cost, h_cost, step_num):
    """Visualizes a single step of the A* search, highlighting the current path."""
    G = nx.Graph()
    for node, edges in graph_data.items():
        G.add_node(node)
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42) # Use a seed for consistent layout
    path_edges = list(zip(path, path[1:]))

    # Draw the base graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight the current path being processed
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    f_cost = g_cost + h_cost
    title = (f"A* Step {step_num}: Processing Route\n"
             f"Route: {' -> '.join(path)}\n"
             f"f(n) = {f_cost:.2f} (g={g_cost} + h={h_cost:.2f})")
    plt.title(title)
    plt.show()

def visualize_final_path(graph_data, path, cost, title):
    """Visualizes the final, optimal path found."""
    G = nx.Graph()
    for node, edges in graph_data.items():
        G.add_node(node)
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    path_edges = list(zip(path, path[1:]))

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if path: # If there is a path, it's the solution
        final_title = (f"{title}\n"
                       f"Route: {' -> '.join(path)}\n"
                       f"Total Walking Time: {cost} minutes")
    else: # Otherwise, it's the initial problem display
        final_title = title

    plt.title(final_title, size=15)
    plt.show()

# --- Helper function for MST calculation (Prim's Algorithm) ---
def calculate_mst_cost(nodes, graph):
    if not nodes: return 0
    nodes = set(nodes)
    start_node_mst = list(nodes)[0]
    visited = {start_node_mst}
    edges = []
    for neighbor, weight in graph[start_node_mst].items():
        if neighbor in nodes:
            heapq.heappush(edges, (weight, start_node_mst, neighbor))
    mst_cost = 0
    while edges and len(visited) < len(nodes):
        weight, _, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst_cost += weight
            for neighbor, new_weight in graph[to].items():
                if neighbor in nodes and neighbor not in visited:
                    heapq.heappush(edges, (new_weight, to, neighbor))
    return mst_cost

# --- A* Algorithm for TSP (Universal Logic) ---
def solve_tsp_a_star_with_viz(graph, start_node):
    all_nodes = set(graph.keys())
    open_list = [(0, 0, start_node, (start_node,))]
    visited = {}
    step = 0

    print("--- Starting A* Search for Campus Errands ---")

    while open_list:
        step += 1
        f_cost_ignored, g_cost, current_node, path = heapq.heappop(open_list)
        path_set = set(path)

        unvisited_nodes_viz = all_nodes - path_set
        mst_cost_viz = calculate_mst_cost(unvisited_nodes_viz, graph)
        min_to_unvisited_viz = min((graph[current_node][n] for n in unvisited_nodes_viz), default=0)
        min_from_unvisited_to_start_viz = min((graph[n][start_node] for n in unvisited_nodes_viz), default=0)
        h_cost_viz = mst_cost_viz + min_to_unvisited_viz + min_from_unvisited_to_start_viz

        print(f"\n--- Step {step} ---")
        print(f"Choosing route to expand: {' -> '.join(path)}")
        print(f"  - Time spent so far (g(n)): {g_cost} min")
        print(f"  - Estimated time remaining (h(n)): {h_cost_viz} min")
        print(f"  - Total estimated time (f(n)): {g_cost + h_cost_viz} min")

        visualize_step(graph, path, g_cost, h_cost_viz, step)

        if (current_node, path) in visited and visited[(current_node, path)] <= g_cost:
            print("  - Route already found with a shorter or equal time. Pruning.")
            continue

        visited[(current_node, path)] = g_cost

        if len(path) == len(all_nodes):
            final_cost = g_cost + graph[current_node][start_node]
            print("\n*** Found the optimal route! ***")
            return path + (start_node,), final_cost

        unvisited_nodes = all_nodes - path_set
        for neighbor in unvisited_nodes:
            new_g_cost = g_cost + graph[current_node][neighbor]
            new_path = path + (neighbor,)

            new_unvisited = all_nodes - set(new_path)
            new_mst_cost = calculate_mst_cost(new_unvisited, graph)
            new_min_to_unvisited = min((graph[neighbor][n] for n in new_unvisited), default=0)
            new_min_from_unvisited_to_start = min((graph[n][start_node] for n in new_unvisited), default=0)

            new_h_cost = new_mst_cost + new_min_to_unvisited + new_min_from_unvisited_to_start
            new_f_cost = new_g_cost + new_h_cost

            heapq.heappush(open_list, (new_f_cost, new_g_cost, neighbor, new_path))

    return None, float('inf')

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define the campus locations and walking times (structurally identical to the A,B,C,D graph)
    campus_graph = {
        'Dorm':      {'Library': 10, 'Cafeteria': 15, 'Gym': 20},
        'Library':   {'Dorm': 10, 'Cafeteria': 35, 'Gym': 25},
        'Cafeteria': {'Dorm': 15, 'Library': 35, 'Gym': 30},
        'Gym':       {'Dorm': 20, 'Library': 25, 'Cafeteria': 30}
    }

    start_location = 'Dorm'

    # Show the initial problem
    visualize_final_path(campus_graph, [], 0, "Campus Errands: Walking Times")

    # Run the A* algorithm to find the most efficient route
    solution_path, solution_cost = solve_tsp_a_star_with_viz(campus_graph, start_location)

    if solution_path:
        print("\n--- A* SEARCH COMPLETE ---")
        print("Optimal route found for the student.")
        print(f"Route: {' -> '.join(solution_path)}")
        print(f"Total Walking Time: {solution_cost} minutes")

        # Show the final solution path
        visualize_final_path(campus_graph, solution_path, solution_cost, "Optimal Errand Route")
    else:
        print("No solution was found.")