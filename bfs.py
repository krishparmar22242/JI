# File: bfs_tsp_visual_trace.py

import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# --- Visualization Functions ---

def visualize_graph(graph_data, title="Graph"):
    """Visualizes the initial graph layout."""
    G = nx.Graph()
    for node, edges in graph_data.items():
        G.add_node(node)
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42) # Use a seed for consistent layout
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title, size=15)
    plt.show()

def visualize_step(graph_data, current_path, current_cost, step_num):
    """Visualizes a single step of the search, highlighting the current path."""
    G = nx.Graph()
    for node, edges in graph_data.items():
        G.add_node(node)
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42) # Use the same seed for consistency
    path_edges = list(zip(current_path, current_path[1:]))

    # Draw the base graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, edge_color='gray', style='dashed')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight the current path being processed
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3)

    # Redraw nodes on top
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    title = (f"BFS - Step {step_num}\n"
             f"Processing: {' -> '.join(current_path)}\n"
             f"Cost so far: {current_cost}")
    plt.title(title)
    plt.show() # Pauses execution until the plot is closed

def visualize_final_path(graph_data, path, cost):
    """Visualizes the final, optimal path found."""
    if not path:
        print("BFS: No solution path found to visualize.")
        return
    visualize_path(graph_data, path, cost, "BFS Final Optimal Path")

# Re-using the generic path visualizer for the final result
def visualize_path(graph_data, path, cost, title=""):
    """Generic function to visualize a path in red."""
    G = nx.Graph()
    for node in graph_data:
        G.add_node(node)
        for neighbor, weight in graph_data[node].items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700)
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"{title}\nPath: {' -> '.join(path)}\nCost: {cost}", size=15)
    plt.show()

# --- BFS Algorithm with Visual Tracing ---

def bfs_tsp_visual_trace(graph, start_node):
    """Solves TSP using BFS with step-by-step visual tracing."""
    queue = deque([([start_node], 0)])
    min_cost = float('inf')
    best_path = None
    num_nodes = len(graph)
    step = 0

    while queue:
        step += 1
        current_path, current_cost = queue.popleft()

        # Visualize the current step
        visualize_step(graph, current_path, current_cost, step)

        if len(current_path) == num_nodes:
            last_node = current_path[-1]
            if start_node in graph[last_node]:
                final_cost = current_cost + graph[last_node][start_node]
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_path = current_path + [start_node]
            continue

        last_node = current_path[-1]
        for neighbor, weight in graph[last_node].items():
            if neighbor not in current_path:
                new_cost = current_cost + weight
                if new_cost < min_cost:
                    new_path = current_path + [neighbor]
                    queue.append((new_path, new_cost))

    return best_path, min_cost

# --- Main Execution Block ---

if __name__ == '__main__':
    graph_data = {
        'A': {'B': 10, 'C': 15, 'D': 20},
        'B': {'A': 10, 'C': 35, 'D': 25},
        'C': {'A': 15, 'B': 35, 'D': 30},
        'D': {'A': 20, 'B': 25, 'C': 30}
    }
    start_node = 'A'

    # 1. Show the initial problem graph
    visualize_graph(graph_data, title="TSP Problem: Initial Cities")

    # 2. Run the BFS algorithm with visual tracing
    bfs_path, bfs_cost = bfs_tsp_visual_trace(graph_data, start_node)

    # 3. Show the final result
    print(f"BFS Search Complete. Final Path: {bfs_path}, Cost: {bfs_cost}")
    visualize_final_path(graph_data, bfs_path, bfs_cost)