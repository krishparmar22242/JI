import math
import matplotlib.pyplot as plt
import networkx as nx

def sigmoid(x):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(output):
    """
    Derivative of the Sigmoid activation function.
    Note: Takes the output of the sigmoid function as input.
    f'(x) = f(x) * (1 - f(x))
    """
    return output * (1 - output)

# --- Initial Values from the Image ---
# Input Layer
x1 = 0
x2 = 1

# Target Output
t = 1

# --- Hidden Layer Weights ---
v11 = 0.6
v21 = -0.1
v01 = 0.3  # Bias for Z1

v12 = -0.3
v22 = 0.4
v02 = 0.5  # Bias for Z2

# --- Output Layer Weights ---
w1 = 0.4
w2 = 0.1
w0 = -0.2  # Bias for Y

# Learning Rate
alpha = 0.25

print("--- Initial Parameters ---")
print(f"Inputs: x1 = {x1}, x2 = {x2}")
print(f"Target: t = {t}")
print(f"Learning Rate (alpha): {alpha}\n")
print("Initial Weights:")
print(f"  v11 = {v11}, v21 = {v21}, v01 = {v01}")
print(f"  v12 = {v12}, v22 = {v22}, v02 = {v02}")
print(f"  w1 = {w1}, w2 = {w2}, w0 = {w0}\n")

# ===============================================
# --- Forward Propagation ---
# ===============================================
print("--- Step 1: Forward Propagation ---\n")

# --- Hidden Layer Calculations ---
print("  --- Hidden Layer ---")
# Net input to Z1
z_in1 = v01 + (x1 * v11) + (x2 * v21)
print(f"  Net input to Z1 (z_in1) = {v01} + ({x1} * {v11}) + ({x2} * {v21}) = {z_in1:.4f}")

# Output of Z1
z1 = sigmoid(z_in1)
print(f"  Output of Z1 (z1) = f({z_in1:.4f}) = {z1:.4f}\n")

# Net input to Z2
z_in2 = v02 + (x1 * v12) + (x2 * v22)
print(f"  Net input to Z2 (z_in2) = {v02} + ({x1} * {v12}) + ({x2} * {v22}) = {z_in2:.4f}")

# Output of Z2
z2 = sigmoid(z_in2)
print(f"  Output of Z2 (z2) = f({z_in2:.4f}) = {z2:.4f}\n")

# --- Output Layer Calculations ---
print("  --- Output Layer ---")
# Net input to Y
y_in = w0 + (z1 * w1) + (z2 * w2)
print(f"  Net input to Y (y_in) = {w0} + ({z1:.4f} * {w1}) + ({z2:.4f} * {w2}) = {y_in:.4f}")

# Final Output of Y
y = sigmoid(y_in)
print(f"  Final Output (y) = f({y_in:.4f}) = {y:.4f}\n")

# ===============================================
# --- Backward Propagation ---
# ===============================================
print("--- Step 2: Backward Propagation (Error Calculation and Weight Updates) ---\n")

# --- Error Calculation at the Output Layer ---
print("  --- Error Calculation ---")
# Error
error = t - y
print(f"  Error = t - y = {t} - {y:.4f} = {error:.4f}\n")

# --- Weight Updates for the Output Layer ---
print("  --- Output Layer Weight Updates ---")
# Error term for the output neuron
delta_k = error * sigmoid_derivative(y)
print(f"  Error term for output neuron (delta_k) = (t - y) * f'(y_in) = {error:.4f} * {sigmoid_derivative(y):.4f} = {delta_k:.4f}\n")

# Update weights for w1, w2, and w0
delta_w1 = alpha * delta_k * z1
delta_w2 = alpha * delta_k * z2
delta_w0 = alpha * delta_k

w1_new = w1 + delta_w1
w2_new = w2 + delta_w2
w0_new = w0 + delta_w0

print(f"  Change in w1 (delta_w1) = alpha * delta_k * z1 = {alpha} * {delta_k:.4f} * {z1:.4f} = {delta_w1:.4f}")
print(f"  New w1 = {w1} + {delta_w1:.4f} = {w1_new:.4f}\n")
print(f"  Change in w2 (delta_w2) = alpha * delta_k * z2 = {alpha} * {delta_k:.4f} * {z2:.4f} = {delta_w2:.4f}")
print(f"  New w2 = {w2} + {delta_w2:.4f} = {w2_new:.4f}\n")
print(f"  Change in w0 (delta_w0) = alpha * delta_k = {alpha} * {delta_k:.4f} = {delta_w0:.4f}")
print(f"  New w0 = {w0} + {delta_w0:.4f} = {w0_new:.4f}\n")

# --- Weight Updates for the Hidden Layer ---
print("  --- Hidden Layer Weight Updates ---")
# Error terms for the hidden layer neurons
delta_in1 = delta_k * w1
delta_in2 = delta_k * w2
print(f"  Error contribution of Z1 (delta_in1) = delta_k * w1 = {delta_k:.4f} * {w1} = {delta_in1:.4f}")
print(f"  Error contribution of Z2 (delta_in2) = delta_k * w2 = {delta_k:.4f} * {w2} = {delta_in2:.4f}\n")

delta_j1 = delta_in1 * sigmoid_derivative(z1)
delta_j2 = delta_in2 * sigmoid_derivative(z2)
print(f"  Error term for Z1 (delta_j1) = delta_in1 * f'(z_in1) = {delta_in1:.4f} * {sigmoid_derivative(z1):.4f} = {delta_j1:.4f}")
print(f"  Error term for Z2 (delta_j2) = delta_in2 * f'(z_in2) = {delta_in2:.4f} * {sigmoid_derivative(z2):.4f} = {delta_j2:.4f}\n")

# Update weights for v11, v21, v01
delta_v11 = alpha * delta_j1 * x1
delta_v21 = alpha * delta_j1 * x2
delta_v01 = alpha * delta_j1

v11_new = v11 + delta_v11
v21_new = v21 + delta_v21
v01_new = v01 + delta_v01

print(f"  Change in v11 (delta_v11) = alpha * delta_j1 * x1 = {alpha} * {delta_j1:.4f} * {x1} = {delta_v11:.4f}")
print(f"  New v11 = {v11} + {delta_v11:.4f} = {v11_new:.4f}\n")
print(f"  Change in v21 (delta_v21) = alpha * delta_j1 * x2 = {alpha} * {delta_j1:.4f} * {x2} = {delta_v21:.4f}")
print(f"  New v21 = {v21} + {delta_v21:.4f} = {v21_new:.4f}\n")
print(f"  Change in v01 (delta_v01) = alpha * delta_j1 = {alpha} * {delta_j1:.4f} = {delta_v01:.4f}")
print(f"  New v01 = {v01} + {delta_v01:.4f} = {v01_new:.4f}\n")

# Update weights for v12, v22, v02
delta_v12 = alpha * delta_j2 * x1
delta_v22 = alpha * delta_j2 * x2
delta_v02 = alpha * delta_j2

v12_new = v12 + delta_v12
v22_new = v22 + delta_v22
v02_new = v02 + delta_v02

print(f"  Change in v12 (delta_v12) = alpha * delta_j2 * x1 = {alpha} * {delta_j2:.4f} * {x1} = {delta_v12:.4f}")
print(f"  New v12 = {v12} + {delta_v12:.4f} = {v12_new:.4f}\n")
print(f"  Change in v22 (delta_v22) = alpha * delta_j2 * x2 = {alpha} * {delta_j2:.4f} * {x2} = {delta_v22:.4f}")
print(f"  New v22 = {v22} + {delta_v22:.4f} = {v22_new:.4f}\n")
print(f"  Change in v02 (delta_v02) = alpha * delta_j2 = {alpha} * {delta_j2:.4f} = {delta_v02:.4f}")
print(f"  New v02 = {v02} + {delta_v02:.4f} = {v02_new:.4f}\n")

print("--- Final Updated Weights ---")
print(f"  v11 = {v11_new:.4f}, v21 = {v21_new:.4f}, v01 = {v01_new:.4f}")
print(f"  v12 = {v12_new:.4f}, v22 = {v22_new:.4f}, v02 = {v02_new:.4f}")
print(f"  w1 = {w1_new:.4f}, w2 = {w2_new:.4f}, w0 = {w0_new:.4f}")

# --- Visualization ---

def draw_network():
    G = nx.DiGraph()

    # Nodes and layers
    layers = {
        'Input': ['x1=0', 'x2=1'],
        'Hidden': [f'Z1={z1:.2f}', f'Z2={z2:.2f}'],
        'Output': [f'Y={y:.2f}']
    }

    pos = {}
    x_gap = 3
    y_gap = 2

    # Position input layer nodes
    for i, node in enumerate(layers['Input']):
        G.add_node(node, layer='Input')
        pos[node] = (0, i * y_gap)

    # Position hidden layer nodes
    for i, node in enumerate(layers['Hidden']):
        G.add_node(node, layer='Hidden')
        pos[node] = (x_gap, i * y_gap)

    # Position output layer node
    for i, node in enumerate(layers['Output']):
        G.add_node(node, layer='Output')
        pos[node] = (2 * x_gap, 0.5 * y_gap)

    # Add edges with weights (showing updated weights)
    edges = [
        ('x1=0', f'Z1={z1:.2f}', v11_new),
        ('x2=1', f'Z1={z1:.2f}', v21_new),
        ('x1=0', f'Z2={z2:.2f}', v12_new),
        ('x2=1', f'Z2={z2:.2f}', v22_new),

        (f'Z1={z1:.2f}', f'Y={y:.2f}', w1_new),
        (f'Z2={z2:.2f}', f'Y={y:.2f}', w2_new),
    ]

    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges with weights as labels
    edge_labels = {(src, dst): f"{weight:.2f}" for src, dst, weight in edges}
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Neural Network Forward Pass with Updated Weights")
    plt.axis('off')
    plt.show()

draw_network()
