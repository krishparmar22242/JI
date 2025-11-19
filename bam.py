import numpy as np

# --- Helper Functions (Simple Output) ---

def display_pattern(vector, shape=(5, 3), title="Pattern:"):
    """Displays the vector as a 2D pattern of 1 and -1."""
    print(title)
    grid = vector.reshape(shape)
    for row in grid:
        print(" ".join(str(int(v)) for v in row))
    print()

def activation_function(net_input):
    """Bipolar activation function."""
    return np.where(net_input > 0, 1, -1)

# --- BAM Network Functions ---

def train_bam(associations):
    """Constructs the BAM weight matrix from associations."""
    print("--- Training BAM Network ---")

    first_key = next(iter(associations))
    s_len = len(associations[first_key][0])
    t_len = len(associations[first_key][1])
    weight_matrix = np.zeros((s_len, t_len))

    print("Formula: W = Σ (s_p^T * t_p)\n")

    for name, (s, t) in associations.items():
        print(f"Association: {name}")
        print("Input vector s:", s)
        print("Target vector t:", t)

        s_col = s.reshape(-1, 1)
        t_row = t.reshape(1, -1)

        w_individual = np.dot(s_col, t_row)
        print("Individual weight matrix W_{}:".format(name))
        print(w_individual, "\n")

        weight_matrix += w_individual

    print("Final Weight Matrix W:")
    print(weight_matrix)
    print("-" * 50)
    return weight_matrix

def test_bam_forward(name, s_input, weight_matrix, t_expected):
    """Forward pass (s → t)."""
    print(f"\n=== Forward Test for '{name}' ===")
    display_pattern(s_input, title=f"Input Pattern '{name}':")

    print("Step 1: Compute y_in = s * W")
    y_in = np.dot(s_input, weight_matrix)
    print("y_in =", y_in)

    print("\nStep 2: Apply activation function")
    y_recalled = activation_function(y_in)
    print("Recalled t =", y_recalled)

    print("\nExpected t:", t_expected)
    print("Recalled t:", y_recalled)

    if np.array_equal(y_recalled, t_expected):
        print("Result: SUCCESS")
    else:
        print("Result: FAILURE")
    print("-" * 50)

def test_bam_backward(name, t_input, weight_matrix, s_expected):
    """Backward pass (t → s)."""
    print(f"\n=== Backward Test for '{name}' ===")
    print("Target vector t:", t_input)

    print("\nStep 1: Compute x_in = t * W^T")
    x_in = np.dot(t_input, weight_matrix.T)
    print("x_in =", x_in)

    print("\nStep 2: Apply activation function")
    s_recalled = activation_function(x_in)
    print("Recalled s =", s_recalled)

    print("\nExpected Pattern:")
    display_pattern(s_expected)

    print("Recalled Pattern:")
    display_pattern(s_recalled)

    if np.array_equal(s_recalled, s_expected):
        print("Result: SUCCESS")
    else:
        print("Result: FAILURE")
    print("-" * 50)

# --- Main Execution ---

if __name__ == "__main__":
    # Input vectors
    s_E = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1])
    t_E = np.array([-1, 1])

    s_F = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1])
    t_F = np.array([1, 1])

    associations = {"E": (s_E, t_E), "F": (s_F, t_F)}

    print("--- BAM Network Initialization ---\n")

    display_pattern(s_E, title="Pattern E:")
    display_pattern(s_F, title="Pattern F:")

    # Training
    W = train_bam(associations)

    # Testing
    print("--- Testing All Associations ---")

    test_bam_forward("E", s_E, W, t_E)
    test_bam_forward("F", s_F, W, t_F)
    test_bam_backward("E", t_E, W, s_E)
    test_bam_backward("F", t_F, W, s_F)

    print("\n--- All Tests Complete ---")
