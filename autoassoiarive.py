import numpy as np

def train_autoassociative_network(input_vector):
    """
    Trains the network and calculates the weight matrix using the outer product rule.

    Args:
        input_vector (np.array): The input pattern to be stored.

    Returns:
        np.array: The calculated weight matrix.
    """
    print("--- 1. Training the Network ---")
    print(f"The input vector is s = {input_vector}")

    # Reshape the vector to be a column vector for the formula s^T * s
    # In numpy, np.outer(s, s) achieves this directly.
    input_vector_col = input_vector.reshape(-1, 1)

    print("\nThe weight matrix W is calculated using the formula: W = s^T * s")
    print("Where s^T is the column vector (transpose) and s is the row vector.")

    # Calculate the weight matrix using the outer product
    weight_matrix = np.outer(input_vector, input_vector)

    print("\nCalculation:")
    print(f"W = \n{input_vector_col}\n    * \n    {input_vector}\n")
    print(f"Resulting Weight Matrix W:\n{weight_matrix}")
    print("-" * 30 + "\n")
    return weight_matrix

def test_network(test_vector, weight_matrix, original_vector):
    """
    Tests the network with a given test vector.

    Args:
        test_vector (np.array): The input vector for testing.
        weight_matrix (np.array): The trained weight matrix.
        original_vector (np.array): The original vector for comparison.
    """
    print(f"\n--- Testing with input vector: {test_vector} ---")

    # Step 1: Compute the net input
    print("\nStep A: Compute the input to the output layer using the formula: y_in = test_input * W")
    y_in = np.dot(test_vector, weight_matrix)
    print(f"y_in = {test_vector} * \n{weight_matrix}")
    print(f"y_in = {y_in}")

    # Step 2: Apply the activation function
    print("\nStep B: Apply the activation function to get the output vector y_out.")
    print("The activation function is a sign function:")
    print("  - f(x) = 1  if x >= 0")
    print("  - f(x) = -1 if x < 0")

    # Apply activation function
    y_out = np.where(y_in >= 0, 1, -1)

    print(f"\nApplying activation to y_in = {y_in} results in:")
    print(f"y_out = {y_out}")

    # Step 3: Compare the result
    print("\nStep C: Compare the output with the original stored vector.")
    print(f"Original Vector: {original_vector}")
    print(f"Recalled Vector: {y_out}")

    if np.array_equal(y_out, original_vector):
        print("Result: The network successfully recalled the original vector.")
    else:
        print("Result: The network did not recall the original vector.")
    print("-" * 30)


# Main program execution
if __name__ == "__main__":
    # Define the original input vector for training
    s = np.array([-1, 1, 1, 1])

    # 1. Train the network
    W = train_autoassociative_network(s)

    # 2. Define the test cases
    print("--- 2. Testing the Network with Various Inputs ---")

    test_cases = {
        "Same Input Vector": np.array([-1, 1, 1, 1]),
        "One Missing Entry (represented by 0)": np.array([0, 1, 1, 1]),
        "One Mistake Entry (flipped sign)": np.array([1, 1, 1, 1]),
        "Two Missing Entries": np.array([0, 0, 1, 1]),
        "Two Mistake Entries": np.array([1, -1, 1, 1])
    }

    # Run all test cases
    for description, test_vec in test_cases.items():
        print(f"\n{'='*10} Test Case: {description} {'='*10}")
        test_network(test_vector=test_vec, weight_matrix=W, original_vector=s)