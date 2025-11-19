import numpy as np # Using numpy for easier matrix operations

# Helper function to format a fuzzy relation matrix for printing
def format_matrix(matrix, row_labels, col_labels, title):
    """Formats a matrix (list of lists) into a clean string format."""
    output = f"--- {title} ---\n"
    # Print column headers
    header = "      " + "   ".join(col_labels)
    output += header + "\n"
    output += "    " + "-" * len(header) + "\n"
    # Print rows
    for i, row in enumerate(matrix):
        row_str = "  ".join([f"{val:g}" for val in row])
        output += f"{row_labels[i]} | [ {row_str} ]\n"
    return output

# --- 1. Define the Initial Fuzzy Sets and a Second Relation for Composition ---

# Fuzzy set A
universe_X = ['x1', 'x2', 'x3']
set_A = {
    'x1': 0.3,
    'x2': 0.7,
    'x3': 1.0
}

# Fuzzy set B
universe_Y = ['y1', 'y2']
set_B = {
    'y1': 0.4,
    'y2': 0.9
}

print("--- Given Fuzzy Sets ---")
print(f"A = {{ {', '.join([f'{v}/{k}' for k, v in set_A.items()])} }}")
print(f"B = {{ {', '.join([f'{v}/{k}' for k, v in set_B.items()])} }}\n")

# To demonstrate composition, we need a second relation, S, from Y to a new universe Z.
# Let's define relation S(Y, Z)
universe_Z = ['z1', 'z2']
relation_S = [
    # z1   z2
    [0.8, 0.2], # y1
    [0.1, 0.6]  # y2
]
print(format_matrix(relation_S, universe_Y, universe_Z, "Given Second Relation S(Y, Z)"))
print("-" * 25 + "\n")


# --- 2. Perform and Explain Each Operation ---

# (a) Cartesian Product (R = A x B)
print("--- (a) Cartesian Product (R = A x B) ---")
print("This operation creates a fuzzy relation R by combining sets A and B.")
print("The membership value for each pair (x, y) is the minimum of their individual memberships.")
print("Formula: μ_R(x, y) = min(μ_A(x), μ_B(y))\n")

relation_R = []
print("Step-by-step Calculation:")
for x_key in universe_X:
    row = []
    for y_key in universe_Y:
        val_A = set_A[x_key]
        val_B = set_B[y_key]
        result = min(val_A, val_B)
        row.append(result)
        print(f"  μ_R({x_key}, {y_key}) = min(μ_A({x_key}), μ_B({y_key})) = min({val_A}, {val_B}) = {result}")
    relation_R.append(row)

print("\nFinal Result:")
print(format_matrix(relation_R, universe_X, universe_Y, "Fuzzy Relation R = A x B"))
print("-" * 25 + "\n")


# (b) Max-Min Composition (T = R ∘ S)
print("--- (b) Max-Min Composition (T = R ∘ S) ---")
print("This operation combines relation R(X, Y) and S(Y, Z) to create a new relation T(X, Z).")
print("Formula: μ_T(x, z) = max_over_y { min(μ_R(x, y), μ_S(y, z)) }\n")

relation_T = []
print("Step-by-step Calculation:")
for i, x_key in enumerate(universe_X):
    row_T = []
    for k, z_key in enumerate(universe_Z):
        min_values = []
        calc_str = f"  μ_T({x_key}, {z_key}) = max("
        for j, y_key in enumerate(universe_Y):
            val_R = relation_R[i][j]
            val_S = relation_S[j][k]
            min_val = min(val_R, val_S)
            min_values.append(min_val)
            calc_str += f"min({val_R}, {val_S}), "

        # Final result for this element
        result = max(min_values)
        row_T.append(result)

        # Print the detailed calculation
        calc_str = calc_str[:-2] + ")" # Remove trailing comma and space
        calc_str += f" = max({', '.join(map(str, min_values))}) = {result}"
        print(calc_str)
    relation_T.append(row_T)

print("\nFinal Result:")
print(format_matrix(relation_T, universe_X, universe_Z, "Max-Min Composition T = R ∘ S"))
print("-" * 25 + "\n")


# (c) Max-Product Composition (U = R ⋅ S)
print("--- (c) Max-Product Composition (U = R ⋅ S) ---")
print("This operation is similar to Max-Min but uses multiplication instead of the minimum.")
print("Formula: μ_U(x, z) = max_over_y { μ_R(x, y) * μ_S(y, z) }\n")

relation_U = []
print("Step-by-step Calculation:")
for i, x_key in enumerate(universe_X):
    row_U = []
    for k, z_key in enumerate(universe_Z):
        prod_values = []
        calc_str = f"  μ_U({x_key}, {z_key}) = max("
        for j, y_key in enumerate(universe_Y):
            val_R = relation_R[i][j]
            val_S = relation_S[j][k]
            prod_val = val_R * val_S
            prod_values.append(prod_val)
            calc_str += f"({val_R} * {val_S}), "

        # Final result for this element
        result = max(prod_values)
        row_U.append(round(result, 4)) # Round for clean display

        # Print the detailed calculation
        calc_str = calc_str[:-2] + ")" # Remove trailing comma and space
        calc_str += f" = max({', '.join([f'{v:.2f}' for v in prod_values])}) = {result:.2f}"
        print(calc_str)
    relation_U.append(row_U)

print("\nFinal Result:")
print(format_matrix(relation_U, universe_X, universe_Z, "Max-Product Composition U = R ⋅ S"))
print("-" * 25 + "\n")