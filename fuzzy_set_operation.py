# Helper function to format a fuzzy set dictionary for printing
def format_fuzzy_set(fs_dict, universe_order, precision=2):
    """Formats a fuzzy set dictionary into the required string format."""
    # Using a general format specifier to handle integers and floats cleanly
    parts = [f"{fs_dict[elem]:g}/{elem}" for elem in universe_order]
    return "{ " + " + ".join(parts) + " }"

# --- 1. Define the Initial Fuzzy Sets and Universe of Discourse ---

# The universe of discourse (the set of mach numbers)
universe = ['0.64', '0.645', '0.65', '0.655', '0.66']

# Fuzzy set A: "near mach 0.65"
set_A = {
    '0.64': 0,
    '0.645': 0.75,
    '0.65': 1,
    '0.655': 0.5,
    '0.66': 0
}

# Fuzzy set B: "in the region of mach 0.65"
set_B = {
    '0.64': 0,
    '0.645': 0.25,
    '0.65': 0.75,
    '0.655': 1,
    '0.66': 0.5
}

print("--- Given Fuzzy Sets ---")
print(f"A = {format_fuzzy_set(set_A, universe)}")
print(f"B = {format_fuzzy_set(set_B, universe)}")
print("-" * 25 + "\n")

# --- 2. Perform and Explain Each Operation ---

# --- Operations from the Problem Image ---

# (a) A ∪ B (Union)
print("--- (a) A ∪ B (Standard Union) ---")
print("Formula: μ(x) = max(μ_A(x), μ_B(x))\n")
union_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = max(set_A[elem], set_B[elem])
    union_result[elem] = result
    print(f"  - For mach '{elem}': max({set_A[elem]}, {set_B[elem]}) = {result}")
print("\nFinal Result:")
print(f"A ∪ B = {format_fuzzy_set(union_result, universe)}\n")
print("-" * 25 + "\n")

# (b) A ∩ B (Intersection)
print("--- (b) A ∩ B (Standard Intersection) ---")
print("Formula: μ(x) = min(μ_A(x), μ_B(x))\n")
intersection_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = min(set_A[elem], set_B[elem])
    intersection_result[elem] = result
    print(f"  - For mach '{elem}': min({set_A[elem]}, {set_B[elem]}) = {result}")
print("\nFinal Result:")
print(f"A ∩ B = {format_fuzzy_set(intersection_result, universe)}\n")
print("-" * 25 + "\n")

# (c) A Complement (Ā)
print("--- (c) A Complement (Ā) ---")
print("Formula: μ(x) = 1 - μ_A(x)\n")
A_comp_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = 1 - set_A[elem]
    A_comp_result[elem] = result
    print(f"  - For mach '{elem}': 1 - {set_A[elem]} = {result}")
print("\nFinal Result:")
print(f"Ā = {format_fuzzy_set(A_comp_result, universe)}\n")
print("-" * 25 + "\n")

# (d) B Complement (B̄)
print("--- (d) B Complement (B̄) ---")
print("Formula: μ(x) = 1 - μ_B(x)\n")
B_comp_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = 1 - set_B[elem]
    B_comp_result[elem] = result
    print(f"  - For mach '{elem}': 1 - {set_B[elem]} = {result}")
print("\nFinal Result:")
print(f"B̄ = {format_fuzzy_set(B_comp_result, universe)}\n")
print("-" * 25 + "\n")

# (e) (A ∪ B)̄ (Complement of the Union)
print("--- (e) (A ∪ B)̄ ---")
print("Formula: μ(x) = 1 - max(μ_A(x), μ_B(x))\n")
comp_union_result = {elem: 1 - union_result[elem] for elem in universe}
print("Step-by-step Calculation (using result from part a):")
for elem in universe:
    print(f"  - For mach '{elem}': 1 - {union_result[elem]} = {comp_union_result[elem]}")
print("\nFinal Result:")
print(f"(A ∪ B)̄ = {format_fuzzy_set(comp_union_result, universe)}\n")
print("-" * 25 + "\n")

# (f) (A ∩ B)̄ (Complement of the Intersection)
print("--- (f) (A ∩ B)̄ ---")
print("Formula: μ(x) = 1 - min(μ_A(x), μ_B(x))\n")
comp_intersection_result = {elem: 1 - intersection_result[elem] for elem in universe}
print("Step-by-step Calculation (using result from part b):")
for elem in universe:
    print(f"  - For mach '{elem}': 1 - {intersection_result[elem]} = {comp_intersection_result[elem]}")
print("\nFinal Result:")
print(f"(A ∩ B)̄ = {format_fuzzy_set(comp_intersection_result, universe)}\n")
print("-" * 25 + "\n")

# --- Additional Operations for a Comprehensive Analysis ---

# (g) A ∪ B̄ (Union of A and B Complement)
print("--- (g) A ∪ B̄ ---")
print("Formula: μ(x) = max(μ_A(x), μ_B̄(x)) = max(μ_A(x), 1 - μ_B(x))\n")
g_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = max(set_A[elem], B_comp_result[elem])
    g_result[elem] = result
    print(f"  - For mach '{elem}': max({set_A[elem]}, {B_comp_result[elem]}) = {result}")
print("\nFinal Result:")
print(f"A ∪ B̄ = {format_fuzzy_set(g_result, universe)}\n")
print("-" * 25 + "\n")

# (h) Ā ∩ B (Intersection of A Complement and B)
print("--- (h) Ā ∩ B ---")
print("Formula: μ(x) = min(μ_Ā(x), μ_B(x)) = min(1 - μ_A(x), μ_B(x))\n")
h_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = min(A_comp_result[elem], set_B[elem])
    h_result[elem] = result
    print(f"  - For mach '{elem}': min({A_comp_result[elem]}, {set_B[elem]}) = {result}")
print("\nFinal Result:")
print(f"Ā ∩ B = {format_fuzzy_set(h_result, universe)}\n")
print("-" * 25 + "\n")

# (i) A ∪ Ā (Law of Excluded Middle for A)
print("--- (i) A ∪ Ā ---")
print("Formula: μ(x) = max(μ_A(x), 1 - μ_A(x))\n")
i_result = {elem: max(set_A[elem], A_comp_result[elem]) for elem in universe}
print("Step-by-step Calculation:")
for elem in universe:
    print(f"  - For mach '{elem}': max({set_A[elem]}, {A_comp_result[elem]}) = {i_result[elem]}")
print("\nFinal Result:")
print(f"A ∪ Ā = {format_fuzzy_set(i_result, universe)}\n")
print("-" * 25 + "\n")

# (j) A ∩ Ā (Law of Contradiction for A)
print("--- (j) A ∩ Ā ---")
print("Formula: μ(x) = min(μ_A(x), 1 - μ_A(x))\n")
j_result = {elem: min(set_A[elem], A_comp_result[elem]) for elem in universe}
print("Step-by-step Calculation:")
for elem in universe:
    print(f"  - For mach '{elem}': min({set_A[elem]}, {A_comp_result[elem]}) = {j_result[elem]}")
print("\nFinal Result:")
print(f"A ∩ Ā = {format_fuzzy_set(j_result, universe)}\n")
print("-" * 25 + "\n")

# (k) B ∪ B̄ (Law of Excluded Middle for B)
print("--- (k) B ∪ B̄ ---")
print("Formula: μ(x) = max(μ_B(x), 1 - μ_B(x))\n")
k_result = {elem: max(set_B[elem], B_comp_result[elem]) for elem in universe}
print("Step-by-step Calculation:")
for elem in universe:
    print(f"  - For mach '{elem}': max({set_B[elem]}, {B_comp_result[elem]}) = {k_result[elem]}")
print("\nFinal Result:")
print(f"B ∪ B̄ = {format_fuzzy_set(k_result, universe)}\n")
print("-" * 25 + "\n")

# (l) B ∩ B̄ (Law of Contradiction for B)
print("--- (l) B ∩ B̄ ---")
print("Formula: μ(x) = min(μ_B(x), 1 - μ_B(x))\n")
l_result = {elem: min(set_B[elem], B_comp_result[elem]) for elem in universe}
print("Step-by-step Calculation:")
for elem in universe:
    print(f"  - For mach '{elem}': min({set_B[elem]}, {B_comp_result[elem]}) = {l_result[elem]}")
print("\nFinal Result:")
print(f"B ∩ B̄ = {format_fuzzy_set(l_result, universe)}\n")
print("-" * 25 + "\n")


# (m) Algebraic Sum (A ⊕ B)
print("--- (m) Algebraic Sum (A ⊕ B) ---")
print("Formula: μ(x) = μ_A(x) + μ_B(x) - (μ_A(x) * μ_B(x))\n")
algebraic_sum_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    val_A = set_A[elem]
    val_B = set_B[elem]
    result = val_A + val_B - (val_A * val_B)
    algebraic_sum_result[elem] = result
    print(f"  - For mach '{elem}': {val_A} + {val_B} - ({val_A} * {val_B}) = {result}")
print("\nFinal Result:")
print(f"A ⊕ B = {format_fuzzy_set(algebraic_sum_result, universe)}\n")
print("-" * 25 + "\n")

# (n) Algebraic Product (A ⋅ B)
print("--- (n) Algebraic Product (A ⋅ B) ---")
print("Formula: μ(x) = μ_A(x) * μ_B(x)\n")
algebraic_prod_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    result = set_A[elem] * set_B[elem]
    algebraic_prod_result[elem] = result
    print(f"  - For mach '{elem}': {set_A[elem]} * {set_B[elem]} = {result}")
print("\nFinal Result:")
print(f"A ⋅ B = {format_fuzzy_set(algebraic_prod_result, universe)}\n")
print("-" * 25 + "\n")

# (o) Bounded Sum (A ⊎ B)
print("--- (o) Bounded Sum (A ⊎ B) ---")
print("Formula: μ(x) = min(1, μ_A(x) + μ_B(x))\n")
bounded_sum_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    val_A = set_A[elem]
    val_B = set_B[elem]
    result = min(1, val_A + val_B)
    bounded_sum_result[elem] = result
    print(f"  - For mach '{elem}': min(1, {val_A} + {val_B}) = min(1, {val_A + val_B}) = {result}")
print("\nFinal Result:")
print(f"A ⊎ B = {format_fuzzy_set(bounded_sum_result, universe)}\n")
print("-" * 25 + "\n")

# (p) Bounded Difference (A ⊖ B)
print("--- (p) Bounded Difference (A ⊖ B) ---")
print("Formula: μ(x) = max(0, μ_A(x) - μ_B(x))\n")
bounded_diff_result = {}
print("Step-by-step Calculation:")
for elem in universe:
    val_A = set_A[elem]
    val_B = set_B[elem]
    result = max(0, val_A - val_B)
    bounded_diff_result[elem] = result
    print(f"  - For mach '{elem}': max(0, {val_A} - {val_B}) = max(0, {val_A - val_B}) = {result}")
print("\nFinal Result:")
print(f"A ⊖ B = {format_fuzzy_set(bounded_diff_result, universe)}\n")
print("-" * 25 + "\n")