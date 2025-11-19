# ---------------------------------------------------------
# McCulloch–Pitts Neuron Model
# ---------------------------------------------------------
def mcp_neuron(inputs, weights, threshold):
    """
    Compute MCP neuron output:
    Net = Σ (wi * xi)
    Output = 1 if Net >= threshold else 0
    """
    net = sum(i * w for i, w in zip(inputs, weights))
    output = 1 if net >= threshold else 0
    return net, output


# ---------------------------------------------------------
# AND Gate using MCP Neuron
# ---------------------------------------------------------
def and_gate():
    print("\n===== AND GATE USING McCULLOCH–PITTS NEURON =====")
    weights = [1, 1]
    threshold = 2

    print("Weights = [1, 1], Threshold = 2\n")

    print("X1 X2 | Net | Output")
    print("----------------------")

    for x1 in [0, 1]:
        for x2 in [0, 1]:
            net, out = mcp_neuron([x1, x2], weights, threshold)
            print(f"{x1}  {x2}  |  {net}  |   {out}")


# ---------------------------------------------------------
# HALF ADDER (Sum = XOR, Carry = AND)
# ---------------------------------------------------------
def half_adder():
    print("\n\n===== HALF ADDER USING McCULLOCH–PITTS NEURONS =====")

    print("\nA B | SUM | CARRY | Detail")
    print("-------------------------------------------")

    for A in [0, 1]:
        for B in [0, 1]:

            # ----- Carry = AND(A,B) -----
            _, CARRY = mcp_neuron([A, B], [1, 1], 2)

            # ----- XOR(A,B) = A⊕B -----
            # XOR = (A AND NOT B) OR (NOT A AND B)

            # NOT A
            _, nA = mcp_neuron([A], [-1], 0)
            # NOT B
            _, nB = mcp_neuron([B], [-1], 0)

            # A AND NOT B
            _, t1 = mcp_neuron([A, nB], [1, 1], 2)

            # NOT A AND B
            _, t2 = mcp_neuron([nA, B], [1, 1], 2)

            # SUM = OR(t1, t2)
            _, SUM = mcp_neuron([t1, t2], [1, 1], 1)

            print(f"{A} {B} |  {SUM}   |   {CARRY}    | "
                  f"NOT A={nA}, NOT B={nB}, A·¬B={t1}, ¬A·B={t2}")


# ---------------------------------------------------------
# FULL ADDER
# ---------------------------------------------------------
def full_adder():
    print("\n\n===== FULL ADDER USING McCULLOCH–PITTS NEURONS =====\n")

    print("A B Cin | SUM | Cout | Details")
    print("--------------------------------------------------------------")

    for A in [0, 1]:
        for B in [0, 1]:
            for Cin in [0, 1]:

                # ---------------------------
                # FIRST XOR: X1 = A XOR B
                # ---------------------------
                _, nA = mcp_neuron([A], [-1], 0)
                _, nB = mcp_neuron([B], [-1], 0)

                # A·¬B
                _, t1 = mcp_neuron([A, nB], [1, 1], 2)
                # ¬A·B
                _, t2 = mcp_neuron([nA, B], [1, 1], 2)

                _, X1 = mcp_neuron([t1, t2], [1, 1], 1)  # XOR output

                # ---------------------------
                # SECOND XOR: SUM = X1 XOR Cin
                # ---------------------------
                _, nX1 = mcp_neuron([X1], [-1], 0)
                _, nCin = mcp_neuron([Cin], [-1], 0)

                _, u1 = mcp_neuron([X1, nCin], [1, 1], 2)
                _, u2 = mcp_neuron([nX1, Cin], [1, 1], 2)

                _, SUM = mcp_neuron([u1, u2], [1, 1], 1)

                # ---------------------------
                # CARRY OUT
                # Cout = (A AND B) OR (Cin AND X1)
                # ---------------------------
                _, AB = mcp_neuron([A, B], [1, 1], 2)
                _, CinX1 = mcp_neuron([Cin, X1], [1, 1], 2)

                _, Cout = mcp_neuron([AB, CinX1], [1, 1], 1)

                print(f"{A}  {B}   {Cin}  |  {SUM}   |   {Cout}   | "
                      f"X1={X1}, u1={u1}, u2={u2}, AB={AB}, Cin·X1={CinX1}")


# ---------------------------------------------------------
# Run All
# ---------------------------------------------------------
and_gate()
half_adder()
full_adder()
