import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# ---------------------------------------------------------------
#  LVQ TRAINING FUNCTION
# ---------------------------------------------------------------
def lvq_train(X, y, prototypes, proto_labels, lr=0.1, epochs=5):
    print("\nInitial prototypes:")
    for i, p in enumerate(prototypes):
        print(f"P{i} (label {proto_labels[i]}): {p}")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch} ===\n")

        for i in range(len(X)):
            x = X[i]
            label = y[i]

            # Compute distances to prototypes
            distances = [euclidean_distance(x, p) for p in prototypes]

            print(f"Step {i + 1}: Sample {i}: x={x}, label={label}")
            print(f" Distances: to P0={distances[0]:.4f}, to P1={distances[1]:.4f}")

            # Winner = closest prototype
            winner = np.argmin(distances)

            print(f" Winner: P{winner} (label {proto_labels[winner]}), distance={distances[winner]:.4f}")

            # Update rule
            update = lr * (x - prototypes[winner])

            if proto_labels[winner] == label:
                # Move towards
                prototypes[winner] += update
                print(f" Update vector = {update}")
                print(" Action: Prototype moved TOWARDS (same label)")
            else:
                # Move away
                prototypes[winner] -= update
                print(f" Update vector = {update}")
                print(" Action: Prototype moved AWAY (different label)")

            print(f" New coords of P{winner} = {prototypes[winner]}\n")

    print("\nFinal Prototypes:")
    for i, p in enumerate(prototypes):
        print(f"P{i} (label {proto_labels[i]}): {p}")

    return prototypes


# ---------------------------------------------------------------
#  LVQ PREDICTION
# ---------------------------------------------------------------
def lvq_predict(X, prototypes, proto_labels):
    preds = []
    for x in X:
        distances = [euclidean_distance(x, p) for p in prototypes]
        pred = proto_labels[np.argmin(distances)]
        preds.append(pred)
    return preds


# ---------------------------------------------------------------
#  SAMPLE DATASET (TWO CLUSTERS)
# ---------------------------------------------------------------

X = np.array([
    [0.1, 0.0],
    [0.0, 0.2],
    [3.1, 3.0],
    [2.9, 3.2]
])

y = np.array([0, 0, 1, 1])  # Class labels

# Initial prototypes (1 per class)
prototypes = np.array([
    [0.1, 0.0],   # Prototype for Class 0
    [3.1, 3.0]    # Prototype for Class 1
])

proto_labels = [0, 1]  # Labels of prototypes

# ---------------------------------------------------------------
# TRAIN LVQ
# ---------------------------------------------------------------
trained_prototypes = lvq_train(X, y, prototypes, proto_labels, lr=0.1, epochs=3)

# ---------------------------------------------------------------
# TEST PREDICTIONS
# ---------------------------------------------------------------
preds = lvq_predict(X, trained_prototypes, proto_labels)

print("\nFinal Predictions:")
for i in range(len(X)):
    print(f" Sample {i}: {X[i]}, True Label={y[i]}, Predicted={preds[i]}")
