import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DetailedKSOM:
    def __init__(self, initial_weights, learning_rate):
        self.weights = np.array(initial_weights, dtype=float)
        self.learning_rate = learning_rate
        print("KSOM Initialized:")
        print("===================")
        print(f"Initial Learning Rate (alpha): {self.learning_rate}")
        print("Initial Weights (w_ij):")
        print(self.weights)
        print("===================\n")

    def find_best_matching_unit(self, input_vector):

        print("Step 1 & 2: Competition and Selection (Detailed Calculation)")
        print("---------------------------------------------------------")
        print(f"Input Vector x = {input_vector}")
        print("Formula: D(j) = Σ (xᵢ - wᵢⱼ)²\n")

        distances = []
        # Iterate through each output neuron j (0 for Y1, 1 for Y2)
        for j in range(self.weights.shape[1]):
            print(f"---> Calculating distance for Neuron Y{j+1}:")
            weight_vector = self.weights[:, j]
            print(f"     Current weight vector w{j+1} = {np.round(weight_vector, 4)}")

            squared_diffs = (input_vector - weight_vector) ** 2

            # Build the calculation string for clarity
            calculation_str = []
            for i in range(len(input_vector)):
                calculation_str.append(f"({input_vector[i]} - {np.round(weight_vector[i], 2)})²")

            print(f"     D({j+1}) = {' + '.join(calculation_str)}")
            print(f"          = {' + '.join([f'{d:.4f}' for d in squared_diffs])}")
            print(f"          = {np.sum(squared_diffs):.4f}\n")
            distances.append(np.sum(squared_diffs))

        bmu_index = np.argmin(distances)

        print("---> Comparison:")
        print(f"     D(1) = {distances[0]:.4f}")
        print(f"     D(2) = {distances[1]:.4f}")
        print(f"Since D({bmu_index+1}) is the minimum, Neuron Y{bmu_index+1} is the winning neuron (BMU).\n")

        return bmu_index

    def update_weights(self, input_vector, bmu_index):

        print("Step 3: Weight Update (Detailed Calculation)")
        print("------------------------------------------")
        print(f"Updating weights for the winning neuron, Y{bmu_index + 1}.")
        print("Formula: wᵢⱼ(new) = wᵢⱼ(old) + α * [xᵢ - wᵢⱼ(old)]")
        print(f"Learning Rate α = {self.learning_rate}\n")

        old_bmu_weights = self.weights[:, bmu_index].copy()

        for i in range(len(input_vector)):
            old_weight = old_bmu_weights[i]
            diff = input_vector[i] - old_weight
            update_value = self.learning_rate * diff
            new_weight = old_weight + update_value

            print(f"---> Calculating new w{i+1}{bmu_index+1}:")
            print(f"     w{i+1}{bmu_index+1}(new) = {np.round(old_weight, 2)} + {self.learning_rate} * ({input_vector[i]} - {np.round(old_weight, 2)})")
            print(f"                   = {np.round(old_weight, 2)} + {self.learning_rate} * ({np.round(diff, 2)})")
            print(f"                   = {np.round(old_weight, 2)} + {np.round(update_value, 4)}")
            print(f"                   = {np.round(new_weight, 4)}\n")

            self.weights[i, bmu_index] = new_weight

        print("The weights for the non-winning neuron(s) remain unchanged as the neighborhood radius R=0.\n")


    def train(self, input_vector):

        input_vector = np.array(input_vector)
        bmu_index = self.find_best_matching_unit(input_vector)
        self.update_weights(input_vector, bmu_index)
        print("Current Weight Matrix after this iteration:")
        print("==========================================")
        print(self.weights)
        print("==========================================\n\n")

# --- NEW: Function to visualize the final clusters ---
def visualize_clusters(ksom_net, input_vectors):
    print("########## Final Cluster Visualization ##########\n")
    final_weights = ksom_net.weights

    # Determine the final cluster for each input vector
    cluster_assignments = []
    for vector in input_vectors:
        distances = [np.sum((vector - final_weights[:, j])**2) for j in range(final_weights.shape[1])]
        cluster_assignments.append(np.argmin(distances))

    # Combine inputs and final weights (cluster centers) for PCA projection
    # We transpose weights because they are stored as columns
    all_data = np.vstack([input_vectors, final_weights.T])

    # Use PCA to reduce from 4D to 2D
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(all_data)

    # Separate the transformed points back into inputs and cluster centers
    transformed_inputs = transformed_data[:len(input_vectors)]
    transformed_centers = transformed_data[len(input_vectors):]

    # Create the plot
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red']

    # Plot the input vectors, colored by their assigned cluster
    for i, point in enumerate(transformed_inputs):
        cluster_index = cluster_assignments[i]
        plt.scatter(point[0], point[1],
                    color=colors[cluster_index],
                    label=f'Cluster {cluster_index + 1}' if f'Cluster {cluster_index + 1}' not in plt.gca().get_legend_handles_labels()[1] else "",
                    s=100) # s is size
        # Add text labels for clarity
        plt.text(point[0] + 0.05, point[1], str(input_vectors[i]))

    # Plot the final cluster centers (neurons) as large stars
    for i, center in enumerate(transformed_centers):
        plt.scatter(center[0], center[1],
                    color=colors[i],
                    marker='*',
                    s=500,  # make stars much larger
                    edgecolors='black',
                    label=f'Cluster Center {i+1} (Neuron Y{i+1})')

    plt.title('Final KSOM Clusters (Projected to 2D using PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.show()


# Main execution
if __name__ == "__main__":
    initial_weights = [
        [0.2, 0.9], [0.4, 0.7], [0.6, 0.5], [0.8, 0.3]
    ]
    learning_rate = 0.5

    ksom_net = DetailedKSOM(initial_weights, learning_rate)

    input_vectors = [
        [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]
    ]

    for i, vector in enumerate(input_vectors):
        print(f"########## Processing Input {i+1} of {len(input_vectors)} ##########")
        ksom_net.train(vector)

    # After all training is done, call the visualization function
    visualize_clusters(ksom_net, input_vectors)