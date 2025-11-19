import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc # Import for ROC curve

# Set random seed for reproducibility
np.random.seed(42)

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Preprocessing:
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()

# Convert 'sex' to numeric
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# Prepare features and target
X = titanic[['pclass', 'sex', 'age', 'fare']].values
y = titanic['survived'].values

# Normalize features (Min-Max scaling)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")

# Initialize weights and bias
input_size = X_train.shape[1]
weights = np.random.randn(input_size)
bias = np.random.randn()

def heaviside(x):
    return 1 if x >= 0 else 0

# Function to get the raw weighted sum (score) before activation
def predict_scores(X_data, weights, bias):
    return np.array([np.dot(xi, weights) + bias for xi in X_data])

learning_rate = 0.01
max_epochs = 50

print(f"Initial weights: {weights}")
print(f"Initial bias: {bias}\n")

# Store loss history for plotting later
loss_history = []

for epoch in range(max_epochs):
    # Shuffle training data at each epoch for better generalization
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    current_epoch_loss = 0
    for i, (xi, yi) in enumerate(zip(X_train_shuffled, y_train_shuffled), 1):
        weighted_sum = np.dot(xi, weights) + bias
        y_pred = heaviside(weighted_sum)
        error = yi - y_pred # Error is 0, -1, or 1
        loss = error ** 2 # Squared error for tracking

        # Update weights and bias only if there's an error
        if error != 0:
            delta_w = learning_rate * error * xi
            delta_b = learning_rate * error

            weights += delta_w
            bias += delta_b

        current_epoch_loss += loss

        # Print detailed calculation for first 5 iterations of the first epoch only
        if epoch == 0 and i <= 5:
            print(f"Epoch {epoch+1}, Iteration {i}:")
            print(f" Input: {xi}")
            print(f" Target: {yi}")
            print(f" Weighted sum: {weighted_sum:.4f}")
            print(f" Prediction: {y_pred}")
            print(f" Error: {error}")
            print(f" Weight update: {delta_w if error != 0 else np.zeros_like(weights)}")
            print(f" Bias update: {delta_b:.4f} if error != 0 else 0.0")
            print(f" Updated weights: {weights}")
            print(f" Updated bias: {bias:.4f}")
            print(f" Loss: {loss:.4f}\n")
        elif epoch == 0 and i == 6:
            print("...Skipping further detailed prints for this epoch...\n")

    loss_history.append(current_epoch_loss / len(X_train_shuffled)) # Average loss per epoch
    print(f"Epoch {epoch+1} average training loss: {loss_history[-1]:.4f}")

print("\nTraining complete.")
print(f"Final weights: {weights}")
print(f"Final bias: {bias}\n")

# --- Evaluation on Training Data ---
y_pred_train = np.array([heaviside(np.dot(xi, weights) + bias) for xi in X_train])
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# --- Evaluation on Test Data ---
y_pred_test = np.array([heaviside(np.dot(xi, weights) + bias) for xi in X_test])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix (Test Data):")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not survive', 'Survived'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Data)")
plt.show()

# Print evaluation metrics for Test Data
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
accuracy = accuracy_score(y_test, y_pred_test)

print(f"Precision (Test Data): {precision:.4f}")
print(f"Recall (Test Data): {recall:.4f}")
print(f"F1 Score (Test Data): {f1:.4f}")
print(f"Accuracy (Test Data): {accuracy:.4f}")

# --- ROC Curve for Test Data ---
y_scores_test = predict_scores(X_test, weights, bias)
fpr, tpr, thresholds = roc_curve(y_test, y_scores_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Data')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Visualization: Training Loss over Epochs ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_epochs + 1), loss_history, marker='o', linestyle='-')
plt.title('Average Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss (Squared Error)')
plt.grid(True)
plt.show()

# --- Visualization: 2D plot (pclass vs sex) for Test Data ---
plt.figure(figsize=(8,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.7, edgecolors='k')
plt.xlabel('Normalized Pclass')
plt.ylabel('Normalized Sex (0=male,1=female)')
plt.title('Titanic Dataset (Test Data) - Survived (red) vs Not Survived (blue) on Pclass & Sex')
plt.show()