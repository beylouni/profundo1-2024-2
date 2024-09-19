import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from app import MLP

# Load Iris dataset
iris = load_iris()
X = iris.data  # Shape: (150 samples, 4 features)
y = iris.target.reshape(-1, 1)  # Shape: (150 samples, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)  # Shape: (150 samples, 3 classes)

# Normalize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Shape: (150 samples, 4 features)

# Split into training and test sets (no transposition needed here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transpose X_train and y_train for the MLP (MLP expects (features, samples) shape)
X_train, X_test = X_train.T, X_test.T  # Now shape is (features, samples)
y_train, y_test = y_train.T, y_test.T  # Now shape is (classes, samples)

# Initialize the MLP model
mlp = MLP(
    layer_sizes=[X_train.shape[0], 10, 3],  # 4 input features, 10 units in hidden layer, 3 output classes
    activations=['relu', 'softmax'],        # Hidden: ReLU, Output: Softmax for multi-class classification
    cost_fn='bce',                          # Use Cross-Entropy Loss
    optimizer='sgd',                       # Use Adam optimizer
    init_method='xavier'
)

# Train the model
mlp.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01)

# Test the model (Forward pass for prediction)
activations, _ = mlp.forward(X_test)
y_pred = activations[-1]

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=0)
y_true_labels = np.argmax(y_test, axis=0)

# Calculate accuracy
accuracy = np.mean(y_pred_labels == y_true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Assuming you have y_pred_labels and y_true_labels already computed
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print("Confusion Matrix:")
print(conf_matrix)