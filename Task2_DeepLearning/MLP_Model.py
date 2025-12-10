"""
Multi-Layer Perceptron (MLP) Implementation
Architecture:
- Input Layer: 2 neurons (x)
- Hidden Layer 0: 10 neurons (h0)
- Hidden Layer 1: 10 neurons (h1)
- Output Layer: 2 neurons (o)
"""

import numpy as np


class MLP:
    """
    MLP with customizable architecture for forward and backward propagation.
    """

    def __init__(
        self,
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=0.01,
        activation="sigmoid",
        init_scale=0.1,
    ):
        """
        Initialize the MLP network.

        Parameters:
        -----------
        input_size : int
            Number of input neurons (default: 2)
        hidden_sizes : list
            List of hidden layer sizes (default: [10, 10])
        output_size : int
            Number of output neurons (default: 2)
        learning_rate : float
            Learning rate for gradient descent (default: 0.01)
        activation : str
            Activation function to use ('sigmoid' or 'relu')
        init_scale : float
            Standard deviation of the normal distribution used for
            weight initialization (default: 0.1)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.init_scale = init_scale

        #########################################################
        # Initialize weights and biases for each layer
        #########################################################
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        # (input_size, hidden_sizes[0]) matrix for weights
        self.weights.append(
            np.random.randn(input_size, hidden_sizes[0]) * self.init_scale
        )
        # (1, hidden_sizes[0]) matrix for biases, zeros
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            # (hidden_sizes[i], hidden_sizes[i+1]) matrix for weights
            self.weights.append(
                np.random.randn(hidden_sizes[i], hidden_sizes[i + 1]) * self.init_scale
            )
            # (1, hidden_sizes[i+1]) matrix for biases
            self.biases.append(np.zeros((1, hidden_sizes[i + 1])))

        # Last hidden layer to output
        self.weights.append(
            np.random.randn(hidden_sizes[-1], output_size) * self.init_scale
        )
        self.biases.append(np.zeros((1, output_size)))

        # Storage for activations and pre-activations (for backpropagation)
        self.z_values = []  # Pre-activation values
        self.activations = []  # Post-activation values

    # ===== Activation functions and derivatives =====

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation function for output layer."""
        # subtract max to prevent overflow of exp
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activation_function(self, x):
        """Apply the chosen activation function."""
        if self.activation_name == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_name == "relu":
            return self.relu(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    def activation_derivative(self, x):
        """Apply the derivative of the chosen activation function."""
        if self.activation_name == "sigmoid":
            return self.sigmoid_derivative(x)
        elif self.activation_name == "relu":
            return self.relu_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    # ===== Forward and backward =====

    def forward(self, X):
        """
        Forward propagation through the network.

        Parameters:
        -----------
        X : ndarray, shape = (n_samples, input_size)
            Input data

        Returns:
        --------
        output : ndarray, shape = (n_samples, output_size)
            Network output probabilities
        """
        # Reset storage for this pass
        self.z_values = []
        self.activations = [X]

        current_input = X

        # Forward through all layers except the last
        for i in range(len(self.weights) - 1):
            # (n_samples, input_size) * (input_size, hidden_sizes[0]) + (1, hidden_sizes[0])
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            a = self.activation_function(z)
            self.activations.append(a)
            current_input = a

        # Last hidden layer to output layer (softmax for classification)
        # (n_samples, hidden_sizes[-1]) * (hidden_sizes[-1], output_size) + (1, output_size)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)

        output = self.softmax(z_output)
        self.activations.append(output)

        return output

    def backward(self, X, y, output):
        """
        Backward propagation to compute gradients.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, input_size)
            Input data
        y : ndarray, shape (n_samples, output_size)
            True labels (one-hot encoded)
        output : ndarray, shape (n_samples, output_size)
            Network predictions
        """
        m = X.shape[0]  # Number of samples

        # Initialize gradients
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        # Output layer error (softmax + cross-entropy)
        delta = output - y  # (m, output_size)

        # Gradients for output layer
        weight_gradients[-1] = np.dot(self.activations[-2].T, delta) / m
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # delta: (m, hidden_sizes[i+1]) -> (m, hidden_sizes[i])
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(
                self.z_values[i]
            )

            weight_gradients[i] = np.dot(self.activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m

        # Update weights and biases (gradient descent)
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    # ===== Simple training (full-batch) – still available if you want it =====

    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the MLP network with full-batch gradient descent.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, input_size)
            Training data
        y : ndarray, shape (n_samples, output_size)
            Training labels (one-hot encoded)
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        losses : list
            List of loss values per epoch
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            losses.append(loss)

            # Backward pass + update
            self.backward(X, y, output)

            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}"
                )

        return losses

    # ===== Prediction / evaluation helpers =====

    def predict(self, X):
        """
        Predict class labels for input data.
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        """
        return self.forward(X)

    def evaluate(self, X, y):
        """
        Evaluate the model accuracy.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, input_size)
        y : ndarray, shape (n_samples, output_size) one-hot

        Returns:
        --------
        accuracy : float
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def get_network_info(self):
        """Print network architecture information."""
        print("=" * 50)
        print("MLP Network Architecture")
        print("=" * 50)
        print(f"Input Layer (x): {self.input_size} neurons")
        for i, size in enumerate(self.hidden_sizes):
            print(f"Hidden Layer {i} (h({i})): {size} neurons")
        print(f"Output Layer (o): {self.output_size} neurons")
        print("=" * 50)
        print(f"Activation Function: {self.activation_name}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Init Scale (σ): {self.init_scale}")
        print("=" * 50)
        print("\nWeight Matrices:")
        for i, w in enumerate(self.weights):
            print(f"W{i}: {w.shape}")
        print("\nBias Vectors:")
        for i, b in enumerate(self.biases):
            print(f"b{i}: {b.shape}")
        print("=" * 50)


def generate_sample_data(n_samples=1000):
    """
    Generate sample classification data for testing.
    (You won’t use this for THA2, but it's handy for debugging.)

    Returns:
    --------
    X : (n_samples, 2)
    y : (n_samples, 2) one-hot labels
    """
    np.random.seed(42)

    # Class 0
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    # Class 1
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

    X = np.vstack([X_class0, X_class1])

    y = np.zeros((n_samples, 2))
    y[: n_samples // 2, 0] = 1
    y[n_samples // 2 :, 1] = 1

    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y
