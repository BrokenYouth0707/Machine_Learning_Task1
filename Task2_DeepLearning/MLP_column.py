"""
Multi-Layer Perceptron (MLP) Implementation - Column Vector Convention
Architecture:
- Input Layer: 2 neurons (x)
- Hidden Layer 0: 10 neurons (h0)
- Hidden Layer 1: 10 neurons (h1)
- Output Layer: 2 neurons (o)

This implementation uses COLUMN VECTOR CONVENTION where:
- Input: X shape (n_features, n_samples) - each column is a sample
- Weights: W shape (n_out, n_in) - transposed compared to row convention
- Forward: z = W @ x + b
- Backward: delta = W.T @ delta_next
"""

import numpy as np


class MLP_Column:
    """
    MLP with Column Vector Convention for forward and backward propagation.
    Each column represents one sample (traditional mathematical notation).
    """
    
    def __init__(self, input_size=2, hidden_sizes=[10, 10], output_size=2, 
                 learning_rate=0.01, activation='sigmoid'):
        """
        Initialize the MLP network with column vector convention.
        
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
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        #########################################################
        ###### Initialize weights and biases for each layer #####
        # Note: Weights are (n_out, n_in) in column convention
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        # (hidden_sizes[0], input_size) - note the order is REVERSED from row convention
        self.weights.append(np.random.randn(hidden_sizes[0], input_size) * 0.1)
        # (hidden_sizes[0], 1) - column vector for biases
        self.biases.append(np.zeros((hidden_sizes[0], 1)))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            # (hidden_sizes[i+1], hidden_sizes[i])
            self.weights.append(np.random.randn(hidden_sizes[i+1], hidden_sizes[i]) * 0.1)
            # (hidden_sizes[i+1], 1)
            self.biases.append(np.zeros((hidden_sizes[i+1], 1)))
        
        # Last hidden layer to output
        # (output_size, hidden_sizes[-1])
        self.weights.append(np.random.randn(output_size, hidden_sizes[-1]) * 0.1)
        # (output_size, 1)
        self.biases.append(np.zeros((output_size, 1)))
        
        # Storage for activations and pre-activations (for backpropagation)
        self.z_values = []  # Pre-activation values
        self.activations = []  # Post-activation values
    
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
        """Softmax activation function for output layer (column-wise)."""
        # Subtract max for numerical stability (column-wise)
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def activation_function(self, x):
        """Apply the chosen activation function."""
        if self.activation_name == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_name == 'relu':
            return self.relu(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")
    
    def activation_derivative(self, x):
        """Apply the derivative of the chosen activation function."""
        if self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_name == 'relu':
            return self.relu_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")
    
    def forward(self, X):
        """
        Forward propagation through the network (Column Convention).
        
        Parameters:
        -----------
        X : ndarray, shape = (input_size, n_samples)
            Input data - each COLUMN is a sample
            
        Returns:
        --------
        output : ndarray, shape = (output_size, n_samples)
            Network output probabilities - each COLUMN is a sample's output
        """
        # Storage for activations and pre-activations
        self.z_values = []
        self.activations = [X]
        
        current_input = X
        
        # Forward through all layers except the last
        for i in range(len(self.weights) - 1):
            # Column convention: z = W @ a + b
            # 1. For input layer to 1st hidden layer:
            #    (hidden_sizes[0], input_size) @ (input_size, n_samples) + (hidden_sizes[0], 1)
            #    Result: (hidden_sizes[0], n_samples)
            # 2. For ith hidden layer to (i+1)th hidden layer:
            #    (hidden_sizes[i+1], hidden_sizes[i]) @ (hidden_sizes[i], n_samples) + (hidden_sizes[i+1], 1)
            #    Result: (hidden_sizes[i+1], n_samples)
            z = np.dot(self.weights[i], current_input) + self.biases[i]
            # Store pre-activation values
            self.z_values.append(z)
            
            # Apply activation function
            a = self.activation_function(z)
            # Store post-activation values
            self.activations.append(a)
            # Update current input for next layer
            current_input = a
        
        # For last hidden layer to output layer (using softmax)
        # (output_size, hidden_sizes[-1]) @ (hidden_sizes[-1], n_samples) + (output_size, 1)
        # Result: (output_size, n_samples)
        z_output = np.dot(self.weights[-1], current_input) + self.biases[-1]
        self.z_values.append(z_output)

        # Apply softmax activation to get output probabilities
        output = self.softmax(z_output)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """
        Backward propagation to compute gradients (Column Convention).
        
        Parameters:
        -----------
        X : ndarray, shape (input_size, n_samples)
            Input data - each column is a sample
        y : ndarray, shape (output_size, n_samples)
            True labels (one-hot encoded) - each column is a sample
        output : ndarray, shape (output_size, n_samples)
            Network predictions - each column is a sample
        """
        m = X.shape[1]  # Number of samples (second dimension in column convention)
        
        # Initialize gradients
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        
        # Output layer error (for softmax + cross-entropy)
        # = Gradient of loss w.r.t. z_output
        # Shape: (output_size, n_samples)
        delta = output - y
        
        # Gradient for output layer w.r.t. weights and biases
        # Column convention: dL/dW = delta @ a.T
        # (output_size, n_samples) @ (n_samples, hidden_sizes[-1])
        # Result: (output_size, hidden_sizes[-1])
        weight_gradients[-1] = np.dot(delta, self.activations[-2].T) / m
        
        # Bias gradient: average over samples (sum along columns)
        # (output_size, n_samples) -> (output_size, 1)
        bias_gradients[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Compute delta for current layer using chain rule
            # Column convention: delta^(l) = W^(l+1).T @ delta^(l+1) ⊙ σ'(z^(l))
            # (hidden_sizes[i], hidden_sizes[i+1]) @ (hidden_sizes[i+1], n_samples)
            # Result: (hidden_sizes[i], n_samples)
            delta = np.dot(self.weights[i + 1].T, delta) * self.activation_derivative(self.z_values[i])
            
            # Gradient for current layer w.r.t. weights
            # delta @ a.T
            # (hidden_sizes[i], n_samples) @ (n_samples, hidden_sizes[i-1] or input_size)
            # Result: (hidden_sizes[i], hidden_sizes[i-1] or input_size)
            weight_gradients[i] = np.dot(delta, self.activations[i].T) / m
            
            # Bias gradient: average over samples
            # (hidden_sizes[i], n_samples) -> (hidden_sizes[i], 1)
            bias_gradients[i] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the MLP network.
        
        Parameters:
        -----------
        X : ndarray, shape (input_size, n_samples)
            Training data - each column is a sample
        y : ndarray, shape (output_size, n_samples)
            Training labels (one-hot encoded) - each column is a sample
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
            
            # Compute loss (cross-entropy)
            # Sum over features (axis=0), mean over samples (axis=1)
            # Add 1e-8 to prevent log(0)
            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=0))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress every 100 epochs
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray, shape (input_size, n_samples)
            Input data - each column is a sample
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples,)
            Predicted class labels
        """
        output = self.forward(X)
        # Return the class with highest probability for each column
        return np.argmax(output, axis=0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : ndarray, shape (input_size, n_samples)
            Input data - each column is a sample
        
        Returns:
        --------
        probabilities : ndarray, shape (output_size, n_samples)
            Predicted probabilities for each class - each column is a sample
        """
        return self.forward(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model accuracy.
        
        Parameters:
        -----------
        X : ndarray, shape (input_size, n_samples)
            Input data - each column is a sample
        y : ndarray, shape (output_size, n_samples)
            True labels (one-hot encoded) - each column is a sample
        
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    
    def get_network_info(self):
        """Print network architecture information."""
        print("=" * 60)
        print("MLP Network Architecture (Column Vector Convention)")
        print("=" * 60)
        print(f"Input Layer (x): {self.input_size} neurons")
        for i, size in enumerate(self.hidden_sizes):
            print(f"Hidden Layer {i} (h({i})): {size} neurons")
        print(f"Output Layer (o): {self.output_size} neurons")
        print("=" * 60)
        print(f"Activation Function: {self.activation_name}")
        print(f"Learning Rate: {self.learning_rate}")
        print("=" * 60)
        print("\nWeight Matrices (note: n_out × n_in):")
        for i, w in enumerate(self.weights):
            print(f"W{i}: {w.shape}")
        print("\nBias Vectors (column vectors):")
        for i, b in enumerate(self.biases):
            print(f"b{i}: {b.shape}")
        print("=" * 60)


def generate_sample_data(n_samples=1000):
    """
    Generate sample classification data for testing.
    Returns data in COLUMN format (features × samples).
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    X : ndarray, shape (2, n_samples)
        Input features - each COLUMN is a sample
    y : ndarray, shape (2, n_samples)
        One-hot encoded labels - each COLUMN is a sample
    """
    # Generate random 2D points
    np.random.seed(42)
    
    # Class 0: points around (2, 2)
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    # Class 1: points around (-2, -2)
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    
    X = np.vstack([X_class0, X_class1])
    
    # Create one-hot encoded labels
    y = np.zeros((n_samples, 2))
    y[:n_samples // 2, 0] = 1  # Class 0
    y[n_samples // 2:, 1] = 1  # Class 1
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # TRANSPOSE to column format: (features, samples)
    X = X.T  # (2, n_samples)
    y = y.T  # (2, n_samples)
    
    return X, y


if __name__ == "__main__":
    print("Multi-Layer Perceptron Implementation")
    print("Using Column Vector Convention\n")
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000)
    
    print(f"Data format: X shape = {X.shape} (features × samples)")
    print(f"Data format: y shape = {y.shape} (classes × samples)")
    
    # Split into train and test sets
    split_idx = int(0.8 * X.shape[1])
    X_train, X_test = X[:, :split_idx], X[:, split_idx:]
    y_train, y_test = y[:, :split_idx], y[:, split_idx:]
    
    print(f"Training samples: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}\n")
    
    # Create and initialize MLP
    mlp = MLP_Column(input_size=2, 
                     hidden_sizes=[10, 10], 
                     output_size=2, 
                     learning_rate=0.1,
                     activation='sigmoid')
    
    # Display network architecture
    mlp.get_network_info()
    
    # Train the network
    print("\nTraining the network...")
    losses = mlp.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = mlp.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make sample predictions
    print("\nSample predictions:")
    # Note: input must be in column format (features, samples)
    sample_X = np.array([[2.0, -2.0, 0.0],   # feature 1
                         [2.0, -2.0, 0.0]])  # feature 2
    predictions = mlp.predict(sample_X)
    probabilities = mlp.predict_proba(sample_X)
    
    for i in range(sample_X.shape[1]):
        x = sample_X[:, i]
        pred = predictions[i]
        prob = probabilities[:, i]
        print(f"Input: {x} -> Predicted Class: {pred}, Probabilities: {prob}")
    
    print("\n" + "=" * 60)
    print("Note: This implementation uses COLUMN vector convention")
    print("where each column represents one sample, matching")
    print("traditional mathematical notation in textbooks.")
    print("=" * 60)
