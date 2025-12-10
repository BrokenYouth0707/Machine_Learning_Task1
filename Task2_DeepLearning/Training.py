"""
Multi-Layer Perceptron Training
"""

import numpy as np
from MLP_Model import MLP, generate_sample_data


if __name__ == "__main__":
    print("Multi-Layer Perceptron Implementation\n")
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000)
    
    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}\n")
    
    # Create and initialize MLP
    mlp = MLP(input_size=2, 
              hidden_sizes=[10, 10], 
              output_size=2, 
              learning_rate=0.1,
              activation='relu')
    
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
    sample_X = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 0.0]])
    predictions = mlp.predict(sample_X)
    probabilities = mlp.predict_proba(sample_X)
    
    for i, (x, pred, prob) in enumerate(zip(sample_X, predictions, probabilities)):
        print(f"Input: {x} -> Predicted Class: {pred}, Probabilities: {prob}")