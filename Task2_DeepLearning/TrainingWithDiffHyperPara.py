"""
Flexible MLP Training Script with Hyperparameter Configuration

This script provides a flexible framework for training MLP models with various
architectures and hyperparameters on the THA2 dataset.

Features:
- Easy configuration of network architecture (layers, neurons)
- Adjustable hyperparameters (learning rate, batch size, activation function, etc.)
- Training with validation monitoring
- Performance visualization
- Model comparison capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP_Model import MLP
from itertools import product
import json
from datetime import datetime


# =========================================================
# 1. Data Loading
# =========================================================
def load_tha2_data(train_path="THA2train.xlsx", val_path="THA2validate.xlsx"):
    """
    Load THA2 train and validation sets from Excel.
    Assumes header row: X_0, X_1, y
    
    Returns:
    --------
    X_train, y_train, y_train_int, X_val, y_val, y_val_int
    """
    df_train = pd.read_excel(train_path)
    df_val = pd.read_excel(val_path)

    # Extract features
    X_train = df_train[["X_0", "X_1"]].values.astype(np.float32)
    X_val = df_val[["X_0", "X_1"]].values.astype(np.float32)

    # Extract labels (0/1)
    y_train_int = df_train["y"].values.astype(int)
    y_val_int = df_val["y"].values.astype(int)

    # Standardize features using training statistics
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # One-hot encode labels
    def one_hot(y, num_classes=2):
        y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
        y_oh[np.arange(len(y)), y] = 1.0
        return y_oh

    y_train = one_hot(y_train_int)
    y_val = one_hot(y_val_int)

    return X_train, y_train, y_train_int, X_val, y_val, y_val_int


# =========================================================
# 2. Utility Functions
# =========================================================
def compute_loss(y_true, y_pred):
    """
    Cross-entropy loss for one-hot labels.
    """
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


def compute_accuracy(y_pred, y_true_int):
    """
    Compute classification accuracy.
    """
    return np.mean(y_pred == y_true_int)


def confusion_matrix_2x2(y_true, y_pred):
    """
    Simple 2x2 confusion matrix.
    Rows = true labels, Cols = predicted labels.
    """
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         title="Training Curves", save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss_only(train_losses, val_losses, title="Loss Curves", save_path=None):
    """
    Plot only training and validation loss curves (for best/worst models).
    """
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", linewidth=2.5, marker='o', 
             markersize=3, markevery=max(1, len(epochs)//20))
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2.5, marker='s', 
             markersize=3, markevery=max(1, len(epochs)//20))
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_coordinates(results_df, save_path=None, top_k=10):
    """
    Create an optimized Parallel Coordinates Plot with two-layer visualization.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter configurations and results
    save_path : str, optional
        Path to save the plot
    top_k : int, optional
        Number of top models to highlight (default: 10 for top 10 models)
    
    Features:
    ---------
    - Two-layer rendering: all models in light gray background + top models in color
    - Reordered axes: val_acc first to better show parameter influence
    - Log scale for learning_rate and init_scale
    - Enhanced visual contrast for best performers
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # Create a copy for manipulation
    plot_data = results_df.copy()
    
    # Define columns with strategic ordering:
    # 1. Put performance metrics (train_acc, val_acc) in positions 3 and 5
    # 2. Group related parameters together
    # 3. Use parameters that show variation
    param_columns = ['learning_rate', 'init_scale', 'final_train_acc', 'batch_size', 
                     'final_val_acc', 'num_layers', 'avg_neurons']
    
    # Filter columns that exist
    param_columns = [col for col in param_columns if col in plot_data.columns]
    
    if len(param_columns) == 0:
        print("Error: No valid columns found in DataFrame!")
        print("Available columns:", plot_data.columns.tolist())
        return
    
    # Apply log transformation to learning_rate and init_scale for better visualization
    log_transform_cols = ['learning_rate', 'init_scale']
    for col in log_transform_cols:
        if col in param_columns:
            # Add small epsilon to avoid log(0)
            plot_data[f'{col}_log'] = np.log10(plot_data[col] + 1e-10)
            # Replace original column with log version in param_columns
            idx = param_columns.index(col)
            param_columns[idx] = f'{col}_log'
    
    # Normalize each parameter to [0, 1]
    normalized_data = pd.DataFrame()
    original_ranges = {}  # Store original ranges for labels
    
    for col in param_columns:
        min_val = plot_data[col].min()
        max_val = plot_data[col].max()
        original_ranges[col] = (min_val, max_val)
        
        if max_val > min_val:
            normalized_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col] = 0.5  # If all values are the same
    
    # Identify top K performers by sorting by val_acc (descending) then val_loss (ascending)
    val_acc = results_df['final_val_acc'].values
    val_loss = results_df['final_val_loss'].values
    
    # Create sorting keys: primary by val_acc (descending), secondary by val_loss (ascending)
    # lexsort sorts in ascending order by default
    # To get: high val_acc first, then low val_loss when val_acc is same
    # We use: lexsort((-val_loss, val_acc)) so last K elements have highest val_acc and lowest val_loss
    sort_indices = np.lexsort((-val_loss, val_acc))  # Sort by val_acc ascending, then -val_loss ascending
    
    # Get top K indices (last K elements have highest val_acc and lowest val_loss)
    top_k_actual = min(top_k, len(val_acc))  # In case dataset is smaller than top_k
    top_indices = sort_indices[-top_k_actual:]  # Get top K indices
    top_mask = np.zeros(len(val_acc), dtype=bool)
    top_mask[top_indices] = True
    
    # Setup colormap for top performers based on rank
    # Use rank (0 to 1) for coloring: 0 = worst of top-K, 1 = best of top-K
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('coolwarm')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # LAYER 1: Plot ALL models in light gray as background
    for idx in range(len(normalized_data)):
        row = normalized_data.iloc[idx].values
        ax.plot(range(len(param_columns)), row, 
                color='lightgray', alpha=0.15, linewidth=1.5, zorder=1)
    
    # LAYER 2: Plot TOP performers in color with emphasis
    # Sort top performers by their position in sort_indices (rank order)
    top_indices = np.where(top_mask)[0]
    # Get the rank of each top model in the overall sorted list
    top_ranks = {idx: np.where(sort_indices == idx)[0][0] for idx in top_indices}
    # Sort by rank (ascending), so lowest rank drawn first
    top_sorted_indices = sorted(top_indices, key=lambda x: top_ranks[x])
    
    for i, idx in enumerate(top_sorted_indices):
        row = normalized_data.iloc[idx].values
        # Normalize rank within top-K: 0 (worst of top-K) to 1 (best of top-K)
        rank_normalized = i / max(1, len(top_sorted_indices) - 1)
        color = cmap(norm(rank_normalized))
        
        # Emphasize the very best (top 3 models get extra thick lines)
        # i ranges from 0 (worst of top-K) to len-1 (best of top-K)
        if i >= len(top_sorted_indices) - 3:  # Top 3 models
            alpha = 1.0
            linewidth = 4
            zorder = 200 + i  # Higher z-order for better models
        else:
            alpha = 0.85
            linewidth = 3
            zorder = 100 + i
        
        ax.plot(range(len(param_columns)), row, 
                color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
    
    # Customize axis labels (show original parameter names without _log suffix)
    display_labels = []
    for col in param_columns:
        if col.endswith('_log'):
            display_labels.append(col.replace('_log', '') + ' (log)')
        else:
            display_labels.append(col)
    
    ax.set_xticks(range(len(param_columns)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalized Value', fontsize=12)
    title = f'Parallel Coordinates Plot: Hyperparameter Effects on Model Performance\n'
    title += f'(Gray: all {len(results_df)} models | Color: top {sum(top_mask)} models)'
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add colorbar for top performers
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f'Relative Rank in Top {sum(top_mask)} Models (Blue=Lower, Red=Higher)', fontsize=12)
    
    # Add text box showing actual val_acc range of top-K models
    # top_val_acc = val_acc[top_mask]
    # if len(top_val_acc) > 0:
    #     text_str = f'Val Acc Range: {top_val_acc.min():.4f} - {top_val_acc.max():.4f}'
    #     ax.text(0.8, 0.9, text_str, transform=ax.transAxes, 
    #             fontsize=10, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add original value labels on y-axis
    for i, col in enumerate(param_columns):
        min_val, max_val = original_ranges[col]
        
        # Format numbers appropriately (scientific notation for log scale)
        if col.endswith('_log'):
            # Convert back from log scale for display
            min_original = 10 ** min_val
            max_original = 10 ** max_val
            ax.text(i, -0.02, f'{min_original:.1e}', ha='center', va='top', 
                    fontsize=8, color='gray')
            ax.text(i, 1.02, f'{max_original:.1e}', ha='center', va='bottom', 
                    fontsize=8, color='gray')
        else:
            ax.text(i, -0.02, f'{min_val:.3f}', ha='center', va='top', 
                    fontsize=8, color='gray')
            ax.text(i, 1.02, f'{max_val:.3f}', ha='center', va='bottom', 
                    fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =========================================================
# 3. Training Function
# =========================================================
def train_mlp(model, X_train, y_train, y_train_int, X_val, y_val, y_val_int,
              epochs=200, batch_size=16, verbose=True, print_interval=10):
    """
    Train the MLP model using mini-batch SGD with validation monitoring.
    
    Parameters:
    -----------
    model : MLP
        The MLP model to train
    X_train, y_train, y_train_int : ndarray
        Training data and labels
    X_val, y_val, y_val_int : ndarray
        Validation data and labels
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for mini-batch SGD (None or <= 0 for full-batch)
    verbose : bool
        Whether to print training progress
    print_interval : int
        Print progress every N epochs
        
    Returns:
    --------
    history : dict
        Dictionary containing training history
    """
    n_train = X_train.shape[0]
    
    # Handle batch size
    if batch_size is None or batch_size <= 0 or batch_size > n_train:
        batch_size = n_train
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Training samples: {n_train}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Batch size: {batch_size if batch_size < n_train else 'full-batch'}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        
        # Mini-batch training
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            y_pred_batch = model.forward(X_batch)
            loss_batch = compute_loss(y_batch, y_pred_batch)
            
            # Backward pass
            model.backward(X_batch, y_batch, y_pred_batch)
            
            # Accumulate loss
            epoch_loss += loss_batch * len(X_batch)
        
        # Average training loss
        epoch_loss /= n_train
        history['train_loss'].append(epoch_loss)
        
        # Training accuracy
        y_pred_train = model.predict(X_train)
        train_acc = compute_accuracy(y_pred_train, y_train_int)
        history['train_acc'].append(train_acc)
        
        # Validation loss and accuracy
        y_pred_val = model.forward(X_val)
        val_loss = compute_loss(y_val, y_pred_val)
        history['val_loss'].append(val_loss)
        
        y_pred_val_class = model.predict(X_val)
        val_acc = compute_accuracy(y_pred_val_class, y_val_int)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}\n")
    
    return history


# =========================================================
# 4. Model Evaluation
# =========================================================
def evaluate_model(model, X, y, y_int, dataset_name="Test"):
    """
    Evaluate the model and print detailed metrics.
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Evaluation")
    print(f"{'='*60}")
    
    # Predictions
    y_pred_class = model.predict(X)
    y_pred_proba = model.forward(X)
    
    # Metrics
    loss = compute_loss(y, y_pred_proba)
    accuracy = compute_accuracy(y_pred_class, y_int)
    cm = confusion_matrix_2x2(y_int, y_pred_class)
    
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    
    # Calculate precision, recall, F1 for each class
    for i in range(2):
        tp = cm[i, i]
        fp = cm[1-i, i]
        fn = cm[i, 1-i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass {i}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    print(f"{'='*60}\n")
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred_class,
        'probabilities': y_pred_proba
    }


# =========================================================
# 5. Hyperparameter Grid Search
# =========================================================
def run_hyperparameter_experiments(X_train, y_train, y_train_int, 
                                   X_val, y_val, y_val_int,
                                   hyperparameter_grid, 
                                   epochs=200,
                                   verbose=False):
    """
    Run experiments with different hyperparameter configurations.
    
    Parameters:
    -----------
    X_train, y_train, y_train_int : ndarray
        Training data and labels
    X_val, y_val, y_val_int : ndarray
        Validation data and labels
    hyperparameter_grid : dict
        Dictionary with lists of hyperparameters to try
    epochs : int
        Number of epochs for each experiment
    verbose : bool
        Whether to print progress for each experiment
        
    Returns:
    --------
    results : list of dict
        List containing results for each configuration
    """
    results = []
    
    # Generate all combinations
    keys = list(hyperparameter_grid.keys())
    values = list(hyperparameter_grid.values())
    combinations = list(product(*values))
    
    total_experiments = len(combinations)
    print(f"\n{'='*70}")
    print(f"Running {total_experiments} hyperparameter experiments")
    print(f"{'='*70}\n")
    
    for idx, combo in enumerate(combinations, 1):
        config = dict(zip(keys, combo))
        
        print(f"\n[Experiment {idx}/{total_experiments}]")
        print(f"Config: {config}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create model
        model = MLP(
            input_size=2,
            hidden_sizes=config['hidden_sizes'],
            output_size=2,
            learning_rate=config['learning_rate'],
            activation=config['activation'],
            init_scale=config['init_scale']
        )
        
        # Train model
        history = train_mlp(
            model=model,
            X_train=X_train,
            y_train=y_train,
            y_train_int=y_train_int,
            X_val=X_val,
            y_val=y_val,
            y_val_int=y_val_int,
            epochs=epochs,
            batch_size=config['batch_size'],
            verbose=verbose,
            print_interval=50
        )
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        train_acc = compute_accuracy(y_pred_train, y_train_int)
        
        y_pred_val = model.predict(X_val)
        val_acc = compute_accuracy(y_pred_val, y_val_int)
        
        y_prob_train = model.forward(X_train)
        train_loss = compute_loss(y_train, y_prob_train)
        
        y_prob_val = model.forward(X_val)
        val_loss = compute_loss(y_val, y_prob_val)
        
        # Store results
        result = {
            'experiment_id': idx,
            'hidden_sizes': config['hidden_sizes'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'activation': config['activation'],
            'init_scale': config['init_scale'],
            'num_layers': len(config['hidden_sizes']),
            'total_neurons': sum(config['hidden_sizes']),
            'avg_neurons': np.mean(config['hidden_sizes']),
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'train_loss_history': history['train_loss'],
            'val_loss_history': history['val_loss'],
            'train_acc_history': history['train_acc'],
            'val_acc_history': history['val_acc'],
            'overfitting_gap': train_acc - val_acc,
            'model': model  # Store model for later analysis
        }
        
        results.append(result)
        
        print(f"Results: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
              f"Overfit Gap={train_acc - val_acc:.4f}")
    
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"{'='*70}\n")
    
    return results


def analyze_results(results, save_dir='Plots/'):
    """
    Analyze and visualize results from hyperparameter experiments.
    
    Parameters:
    -----------
    results : list of dict
        Results from run_hyperparameter_experiments
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame([
        {
            'experiment_id': r['experiment_id'],
            'hidden_sizes': str(r['hidden_sizes']),
            'learning_rate': r['learning_rate'],
            'batch_size': r['batch_size'],
            'activation': r['activation'],
            'init_scale': r['init_scale'],
            'num_layers': r['num_layers'],
            'total_neurons': r['total_neurons'],
            'avg_neurons': r['avg_neurons'],
            'final_train_loss': r['final_train_loss'],
            'final_val_loss': r['final_val_loss'],
            'final_train_acc': r['final_train_acc'],
            'final_val_acc': r['final_val_acc'],
            'overfitting_gap': r['overfitting_gap']
        }
        for r in results
    ])
    
    # Save results to CSV
    csv_path = os.path.join(save_dir, 'hyperparameter_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}\n")
    
    # Sort by validation accuracy (descending) then by validation loss (ascending)
    results_df_sorted = results_df.sort_values(
        by=['final_val_acc', 'final_val_loss'], 
        ascending=[False, True]
    )
    
    # Print top 5 and bottom 5 configurations
    print("="*70)
    print("TOP 5 CONFIGURATIONS (by Val Accuracy ↓, then Val Loss ↑)")
    print("="*70)
    print(results_df_sorted.head(5).to_string(index=False))
    
    print("\n" + "="*70)
    print("BOTTOM 5 CONFIGURATIONS (by Val Accuracy ↓, then Val Loss ↑)")
    print("="*70)
    print(results_df_sorted.tail(5).to_string(index=False))
    
    # Find best and worst models using the same sorting criteria
    # Best: highest val_acc, then lowest val_loss
    best_idx = results_df_sorted.index[0]
    # Worst: lowest val_acc, then highest val_loss
    worst_idx = results_df_sorted.index[-1]
    
    best_result = results[best_idx]
    worst_result = results[worst_idx]
    
    print(f"\n{'='*70}")
    print("BEST MODEL")
    print(f"{'='*70}")
    print(f"Experiment ID: {best_result['experiment_id']}")
    print(f"Architecture: {best_result['hidden_sizes']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Activation: {best_result['activation']}")
    print(f"Init Scale: {best_result['init_scale']}")
    print(f"Train Accuracy: {best_result['final_train_acc']:.4f}")
    print(f"Val Accuracy: {best_result['final_val_acc']:.4f}")
    print(f"Overfitting Gap: {best_result['overfitting_gap']:.4f}")
    
    print(f"\n{'='*70}")
    print("WORST MODEL")
    print(f"{'='*70}")
    print(f"Experiment ID: {worst_result['experiment_id']}")
    print(f"Architecture: {worst_result['hidden_sizes']}")
    print(f"Learning Rate: {worst_result['learning_rate']}")
    print(f"Batch Size: {worst_result['batch_size']}")
    print(f"Activation: {worst_result['activation']}")
    print(f"Init Scale: {worst_result['init_scale']}")
    print(f"Train Accuracy: {worst_result['final_train_acc']:.4f}")
    print(f"Val Accuracy: {worst_result['final_val_acc']:.4f}")
    print(f"Overfitting Gap: {worst_result['overfitting_gap']:.4f}")
    
    # Plot parallel coordinates
    print(f"\n{'='*70}")
    print("Generating Parallel Coordinates Plot...")
    print(f"{'='*70}\n")
    # For parallel coordinates, use only numeric columns
    numeric_df = results_df.select_dtypes(include=[np.number])
    plot_parallel_coordinates(
        numeric_df, 
        save_path=os.path.join(save_dir, 'parallel_coordinates.png'),
        top_k=20  # Show top 10 models (can be adjusted: 5, 15, 20, etc.)
    )
    
    # Plot best model loss curves
    print("Generating Best Model Loss Curves...")
    plot_loss_only(
        best_result['train_loss_history'],
        best_result['val_loss_history'],
        title=f"Best Model Loss Curves (Val Acc: {best_result['final_val_acc']:.4f})\n"
              f"Architecture: {best_result['hidden_sizes']}, LR: {best_result['learning_rate']}, "
              f"BS: {best_result['batch_size']}",
        save_path=os.path.join(save_dir, 'best_model_loss.png')
    )
    
    # Plot worst model loss curves
    print("Generating Worst Model Loss Curves...")
    plot_loss_only(
        worst_result['train_loss_history'],
        worst_result['val_loss_history'],
        title=f"Worst Model Loss Curves (Val Acc: {worst_result['final_val_acc']:.4f})\n"
              f"Architecture: {worst_result['hidden_sizes']}, LR: {worst_result['learning_rate']}, "
              f"BS: {worst_result['batch_size']}",
        save_path=os.path.join(save_dir, 'worst_model_loss.png')
    )
    
    # Analysis of overfitting/underfitting
    print(f"\n{'='*70}")
    print("GENERALIZATION ANALYSIS")
    print(f"{'='*70}\n")
    
    print("BEST MODEL Analysis:")
    print("-" * 70)
    analyze_generalization(best_result, "Best")
    
    print("\nWORST MODEL Analysis:")
    print("-" * 70)
    analyze_generalization(worst_result, "Worst")
    
    return results_df, best_result, worst_result


def analyze_generalization(result, model_name):
    """
    Analyze generalization performance (overfitting/underfitting).
    """
    train_acc = result['final_train_acc']
    val_acc = result['final_val_acc']
    gap = result['overfitting_gap']
    
    train_loss = result['final_train_loss']
    val_loss = result['final_val_loss']
    
    print(f"{model_name} Model Generalization:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Accuracy Gap:        {gap:.4f}")
    print(f"  Training Loss:       {train_loss:.4f}")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Loss Gap:            {val_loss - train_loss:.4f}")
    
    # Diagnose
    print(f"\nDiagnosis:")
    
    if train_acc < 0.7 and val_acc < 0.7:
        print("  ⚠️  UNDERFITTING: Both training and validation accuracies are low.")
        print("     The model is too simple or learning rate is too low.")
        print("     Suggestions: Increase model complexity, increase learning rate,")
        print("                 train for more epochs, or check data quality.")
    
    elif gap > 0.15:  # Large gap between train and val
        print("  ⚠️  OVERFITTING: Training accuracy is much higher than validation.")
        print("     The model memorizes training data but doesn't generalize well.")
        print("     Suggestions: Reduce model complexity, use regularization,")
        print("                 increase training data, or use early stopping.")
    
    elif val_acc > train_acc:
        print("  ℹ️  UNUSUAL: Validation accuracy exceeds training accuracy.")
        print("     This might indicate: small dataset, lucky validation split,")
        print("     or insufficient training.")
    
    elif gap < 0.05 and val_acc > 0.8:
        print("  ✅ GOOD GENERALIZATION: Small gap and good performance.")
        print("     The model generalizes well to unseen data.")
    
    elif gap < 0.10 and val_acc > 0.7:
        print("  ✅ ACCEPTABLE GENERALIZATION: Reasonable gap and decent performance.")
        print("     The model has acceptable generalization capability.")
    
    else:
        print("  ℹ️  MODERATE: The model shows moderate generalization.")
        print("     Performance is acceptable but could be improved.")
    
    # Analyze loss trends
    train_loss_history = result['train_loss_history']
    val_loss_history = result['val_loss_history']
    
    # Check if validation loss increases while training loss decreases
    mid_point = len(train_loss_history) // 2
    train_loss_trend = train_loss_history[-1] - train_loss_history[mid_point]
    val_loss_trend = val_loss_history[-1] - val_loss_history[mid_point]
    
    print(f"\nLoss Trend Analysis (2nd half of training):")
    print(f"  Training loss change:   {train_loss_trend:+.4f}")
    print(f"  Validation loss change: {val_loss_trend:+.4f}")
    
    if train_loss_trend < -0.05 and val_loss_trend > 0.05:
        print("  ⚠️  Classic overfitting pattern: Training loss decreases while")
        print("     validation loss increases in later epochs.")
    elif train_loss_trend < 0 and val_loss_trend < 0:
        print("  ✅ Both losses decreasing: Model is still learning effectively.")
    elif abs(train_loss_trend) < 0.01 and abs(val_loss_trend) < 0.01:
        print("  ℹ️  Losses plateaued: Training has converged.")


# =========================================================
# 6. Main Execution
# =========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MLP Training with Flexible Hyperparameters")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # =====================================================
    # CONFIGURATION: Choose mode
    # =====================================================
    # MODE 1: Single model training
    # MODE 2: Hyperparameter grid search with parallel coordinates
    
    MODE = 2  # Change to 1 for single model, 2 for grid search
    
    # =====================================================
    # LOAD DATA
    # =====================================================
    print("Loading data...")
    X_train, y_train, y_train_int, X_val, y_val, y_val_int = load_tha2_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {y_train.shape[1]}\n")
    
    # =====================================================
    # MODE 1: Single Model Training
    # =====================================================
    if MODE == 1:
        print("\n" + "="*70)
        print("MODE 1: Single Model Training")
        print("="*70 + "\n")
        
        CONFIG = {
            # Network Architecture
            'input_size': 2,
            'hidden_sizes': [20, 15, 10],  # Try: [10, 10], [20, 15, 10], [50, 30, 20, 10]
            'output_size': 2,
            
            # Training Hyperparameters
            'learning_rate': 0.01,         # Try: 0.001, 0.01, 0.1
            'batch_size': 16,              # Try: 4, 16, 32, None (full-batch)
            'epochs': 250,                 # Try: 100, 200, 300
            'init_scale': 0.1,             # Try: 0.01, 0.1, 0.5
            'activation': 'relu',          # Try: 'sigmoid', 'relu'
            
            # Verbosity
            'verbose': True,
            'print_interval': 20,
            
            # Visualization
            'plot_results': True,
            'save_plots': True,
        }
        
        # Print configuration
        print("Configuration:")
        print("-" * 70)
        for key, value in CONFIG.items():
            print(f"  {key:20s}: {value}")
        print("-" * 70 + "\n")
        
        # Create model
        print("Creating MLP model...")
        model = MLP(
            input_size=CONFIG['input_size'],
            hidden_sizes=CONFIG['hidden_sizes'],
            output_size=CONFIG['output_size'],
            learning_rate=CONFIG['learning_rate'],
            activation=CONFIG['activation'],
            init_scale=CONFIG['init_scale']
        )
        
        # Display network architecture
        model.get_network_info()
        
        # Train model
        history = train_mlp(
            model=model,
            X_train=X_train,
            y_train=y_train,
            y_train_int=y_train_int,
            X_val=X_val,
            y_val=y_val,
            y_val_int=y_val_int,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            verbose=CONFIG['verbose'],
            print_interval=CONFIG['print_interval']
        )
        
        # Evaluate model
        train_results = evaluate_model(
            model, X_train, y_train, y_train_int, dataset_name="Training"
        )
        
        val_results = evaluate_model(
            model, X_val, y_val, y_val_int, dataset_name="Validation"
        )
        
        # Visualize results
        if CONFIG['plot_results']:
            arch_str = "_".join(map(str, CONFIG['hidden_sizes']))
            lr_str = str(CONFIG['learning_rate']).replace('.', 'p')
            bs_str = str(CONFIG['batch_size']) if CONFIG['batch_size'] else 'full'
            
            save_path = None
            if CONFIG['save_plots']:
                save_path = f"training_curves_arch{arch_str}_lr{lr_str}_bs{bs_str}.png"
            
            plot_training_curves(
                train_losses=history['train_loss'],
                val_losses=history['val_loss'],
                train_accs=history['train_acc'],
                val_accs=history['val_acc'],
                title=f"MLP Training (layers={CONFIG['hidden_sizes']}, "
                      f"lr={CONFIG['learning_rate']}, bs={CONFIG['batch_size']})",
                save_path=save_path
            )
        
        # Summary
        print("\n" + "="*70)
        print("Training Summary")
        print("="*70)
        print(f"Architecture: {CONFIG['input_size']} -> {' -> '.join(map(str, CONFIG['hidden_sizes']))} -> {CONFIG['output_size']}")
        print(f"Activation: {CONFIG['activation']}")
        print(f"Learning Rate: {CONFIG['learning_rate']}")
        print(f"Batch Size: {CONFIG['batch_size']}")
        print(f"Epochs: {CONFIG['epochs']}")
        print(f"\nFinal Results:")
        print(f"  Training Accuracy:   {train_results['accuracy']:.4f}")
        print(f"  Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"  Training Loss:       {train_results['loss']:.4f}")
        print(f"  Validation Loss:     {val_results['loss']:.4f}")
        print("="*70 + "\n")
    
    # =====================================================
    # MODE 2: Hyperparameter Grid Search
    # =====================================================
    elif MODE == 2:
        print("\n" + "="*70)
        print("MODE 2: Hyperparameter Grid Search with Parallel Coordinates")
        print("="*70 + "\n")
        
        # Define hyperparameter grid
        # Modify these to test different combinations
        HYPERPARAMETER_GRID = {
            'hidden_sizes': [
                [7, 7],
                [10, 10],
                [15, 15],
                [10, 15],
                [10, 15, 10],
                [15, 10, 15],
                [30, 15, 10],
            ],
            'learning_rate': [0.01, 0.05, 0.1],
            'batch_size': [4, 8, 16],
            'activation': ['relu', 'sigmoid'],
            'init_scale': [0.05, 0.1, 0.5],
        }
        
        # Print grid configuration
        print("Hyperparameter Grid:")
        print("-" * 70)
        for key, values in HYPERPARAMETER_GRID.items():
            print(f"  {key:20s}: {values}")
        
        total_combinations = np.prod([len(v) for v in HYPERPARAMETER_GRID.values()])
        print(f"\nTotal combinations to test: {total_combinations}")
        print("-" * 70 + "\n")
        
        # Run experiments
        results = run_hyperparameter_experiments(
            X_train, y_train, y_train_int,
            X_val, y_val, y_val_int,
            HYPERPARAMETER_GRID,
            epochs=150,  # Use fewer epochs for faster grid search
            verbose=False  # Set to True to see detailed training for each experiment
        )
        
        # Analyze results
        results_df, best_result, worst_result = analyze_results(results, save_dir='Plots_B4/')
        
        # Additional summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Average Validation Accuracy: {results_df['final_val_acc'].mean():.4f} ± {results_df['final_val_acc'].std():.4f}")
        print(f"Best Validation Accuracy:    {results_df['final_val_acc'].max():.4f}")
        print(f"Worst Validation Accuracy:   {results_df['final_val_acc'].min():.4f}")
        print(f"Average Overfitting Gap:     {results_df['overfitting_gap'].mean():.4f} ± {results_df['overfitting_gap'].std():.4f}")
        print(f"{'='*70}\n")
    
    # =====================================================
    # EXPERIMENT SUGGESTIONS (for single model mode)
    # =====================================================
    if MODE == 1:
        print("\n" + "="*70)
        print("Experimentation Tips")
        print("="*70)
        print("""
1. Network Architecture:
   - Shallow: [10, 10] or [20, 15]
   - Medium: [20, 15, 10] or [30, 20, 10]
   - Deep: [50, 30, 20, 10] or [40, 30, 20, 10, 5]

2. Learning Rate:
   - Small: 0.001 (slower, more stable)
   - Medium: 0.01 (balanced)
   - Large: 0.1 (faster, may be unstable)

3. Batch Size:
   - Small: 4 or 8 (noisy gradients, regularization effect)
   - Medium: 16 or 32 (balanced)
   - Large: 64 or None (full-batch, smooth convergence)

4. Activation Function:
   - 'sigmoid': Traditional, smooth gradients
   - 'relu': Modern, faster training, helps with deep networks

5. Initialization Scale:
   - Small: 0.01 (conservative)
   - Medium: 0.1 (standard)
   - Large: 0.5 or 1.0 (bold, may help break symmetry)

To experiment, simply modify the CONFIG dictionary at the top of the
main section and run the script again!

To run hyperparameter grid search, set MODE = 2 and modify HYPERPARAMETER_GRID.
        """)
        print("="*70 + "\n")
