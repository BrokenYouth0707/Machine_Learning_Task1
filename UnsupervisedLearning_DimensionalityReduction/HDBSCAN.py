import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import os


def fit_hdbscan(X: np.ndarray, 
                min_cluster_size: int = 5, 
                min_samples: Optional[int] = None,
                metric: str = 'euclidean', 
                cluster_selection_epsilon: float = 0.0) -> Tuple[HDBSCAN, np.ndarray, np.ndarray]:
    """
    Fit HDBSCAN model on data.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        min_cluster_size: The minimum number of samples in a group for that group to be considered a cluster
        min_samples: The parameter k used to calculate the distance between a point and its k-th nearest neighbor
        metric: Distance metric to use
        cluster_selection_epsilon: A distance threshold. Clusters below this value will be merged. 
        
    Returns:
        Tuple of (model, labels, probabilities)
        - labels: Cluster labels for each point in the dataset given to fit. Outliers are labeled as follows:
            1. Noisy samples are given the label -1
            2. Samples with infinite elements (+/- np.inf) are given the label -2.
            3. Samples with missing data are given the label -3, even if they also have infinite elements.
        - probabilities: The strength with which each sample is a member of its assigned cluster.
            1. Clustered samples have probabilities proportional to the degree that they persist as part of the cluster.
            2. Noisy samples have probability zero.
            3. Samples with infinite elements (+/- np.inf) have probability 0.
            4. Samples with missing data have probability np.nan.
    """
    print(f"\n{'='*60}")
    print("Training HDBSCAN Model")
    print(f"{'='*60}")
    print(f"Data shape: {X.shape}")
    print(f"Parameters:")
    print(f"  - min_cluster_size: {min_cluster_size}")
    print(f"  - min_samples: {min_samples}")
    print(f"  - metric: {metric}")
    print(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
    print(f"  - cluster_selection_method: EOM")
    
    # Create and fit HDBSCAN model
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method='eom'
    )
    # Cluster X and return the associated cluster labels.
    labels = model.fit_predict(X)
    # Get membership probabilities for each point
    probabilities = model.probabilities_
    
    # Print clustering results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clustered_samples = 0
    for lbl in labels:
        n_clustered_samples += (1 if lbl >= 0 else 0)
    n_noise = list(labels).count(-1)

    # For each cluster, print the corresponding number of samples, except noise
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    n_clustered_samples = 0
    if len(unique_labels) > 0:
        print(f"\nCluster Sizes:")
        for label, count in zip(unique_labels, counts):
            n_clustered_samples += count
            print(f"  - Cluster {label}: {count} samples")

    print(f"\nClustering Results:")
    print(f"  - Number of clusters found: {n_clusters}")
    print(f"  - Number of clustered samples: {n_clustered_samples} ({n_clustered_samples/len(X)*100:.2f}%)")
    print(f"  - Number of noise points: {n_noise} ({n_noise/len(X)*100:.2f}%)")

    return model, labels, probabilities


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, 
                        y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate clustering performance.
    
    Args:
        X: Input data
        labels: Predicted cluster labels
        y_true: True labels (optional, for supervised metrics)
        
    Returns:
        Dictionary of evaluation metrics
        - silhouette_score: Silhouette Score of the clustering (higher is better)
        - adjusted_rand_score: Adjusted Rand Index (if y_true provided)
    """
    metrics = {}
    
    # Calculate silhouette score (exclude noise points)
    valid_mask = labels != -1
    if valid_mask.sum() > 0 and len(set(labels[valid_mask])) > 1:
        metrics['silhouette_score'] = silhouette_score(
            X[valid_mask], labels[valid_mask]
        )
    else:
        metrics['silhouette_score'] = None
    
    # If true labels provided, calculate supervised metrics
    if y_true is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(y_true, labels)
    
    print(f"\n{'='*60}")
    print("Evaluation Metrics")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: N/A")
    
    return metrics


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray,
                     title: str = "HDBSCAN Clustering", 
                     save_path: Optional[str] = None):
    """
    Plot 2D visualization of clusters.
    
    Args:
        X_2d: 2D projection of data (n_samples, 2)
        labels: Cluster labels
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster with different color
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'k'
            marker = 'x'
            label_name = 'Noise'
            alpha = 0.3
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
            alpha = 0.6
        
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], marker=marker, s=50, alpha=alpha,
            edgecolors='k', linewidth=0.5, label=label_name
        )
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot] saved {save_path}")
    
    plt.close()


def train_hdbscan_olivetti(X: np.ndarray, y: np.ndarray, 
                          min_cluster_size: int = 5,
                          min_samples: Optional[int] = None,
                          output_dir: str = "clustering_results") -> Tuple[HDBSCAN, np.ndarray, Dict[str, float]]:
    """
    Train HDBSCAN on Olivetti faces dataset with complete workflow.
    
    Args:
        X: PCA-transformed features (n_samples, n_components)
        y: True labels (for evaluation only)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples parameter
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (model, labels, metrics dictionary)
    """
    # Fit HDBSCAN model
    model, labels, probabilities = fit_hdbscan(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    # Evaluate
    metrics = evaluate_clustering(X, labels, y_true=y)
    
    # # Create 2D visualization using PCA
    # print("\nCreating 2D visualization...")
    # pca_2d = PCA(n_components=2, random_state=42)
    # X_2d = pca_2d.fit_transform(X)
    
    # # Create output directory
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Plot results
    # plot_clusters_2d(
    #     X_2d, labels,
    #     title="HDBSCAN Clustering - Olivetti Faces (PCA 2D projection)",
    #     save_path=os.path.join(output_dir, "hdbscan_olivetti_clusters.png")
    # )
    
    # print(f"\nAll results saved in: {output_dir}")
    
    return model, labels, metrics


if __name__ == "__main__":
    # Test with example data
    from Datasets.Olivetti_Faces_Dataset import load_olivetti_pca, load_olivetti_raw
    
    print("Loading Olivetti Faces Dataset...")
    X, y = load_olivetti_pca(n_components=40)
    #X, y = load_olivetti_raw()
    
    print("Dataset info:")
    print(f"  - Shape: {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    print(f"  - Number of unique labels: {len(np.unique(y))}")
    
    # Train HDBSCAN with complete workflow
    model, labels, metrics = train_hdbscan_olivetti(
        X, y, 
        min_cluster_size=5, 
        min_samples=3,
        output_dir="clustering_results"
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
